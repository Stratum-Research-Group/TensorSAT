import math
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tensorsat.triton_kernel import triton_singlet_proj
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class KVCache:
    """
    Key-Value & Singlet Cache for fast autoregressive text generation.
    Stores past projected singlet representations to achieve O(1) step latency.
    """
    def __init__(self):
        self.cached_k: Optional[torch.Tensor] = None
        self.cached_v: Optional[torch.Tensor] = None
        self.cached_kv_context: Optional[torch.Tensor] = None
        self.seq_len: int = 0

    def update(self, k: torch.Tensor, v: torch.Tensor, kv_context: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if self.cached_kv_context is None:
            self.cached_kv_context = kv_context
            self.seq_len = k.shape[2]
        else:
            self.cached_kv_context = self.cached_kv_context + kv_context
            self.seq_len += k.shape[2]
        return self.cached_k, self.cached_v, self.cached_kv_context

    def reset(self):
        self.cached_k = None
        self.cached_v = None
        self.cached_kv_context = None
        self.seq_len = 0


class SingletAttention(nn.Module):
    """
    Production-grade Sub-quadratic SingletAttention layer.
    Features:
      1. Triton JIT fused GPU kernels for low-rank singlet projections.
      2. Memory-efficient chunked causal masking (O(chunk) memory instead of O(N*R*d)).
      3. Incremental KV Caching for instant token-by-token generation.
      4. Automatic fallback to PyTorch tensor contractions on CPU.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 64,
        num_singlets: int = 16,
        is_causal: bool = True,
        use_triton: bool = True,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.num_singlets = num_singlets
        self.is_causal = is_causal
        self.use_triton = use_triton and HAS_TRITON
        self.chunk_size = chunk_size

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert chunk_size > 0, "chunk_size must be positive"
        assert num_singlets > 0, "num_singlets must be positive"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.singlet_u = nn.Parameter(torch.randn(num_heads, self.head_dim, num_singlets) * 0.02)
        self.singlet_v = nn.Parameter(torch.randn(num_heads, self.head_dim, num_singlets) * 0.02)

    def _project_singlets(self, X: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if self.use_triton and X.is_cuda and basis.is_cuda and not (torch.is_grad_enabled() and (X.requires_grad or basis.requires_grad)):
            return triton_singlet_proj(X, basis)
        return torch.einsum('bhnd,hdr->bhnr', X, basis)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project Q, K, V
        Q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        K = self.k_proj(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        V = self.v_proj(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)

        # Compute Low-Rank Singlet Projections
        Q_s = self._project_singlets(Q, self.singlet_u)  # (B, H, N, R)
        K_s = self._project_singlets(K, self.singlet_v)  # (B, H, N, R)

        # Autoregressive Generation with KV Cache
        if use_cache:
            if kv_cache is None:
                kv_cache = KVCache()
            start_pos = kv_cache.seq_len
            prev_context = kv_cache.cached_kv_context
            if prev_context is None:
                prev_context = torch.zeros((B, H, self.num_singlets, d), dtype=Q_s.dtype, device=x.device)

            out_global = torch.empty((B, H, N, d), dtype=Q_s.dtype, device=x.device)
            if self.is_causal and N > 1:
                # Process cached appends in chunks bounded by self.chunk_size
                chunk = self.chunk_size
                running_context = prev_context
                for start in range(0, N, chunk):
                    end = min(start + chunk, N)
                    C = end - start
                    qc = Q_s[:, :, start:end, :]
                    kc = K_s[:, :, start:end, :]
                    vc = V[:, :, start:end, :]

                    # 1. Prior context contribution (from previous cache & prior chunks in this append)
                    out_chunk = torch.einsum('bhcr,bhrd->bhcd', qc, running_context)

                    # 2. Causal intra-chunk contribution
                    token_kv = kc.unsqueeze(-1) * vc.unsqueeze(-2)  # (B, H, C, R, d)
                    causal_intra = torch.cumsum(token_kv, dim=2)
                    out_intra = torch.einsum('bhcr,bhcrd->bhcd', qc, causal_intra)
                    out_chunk = out_chunk + out_intra

                    # Normalize by causal position square-root in float32 to prevent FP16 denominator overflow
                    curr_start_pos = start_pos + start
                    pos_norm = torch.sqrt(
                        torch.arange(curr_start_pos + 1, curr_start_pos + C + 1, device=x.device, dtype=torch.float32)
                    ).view(1, 1, C, 1) * math.sqrt(d)
                    out_global[:, :, start:end, :] = (out_chunk.float() / pos_norm).to(out_chunk.dtype)

                    # Accumulate chunk context into running state
                    running_context = running_context + torch.einsum('bhcr,bhcd->bhrd', kc, vc)
            elif self.is_causal and N == 1:
                token_kv = torch.einsum('bhnr,bhnd->bhnrd', K_s, V)
                out_chunk = torch.einsum('bhnr,bhrd->bhnd', Q_s, prev_context) + torch.einsum('bhnr,bhnrd->bhnd', Q_s, token_kv)
                pos_norm = math.sqrt(start_pos + 1) * math.sqrt(d)
                out_global = (out_chunk.float() / pos_norm).to(out_chunk.dtype)
            else:
                new_kv = torch.einsum('bhnr,bhnd->bhrd', K_s, V)
                out_chunk = torch.einsum('bhnr,bhrd->bhnd', Q_s, prev_context + new_kv)
                total_seq_len = start_pos + N
                pos_norm = math.sqrt(d * total_seq_len)
                out_global = (out_chunk.float() / pos_norm).to(out_chunk.dtype)

            chunk_kv_context = torch.einsum('bhnr,bhnd->bhrd', K_s, V)
            kv_cache.update(K, V, chunk_kv_context)

            out = out_global.transpose(1, 2).contiguous().view(B, N, D)
            return self.out_proj(out), kv_cache

        # Prefill / Training Mode (Parallel sequence)
        if self.is_causal and N > 1:
            # Memory-Efficient Chunked Causal Prefix Evaluation
            # Avoids allocating massive (B, H, N, R, d) tensors in VRAM
            out_global = torch.empty((B, H, N, d), dtype=Q_s.dtype, device=x.device)
            running_kv_context = torch.zeros((B, H, self.num_singlets, d), dtype=Q_s.dtype, device=x.device)

            chunk = self.chunk_size
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                q_chunk = Q_s[:, :, start:end, :]  # (B, H, C, R)
                k_chunk = K_s[:, :, start:end, :]  # (B, H, C, R)
                v_chunk = V[:, :, start:end, :]    # (B, H, C, d)
                C = end - start

                # 1. Contribution from all prior tokens (inter-chunk context)
                out_chunk = torch.einsum('bhcr,bhrd->bhcd', q_chunk, running_kv_context)

                # 2. Intra-chunk causal contribution
                token_kv_chunk = k_chunk.unsqueeze(-1) * v_chunk.unsqueeze(-2)  # (B, H, C, R, d)
                causal_intra = torch.cumsum(token_kv_chunk, dim=2)
                out_intra = torch.einsum('bhcr,bhcrd->bhcd', q_chunk, causal_intra)

                out_chunk = out_chunk + out_intra

                # Normalize by causal position square-root in float32
                pos_norm = torch.sqrt(torch.arange(start + 1, end + 1, device=x.device, dtype=torch.float32)).view(1, 1, C, 1) * math.sqrt(d)
                out_global[:, :, start:end, :] = (out_chunk.float() / pos_norm).to(out_chunk.dtype)

                # Update running inter-chunk context
                running_kv_context = running_kv_context + torch.einsum('bhcr,bhcd->bhrd', k_chunk, v_chunk)
        else:
            kv_context = torch.einsum('bhnr,bhnd->bhrd', K_s, V) / math.sqrt(N)
            out_global = torch.einsum('bhnr,bhrd->bhnd', Q_s, kv_context) / math.sqrt(d)

        out = out_global.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out), None
