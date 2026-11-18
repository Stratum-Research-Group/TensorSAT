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

    def update(self, k: torch.Tensor, v: torch.Tensor, kv_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cached_k is None or self.cached_v is None or self.cached_kv_context is None:
            self.cached_k = k
            self.cached_v = v
            self.cached_kv_context = kv_context
        else:
            self.cached_k = torch.cat([self.cached_k, k], dim=2)
            self.cached_v = torch.cat([self.cached_v, v], dim=2)
            self.cached_kv_context = self.cached_kv_context + kv_context
        self.seq_len = self.cached_k.shape[2]
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

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.singlet_u = nn.Parameter(torch.randn(num_heads, self.head_dim, num_singlets) * 0.02)
        self.singlet_v = nn.Parameter(torch.randn(num_heads, self.head_dim, num_singlets) * 0.02)

    def _project_singlets(self, X: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if self.use_triton and X.is_cuda and basis.is_cuda:
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
        if use_cache and kv_cache is not None:
            new_kv_context = torch.einsum('bhnr,bhnd->bhrd', K_s, V)
            all_k, all_v, total_kv_context = kv_cache.update(K, V, new_kv_context)
            total_seq_len = kv_cache.seq_len

            out_global = torch.einsum('bhnr,bhrd->bhnd', Q_s, total_kv_context) / math.sqrt(d * total_seq_len)
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

                # Normalize by causal position square-root
                pos_norm = torch.sqrt(torch.arange(start + 1, end + 1, device=x.device, dtype=x.dtype)).view(1, 1, C, 1) * math.sqrt(d)
                out_global[:, :, start:end, :] = out_chunk / pos_norm

                # Update running inter-chunk context
                running_kv_context = running_kv_context + torch.einsum('bhcr,bhcd->bhrd', k_chunk, v_chunk)
        else:
            kv_context = torch.einsum('bhnr,bhnd->bhrd', K_s, V) / math.sqrt(N)
            out_global = torch.einsum('bhnr,bhrd->bhnd', Q_s, kv_context) / math.sqrt(d)

        out = out_global.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out), None
