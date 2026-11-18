"""
TensorSAT Pre-Trained Weight Adapter: Softmax-to-Singlet Translation Layer.

Addresses Trade-Off B:
Enables existing pre-trained transformer weights (e.g., Llama-3, Mistral, Gemma)
to run through TensorSAT linear singlet attention without full re-training:
1. Wraps existing Q, K, V projections and maps their exponential softmax geometry
   into an equivalent planar singlet invariant space via learned Kernel Transduction.
2. Supports fast feature-matching alignment (distillation) in hours rather than months.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxToSingletAdapter(nn.Module):
    """
    Adapter layer placed between pre-trained Q/K projections and Singlet Attention.
    Learns the feature map phi(Q), phi(K) such that:
        phi(q_i)^T phi(k_j) ~= exp(q_i k_j^T / sqrt(d))
    Using normalized positive random/learned feature kernel basis.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_singlets: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_singlets = num_singlets

        # Learned transduction kernels for queries and keys
        # Initialize orthogonal / normalized feature projections
        self.q_trans = nn.Parameter(
            torch.randn(num_heads, self.head_dim, num_singlets) * (1.0 / math.sqrt(self.head_dim))
        )
        self.k_trans = nn.Parameter(
            torch.randn(num_heads, self.head_dim, num_singlets) * (1.0 / math.sqrt(self.head_dim))
        )

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Q: (B, H, N, d) pre-trained queries
        K: (B, H, N, d) pre-trained keys
        Returns:
            Q_s: (B, H, N, R) transformed query singlets
            K_s: (B, H, N, R) transformed key singlets
        """
        # Unit-normalize Q and K along head_dim to stabilize kernel magnitude
        q_norm = F.normalize(Q, p=2, dim=-1)
        k_norm = F.normalize(K, p=2, dim=-1)

        q_proj = torch.einsum("bhnd,hdr->bhnr", q_norm, self.q_trans)
        k_proj = torch.einsum("bhnd,hdr->bhnr", k_norm, self.k_trans)

        # Smooth positive kernel: ELU(x) + 1.0 > 0
        Q_s = F.elu(q_proj) + 1.0
        K_s = F.elu(k_proj) + 1.0

        return Q_s, K_s

    def compute_alignment_loss(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Feature Matching Distillation Loss:
        Computes cosine similarity + MSE between standard Softmax Attention output
        and adapted Singlet Linear Attention output.
        """
        # 1. Target output: Standard Softmax Attention
        target_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # 2. Predicted output: Adapted Singlet Linear Attention
        Q_s, K_s = self.forward(Q, K)  # (B, H, N, R)

        token_kv = K_s.unsqueeze(-1) * V.unsqueeze(-2)  # (B, H, N, R, d)
        causal_kv = torch.cumsum(token_kv, dim=2)
        pred_unnorm = torch.einsum("bhnr,bhnrd->bhnd", Q_s, causal_kv)

        # Normalization denominator
        causal_k = torch.cumsum(K_s, dim=2)  # (B, H, N, R)
        pred_norm = torch.einsum("bhnr,bhnr->bhn", Q_s, causal_k).unsqueeze(-1) + 1e-5
        pred_out = pred_unnorm / pred_norm

        # Cosine embedding loss + MSE for stable convergence
        cos_loss = 1.0 - F.cosine_similarity(pred_out, target_out, dim=-1).mean()
        mse_loss = F.mse_loss(pred_out, target_out)
        return cos_loss + 0.1 * mse_loss
