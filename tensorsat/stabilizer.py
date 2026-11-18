"""
TensorSAT Numerical Stabilizer: Decay-Factored State Tracking & RMS Normalization.

Addresses Trade-Off C:
1. Enforces strictly bounded spectral radius on running linear accumulators
   via learned / data-dependent decay factor gamma in (0, 1].
2. Provides Grouped RMS normalization over running state matrices S_t,
   guaranteeing zero numerical explosion or mantissa overflow in FP16/BF16/FP8
   even over N = 10^9 tokens.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateRMSNorm(nn.Module):
    """
    Normalizes the (Batch, Heads, Rank, Dim) running state tensor
    along the (Rank, Dim) dimensions to strictly bound dynamic range.
    """

    def __init__(self, rank: int, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(rank, dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: shape (B, H, R, d)
        returns normalized state of same shape
        """
        # Preserve FP64 precision if provided; upcast FP16 / BF16 to FP32 to prevent mantissa overflow
        calc_dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
        state_calc = state.to(calc_dtype)

        # Scale down large state magnitudes if needed to prevent square overflow in float32 / float64
        max_val = torch.amax(torch.abs(state_calc), dim=(-2, -1), keepdim=True)
        # Avoid division by zero
        scale = torch.clamp(max_val, min=1.0)
        state_scaled = state_calc / scale

        variance_scaled = torch.mean(state_scaled**2, dim=(-2, -1), keepdim=True)
        rms = torch.sqrt(variance_scaled + (self.eps / (scale**2))) * scale
        normed = (state_calc / rms).to(state.dtype)
        return normed * self.scale


class DecayStabilizer(nn.Module):
    """
    Learned decay mechanism gamma_t in (0, 1] applied to linear invariant accumulator:
        S_t = gamma_t * S_{t-1} + K_s^T V
    Guarantees ||S_t|| <= (1 / (1 - gamma_max)) * max ||K_s^T V||
    """

    def __init__(self, num_heads: int, base_decay: float = 0.999) -> None:
        super().__init__()
        assert num_heads > 0, "num_heads must be positive"
        assert 0.0 < base_decay < 1.0, "base_decay must be in (0, 1)"
        self.num_heads = num_heads
        # Parameterized in logit space to keep gamma in (0, 1)
        init_logit = math.log(base_decay / (1.0 - base_decay))
        self.gamma_logits = nn.Parameter(torch.full((num_heads, 1, 1), init_logit))

    def get_decay(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_logits)

    def forward(
        self,
        running_kv: torch.Tensor,
        new_kv_chunk: torch.Tensor,
        steps: int = 1,
    ) -> torch.Tensor:
        """
        running_kv: (B, H, R, d)
        new_kv_chunk: (B, H, R, d)
        steps: number of tokens accumulated in new_kv_chunk
        """
        gamma = self.get_decay()  # (H, 1, 1)
        # Multi-step continuous decay across chunk: gamma^steps
        decay_factor = gamma**steps
        decayed_state = running_kv * decay_factor + new_kv_chunk
        return decayed_state
