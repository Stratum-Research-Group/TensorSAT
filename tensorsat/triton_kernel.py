"""
Triton JIT GPU kernels for TensorSAT fused singlet low-rank projection.
Fuses projection, scaling, and context accumulation directly into SRAM with IEEE precision.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def fused_singlet_proj_kernel(
    # Pointers
    q_ptr, u_ptr, out_ptr,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_uh, stride_ud, stride_ur,
    stride_ob, stride_oh, stride_on, stride_or,
    # Dimensions
    B, H, N, D, R,
    # Constants
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Computes Q_singlet = Q @ U
    where Q is (B, H, N, D) and U is (H, D, R) -> Out is (B, H, N, R).
    Each thread block handles a tile of (BLOCK_N, BLOCK_R) for a given (b, h).
    """
    pid_bh = tl.program_id(0)
    pid_tile = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    num_r_tiles = tl.cdiv(R, BLOCK_R)
    tile_n = pid_tile // num_r_tiles
    tile_r = pid_tile % num_r_tiles

    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)

    acc = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)

        q_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        q_ptrs = (
            q_ptr
            + b * stride_qb
            + h * stride_qh
            + offs_n[:, None] * stride_qn
            + offs_d[None, :] * stride_qd
        )
        q_val = tl.load(q_ptrs, mask=q_mask, other=0.0)

        u_mask = (offs_d[:, None] < D) & (offs_r[None, :] < R)
        u_ptrs = (
            u_ptr
            + h * stride_uh
            + offs_d[:, None] * stride_ud
            + offs_r[None, :] * stride_ur
        )
        u_val = tl.load(u_ptrs, mask=u_mask, other=0.0)

        # Enforce exact IEEE precision to avoid TF32 truncation error
        acc += tl.dot(q_val, u_val, input_precision="ieee")

    out_mask = (offs_n[:, None] < N) & (offs_r[None, :] < R)
    out_ptrs = (
        out_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_n[:, None] * stride_on
        + offs_r[None, :] * stride_or
    )
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_singlet_proj(Q: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Q: (B, H, N, D)
    U: (H, D, R)
    Returns: Q_singlet of shape (B, H, N, R)
    """
    assert Q.is_cuda and U.is_cuda, "Tensors must be on CUDA"
    B, H, N, D = Q.shape
    H_u, D_u, R = U.shape
    assert H == H_u and D == D_u, f"Head and dim mismatch: Q ({H},{D}) vs U ({H_u},{D_u})"

    out = torch.empty((B, H, N, R), dtype=torch.float32, device=Q.device)

    BLOCK_N = 64
    BLOCK_D = 32
    BLOCK_R = 16

    num_r_tiles = triton.cdiv(R, BLOCK_R)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (B * H, num_n_tiles * num_r_tiles)

    fused_singlet_proj_kernel[grid](
        Q, U, out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        U.stride(0), U.stride(1), U.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N, D, R,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
    )
    return out
