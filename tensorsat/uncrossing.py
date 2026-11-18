import itertools
from typing import Tuple, List
import torch

def s_factor(p: int, q: int) -> int:
    """
    Phase factor s(p, q) = (-1)^p [p != q].
    Matches Lean 4 definition: Groundwork.s
    """
    if p == q:
        return 0
    return -1 if p == 1 else 1

def eval_slice_A(n_bits: int, i: int, x: int, y: int, u: int, v: int) -> int:
    """
    Evaluates slice A_i(x, y, u, v) without dense matrix allocation.
    sum over subsets S of size i:
      term = prod_{k in S} (-s(x_k, y_k)*s(u_k, v_k)) * prod_{k not in S} [x_k == u_k and y_k == v_k]
    """
    indices = range(n_bits)
    all_subsets = list(itertools.combinations(indices, i))
    total = 0
    
    for S in all_subsets:
        S_set = set(S)
        prod_S = 1
        for k in S:
            xk = (x >> k) & 1
            yk = (y >> k) & 1
            uk = (u >> k) & 1
            vk = (v >> k) & 1
            prod_S *= -(s_factor(xk, yk) * s_factor(uk, vk))
            if prod_S == 0:
                break
        if prod_S == 0:
            continue
        
        match_rest = True
        for k in range(n_bits):
            if k not in S_set:
                xk = (x >> k) & 1
                yk = (y >> k) & 1
                uk = (u >> k) & 1
                vk = (v >> k) & 1
                if not (xk == uk and yk == vk):
                    match_rest = False
                    break
        if match_rest:
            total += prod_S
    return total

def generate_uncrossing_tensor(n_bits: int, device: str = "cpu") -> List[torch.Tensor]:
    """
    Generates the n_bits + 1 slice tensors A_0 ... A_n as PyTorch tensors.
    Each slice has shape (2^n, 2^n, 2^n, 2^n).
    """
    dim = 1 << n_bits
    slices = []
    for i in range(n_bits + 1):
        tensor = torch.zeros((dim, dim, dim, dim), dtype=torch.float32, device=device)
        for x in range(dim):
            for y in range(dim):
                for u in range(dim):
                    for v in range(dim):
                        val = eval_slice_A(n_bits, i, x, y, u, v)
                        if val != 0:
                            tensor[x, y, u, v] = float(val)
        slices.append(tensor)
    return slices
