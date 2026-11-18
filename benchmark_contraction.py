import itertools
import time

def s(p: int, q: int) -> int:
    """Singlet phase factor: s(p,q) = (-1)^p [p != q]"""
    if p == q:
        return 0
    return -1 if p == 1 else 1

def uncrossing_slice_eval(n: int, i: int, x: int, y: int, u: int, v: int) -> int:
    """
    Evaluates slice A_i(x, y, u, v) without allocating dense matrices.
    """
    total = 0
    indices = range(n)
    all_subsets = list(itertools.combinations(indices, i))
    
    for S in all_subsets:
        S_set = set(S)
        prod_S = 1
        for k in S:
            xk = (x >> k) & 1
            yk = (y >> k) & 1
            uk = (u >> k) & 1
            vk = (v >> k) & 1
            prod_S *= -(s(xk, yk) * s(uk, vk))
            if prod_S == 0:
                break
        if prod_S == 0:
            continue
        
        match_rest = True
        for k in range(n):
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

def benchmark_contraction(n: int = 2):
    """
    Pure Python benchmark comparing direct diagonal crossing contraction
    versus singlet-decomposed branch contraction.
    """
    dim = 1 << n
    print(f"\n=======================================================")
    print(f"TensorSAT Contraction Benchmark (Pure Python)")
    print(f"Parameters: lambda = {n} bits per line, dimension = {dim}")
    print(f"=======================================================")
    
    # Generate deterministic boundary tensors W1(x, y) and W2(u, v)
    W1 = {}
    W2 = {}
    for a in range(dim):
        for b in range(dim):
            W1[(a, b)] = ((a * 7 + b * 13 + 3) % 11) - 5
            W2[(a, b)] = ((a * 17 + b * 5 + 7) % 13) - 6
            
    # Method 1: Direct Naive Contraction
    # Z_direct = sum_{x,y} W1(x,y) * W2(y,x)  (since crossing enforces u=y, v=x)
    t0 = time.perf_counter()
    Z_direct = 0
    for x in range(dim):
        for y in range(dim):
            Z_direct += W1[(x, y)] * W2[(y, x)]
    t_direct = time.perf_counter() - t0
    
    # Method 2: Singlet Decomposition Slices
    # Z_decomposed = sum_{i=0}^n [ sum_{x,y,u,v} W1(x,y) * A_i(x,y,u,v) * W2(u,v) ]
    t0 = time.perf_counter()
    Z_decomposed = 0
    branch_contributions = []
    
    for i in range(n + 1):
        Z_slice = 0
        for x in range(dim):
            for y in range(dim):
                w1_val = W1[(x, y)]
                if w1_val == 0:
                    continue
                for u in range(dim):
                    for v in range(dim):
                        w2_val = W2[(u, v)]
                        if w2_val == 0:
                            continue
                        coeff = uncrossing_slice_eval(n, i, x, y, u, v)
                        if coeff != 0:
                            Z_slice += w1_val * coeff * w2_val
        branch_contributions.append(Z_slice)
        Z_decomposed += Z_slice
    t_decomposed = time.perf_counter() - t0
    
    print(f"Direct Contraction Result:     Z = {Z_direct}")
    print(f"Decomposed Slices Sum:         Z = {Z_decomposed}")
    print(f"Branch Contributions A_0..A_{n}: {branch_contributions}")
    print(f"Exact Match:                   {Z_direct == Z_decomposed}")
    print(f"Direct Contraction Time:       {t_direct * 1000:.3f} ms")
    print(f"Decomposed Evaluation Time:    {t_decomposed * 1000:.3f} ms")

if __name__ == "__main__":
    benchmark_contraction(n=1)
    benchmark_contraction(n=2)
    benchmark_contraction(n=3)
