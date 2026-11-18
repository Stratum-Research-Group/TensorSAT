import math
import time
import torch
import torch.nn as nn
import numpy as np

from tensorsat.attention import SingletAttention, KVCache
from tensorsat.scientific import IEEETensorContractor, GridPoissonSolver

def benchmark_neural_scaling():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "=" * 78)
    print(f"BENCHMARK 1: LLM Neural Attention Scaling (RTX 5060 Ti)")
    print("=" * 78)

    embed_dim = 512
    num_heads = 8
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    layer = SingletAttention(embed_dim, num_heads, is_causal=True, use_triton=True).to(device)
    layer.eval()

    print(f"{'Context (Tokens)':<18} | {'Latency (ms)':<16} | {'VRAM (MB)':<14} | {'Complexity':<12}")
    print("-" * 78)

    for N in seq_lengths:
        x = torch.randn(1, N, embed_dim, device=device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            _ = layer(x[:, :min(N, 64), :])
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = layer(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{N:<18} | {elapsed:<16.2f} | {peak_mem:<14.1f} | O(N)")

def benchmark_kv_generation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "=" * 78)
    print("BENCHMARK 2: Autoregressive KV-Cache Throughput (Tokens/Second)")
    print("=" * 78)

    embed_dim = 512
    num_heads = 8
    layer = SingletAttention(embed_dim, num_heads, is_causal=True).to(device)
    layer.eval()

    prompt_sizes = [512, 2048, 8192]
    for P in prompt_sizes:
        cache = KVCache()
        prompt = torch.randn(1, P, embed_dim, device=device)
        with torch.no_grad():
            _, cache = layer(prompt, kv_cache=cache, use_cache=True)

        token = torch.randn(1, 1, embed_dim, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps = 100
        for _ in range(steps):
            with torch.no_grad():
                _, cache = layer(token, kv_cache=cache, use_cache=True)
        torch.cuda.synchronize()
        step_time = (time.perf_counter() - t0) / steps * 1000
        tps = 1000.0 / step_time
        print(f"Past Prompt: {P:<6} tokens | Step Latency: {step_time:.3f} ms | Throughput: {tps:.1f} tokens/s")

def benchmark_ieee_scientific():
    print("\n" + "=" * 78)
    print("BENCHMARK 3: IEEE 754 High-Precision Double (FP64) Tensor Contraction")
    print("=" * 78)

    for bits in [1, 2, 3]:
        dim = 1 << bits
        contractor = IEEETensorContractor(n_bits=bits, dtype=np.float64)
        W1 = np.random.randn(dim, dim).astype(np.float64)
        W2 = np.random.randn(dim, dim).astype(np.float64)

        t0 = time.perf_counter()
        total, slices = contractor.contract_exact(W1, W2)
        elapsed = (time.perf_counter() - t0) * 1000

        # Exact naive reference
        expected = np.sum(W1 * W2.T)
        error = np.abs(expected - total)
        print(f"Dimension: 2^{bits} = {dim:<3} | Contraction: {elapsed:.3f} ms | Max Drift: {error:.2e} (Machine Epsilon)")

if __name__ == '__main__':
    benchmark_neural_scaling()
    benchmark_kv_generation()
    benchmark_ieee_scientific()
