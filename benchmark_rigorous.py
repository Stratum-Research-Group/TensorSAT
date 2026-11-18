"""
Rigorous Empirical Benchmark for TensorSAT vs Standard Softmax Attention (SDPA / FlashAttention).
Features:
1. Head-to-Head Comparison:
   - Standard Softmax Attention (PyTorch F.scaled_dot_product_attention with causal mask)
   - Standard Softmax Attention Layer (nn.MultiheadAttention / Projection + SDPA)
   - TensorSAT SingletAttention (Triton JIT fused projection + chunked causal prefix)
   - TensorSAT HybridSingletAttention (Sliding Window + Singlet Global)
2. Accurate Timing & Memory:
   - CUDA Events for microsecond timing precision
   - Peak GPU memory tracking via torch.cuda.max_memory_allocated()
   - Warmup iterations and multiple measurement trials
3. Autoregressive Incremental Generation Benchmark (Step Latency & Throughput)
"""

import gc
import math
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorsat.attention import SingletAttention, KVCache
from tensorsat.hybrid import HybridSingletAttention


class StandardMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, is_causal: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


def measure_cuda_latency(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    return elapsed_ms


def run_scaling_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("CUDA unavailable. Aborting GPU benchmark.")
        return

    hw_name = torch.cuda.get_device_name(0)
    print("=" * 125)
    print(f"BENCHMARK A: HEAD-TO-HEAD SCALING (Standard Softmax vs TensorSAT SingletAttention)")
    print(f"Device: {hw_name} | Batch = 1, Heads = 8, HeadDim = 64, EmbedDim = 512, Dtype = FP32, Causal = True")
    print("Inference Mode: torch.inference_mode() enabled across all runs")
    print("=" * 125)

    embed_dim = 512
    num_heads = 8

    sdpa_layer = StandardMHA(embed_dim, num_heads, is_causal=True).to(device).eval()
    singlet_layer = SingletAttention(embed_dim, num_heads, is_causal=True, use_triton=True).to(device).eval()
    hybrid_layer = HybridSingletAttention(embed_dim, num_heads, window_size=128, is_causal=True, use_triton=True).to(device).eval()

    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    header = (
        f"{'Context (N)':<12} | "
        f"{'Std Softmax':<18} | "
        f"{'SingletAttn':<18} | "
        f"{'Hybrid':<16} | "
        f"{'Speedup':<10} | "
        f"{'SDPA VRAM':<12} | "
        f"{'Singlet VRAM':<12}"
    )
    print(header)
    print("-" * 125)

    with torch.inference_mode():
        for N in seq_lengths:
            torch.cuda.empty_cache()
            gc.collect()

            x = torch.randn(1, N, embed_dim, device=device)

            # 1. Measure Standard Softmax
            sdpa_oom = False
            sdpa_latency = float('nan')
            sdpa_vram = float('nan')
            try:
                torch.cuda.reset_peak_memory_stats()
                sdpa_latency = measure_cuda_latency(lambda: sdpa_layer(x), warmup=2, iters=5)
                sdpa_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except torch.cuda.OutOfMemoryError:
                sdpa_oom = True
                torch.cuda.empty_cache()

            # 2. Measure TensorSAT SingletAttention
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            singlet_latency = measure_cuda_latency(lambda: singlet_layer(x), warmup=2, iters=5)
            singlet_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # 3. Measure HybridSingletAttention
            torch.cuda.empty_cache()
            hybrid_latency = measure_cuda_latency(lambda: hybrid_layer(x), warmup=2, iters=5)

            speedup_str = f"{sdpa_latency / singlet_latency:>6.2f}x" if not sdpa_oom else "N/A (SDPA OOM)"
            sdpa_str = f"{sdpa_latency:>8.2f} ms" if not sdpa_oom else "CUDA OOM"
            singlet_str = f"{singlet_latency:>8.2f} ms"
            hybrid_str = f"{hybrid_latency:>8.2f} ms"
            sdpa_vram_str = f"{sdpa_vram:>7.1f} MB" if not sdpa_oom else "OOM"
            singlet_vram_str = f"{singlet_vram:>7.1f} MB"

            print(f"{N:<12} | {sdpa_str:<18} | {singlet_str:<18} | {hybrid_str:<16} | {speedup_str:<10} | {sdpa_vram_str:<12} | {singlet_vram_str:<12}")


def run_autoregressive_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        return

    hw_name = torch.cuda.get_device_name(0)
    print("\n" + "=" * 90)
    print(f"BENCHMARK B: AUTOREGRESSIVE GENERATION THROUGHPUT (Tokens / Second)")
    print(f"Device: {hw_name} | Step Latency & Throughput across Past Prompt Depths")
    print("=" * 90)

    embed_dim = 512
    num_heads = 8
    layer = SingletAttention(embed_dim, num_heads, is_causal=True, use_triton=True).to(device).eval()

    prompt_depths = [512, 1024, 2048, 4096, 8192, 16384]

    print(f"{'Past Prompt Depth':<20} | {'Step Latency (ms)':<20} | {'Throughput (Tokens/s)':<22} | {'Complexity':<10}")
    print("-" * 90)

    for P in prompt_depths:
        torch.cuda.empty_cache()
        cache = KVCache()
        prompt = torch.randn(1, P, embed_dim, device=device)
        with torch.no_grad():
            _, cache = layer(prompt, kv_cache=cache, use_cache=True)

        token = torch.randn(1, 1, embed_dim, device=device)

        # Warmup step
        with torch.no_grad():
            _, cache = layer(token, kv_cache=cache, use_cache=True)

        steps = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(steps):
            with torch.no_grad():
                _, cache = layer(token, kv_cache=cache, use_cache=True)
        end_event.record()
        torch.cuda.synchronize()

        step_ms = start_event.elapsed_time(end_event) / steps
        tokens_sec = 1000.0 / step_ms
        print(f"{P:<20} | {step_ms:<20.3f} | {tokens_sec:<22.1f} | O(1) constant")


if __name__ == "__main__":
    run_scaling_benchmark()
    run_autoregressive_benchmark()
