import math
import time
import torch
import torch.nn as nn
from tensorsat.attention import SingletAttention, KVCache

def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=======================================================")
    print(f"TensorSAT Production SingletAttention Benchmark")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Features Enabled: Triton Fused GPU Kernel + Causal Masking")
    print(f"=======================================================")

    embed_dim = 512
    num_heads = 8
    batch_size = 1

    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    layer = SingletAttention(embed_dim, num_heads, is_causal=True, use_triton=True).to(device)
    layer.eval()

    print(f"{'Context (N tokens)':<20} | {'SingletAttn Latency':<25} | {'Memory Allocated':<20}")
    print("-" * 70)

    for N in seq_lengths:
        x = torch.randn(batch_size, N, embed_dim, device=device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            _ = layer(x[:, :min(N, 128), :])
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = layer(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{N:<20} | {elapsed:<15.2f} ms{'':<8} | {peak_mem:<12.1f} MB")

    print("\n-------------------------------------------------------")
    print("Testing Autoregressive Step Latency with KV Cache:")
    print("-------------------------------------------------------")
    cache = KVCache()
    prompt = torch.randn(batch_size, 2048, embed_dim, device=device)
    with torch.no_grad():
        _, cache = layer(prompt, kv_cache=cache, use_cache=True)

    # Measure single-token generation step at context 2048
    new_token = torch.randn(batch_size, 1, embed_dim, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            step_out, cache = layer(new_token, kv_cache=cache, use_cache=True)
    torch.cuda.synchronize()
    step_latency = (time.perf_counter() - t0) / 50 * 1000
    throughput = 1000.0 / step_latency
    print(f"Per-Token Step Latency: {step_latency:.3f} ms ({throughput:.1f} tokens/sec)")

if __name__ == "__main__":
    benchmark()
