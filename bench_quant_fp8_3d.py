"""
Benchmark for quant_fp8_3d with realistic MoE token loads.

Focus on E=8 experts, max_tokens_per_expert=128, hidden H=2560, and average 16 tokens per expert.
"""
import time
import torch

from ptgq_fp8 import quant_fp8_3d


def benchmark(
    E: int,
    T: int,
    H: int,
    avg_tokens_per_expert: int,
    group_size: int = 128,
    repeat: int = 200,
):
    """
    Run quant_fp8_3d E x T x H with a fixed avg_tokens_per_expert load and report average latency.
    """
    if not torch.cuda.is_available():
        print("CUDA is required for benchmark; skipping.")
        return

    # choose available FP8 dtype
    dtype_fp8 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e5m2', None)
    if dtype_fp8 is None:
        raise RuntimeError("No FP8 dtype available in this build of PyTorch")

    device = torch.device('cuda')
    # random activations
    y = torch.randn(E, T, H, dtype=torch.float32, device=device)
    # simulate token counts per expert (constant average load)
    tokens_per_expert = torch.full((E,), avg_tokens_per_expert, dtype=torch.int32, device=device)

    # warmup and pre-allocation for CUDA graph
    for _ in range(2):
        quant_fp8_3d(y, dtype_fp8, tokens_per_expert, group_size)
    torch.cuda.synchronize()

    # static inputs for CUDA graph capture
    static_y = y.clone()
    static_tokens = tokens_per_expert.clone()
    # pre-allocate output memory via a dry run
    quant_fp8_3d(static_y, dtype_fp8, static_tokens, group_size)
    torch.cuda.synchronize()

    # capture CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        quant_fp8_3d(static_y, dtype_fp8, static_tokens, group_size)
    torch.cuda.synchronize()

    # timing loop using CUDA graph replay
    start = time.time()
    for _ in range(repeat):
        g.replay()
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000.0 / repeat
    print(f"quant_fp8_3d: E={E}, T={T}, H={H}, avg_tokens_per_expert={avg_tokens_per_expert},"
          f" group_size={group_size} -> {avg_ms:.3f} ms per call")


if __name__ == '__main__':
    # realistic MoE GPU benchmark parameters
    benchmark(
        E=8,
        T=10240,
        H=2560,
        avg_tokens_per_expert=16,
        group_size=128,
        repeat=200,
    )
