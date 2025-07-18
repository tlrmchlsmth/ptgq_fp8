"""
Benchmark comparing performance of quant_silu_mul_fp8_3d and
silu_and_mul_masked_post_quant_fwd using CUDA graphs.
Sweeps over number of tokens per expert.
"""
import time
import torch

from ptgq_fp8 import quant_silu_mul_fp8_3d
from light_llm import silu_and_mul_masked_post_quant_fwd


def benchmark(
    E: int,
    T: int,
    H: int,
    group_size: int,
    token_sweep,
    repeat: int = 100,
):
    if not torch.cuda.is_available():
        print("CUDA is required for benchmark; skipping.")
        return

    dtype_fp8 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e5m2', None)
    if dtype_fp8 is None:
        print("No FP8 dtype available; skipping benchmark.")
        return

    device = torch.device('cuda')
    # inputs and outputs
    y = torch.randn(E, T, 2 * H, dtype=torch.bfloat16, device=device)
    output = torch.empty((E, T, H), dtype=dtype_fp8, device=device)
    output_scale = torch.empty((E, T, H // group_size), dtype=torch.bfloat16, device=device)
    masked_m = torch.empty((E,), dtype=torch.int32, device=device)

    print(f"Benchmark: E={E}, T={T}, H={H}, group_size={group_size}, repeat={repeat}")
    for tokens in token_sweep:
        masked_m.fill_(tokens)
        tokens_per_expert = masked_m.clone()

        # warmup
        quant_silu_mul_fp8_3d(y, dtype_fp8, tokens_per_expert, group_size)
        silu_and_mul_masked_post_quant_fwd(y, output, output_scale, group_size, masked_m)
        torch.cuda.synchronize()

        # quant_silu_mul_fp8_3d with CUDA graph
        static_y = y.clone()
        static_tokens = tokens_per_expert.clone()
        quant_silu_mul_fp8_3d(static_y, dtype_fp8, static_tokens, group_size)
        torch.cuda.synchronize()
        g_q = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_q):
            quant_silu_mul_fp8_3d(static_y, dtype_fp8, static_tokens, group_size)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            g_q.replay()
        torch.cuda.synchronize()
        t_graph = (time.time() - start) * 1000.0 / repeat

        # silu_and_mul_masked_post_quant_fwd with CUDA graph
        static_out = output.clone()
        static_scale = output_scale.clone()
        static_m = masked_m.clone()
        silu_and_mul_masked_post_quant_fwd(static_y, static_out, static_scale, group_size, static_m)
        torch.cuda.synchronize()
        g_s = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_s):
            silu_and_mul_masked_post_quant_fwd(static_y, static_out, static_scale, group_size, static_m)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            g_s.replay()
        torch.cuda.synchronize()
        t2_graph = (time.time() - start) * 1000.0 / repeat

        print(
            f"tokens={tokens}: "
            f"\tquant_silu_mul {t_graph:.3f}ms "
            f"\tlight_llm {t2_graph:.3f}ms"
        )


if __name__ == '__main__':
    # example configuration
    E, T, H = 256, 2048, 7168
    group_size = 128
    token_sweep = [4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048]
    benchmark(E, T, H, group_size, token_sweep, repeat=200)
