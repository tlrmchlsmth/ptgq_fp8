import unittest
import torch

from ptgq_fp8 import quant_silu_mul_fp8_3d
from light_llm import silu_and_mul_masked_post_quant_fwd


class TestLightLLM(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for test")
        dtype_fp8 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e5m2', None)
        if dtype_fp8 is None:
            self.skipTest("FP8 dtype not available for test")
        self.dtype_fp8 = dtype_fp8
        self.E, self.T, self.H = 4, 16, 256
        self.group_size = 128
        self.y = torch.randn(self.E, self.T, 2 * self.H, dtype=torch.float32, device='cuda')
        self.tokens_per_expert = torch.randint(
            1, self.T + 1, (self.E,), dtype=torch.int32, device=self.y.device
        )

    def test_silu_and_mul_matches_quant_silu_mul(self):
        yq_ref, ys_ref = quant_silu_mul_fp8_3d(
            self.y, self.dtype_fp8, self.tokens_per_expert, self.group_size
        )
        output = torch.empty(
            (self.E, self.T, self.H), dtype=self.dtype_fp8, device=self.y.device
        )
        output_scale = torch.empty(
            (self.E, self.T, self.H // self.group_size), dtype=torch.float32, device=self.y.device
        )
        masked_m = self.tokens_per_expert.clone()
        silu_and_mul_masked_post_quant_fwd(
            self.y, output, output_scale, self.group_size, masked_m
        )

        # outputs must match exactly
        self.assertTrue(torch.equal(yq_ref, output))
        torch.testing.assert_close(ys_ref, output_scale, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
