import triton
import triton.language as tl
import torch


@triton.jit
def _per_token_group_quant_fp8_3d(
    # Pointers ------------------------------------------------------------
    y_ptr,                 # *FP32  activations  (E, T, H)
    y_q_ptr,               # *FP8   quantised activations (same logical shape)

    y_s_ptr,               # *FP32  scales (E, T, G)
    counts_ptr,            # *INT32  number of tokens per expert (E)

    # Sizes ---------------------------------------------------------------
    E: tl.constexpr,       # num_experts
    T: tl.constexpr,       # max_num_tokens
    H: tl.constexpr,       # hidden dimension
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)

    # Strides for y (elements) -------------------------------------------
    stride_y_e,
    stride_y_t,
    stride_y_h,

    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,


    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,

    # Stride for counts (elements)
    stride_counts_e,

    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,

    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
):
    """Dynamic FP8 quantisation over a 3‑D tensor laid out **(E, T, H)**.

    *   Each program instance handles **one** `GROUP_SIZE`‑length slice along H
        for a single (expert *e*, token *t*).
    *   Scales are produced with shape **(E, T, G)** where
        `G = H // GROUP_SIZE` and with *element* strides
        `(T*G, 1, T)` so that the *token* dimension is the fastest‑varying in
        memory – matching the downstream reshape you showed.
    *   *All* strides are expressed **in elements**, not bytes.
    """

    G = H // GROUP_SIZE  # groups per hidden dim

    # ----------------------- map program id -> (e, g) --------------------
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    # block for H dimension
    cols = tl.arange(0, BLOCK)
    cols = cols.to(tl.int64)
    mask_h = cols < BLOCK

    # iterate over tokens for this (expert, group)
    t = tl.zeros([], tl.int64)
    while t < n_tokens:
        base_y_offset = e * stride_y_e + t * stride_y_t + g * GROUP_SIZE * stride_y_h
        base_yq_offset = e * stride_yq_e + t * stride_yq_t + g * GROUP_SIZE * stride_yq_h
        base_ys_offset = e * stride_ys_e + t * stride_ys_t + g * stride_ys_g

        mask = mask_h
        y = tl.load(y_ptr + base_y_offset + cols * stride_y_h,
                    mask=mask, other=0.0).to(tl.float32)

        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + cols * stride_yq_h,
                 y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset, y_s)

        t += 1

@triton.jit
def _per_token_group_silu_mul_quant_fp8_3d(
    # Pointers ------------------------------------------------------------
    input_ptr,              # *FP32 activations (E, T, 2*H)
    y_q_ptr,                # *FP8   quantised activations (E, T, H)
    y_s_ptr,                # *FP32  scales (E, T, G)
    counts_ptr,             # *INT32 number of tokens per expert (E)

    # Sizes ---------------------------------------------------------------
    E: tl.constexpr,        # num_experts
    T: tl.constexpr,        # max_num_tokens
    H: tl.constexpr,        # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)

    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,

    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,

    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,

    # Stride for counts (elements)
    stride_counts_e,

    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,

    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
):
    """Dynamic FP8 quantisation of silu(input[..., :H]) * input[..., H:] over a 3D tensor laid out **(E, T, 2*H)**.

    * Each program instance handles one GROUP_SIZE-length slice along H for a single (expert e, token t).
    * Scales are produced with shape (E, T, G) where G = H // GROUP_SIZE and with element strides (T*G, 1, T).
    * All strides are expressed in elements, not bytes.
    """

    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK)
    cols = cols.to(tl.int64)
    mask_h = cols < BLOCK

    t = tl.zeros([], tl.int64)
    while t < n_tokens:
        base_i_offset = e * stride_i_e + t * stride_i_t + g * GROUP_SIZE * stride_i_h
        base_yq_offset = e * stride_yq_e + t * stride_yq_t + g * GROUP_SIZE * stride_yq_h
        base_ys_offset = e * stride_ys_e + t * stride_ys_t + g * stride_ys_g

        mask = mask_h
        x = tl.load(input_ptr + base_i_offset + cols * stride_i_h, mask=mask, other=0.0).to(tl.float32)
        y2 = tl.load(input_ptr + base_i_offset + H * stride_i_h + cols * stride_i_h, mask=mask, other=0.0).to(tl.float32)

        x = x * (1.0 / (1.0 + tl.exp(-x)))
        y = x * y2

        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + cols * stride_yq_h, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset, y_s)

        t += 1

# -------------------------------------------------------------------------
# Python wrapper -----------------------------------------------------------

def quant_fp8_3d(
    y: torch.Tensor,               # (E, T, H) float32
    fp8_dtype,                     # torch.float8_e4m3fn or torch.float8_e5m2
    tokens_per_expert: torch.Tensor, # (E,) number of valid tokens per expert
    group_size: int = 128,
    eps: float = 1e-6,
):
    """Quantise **y** into FP8 with per‑(expert, token, group) scales.

    Only the first `tokens_per_expert[e]` tokens are quantized per expert;
    the remaining positions in each (E, T, H) slice are treated as padding.

    Returns `(y_q, y_s)` where
    * `y_q` is the FP8 tensor, same shape and **standard PyTorch order** as *y*.
    * `y_s` has shape `(E, T, H // group_size)` and element strides
      `(T * G, 1, T)` so that the *token* dimension is contiguous.
    """

    assert y.ndim == 3, "y must be (E, T, H)"
    E, T, H = y.shape
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E, \
        "tokens_per_expert must be shape (E,)"
    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # ---------------- allocate outputs ----------------------------------
    y_q = torch.empty_like(y, dtype=fp8_dtype)


    # ---------------- stride bookkeeping (elements, not bytes) ----------
    stride_y_e, stride_y_t, stride_y_h = y.stride()

    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T

    # allocate scale buffer with proper shape and stride
    y_s = torch.empty_strided((E, T, G), (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32, device=y.device)

    # stride for tokens_per_expert (elements)
    stride_cnt_e = tokens_per_expert.stride()[0]

    # static grid over experts and H-groups; tokens loop is internal to the kernel
    grid = (E * G,)

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = -f_info.max

    _per_token_group_quant_fp8_3d[grid](
        y, y_q, y_s, tokens_per_expert,
        E, T, H, group_size,
        stride_y_e, stride_y_t, stride_y_h,
        stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g,
        stride_cnt_e,
        eps, fp8_min, fp8_max,
        BLOCK=group_size,
        num_warps=4,
    )

    return y_q, y_s

def quant_silu_mul_fp8_3d(
    y: torch.Tensor,               # (E, T, 2*H) float32
    fp8_dtype,                     # torch.float8_e4m3fn or torch.float8_e5m2
    tokens_per_expert: torch.Tensor, # (E,) number of valid tokens per expert
    group_size: int = 128,
    eps: float = 1e-6,
):
    """Quantise silu(y[..., :H]) * y[..., H:] into FP8 with per-(expert, token, group) scales.

    y has shape (E, T, 2*H). The first half of the last dimension is silu-activated,
    multiplied by the second half, then quantized into FP8.

    Returns `(y_q, y_s)` where
    * `y_q` is the FP8 tensor of shape `(E, T, H)`, same layout as `y[..., :H]`.
    * `y_s` has shape `(E, T, H // group_size)` and element strides `(T*G, 1, T)`.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E, \
        "tokens_per_expert must be shape (E,)"
    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # allocate outputs
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    # strides (elements)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided((E, T, G), (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32, device=y.device)

    stride_cnt_e = tokens_per_expert.stride()[0]

    # static grid over experts and H-groups; tokens loop is internal to the kernel
    grid = (E * G,)

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = -f_info.max

    _per_token_group_silu_mul_quant_fp8_3d[grid](
        y, y_q, y_s, tokens_per_expert,
        E, T, H, group_size,
        stride_i_e, stride_i_t, stride_i_h,
        stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g,
        stride_cnt_e,
        eps, fp8_min, fp8_max,
        BLOCK=group_size,
        num_warps=4,
    )

    return y_q, y_s

# -------------------------------------------------------------------------
# Self‑test ----------------------------------------------------------------

def _self_test():
    dtype_fp8 = getattr(torch, "float8_e4m3fn", None)
    if dtype_fp8 is None:
        print("float8_e4m3fn not available; skipping test")
        return

    if not torch.cuda.is_available():
        print("CUDA not available; skipping test")
        return

    E, T, H = 2, 32, 256
    y = torch.randn(E, T, H, device="cuda", dtype=torch.float32)
    tokens_per_expert = torch.full((E,), T, dtype=torch.int32, device=y.device)
    y_q, y_s = quant_fp8_3d(y, dtype_fp8, tokens_per_expert, group_size=128)

    assert y_q.shape == y.shape, "y_q shape mismatch"
    assert y_q.stride() == y.stride(), "y_q strides should match y"

    expected_strides = (T * (H // 128), 1, T)
    assert y_s.shape == (E, T, H // 128), "y_s shape mismatch"
    assert y_s.stride() == expected_strides, "y_s strides mismatch"

    # quick reconstruction error check (should be small)
    G = y_s.shape[-1]
    group_size = H // G
    y_s_expanded = y_s.unsqueeze(-1).expand(-1, -1, -1, group_size).reshape(E, T, H).to(torch.float32)
    y_deq = y_q.to(torch.float32) * y_s_expanded
    mae = (y - y_deq).abs().mean().item()
    print("MAE after round‑trip (no padding):", mae)
    print("Full-length self-test passed.")

    # ----- test padding behavior: only first tokens_per_expert[e] should be quantized -----
    E, T, H = 2, 32, 256
    y = torch.randn(E, T, H, device="cuda", dtype=torch.float32)
    tokens_per_expert = torch.tensor([7, 3], dtype=torch.int32, device=y.device)
    y_q, y_s = quant_fp8_3d(y, dtype_fp8, tokens_per_expert, group_size=128)
    # check reconstruction error only on valid tokens
    G = y_s.shape[-1]
    grp = H // G
    y_s_exp = y_s.unsqueeze(-1).expand(-1, -1, -1, grp).reshape(E, T, H).to(torch.float32)
    y_deq = y_q.to(torch.float32) * y_s_exp
    # mask for valid tokens per expert
    valid_mask = (torch.arange(T, device=y.device).unsqueeze(0) < tokens_per_expert.unsqueeze(1))
    err = (y - y_deq).abs()
    mae_pad = err[valid_mask.unsqueeze(-1).expand(-1, -1, H)].mean().item()
    assert mae_pad < 1e-1, f"MAE with padding too high: {mae_pad}"
    print("Padding self-test passed.")

    # ----- test silu+mul quant kernel -----
    E, T, H = 2, 32, 256
    y2 = torch.randn(E, T, 2 * H, device="cuda", dtype=torch.float32)
    tokens_per_expert = torch.full((E,), T, dtype=torch.int32, device=y2.device)
    yq2, ys2 = quant_silu_mul_fp8_3d(y2, dtype_fp8, tokens_per_expert, group_size=128)
    assert yq2.shape == (E, T, H), "y_q shape mismatch for silu_mul"
    expected_stride_yq = (T * H, H, 1)
    assert yq2.stride() == expected_stride_yq, f"y_q strides mismatch: {yq2.stride()}"
    G2 = ys2.shape[-1]
    assert ys2.shape == (E, T, H // 128)
    expected_stride_ys2 = (T * G2, 1, T)
    assert ys2.stride() == expected_stride_ys2, f"y_s strides mismatch for silu_mul: {ys2.stride()}"
    fused = torch.nn.functional.silu(y2[..., :H]) * y2[..., H:]
    ys2_exp = ys2.unsqueeze(-1).expand(-1, -1, -1, 128).reshape(E, T, H).to(torch.float32)
    deq = yq2.to(torch.float32) * ys2_exp
    mae2 = (fused - deq).abs().mean().item()
    assert mae2 < 1e-1, f"MAE silu_mul too high: {mae2}"
    print("Silu+mul quant self-test passed.")
    # padding behavior for silu+mul kernel
    tokens_per_expert = torch.tensor([7, 3], dtype=torch.int32, device=y2.device)
    yq2p, ys2p = quant_silu_mul_fp8_3d(y2, dtype_fp8, tokens_per_expert, group_size=128)
    fused = torch.nn.functional.silu(y2[..., :H]) * y2[..., H:]
    ys2p_exp = ys2p.unsqueeze(-1).expand(-1, -1, -1, 128).reshape(E, T, H).to(torch.float32)
    deqp = yq2p.to(torch.float32) * ys2p_exp
    valid = (torch.arange(T, device=y2.device).unsqueeze(0) < tokens_per_expert.unsqueeze(1))
    err2p = (fused - deqp).abs()
    mae2p = err2p[valid.unsqueeze(-1).expand(-1, -1, H)].mean().item()
    assert mae2p < 1e-1, f"MAE silu_mul padding too high: {mae2p}"
    print("Silu+mul quant padding self-test passed.")


if __name__ == "__main__":
    _self_test()

