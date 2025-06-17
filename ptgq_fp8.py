import triton
import triton.language as tl
import torch


@triton.jit
def _per_token_group_quant_fp8_3d(
    # Pointers ------------------------------------------------------------
    y_ptr,                 # *FP32  activations  (E, T, H)
    y_q_ptr,               # *FP8   quantised activations (same logical shape)
    y_s_ptr,               # *FP32  scales (E, T, G)

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

    # ----------------------- map program id -> (e, t, g) ----------------
    pid = tl.program_id(0)
    t = pid % T
    tmp = pid // T
    g = tmp % G
    e = tmp // G

    # ----------------------- base offsets ------------------------------
    base_y_offset = (
        e * stride_y_e +
        t * stride_y_t +
        g * GROUP_SIZE * stride_y_h
    )
    base_yq_offset = (
        e * stride_yq_e +
        t * stride_yq_t +
        g * GROUP_SIZE * stride_yq_h
    )
    base_ys_offset = (
        e * stride_ys_e +
        t * stride_ys_t +
        g * stride_ys_g
    )

    # ----------------------- load FP32 activations ----------------------
    cols = tl.arange(0, BLOCK)
    mask = cols < GROUP_SIZE
    y = tl.load(y_ptr + base_y_offset + cols * stride_y_h,
                mask=mask, other=0.0).to(tl.float32)

    # ----------------------- quantise -----------------------------------
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max                     # scale for this (e,t,g)

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    # ----------------------- store outputs ------------------------------
    tl.store(y_q_ptr + base_yq_offset + cols * stride_yq_h,
             y_q, mask=mask)

    tl.store(y_s_ptr + base_ys_offset, y_s)


# -------------------------------------------------------------------------
# Python wrapper -----------------------------------------------------------

def quant_fp8_3d(
    y: torch.Tensor,          # (E, T, H) float32
    fp8_dtype,                # torch.float8_e4m3fn or torch.float8_e5m2
    group_size: int = 128,
    eps: float = 1e-6,
):
    """Quantise **y** into FP8 with per‑(expert, token, group) scales.

    Returns `(y_q, y_s)` where
    * `y_q` is the FP8 tensor, same shape and **standard PyTorch order** as *y*.
    * `y_s` has shape `(E, T, H // group_size)` and element strides
      `(T * G, 1, T)` so that the *token* dimension is contiguous.
    """

    assert y.ndim == 3, "y must be (E, T, H)"
    E, T, H = y.shape
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"

    # ---------------- allocate outputs ----------------------------------
    y_q = torch.empty_like(y, dtype=fp8_dtype)

    # allocate raw buffer; we'll treat it with custom strides inside the kernel
    y_s_raw = torch.empty((E, T, G), dtype=torch.float32, device=y.device)

    # ---------------- stride bookkeeping (elements, not bytes) ----------
    stride_y_e, stride_y_t, stride_y_h = y.stride()

    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T

    grid = (E * G * T,)

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = -f_info.max

    _per_token_group_quant_fp8_3d[grid](
        y, y_q, y_s_raw,
        E, T, H, group_size,
        stride_y_e, stride_y_t, stride_y_h,
        stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g,
        eps, fp8_min, fp8_max,
        BLOCK=group_size,
        num_warps=4,
    )

    # expose scales with the *requested* strides (T contiguous)
    y_s = y_s_raw.as_strided(
        size=(E, T, G),
        stride=(stride_ys_e, stride_ys_t, stride_ys_g)
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

    E, T, H = 2, 3, 256
    y = torch.randn(E, T, H, device="cuda", dtype=torch.float32)
    y_q, y_s = quant_fp8_3d(y, dtype_fp8, group_size=128)

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
    print("MAE after round‑trip:", mae)
    print("Self‑test passed.")


if __name__ == "__main__":
    _self_test()

