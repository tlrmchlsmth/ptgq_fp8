This is the GPT o3 conversation that spawned this work

I have the following quantize function, used to do dynamic online 1x128 block quantization for LLM inference:

@triton.jit
def _general_per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) *
                                                        group_size)
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

I need to update this to make it more general. The activation and scale tensors need to be 3D and the stride order needs to be (0, 2, 1) in order of slowest to fastest dim. Best for it to handle general strides.


ChatGPT said:

Thought for 49 seconds
Triton Quant Fp8 3d
I've drafted an updated Triton kernel (and a small Python convenience wrapper) that:

Works with 3-D inputs laid out in stride order (0 → slowest, 2, 1 → fastest).

Accepts arbitrary strides for every tensor it touches.

Launches one program per contiguous block of group_size (default 128) along the fastest axis, exactly matching your old “1 × 128” behaviour.

Emits scales per block and stores the quantised FP8 values with their own (also fully stride-controlled) layout.

Feel free to open triton_quant_fp8_3d.py in the canvas to review or tweak.
Let me know if you’d like different output layouts, extra dtypes (e.g. E5M2), or further performance tuning tips!


Thought for 35 seconds
Edited
Fixed the non-ASCII minus sign in the default eps value and added a quick self-test that validates shapes and runs the kernel if an FP8 dtype is available on your PyTorch build.

Give it another run and let me know if anything else crops up!






You said:
One issue: This is the right dim order for y_s (the scales)

The activation tensors (y and y_q) should just be standard pytorch dim order 


ChatGPT said:

Thought for 1m 13s
Just to make sure I wire this the way you need:

Activations (y, y_q) – do you want the standard PyTorch “batch × seq × hidden” layout, i.e. shape (B, T, H) with the last (hidden) dimension contiguous?

Scales (y_s) – should these be shaped (B, ? , ?)?

Would you like them as (B, H//group, T) (batch, hidden-groups, seq) or (B, T, H//group)?

And which dimension should be the fastest-striding one (contiguous in memory)?

Once I have that nailed down, I’ll refactor the kernel and wrapper accordingly.







You said:
For scales: The shape needs to be:
num_experts, max_num_tokens, hidden dim//group

Strides need to be: max_num_tokens * H//group, 1, max_num_tokens


This is a reshape operation that puts the tensors in the right format for this:
+        a2q_scale = a2q_scale.as_strided(size=(E, max_num_tokens, a2q_scale.shape[-1]),
+                                      stride=(max_num_tokens * a2q_scale.shape[-1], 1, max_num_tokens))

(wrong answer without an update to the kernel though)


ChatGPT said:

Thought for 59 seconds
Edited

Answer in chat instead
Done!
Key fixes:

All strides are now in elements (not bytes) – Triton pointer arithmetic is element-based, so this removes the silent 4×/8× mis-offset.

Activations (y, y_q) remain in standard contiguous PyTorch order (E, T, H); the wrapper passes their natural strides straight to the kernel.

Scales (y_s) produced with shape (E, T, G) and element strides (T × G, 1, T), matching the reshape snippet you showed.

Updated self-test verifies both shapes and strides, and checks a quick round-trip MAE.

Give the new version a spin and let me know if anything else crops up!
