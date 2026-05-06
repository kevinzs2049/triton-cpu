"""
TLE-CPU: ARM NEON/SVE2 intrinsics as Triton language builtins.

Usage in @triton.jit kernels:
    from triton.language.extra.cpu import tle_ops as tle_cpu

    @triton.jit
    def my_kernel(...):
        acc = tl.zeros([4], dtype=tl.int32)
        a_bc = ...  # [16] int8, A broadcast
        b_pk = ...  # [16] int8, B pre-packed
        acc = tle_cpu.sdot(acc, a_bc, b_pk)
"""

import triton.language as tl
from triton.language.core import builtin, tensor, _unwrap_if_constexpr


@builtin
def sdot_gemv(a_ptr, b_packed_ptr, c_ptr, K, N, _builder=None):
    """TLE-CPU: M=1 INT8 GEMV micro-kernel using NEON SDOT with pre-packed weights.

    A complete micro-kernel that performs:
      C[N] = A[K] (int8) @ B_packed[K//4, N//4, 4, 4] (int8) → C[N] (int32)

    Uses K-outer loop with NEON SDOT, OMP parallelized across N.
    Calls sdot_gemv_m1_prepacked() from libTritonCPURuntime.so.

    Args:
        a_ptr: pointer to [K] int8 activation
        b_packed_ptr: pointer to pre-packed int8 weights (SDOT format)
        c_ptr: pointer to [N] int32 output
        K: activation/weight inner dimension
        N: output dimension
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_gemv(
        a_ptr.handle, b_packed_ptr.handle, c_ptr.handle, K_val, N_val)
    # Return None (void op)
    return None


@builtin
def fused_decode_step(
    token_id, pos,
    embed_table_ptr, layer_ptrs_ptr,
    k_cache_ptr, v_cache_ptr,
    rope_cos_ptr, rope_sin_ptr,
    final_norm_ptr,
    lm_head_packed_ptr, lm_head_scale_ptr,
    hidden_dim, head_dim, n_heads, n_kv_heads,
    intermediate, vocab_size, n_layers, max_seq,
    rms_eps,
    _builder=None):
    """TLE-CPU: Full decode step. Returns next token ID (i64).

    embedding → n_layers × transformer layer → final norm → lm_head → argmax.
    One Triton kernel launch per token.
    """
    def _i64(v):
        raw = _unwrap_if_constexpr(v)
        if hasattr(raw, 'handle'):
            handle = raw.handle
            i64_ty = _builder.get_int64_ty()
            try:
                handle = _builder.create_int_cast(handle, i64_ty, True)
            except Exception:
                pass
            return handle
        return _builder.get_int64(raw)

    rms_f = float(_unwrap_if_constexpr(rms_eps))

    result = _builder.create_cpu_fused_decode_step(
        _i64(token_id), _i64(pos),
        embed_table_ptr.handle, layer_ptrs_ptr.handle,
        k_cache_ptr.handle, v_cache_ptr.handle,
        rope_cos_ptr.handle, rope_sin_ptr.handle,
        final_norm_ptr.handle,
        lm_head_packed_ptr.handle, lm_head_scale_ptr.handle,
        _i64(hidden_dim), _i64(head_dim), _i64(n_heads), _i64(n_kv_heads),
        _i64(intermediate), _i64(vocab_size), _i64(n_layers), _i64(max_seq),
        rms_f)
    return tensor(result, tl.int64)


@builtin
def fused_transformer_layer(
    hidden_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    wq_s_ptr, wk_s_ptr, wv_s_ptr, wo_s_ptr,
    q_norm_ptr, k_norm_ptr,
    cos_ptr, sin_ptr,
    k_cache_ptr, v_cache_ptr,
    cache_pos, max_seq_len,
    gate_ptr, up_ptr, down_ptr,
    gate_s_ptr, up_s_ptr, down_s_ptr,
    input_norm_ptr, post_norm_ptr,
    hidden_dim, head_dim, n_heads, n_kv_heads, intermediate,
    rms_eps,
    _builder=None):
    """TLE-CPU: Full transformer decode layer in one C call.

    RMSNorm → QKV GEMV → QK_Norm → RoPE → KV_cache → Attention →
    O_GEMV → Residual → RMSNorm → Gate+Up+SWIGLU → Down_GEMV → Residual.

    Zero tensor allocation. Zero Python dispatch within layer.
    """
    def _i64(v):
        raw = _unwrap_if_constexpr(v)
        if hasattr(raw, 'handle'):
            # Always cast to i64 to handle i32 kernel args
            handle = raw.handle
            i64_ty = _builder.get_int64_ty()
            try:
                handle = _builder.create_int_cast(handle, i64_ty, True)
            except Exception:
                pass
            return handle
        return _builder.get_int64(raw)

    rms_f = float(_unwrap_if_constexpr(rms_eps))

    _builder.create_cpu_fused_transformer_layer(
        hidden_ptr.handle,
        wq_ptr.handle, wk_ptr.handle, wv_ptr.handle, wo_ptr.handle,
        wq_s_ptr.handle, wk_s_ptr.handle, wv_s_ptr.handle, wo_s_ptr.handle,
        q_norm_ptr.handle, k_norm_ptr.handle,
        cos_ptr.handle, sin_ptr.handle,
        k_cache_ptr.handle, v_cache_ptr.handle,
        _i64(cache_pos), _i64(max_seq_len),
        gate_ptr.handle, up_ptr.handle, down_ptr.handle,
        gate_s_ptr.handle, up_s_ptr.handle, down_s_ptr.handle,
        input_norm_ptr.handle, post_norm_ptr.handle,
        _i64(hidden_dim), _i64(head_dim), _i64(n_heads), _i64(n_kv_heads),
        _i64(intermediate), rms_f)
    return None


@builtin
def fused_mlp(x_ptr, gate_packed_ptr, up_packed_ptr,
               gate_scale_ptr, up_scale_ptr, out_ptr, K, N, _builder=None):
    """TLE-CPU: Fused MLP = gate SDOT GEMV + up SDOT GEMV + SWIGLU.

    Single OMP region replaces 3 separate ops (gate_proj, up_proj, silu_and_mul).

    Args:
        x_ptr: [K] bf16 activation
        gate_packed_ptr, up_packed_ptr: [K/4, N/4, 16] int8 SDOT-packed weights
        gate_scale_ptr, up_scale_ptr: [N] fp32 per-channel weight scales
        out_ptr: [N] bf16 output
        K, N: dimensions
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_fused_mlp(
        x_ptr.handle, gate_packed_ptr.handle, up_packed_ptr.handle,
        gate_scale_ptr.handle, up_scale_ptr.handle, out_ptr.handle,
        K_val, N_val)
    return None


@builtin
def flash_attn_decode(q_ptr, k_ptr, v_ptr, out_ptr,
                       seq_len, head_dim, sm_scale,
                       num_heads, num_kv_heads,
                       stride_kn, stride_vn, _builder=None):
    """TLE-CPU: M=1 Flash Attention with NEON online softmax.

    Replaces ATen SDPA fallback for decode (M=1). Per-row online softmax,
    NEON dot product for Q·K^T, OMP parallelized across heads.

    Args:
        q_ptr: [num_heads, head_dim] bf16
        k_ptr: [num_kv_heads, seq_len, head_dim] bf16
        v_ptr: [num_kv_heads, seq_len, head_dim] bf16
        out_ptr: [num_heads, head_dim] bf16
        seq_len, head_dim: dimensions
        sm_scale: softmax scale (typically head_dim^-0.5)
        num_heads, num_kv_heads: head counts (GQA support)
        stride_kn, stride_vn: strides for K,V along seq_len dim
    """
    vals = {}
    for name, v in [('seq_len', seq_len), ('head_dim', head_dim),
                     ('num_heads', num_heads), ('num_kv_heads', num_kv_heads),
                     ('stride_kn', stride_kn), ('stride_vn', stride_vn)]:
        raw = _unwrap_if_constexpr(v)
        vals[name] = raw.handle if hasattr(raw, 'handle') else _builder.get_int64(raw)
    sm_scale_val = _unwrap_if_constexpr(sm_scale)
    if hasattr(sm_scale_val, 'handle'):
        sm_f = 0.0  # will be set from handle
    else:
        sm_f = float(sm_scale_val)
    _builder.create_cpu_flash_attn_decode(
        q_ptr.handle, k_ptr.handle, v_ptr.handle, out_ptr.handle,
        vals['seq_len'], vals['head_dim'], sm_f,
        vals['num_heads'], vals['num_kv_heads'],
        vals['stride_kn'], vals['stride_vn'])
    return None


@builtin
def rms_norm_gated(x_ptr, gate_ptr, weight_ptr, out_ptr, M, D, eps, _builder=None):
    """TLE-CPU: RMSNormGated multi-row — out = rms_norm(x) * weight * silu(gate).

    Replaces a 6-op ATen sequence per row × M rows. Per-row OMP parallel.
    Used by Qwen3.5 GDN's Qwen3_5RMSNormGated module (M = num_v_heads,
    D = head_v_dim, typically M=16 D=128).
    """
    M_raw = _unwrap_if_constexpr(M)
    D_raw = _unwrap_if_constexpr(D)
    M_val = M_raw.handle if hasattr(M_raw, 'handle') else _builder.get_int64(M_raw)
    D_val = D_raw.handle if hasattr(D_raw, 'handle') else _builder.get_int64(D_raw)
    eps_f = float(_unwrap_if_constexpr(eps))
    _builder.create_cpu_rms_norm_gated(
        x_ptr.handle, gate_ptr.handle, weight_ptr.handle, out_ptr.handle,
        M_val, D_val, eps_f)
    return None


@builtin
def fused_swiglu_q4_0_v2(x_ptr, gate_packed_ptr, up_packed_ptr, out_ptr,
                          K, N, _builder=None):
    """TLE-CPU: Fused SwiGLU on Q4_0 v2 weights.

      out = silu(gate_proj(x)) * up_proj(x)

    where gate_proj and up_proj are both Q4_0 v2 packed Linears (K → N).
    Replaces 5 ATen ops (gate, up, silu, mul, alloc) and 4 intermediate
    tensors with a single dispatch. Decode-only (M=1). Require K % 32 == 0,
    N % 4 == 0.
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_fused_swiglu_q4_0_v2(
        x_ptr.handle, gate_packed_ptr.handle, up_packed_ptr.handle,
        out_ptr.handle, K_val, N_val)
    return None


@builtin
def rms_norm(x_ptr, weight_ptr, out_ptr, D, eps, _builder=None):
    """TLE-CPU: RMSNorm — out = (x / rms(x)) * weight.

    Single NEON kernel replaces 5 ATen decomposed ops.
    BF16 input/output, single-threaded (optimal for decode D <= 4096).

    Args:
        x_ptr: pointer to [D] bfloat16 input
        weight_ptr: pointer to [D] bfloat16 weight
        out_ptr: pointer to [D] bfloat16 output
        D: hidden dimension
        eps: epsilon for numerical stability
    """
    D_raw = _unwrap_if_constexpr(D)
    D_val = D_raw.handle if hasattr(D_raw, 'handle') else _builder.get_int64(D_raw)
    eps_f = float(_unwrap_if_constexpr(eps))
    _builder.create_cpu_rms_norm(x_ptr.handle, weight_ptr.handle, out_ptr.handle, D_val, eps_f)
    return None


@builtin
def causal_conv1d_update(hidden_ptr, state_ptr, weight_ptr, bias_ptr, out_ptr,
                          B, C, kernel_size, silu, has_bias, _builder=None):
    """TLE-CPU: Depthwise causal conv1d update (T=1, bf16, kernel_size=4).

    Per (b, c): out[b,c] = sum_k v[b,c,k] * weight[c,k] where
    v = [state[b,c,0..kernel_size-2], hidden[b,c]]; then roll state by 1.
    Optional bias (pass null pointer to skip) and SiLU activation.

    Replaces aten::conv1d (groups=C) which has high mkldnn dispatch overhead
    on Qwen3.5/Qwen3-Next conv state update.

    Args:
        hidden_ptr: [B, C] bf16 input
        state_ptr:  [B, C, kernel_size-1] bf16 IN-OUT (rolled)
        weight_ptr: [C, kernel_size] bf16
        bias_ptr:   [C] bf16 or null
        out_ptr:    [B, C] bf16 output
        B, C, kernel_size: dimensions
        silu: 1 to apply SiLU, 0 otherwise
    """
    def _i64(x):
        raw = _unwrap_if_constexpr(x)
        if hasattr(raw, 'handle'):
            handle = raw.handle
            i64_ty = _builder.get_int64_ty()
            try:
                handle = _builder.create_int_cast(handle, i64_ty, True)
            except Exception:
                pass
            return handle
        return _builder.get_int64(raw)

    _builder.create_cpu_causal_conv1d_update(
        hidden_ptr.handle, state_ptr.handle, weight_ptr.handle,
        bias_ptr.handle, out_ptr.handle,
        _i64(B), _i64(C), _i64(kernel_size), _i64(silu), _i64(has_bias))
    return None


@builtin
def gated_delta_decode(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
                       state_ptr, out_ptr,
                       B, H, k_dim, v_dim, use_l2norm,
                       _builder=None):
    """TLE-CPU: Gated Delta Net recurrent decode (T=1, fp32).

    Single fused C kernel replaces ~8 ATen ops per layer per token.
    State update + output dot fused into a single sweep over state
    (matches llama.cpp Metal/SYCL backends).

    Args:
        q_ptr, k_ptr: pointer to [B, H, k_dim] float32
        v_ptr:        pointer to [B, H, v_dim] float32
        g_ptr:        pointer to [B, H] float32 (raw, exp'd in kernel)
        beta_ptr:     pointer to [B, H] float32
        state_ptr:    pointer to [B, H, k_dim, v_dim] float32 IN-OUT
        out_ptr:      pointer to [B, H, v_dim] float32 OUT
        B, H, k_dim, v_dim: dimensions (k_dim, v_dim ≤ 256)
        use_l2norm:   1 to apply L2 norm of q & k along head_dim, 0 otherwise
    """
    def _i64(x):
        raw = _unwrap_if_constexpr(x)
        if hasattr(raw, 'handle'):
            handle = raw.handle
            i64_ty = _builder.get_int64_ty()
            try:
                handle = _builder.create_int_cast(handle, i64_ty, True)
            except Exception:
                pass
            return handle
        return _builder.get_int64(raw)

    _builder.create_cpu_gated_delta_decode(
        q_ptr.handle, k_ptr.handle, v_ptr.handle,
        g_ptr.handle, beta_ptr.handle,
        state_ptr.handle, out_ptr.handle,
        _i64(B), _i64(H), _i64(k_dim), _i64(v_dim), _i64(use_l2norm))
    return None


@builtin
def swiglu(gate_ptr, up_ptr, out_ptr, N, _builder=None):
    """TLE-CPU: Fused SWIGLU activation: out = silu(gate) * up.

    Single NEON kernel replaces F.silu(gate) * up (2 ATen calls).
    BF16 input/output. Single-threaded (optimal for decode N <= 6144).

    Args:
        gate_ptr: pointer to [N] bfloat16 gate values
        up_ptr: pointer to [N] bfloat16 up values
        out_ptr: pointer to [N] bfloat16 output
        N: number of elements
    """
    N_raw = _unwrap_if_constexpr(N)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_swiglu(gate_ptr.handle, up_ptr.handle, out_ptr.handle, N_val)
    return None


@builtin
def sdot_gemv_fused_bf16(x_ptr, b_packed_ptr, w_scale_ptr, out_ptr, K, N, _builder=None):
    """TLE-CPU: Fused BF16→INT8 quant + SDOT GEMV + dequant→BF16.

    Single call replaces: abs().max() → div → clamp → to(int8) → gemv → mul(scale) → to(bf16)

    Args:
        x_ptr: pointer to [K] bfloat16 activation
        b_packed_ptr: pointer to pre-packed int8 weights (SDOT format)
        w_scale_ptr: pointer to [N] float32 per-channel weight scale
        out_ptr: pointer to [N] bfloat16 output
        K, N: dimensions
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_gemv_fused_bf16(
        x_ptr.handle, b_packed_ptr.handle, w_scale_ptr.handle,
        out_ptr.handle, K_val, N_val)
    return None


@builtin
def gemm_q4_0_v2_smmla_bf16(x_ptr, w_packed_ptr, out_ptr, M, K, N, _builder=None):
    """TLE-CPU: Q4_0 v2 SMMLA i8mm prefill GEMM (M >= 2).

    Pairs M and N into 2x2 SMMLA tiles, on-the-fly Q4_0 unpack inside the
    kernel. Per-token int8 activation quant computed inside.

    Args:
        x_ptr: pointer to [M, K] bfloat16 activation
        w_packed_ptr: pointer to packed Q4_0 v2 weights, [N × K/32 × 18] int8
        out_ptr: pointer to [M, N] bfloat16 output
        M, K, N: dimensions; K must be a multiple of 32, N a multiple of 2.
    """
    def _i64(x):
        raw = _unwrap_if_constexpr(x)
        return raw.handle if hasattr(raw, 'handle') else _builder.get_int64(raw)
    _builder.create_cpu_gemm_q4_0_v2_smmla_bf16(
        x_ptr.handle, w_packed_ptr.handle, out_ptr.handle,
        _i64(M), _i64(K), _i64(N))
    return None


@builtin
def sdot_gemv_q4_0_v2_bf16(x_ptr, w_packed_ptr, out_ptr, K, N, _builder=None):
    """TLE-CPU: Q4_0 v2 SDOT GEMV (llama.cpp-style per-N K-major layout, bf16 in/out).

    Single packed buffer of [N × K/32 × 18] bytes (16 bytes nibbles + 2 bytes
    fp16 scale per K-block-32 per output channel). Per-token int8 activation
    quant is computed inside the kernel.

    Args:
        x_ptr: pointer to [K] bfloat16 activation
        w_packed_ptr: pointer to packed weight, [N × K/32 × 18] int8
        out_ptr: pointer to [N] bfloat16 output
        K, N: dimensions; K must be a multiple of 32.
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_gemv_q4_0_v2_bf16(
        x_ptr.handle, w_packed_ptr.handle, out_ptr.handle, K_val, N_val)
    return None


@builtin
def sdot_gemv_q4_0_bf16(x_ptr, b_packed_ptr, block_scales_ptr, out_ptr, K, N, _builder=None):
    """TLE-CPU: Q4_0-style W4A8 SDOT GEMV (per-block-32 fp16 scale, per-token A8, bf16 in/out).

    Mirrors llama.cpp Q4_0 quantization layout (1 fp16 scale per 32 K-elements
    per output channel) to handle outlier-heavy weight rows that per-channel
    W4 can't represent.

    Args:
        x_ptr: pointer to [K] bfloat16 activation
        b_packed_ptr: pointer to packed int4 weights, layout
                       [K/4, N/4, 4, 2] int8 (16 i4 weights per (kb, nb) block)
        block_scales_ptr: pointer to [K/32, N] float16 per-block-32 scale
        out_ptr: pointer to [N] bfloat16 output
        K, N: dimensions; K must be multiple of 32, N multiple of 4.
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_gemv_q4_0_bf16(
        x_ptr.handle, b_packed_ptr.handle, block_scales_ptr.handle,
        out_ptr.handle, K_val, N_val)
    return None


@builtin
def sdot_gemv_w4a8_bf16(x_ptr, b_packed_ptr, w_scale_ptr, out_ptr, K, N, _builder=None):
    """TLE-CPU: Fused W4A8 SDOT GEMV (per-channel symmetric W4, dynamic A8, bf16 in/out).

    Decode-path GEMV using NEON SDOT with on-the-fly 4-bit weight unpack.

    Args:
        x_ptr: pointer to [K] bfloat16 activation
        b_packed_ptr: pointer to packed int4 weights — layout
                      [K/4, N/4, 4, 2] int8 (16 i4 weights / 8 bytes per block)
        w_scale_ptr: pointer to [N] float32 per-channel scale
        out_ptr: pointer to [N] bfloat16 output
        K, N: dimensions (both must be divisible by 4)
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_gemv_w4a8_bf16(
        x_ptr.handle, b_packed_ptr.handle, w_scale_ptr.handle,
        out_ptr.handle, K_val, N_val)
    return None


@builtin
def sdot_pack_weights(b_ptr, b_packed_ptr, K, N, _builder=None):
    """TLE-CPU: Pack INT8 weights from row-major [K,N] to SDOT format [K//4, N//4, 4, 4].

    Args:
        b_ptr: pointer to [K, N] int8 weights (row-major)
        b_packed_ptr: pointer to output buffer (pre-allocated)
        K, N: dimensions
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_pack_weights(
        b_ptr.handle, b_packed_ptr.handle, K_val, N_val)
    return None


@builtin
def sdot(acc, a, b, _builder=None):
    """NEON SDOT: 4-lane signed int8 dot product accumulate.

    acc: tensor([4], int32)  — accumulator
    a:   tensor([16], int8)  — first operand (typically A broadcast)
    b:   tensor([16], int8)  — second operand (typically B pre-packed)

    Returns: tensor([4], int32)

    Each lane computes:
        result[i] = acc[i] + sum_{k=0}^{3}(a[i*4+k] * b[i*4+k])

    Lowered to: llvm.aarch64.neon.sdot.v4i32.v16i8 via TTC_NeonSdotOp.
    """
    return tensor(
        _builder.create_cpu_neon_sdot(acc.handle, a.handle, b.handle),
        acc.type,
    )
