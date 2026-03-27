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
