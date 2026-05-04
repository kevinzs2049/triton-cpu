/*
 * NEON SDOT runtime for M=1 INT8 GEMV.
 *
 * Compiled into TritonCPURuntime.so. Called from FlagGems dispatch layer
 * for high-performance M=1 INT8 GEMV with pre-packed weights.
 *
 * API:
 *   sdot_pack_weights(B_ptr, B_packed_ptr, K, N)
 *     B: [K, N] int8 (row-major)
 *     B_packed: [K//4, N//4, 4, 4] int8 (output, pre-allocated)
 *
 *   sdot_gemv_m1_prepacked(A_ptr, B_packed_ptr, C_ptr, K, N, N4)
 *     A: [K] int8, B_packed: SDOT format, C: [N] int32
 *
 *   sdot_gemv_m1_fused_bf16(x_bf16, B_packed, w_scale, out_bf16, K, N, N4)
 *     Fused: BF16 activation → dynamic quant → SDOT GEMV → dequant → BF16 output
 *     Eliminates Python-side abs/div/clamp/to overhead (~29ms → ~1ms)
 */

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

extern "C" {

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

EXPORT void sdot_pack_weights(const int8_t *B, int8_t *B_packed,
                               int64_t K, int64_t N) {
  int64_t K4 = K / 4;
  int64_t N4 = N / 4;
  for (int64_t kb = 0; kb < K4; kb++) {
    for (int64_t nb = 0; nb < N4; nb++) {
      int8_t *dst = B_packed + (kb * N4 + nb) * 16;
      for (int ni = 0; ni < 4; ni++) {
        for (int ki = 0; ki < 4; ki++) {
          dst[ni * 4 + ki] = B[(kb * 4 + ki) * N + nb * 4 + ni];
        }
      }
    }
  }
}

static void sdot_gemv_range(const int8_t *A, const int8_t *B_packed,
                             int32_t *C, int64_t K4, int64_t N4,
                             int64_t nb_start, int64_t nb_count) {
  // Accumulator array. For lm_head (N=151936, 8 threads) need up to 4748 groups.
  // Use heap allocation for large counts, stack for small.
  std::vector<int32x4_t> acc_heap;
  int32x4_t acc_stack[2048];
  int32x4_t *acc;
  if (nb_count <= 2048) {
    acc = acc_stack;
  } else {
    acc_heap.resize(nb_count);
    acc = acc_heap.data();
  }
  for (int64_t i = 0; i < nb_count; i++)
    acc[i] = vdupq_n_s32(0);

  for (int64_t kb = 0; kb < K4; kb++) {
    int32_t a4;
    std::memcpy(&a4, A + kb * 4, 4);
    int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
    const int8_t *bp = B_packed + (kb * N4 + nb_start) * 16;

    int64_t i = 0;
    for (; i + 4 <= nb_count; i += 4) {
      acc[i]   = vdotq_s32(acc[i],   av, vld1q_s8(bp)); bp += 16;
      acc[i+1] = vdotq_s32(acc[i+1], av, vld1q_s8(bp)); bp += 16;
      acc[i+2] = vdotq_s32(acc[i+2], av, vld1q_s8(bp)); bp += 16;
      acc[i+3] = vdotq_s32(acc[i+3], av, vld1q_s8(bp)); bp += 16;
    }
    for (; i < nb_count; i++) {
      acc[i] = vdotq_s32(acc[i], av, vld1q_s8(bp)); bp += 16;
    }
  }
  for (int64_t i = 0; i < nb_count; i++)
    vst1q_s32(C + (nb_start + i) * 4, acc[i]);
}

EXPORT void sdot_gemv_m1_prepacked(const int8_t *A,
                                    const int8_t *B_packed,
                                    int32_t *C,
                                    int64_t K, int64_t N, int64_t N4) {
  int64_t K4 = K / 4;

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk = (N4 + nt - 1) / nt;
    int64_t start = tid * chunk;
    int64_t count = chunk;
    if (start + count > N4) count = N4 - start;
    if (start >= N4) count = 0;
    if (count > 0)
      sdot_gemv_range(A, B_packed, C, K4, N4, start, count);
  }
}

/*
 * Fused BF16→INT8 dynamic quantization + SDOT GEMV + dequantization→BF16.
 *
 * Replaces the Python-side pipeline:
 *   abs().max() → div → clamp → to(int8) → _int_mm → to(fp32) → mul(scale) → to(bf16)
 *
 * Single C call does: quantize activation (NEON), SDOT GEMV (OMP), dequant to BF16.
 */

// BF16 → FP32: shift left 16 bits
static inline float32x4_t bf16_to_fp32(uint16x4_t bf16) {
  return vreinterpretq_f32_u32(vshll_n_u16(bf16, 16));
}

static float quantize_activation_bf16(const uint16_t *x_bf16, int8_t *x_int8,
                                       int64_t K) {
  float32x4_t vmax = vdupq_n_f32(0.0f);
  int64_t k = 0;
  for (; k + 4 <= K; k += 4) {
    float32x4_t f = bf16_to_fp32(vld1_u16(x_bf16 + k));
    vmax = vmaxq_f32(vmax, vabsq_f32(f));
  }
  float amax = vmaxvq_f32(vmax);
  for (; k < K; k++) {
    uint32_t bits = (uint32_t)x_bf16[k] << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    float av = std::abs(v);
    if (av > amax) amax = av;
  }
  if (amax < 1e-8f) amax = 1e-8f;
  float inv_scale = 127.0f / amax;

  k = 0;
  for (; k + 4 <= K; k += 4) {
    float32x4_t f = bf16_to_fp32(vld1_u16(x_bf16 + k));
    float32x4_t scaled = vmulq_n_f32(f, inv_scale);
    int32x4_t rounded = vcvtnq_s32_f32(scaled);
    int16x4_t n16 = vqmovn_s32(rounded);
    int8x8_t n8 = vqmovn_s16(vcombine_s16(n16, n16));
    vst1_lane_s32(reinterpret_cast<int32_t *>(x_int8 + k),
                  vreinterpret_s32_s8(n8), 0);
  }
  for (; k < K; k++) {
    uint32_t bits = (uint32_t)x_bf16[k] << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    int32_t r = static_cast<int32_t>(std::round(v * inv_scale));
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    x_int8[k] = static_cast<int8_t>(r);
  }
  return amax / 127.0f;
}

static void dequant_range_bf16(const int32_t *out_int32, float x_scale,
                                const float *w_scale, uint16_t *out_bf16,
                                int64_t start, int64_t count) {
  float32x4_t xs = vdupq_n_f32(x_scale);
  int64_t n = start;
  int64_t end = start + count;
  for (; n + 4 <= end; n += 4) {
    float32x4_t oi = vcvtq_f32_s32(vld1q_s32(out_int32 + n));
    float32x4_t ws = vld1q_f32(w_scale + n);
    float32x4_t result = vmulq_f32(vmulq_f32(oi, xs), ws);
    uint32x4_t ru = vreinterpretq_u32_f32(result);
    uint16x4_t bf = vshrn_n_u32(ru, 16);
    vst1_u16(out_bf16 + n, bf);
  }
  for (; n < end; n++) {
    float r = static_cast<float>(out_int32[n]) * x_scale * w_scale[n];
    uint32_t bits;
    std::memcpy(&bits, &r, 4);
    out_bf16[n] = static_cast<uint16_t>(bits >> 16);
  }
}

EXPORT void sdot_gemv_m1_fused_bf16(const uint16_t *x_bf16,
                                     const int8_t *B_packed,
                                     const float *w_scale,
                                     uint16_t *out_bf16,
                                     int64_t K, int64_t N, int64_t N4) {
  // Step 1: Quantize BF16 activation → INT8 (single-threaded, K is small ~2-6K)
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);

  // Step 2+3: SDOT GEMV + Dequantize (fused per-thread to keep data in L1)
  int64_t K4 = K / 4;

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk_n4 = (N4 + nt - 1) / nt;
    int64_t start_n4 = tid * chunk_n4;
    int64_t count_n4 = chunk_n4;
    if (start_n4 + count_n4 > N4) count_n4 = N4 - start_n4;
    if (start_n4 >= N4) count_n4 = 0;

    if (count_n4 > 0) {
      // Allocate thread-local int32 buffer on heap for large N (lm_head: N=151936)
      std::vector<int32_t> local_buf(count_n4 * 4);
      int32_t *local_int32 = local_buf.data() - start_n4 * 4;
      sdot_gemv_range(x_int8, B_packed, local_int32,
                       K4, N4, start_n4, count_n4);
      dequant_range_bf16(local_int32, x_scale, w_scale,
                          out_bf16, start_n4 * 4, count_n4 * 4);
    }
  }
}

/* ═══════════════════════════════════════════════════════════
 * W4A8 per-channel SDOT GEMV (decode T=1, fused BF16 in/out).
 *
 * Weight packing format:  [K/4, N/4, 4, 2] int8
 *   For each (kb, nb) block (4 K-elements × 4 N-channels = 16 i4 weights):
 *     8 packed bytes; byte (ni*2 + p) holds:
 *       low nibble  = w[ni, ki=2p]
 *       high nibble = w[ni, ki=2p+1]
 *   Quantization: per-output-channel symmetric, w in [-7, 7] (4-bit signed).
 *
 * NEON unpack uses vshl/vshr-by-4 sign-extend trick + vzip1q to interleave
 * 8 low + 8 high nibbles into the 16 int8 layout SDOT wants
 * (ni0_ki0..3, ni1_ki0..3, ni2_ki0..3, ni3_ki0..3).
 *
 * Replaces W8 SDOT-fused GEMV at 2× memory throughput on weights (1 byte
 * per 2 weights vs 1 byte per 1 weight).
 * ═══════════════════════════════════════════════════════════ */

static void sdot_gemv_w4_range(const int8_t *A_int8, const int8_t *B_w4,
                                int32_t *C_int32, int64_t K4, int64_t N4,
                                int64_t nb_start, int64_t nb_count) {
  std::vector<int32x4_t> acc_heap;
  int32x4_t acc_stack[2048];
  int32x4_t *acc;
  if (nb_count <= 2048) {
    acc = acc_stack;
  } else {
    acc_heap.resize(nb_count);
    acc = acc_heap.data();
  }
  for (int64_t i = 0; i < nb_count; i++) acc[i] = vdupq_n_s32(0);

  for (int64_t kb = 0; kb < K4; kb++) {
    int32_t a4;
    std::memcpy(&a4, A_int8 + kb * 4, 4);
    int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
    // Each (kb, nb) block is 8 bytes (16 i4 weights).
    const int8_t *bp = B_w4 + (kb * N4 + nb_start) * 8;

    int64_t i = 0;
    for (; i < nb_count; i++) {
      // Load 8 packed bytes.
      int8x8_t packed = vld1_s8(bp);
      // Sign-extend low nibble: (packed << 4) >> 4 (arithmetic).
      int8x8_t lo = vshr_n_s8(vshl_n_s8(packed, 4), 4);
      // Sign-extend high nibble: packed >> 4 (arithmetic).
      int8x8_t hi = vshr_n_s8(packed, 4);
      // Interleave to (lo[0], hi[0], lo[1], hi[1], ..., lo[7], hi[7]) ==
      //   (w[ni0_ki0], w[ni0_ki1], w[ni0_ki2], w[ni0_ki3],
      //    w[ni1_ki0], ..., w[ni3_ki3]).
      int8x16_t bv = vzip1q_s8(
          vcombine_s8(lo, vdup_n_s8(0)),
          vcombine_s8(hi, vdup_n_s8(0)));
      acc[i] = vdotq_s32(acc[i], av, bv);
      bp += 8;
    }
  }
  for (int64_t i = 0; i < nb_count; i++)
    vst1q_s32(C_int32 + (nb_start + i) * 4, acc[i]);
}

EXPORT void sdot_gemv_m1_w4_fused_bf16(const uint16_t *x_bf16,
                                        const int8_t *B_w4,
                                        const float *w_scale,
                                        uint16_t *out_bf16,
                                        int64_t K, int64_t N) {
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);

  int64_t K4 = K / 4;
  int64_t N4 = N / 4;

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk_n4 = (N4 + nt - 1) / nt;
    int64_t start_n4 = tid * chunk_n4;
    int64_t count_n4 = chunk_n4;
    if (start_n4 + count_n4 > N4) count_n4 = N4 - start_n4;
    if (start_n4 >= N4) count_n4 = 0;
    if (count_n4 > 0) {
      std::vector<int32_t> local_buf(count_n4 * 4);
      int32_t *local_int32 = local_buf.data() - start_n4 * 4;
      sdot_gemv_w4_range(x_int8, B_w4, local_int32,
                          K4, N4, start_n4, count_n4);
      dequant_range_bf16(local_int32, x_scale, w_scale,
                          out_bf16, start_n4 * 4, count_n4 * 4);
    }
  }
}

/* ═══════════════════════════════════════════════════════════
 * Q4_0-style W4A8 SDOT GEMV: per-block-32 W4 with fp16 scale per block per
 * output channel. Mirrors llama.cpp Q4_0 quantization (1 fp16 scale per
 * 32 K-elements per output channel).
 *
 * Layouts:
 *   weights:      [K/4, N/4, 4, 2] int8           (16 i4 weights per (kb, nb) block, same as W4 per-channel)
 *   block_scales: [K/32, N] fp16                   (1 scale per K-block-32 per output channel)
 *   activation:   [K] bf16                         (dynamically int8-quantized in kernel)
 *   out:          [N] bf16
 *
 * Per super-block of 32 K, accumulate 8 SDOTs into int32, then multiply by
 * (block_scale × activation_scale) to fp32, accumulate fp32 across blocks,
 * then convert to bf16. Avoids per-channel scale's outlier sensitivity.
 *
 * Constraint: K must be a multiple of 32 (and N a multiple of 4).
 * ═══════════════════════════════════════════════════════════ */

static void sdot_gemv_q4_0_range(const int8_t *A_int8, const int8_t *B_w4,
                                  const uint16_t *block_scales_fp16,
                                  float x_scale,
                                  uint16_t *out_bf16,
                                  int64_t K, int64_t N4,
                                  int64_t nb_start, int64_t nb_count) {
  // K must be multiple of 32 (caller guarantees).
  const int64_t K_super = K / 32;

  // Per-N-stripe accumulators. Two arrays:
  //   int_acc[i]: int32 SDOT accumulator within the current K-block-32
  //   out_fp[i]:  fp32 accumulator across K-blocks (after per-block dequant)
  // Stack-allocate when small; fall back to heap for very large N (e.g.
  // lm_head N=248320 → count_n4 up to 7760 per thread).
  std::vector<int32x4_t> int_heap;
  std::vector<float32x4_t> out_heap;
  int32x4_t int_stack[1024];
  float32x4_t out_stack[1024];
  int32x4_t *int_acc;
  float32x4_t *out_fp;
  if (nb_count <= 1024) {
    int_acc = int_stack;
    out_fp = out_stack;
  } else {
    int_heap.resize(nb_count);
    out_heap.resize(nb_count);
    int_acc = int_heap.data();
    out_fp = out_heap.data();
  }
  for (int64_t i = 0; i < nb_count; i++) out_fp[i] = vdupq_n_f32(0.0f);

  for (int64_t kb_super = 0; kb_super < K_super; kb_super++) {
    // Reset int32 accumulator at the start of each K-block-32.
    for (int64_t i = 0; i < nb_count; i++) int_acc[i] = vdupq_n_s32(0);

    // 8 K-stripes (kb = kb_super*8 .. kb_super*8+7) within this block.
    const int8_t *xp = A_int8 + kb_super * 32;
    for (int j = 0; j < 8; j++) {
      int32_t a4;
      std::memcpy(&a4, xp + j * 4, 4);
      int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));

      // Inner loop: iterate N-stripes sequentially (cache-friendly).
      // Weight ptr: stripe (kb=kb_super*8+j, nb) is at offset
      //   (kb*N4 + nb) * 8 bytes.
      const int8_t *bp = B_w4 + (((kb_super * 8 + j) * N4 + nb_start) * 8);

      int64_t i = 0;
      for (; i + 4 <= nb_count; i += 4) {
        // 4-way unroll for ILP
        int8x8_t p0 = vld1_s8(bp);     bp += 8;
        int8x8_t p1 = vld1_s8(bp);     bp += 8;
        int8x8_t p2 = vld1_s8(bp);     bp += 8;
        int8x8_t p3 = vld1_s8(bp);     bp += 8;
        int8x8_t lo0 = vshr_n_s8(vshl_n_s8(p0, 4), 4);
        int8x8_t lo1 = vshr_n_s8(vshl_n_s8(p1, 4), 4);
        int8x8_t lo2 = vshr_n_s8(vshl_n_s8(p2, 4), 4);
        int8x8_t lo3 = vshr_n_s8(vshl_n_s8(p3, 4), 4);
        int8x8_t hi0 = vshr_n_s8(p0, 4);
        int8x8_t hi1 = vshr_n_s8(p1, 4);
        int8x8_t hi2 = vshr_n_s8(p2, 4);
        int8x8_t hi3 = vshr_n_s8(p3, 4);
        int8x16_t b0 = vzip1q_s8(vcombine_s8(lo0, vdup_n_s8(0)), vcombine_s8(hi0, vdup_n_s8(0)));
        int8x16_t b1 = vzip1q_s8(vcombine_s8(lo1, vdup_n_s8(0)), vcombine_s8(hi1, vdup_n_s8(0)));
        int8x16_t b2 = vzip1q_s8(vcombine_s8(lo2, vdup_n_s8(0)), vcombine_s8(hi2, vdup_n_s8(0)));
        int8x16_t b3 = vzip1q_s8(vcombine_s8(lo3, vdup_n_s8(0)), vcombine_s8(hi3, vdup_n_s8(0)));
        int_acc[i]   = vdotq_s32(int_acc[i],   av, b0);
        int_acc[i+1] = vdotq_s32(int_acc[i+1], av, b1);
        int_acc[i+2] = vdotq_s32(int_acc[i+2], av, b2);
        int_acc[i+3] = vdotq_s32(int_acc[i+3], av, b3);
      }
      for (; i < nb_count; i++) {
        int8x8_t p = vld1_s8(bp);
        int8x8_t lo = vshr_n_s8(vshl_n_s8(p, 4), 4);
        int8x8_t hi = vshr_n_s8(p, 4);
        int8x16_t bv = vzip1q_s8(vcombine_s8(lo, vdup_n_s8(0)),
                                  vcombine_s8(hi, vdup_n_s8(0)));
        int_acc[i] = vdotq_s32(int_acc[i], av, bv);
        bp += 8;
      }
    }

    // After this K-block-32: convert int32 → fp32, multiply by per-channel
    // per-block scale, accumulate into out_fp[i].
    const uint16_t *sp = block_scales_fp16 + kb_super * (N4 * 4) + nb_start * 4;
    for (int64_t i = 0; i < nb_count; i++) {
      float16x4_t scales_h = vld1_f16(reinterpret_cast<const float16_t *>(sp));
      sp += 4;
      float32x4_t scales = vcvt_f32_f16(scales_h);
      out_fp[i] = vfmaq_f32(out_fp[i], vcvtq_f32_s32(int_acc[i]), scales);
    }
  }

  // Final: multiply by per-token activation scale, convert fp32 → bf16.
  float32x4_t xs = vdupq_n_f32(x_scale);
  for (int64_t i = 0; i < nb_count; i++) {
    float32x4_t out = vmulq_f32(out_fp[i], xs);
    uint32x4_t ru = vreinterpretq_u32_f32(out);
    uint16x4_t bf = vshrn_n_u32(ru, 16);
    vst1_u16(out_bf16 + (nb_start + i) * 4, bf);
  }
}

EXPORT void sdot_gemv_m1_q4_0_fused_bf16(const uint16_t *x_bf16,
                                          const int8_t *B_w4,
                                          const uint16_t *block_scales_fp16,
                                          uint16_t *out_bf16,
                                          int64_t K, int64_t N) {
  // Quantize activation
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);

  int64_t N4 = N / 4;

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk_n4 = (N4 + nt - 1) / nt;
    int64_t start_n4 = tid * chunk_n4;
    int64_t count_n4 = chunk_n4;
    if (start_n4 + count_n4 > N4) count_n4 = N4 - start_n4;
    if (start_n4 >= N4) count_n4 = 0;
    if (count_n4 > 0) {
      sdot_gemv_q4_0_range(x_int8, B_w4, block_scales_fp16,
                            x_scale, out_bf16, K, N4,
                            start_n4, count_n4);
    }
  }
}

/* W4 weight packer: [K, N] int8 (values in -7..7) → [K/4, N/4, 4, 2] int8. */
EXPORT void sdot_pack_weights_w4(const int8_t *B, int8_t *B_packed,
                                  int64_t K, int64_t N) {
  int64_t K4 = K / 4;
  int64_t N4 = N / 4;
  for (int64_t kb = 0; kb < K4; kb++) {
    for (int64_t nb = 0; nb < N4; nb++) {
      int8_t *dst = B_packed + (kb * N4 + nb) * 8;
      for (int ni = 0; ni < 4; ni++) {
        for (int p = 0; p < 2; p++) {
          int ki_lo = 2 * p;
          int ki_hi = 2 * p + 1;
          int8_t lo = B[(kb * 4 + ki_lo) * N + nb * 4 + ni];
          int8_t hi = B[(kb * 4 + ki_hi) * N + nb * 4 + ni];
          // Pack low nibble of `lo` into low nibble of byte;
          // low nibble of `hi` into high nibble of byte.
          uint8_t b = (uint8_t)((lo & 0x0F) | ((hi & 0x0F) << 4));
          dst[ni * 2 + p] = (int8_t)b;
        }
      }
    }
  }
}

#else
// Stub for non-ARM platforms
EXPORT void sdot_pack_weights(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_prepacked(const int8_t *, const int8_t *,
                                    int32_t *, int64_t, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_fused_bf16(const uint16_t *, const int8_t *,
                                     const float *, uint16_t *,
                                     int64_t, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_w4_fused_bf16(const uint16_t *, const int8_t *,
                                        const float *, uint16_t *,
                                        int64_t, int64_t) {}
EXPORT void sdot_pack_weights_w4(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_q4_0_fused_bf16(const uint16_t *, const int8_t *,
                                          const uint16_t *, uint16_t *,
                                          int64_t, int64_t) {}
#endif

} // extern "C"
