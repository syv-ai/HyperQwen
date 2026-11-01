// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// softmax.cuh: the online softmax of one warp over its sub-tiles, in the
// exp2 domain, with the per-sub-tile power-of-two V-scale normalisation P''.
//
// State per warp (its 16 rows; each thread holds rows g and g + 8):
//   m[h]  running max of the scores s (log2 units), -inf before any token;
//   l[h]  running sum of the UNSCALED p = 2^(s - m), thread-partial (the
//         thread's own columns; the 4 lanes of a row are added at the end);
//   o     O accumulators in units of 2^c (fp32, D/8 n-tiles x 4);
//   c     the V-scale exponent of the current units (one per warp: the
//         sub-tile's tokens are the same for all rows).
//
// One step over a sub-tile of SUB tokens (the arithmetic of
// pth_testing/emulate.py `_warp`, which the bound of bound.py covers):
//   m_new   = max(m, max_t s_t)            per row
//   m_use   = m_new, or 0 if m_new = -inf  (no visible token yet)
//   alpha   = 2^(m - m_use), or 0 if m = -inf
//   c_new   = ceil(log2 max_t vs_t) over the sub-tile's in-range tokens, or
//             c if there are none (tokens past the CTA's range end are
//             zero-filled by the loader, so max vs > 0 means "some token")
//   o      *= alpha * 2^(c - c_new)        (the power of two is exact)
//   p_t     = 2^(s_t - m_use)              ex2.approx.ftz
//   l       = l * alpha + sum_t p_t
//   P''_t   = fp16_rn(fp32(p_t * vs_t) * 2^-c_new)   <= 1 in magnitude
// after which pv_subtile adds P'' . V_int to o.

#pragma once

#include <cstdint>

#include "dequant.cuh"
#include "ptx.cuh"
#include "tile_traits.cuh"
#include "tiling.cuh"

namespace pth_attn {

// Exponent range of 2^k built from bits: normal f32 exponents are
// [-126, 127]; the pattern for k = -127 is +0.0, which pow2_clamped uses as
// "vanishingly small" (only reached when a factor is multiplied by alpha = 0
// or by an O that the 2^(c_old - c_new) shift would underflow anyway).
constexpr int kPow2MinExp = -127;
constexpr int kPow2MaxExp = 127;
// Fraction field of an f32 (23 bits); nonzero means v is not a power of two.
constexpr uint32_t kSoftmaxF32Frac = (1u << kF32MantBits) - 1u;

#if defined(__CUDACC__)

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float lg2_approx(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// 2^k, exact for -126 <= k <= 127, +0 below (see kPow2MinExp).
__device__ __forceinline__ float pow2_clamped(int k) {
  k = k < kPow2MinExp ? kPow2MinExp : (k > kPow2MaxExp ? kPow2MaxExp : k);
  return pow2_f32(k);
}

// ceil(log2 v) for a positive normal v, from the exponent bits (no MUFU).
__device__ __forceinline__ int ceil_log2_pos(float v) {
  const uint32_t b = __float_as_uint(v);
  const int e =
      static_cast<int>((b >> kF32MantBits) & kF32ExpFieldMask) - kF32ExpBias;
  return (b & kSoftmaxF32Frac) != 0u ? e + 1 : e;
}

// The 4 lanes of a row (same g, q = 0..3) are lanes 4g .. 4g + 3.
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

template <int D>
struct WarpSoftmax {
  float o[D / kMmaN][4];  // O in units of 2^c
  float m[2];             // rows g, g + 8
  float l[2];             // thread-partial
  int c;
};

template <int D>
__device__ __forceinline__ void softmax_init(WarpSoftmax<D>& st) {
  #pragma unroll
  for (int j = 0; j < D / kMmaN; ++j) {
    st.o[j][0] = st.o[j][1] = st.o[j][2] = st.o[j][3] = 0.f;
  }
  st.m[0] = st.m[1] = -INFINITY;
  st.l[0] = st.l[1] = 0.f;
  st.c = 0;
}

// One online-softmax step over a sub-tile (see the file comment). `s` are
// the masked scores in log2 units; the v_scales of tokens t0 .. t0+SUB-1 are
// read from `stage` through the stage reader R; `pa` receives the P'' A
// fragments.
template <int D, int SUB, class R>
__device__ __forceinline__ void softmax_step(WarpSoftmax<D>& st,
                                             const float (&s)[SUB / kMmaN][4],
                                             const uint8_t* stage, int t0,
                                             uint32_t (&pa)[SUB / kMmaN][2],
                                             int lane) {
  constexpr int kNt = SUB / kMmaN;
  const int q = lane & 3;
  float mx[2] = {st.m[0], st.m[1]};
  float vmax = 0.f;
  float2 v[kNt];
  #pragma unroll
  for (int n = 0; n < kNt; ++n) {
    v[n] = load_scale_pair<R>(stage, true, t0 + n * kMmaN + 2 * q);
    vmax = fmaxf(vmax, fmaxf(v[n].x, v[n].y));
  #pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = fmaxf(mx[h], fmaxf(s[n][2 * h], s[n][2 * h + 1]));
    }
  }
  mx[0] = quad_max(mx[0]);
  mx[1] = quad_max(mx[1]);
  vmax = quad_max(vmax);  // over all SUB tokens of the sub-tile
  const int c_new = vmax > 0.f ? ceil_log2_pos(vmax) : st.c;
  const float p_norm = pow2_clamped(-c_new);
  const float c_shift = pow2_clamped(st.c - c_new);

  float m_use[2], alpha_o[2];
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    m_use[h] = mx[h] == -INFINITY ? 0.f : mx[h];
    const float alpha =
        st.m[h] == -INFINITY ? 0.f : ex2_approx(st.m[h] - m_use[h]);
    alpha_o[h] = alpha * c_shift;
    st.l[h] *= alpha;
    st.m[h] = mx[h];
  }
  st.c = c_new;
  #pragma unroll
  for (int j = 0; j < D / kMmaN; ++j) {
    st.o[j][0] *= alpha_o[0];
    st.o[j][1] *= alpha_o[0];
    st.o[j][2] *= alpha_o[1];
    st.o[j][3] *= alpha_o[1];
  }
  #pragma unroll
  for (int n = 0; n < kNt; ++n) {
  #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float p0 = ex2_approx(s[n][2 * h] - m_use[h]);
      const float p1 = ex2_approx(s[n][2 * h + 1] - m_use[h]);
      st.l[h] += p0 + p1;
      pa[n][h] = pack_f16x2_rn((p0 * v[n].x) * p_norm, (p1 * v[n].y) * p_norm);
    }
  }
}

#endif  // __CUDACC__

}  // namespace pth_attn
