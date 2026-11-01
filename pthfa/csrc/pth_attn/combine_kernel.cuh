// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// combine_kernel.cuh: merging split-KV partials, and the 16-bit output stores.
//
// With S > 1 splits every split s of a query row writes its own
// softmax-normalised output o_s (fp32, [S, T, Hq, D]) and the log2-sum-exp of
// its scores lse_s = m_s + log2(l_s) (fp32, [S, T, Hq]); an empty split writes
// lse_s = -inf. The exact output is
//
//   o = sum_s 2^lse_s * o_s / sum_s 2^lse_s,
//
// because 2^lse_s = sum over the split's tokens of 2^(score) is exactly the
// split's share of the softmax denominator.
// To keep 2^x in range the kernel subtracts M = max_s lse_s first (a common
// factor cancels). This is the rule tests/cpu/test_planner_combine.py fixes:
//   - M over the live splits only (lse > -inf);
//   - an empty split is skipped BEFORE its o_s is read: the workspace is
//     uninitialised memory, and 0 * NaN = NaN would poison the row;
//   - sums in the fixed order s = 0, 1, ..., S-1, so the result is a
//     deterministic function of the partials (not batch-invariant: S itself
//     depends on the batch).
//
// The row function is lane-local (no shuffles): every lane of a warp computes
// the same S weights with the same operations, then its own columns
// d = lane + 32 j. The same source therefore runs on the device and, one lane
// at a time, in the host emulator (ref_capi.cu, tests).
//
// The fp32 -> fp16/bf16 conversions are written out bit by bit (round to
// nearest even) instead of using cuda_fp16.h intrinsics, so the host build and
// the device build produce identical bits (checked against torch in
// tests/cpu/test_ref_capi_build.py).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "attn_args.h"
#include "cache_geometry.cuh"

// Full unrolling of the per-lane column loops under nvcc (keeps the lane
// arrays in registers); g++ needs no hint (the host emulator is not tuned).
#if defined(__CUDACC__)
  #define PTH_ATTN_UNROLL _Pragma("unroll")
#else
  #define PTH_ATTN_UNROLL
#endif

namespace pth_attn {

// Lanes of a warp; the reference kernel and the combine give lane l the
// columns d = l + 32 j.
constexpr int32_t kWarpLanes = 32;

// ------------------------------------------------------------ bit casts
// The fp32/fp16/bf16 bit-level helpers and their constants live in a nested
// namespace: the same IEEE field names (kF32AbsMask, kF16Inf, ...) are also
// defined by dequant.cuh and ptx.cuh, and a kernel TU includes all of
// them. Only load_q16/store_out/load_f32/store_f32 below are the API.
namespace ref_bits {

PTH_HD float f32_from_bits(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
#endif
}

PTH_HD uint32_t f32_to_bits(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
#endif
}

// ------------------------------------------------- 16-bit float <-> fp32
// IEEE binary32: 1 sign, 8 exponent (bias 127), 23 fraction bits.
constexpr uint32_t kF32SignBit = 0x80000000u;
constexpr uint32_t kF32AbsMask = 0x7fffffffu;
constexpr uint32_t kF32ExpInf = 0x7f800000u;  // exponent all ones
constexpr uint32_t kF32FracMask = 0x007fffffu;
constexpr uint32_t kF32Hidden = 0x00800000u;  // implicit leading 1
constexpr int32_t kF32FracBits = 23;
// binary16: 1 sign, 5 exponent (bias 15), 10 fraction bits.
constexpr int32_t kF16FracBits = 10;
constexpr int32_t kF16ExpBias = 15;
constexpr int32_t kF32ExpBias = 127;
constexpr uint32_t kF16ExpMax = 0x1fu;
constexpr uint32_t kF16FracMask = 0x3ffu;
constexpr uint32_t kF16Inf = 0x7c00u;
constexpr uint32_t kF16QuietNan = 0x7e00u;
// Halves are 16 bits: the sign moves from bit 31 to bit 15.
constexpr int32_t kHalfShift = 16;
// fp32 bit patterns that bound the fp16 cases:
// 65520 = 65504 + half an ulp: rounds (ties to even) to +inf.
constexpr uint32_t kF32BitsF16Overflow = 0x477ff000u;
// 2^-14: the smallest normal fp16.
constexpr uint32_t kF32BitsF16MinNormal = 0x38800000u;
// 2^-25: half the smallest subnormal; at or below it a value rounds to 0.
constexpr uint32_t kF32BitsF16HalfMinSub = 0x33000000u;
// Rebias fp32 -> fp16 exponent: (127 - 15) << 23.
constexpr uint32_t kF32ToF16Rebias = 0x38000000u;
// Round-to-nearest-even of the 13 dropped fraction bits: add 0xfff + lsb.
constexpr uint32_t kF16RoundBias = 0x0fffu;
constexpr int32_t kF32ToF16Drop = kF32FracBits - kF16FracBits;  // 13
// bf16 = top 16 bits of fp32; RNE adds 0x7fff + lsb before truncating.
constexpr uint32_t kBf16RoundBias = 0x7fffu;
constexpr uint32_t kBf16QuietBit = 0x0040u;
// 2^-24 (exact in fp32): the fp16 subnormal quantum.
constexpr float kF16SubQuantum = 5.9604644775390625e-08f;

PTH_HD float bf16_to_f32(uint16_t h) {
  return f32_from_bits(static_cast<uint32_t>(h) << kHalfShift);
}

PTH_HD float f16_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << kHalfShift;
  const uint32_t e = (static_cast<uint32_t>(h) >> kF16FracBits) & kF16ExpMax;
  const uint32_t m = static_cast<uint32_t>(h) & kF16FracMask;
  if (e == 0u) {
    // Zero or subnormal: m * 2^-24, exact in fp32 (11 bits).
    const float v = static_cast<float>(m) * kF16SubQuantum;
    return sign ? -v : v;
  }
  if (e == kF16ExpMax) {
    return f32_from_bits(sign | kF32ExpInf | (m << kF32ToF16Drop));
  }
  const uint32_t e32 = e - kF16ExpBias + kF32ExpBias;
  return f32_from_bits(sign | (e32 << kF32FracBits) | (m << kF32ToF16Drop));
}

PTH_HD uint16_t f32_to_bf16_rne(float f) {
  const uint32_t u = f32_to_bits(f);
  if ((u & kF32AbsMask) > kF32ExpInf) {  // NaN: keep it a (quiet) NaN
    return static_cast<uint16_t>((u >> kHalfShift) | kBf16QuietBit);
  }
  const uint32_t lsb = (u >> kHalfShift) & 1u;
  return static_cast<uint16_t>((u + kBf16RoundBias + lsb) >> kHalfShift);
}

PTH_HD uint16_t f32_to_f16_rne(float f) {
  const uint32_t u = f32_to_bits(f);
  const uint32_t sign = (u & kF32SignBit) >> kHalfShift;
  const uint32_t a = u & kF32AbsMask;
  if (a > kF32ExpInf) return static_cast<uint16_t>(sign | kF16QuietNan);
  if (a >= kF32BitsF16Overflow) return static_cast<uint16_t>(sign | kF16Inf);
  if (a >= kF32BitsF16MinNormal) {
    // Normal: rebias the exponent, then round the 13 dropped fraction bits
    // to nearest even; a carry out of the fraction correctly bumps the
    // exponent (and a < 65520 keeps the result below inf).
    uint32_t r = a - kF32ToF16Rebias;
    r += kF16RoundBias + ((r >> kF32ToF16Drop) & 1u);
    return static_cast<uint16_t>(sign | (r >> kF32ToF16Drop));
  }
  if (a <= kF32BitsF16HalfMinSub) return static_cast<uint16_t>(sign);
  // Subnormal: value = man * 2^(e - 150) with the hidden bit in man; in units
  // of 2^-24 that is man >> (126 - e), 14 <= shift <= 24 here. Round to
  // nearest even on the bits shifted out.
  const int32_t e = static_cast<int32_t>(a >> kF32FracBits);
  const uint32_t man = (a & kF32FracMask) | kF32Hidden;
  const int32_t shift = (kF32ExpBias - 1) - e;
  uint32_t q = man >> shift;
  const uint32_t rem = man & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (q & 1u))) ++q;
  return static_cast<uint16_t>(sign | q);
}

}  // namespace ref_bits

// Loads one 16-bit q element as fp32 (exact for both formats).
PTH_HD float load_q16(const char* p, int32_t dtype) {
  uint16_t h;
#if defined(__CUDA_ARCH__)
  h = *reinterpret_cast<const uint16_t*>(p);
#else
  std::memcpy(&h, p, sizeof h);
#endif
  return dtype == PTH_ATTN_BF16 ? ref_bits::bf16_to_f32(h)
                                : ref_bits::f16_to_f32(h);
}

// Stores one output element in `dtype` (fp32 as is; fp16/bf16 rounded to
// nearest even).
PTH_HD void store_out(char* p, int32_t dtype, float v) {
  if (dtype == PTH_ATTN_F32) {
#if defined(__CUDA_ARCH__)
    *reinterpret_cast<float*>(p) = v;
#else
    std::memcpy(p, &v, sizeof v);
#endif
    return;
  }
  const uint16_t h = dtype == PTH_ATTN_BF16 ? ref_bits::f32_to_bf16_rne(v)
                                            : ref_bits::f32_to_f16_rne(v);
#if defined(__CUDA_ARCH__)
  *reinterpret_cast<uint16_t*>(p) = h;
#else
  std::memcpy(p, &h, sizeof h);
#endif
}

PTH_HD float load_f32(const char* p) {
#if defined(__CUDA_ARCH__)
  return *reinterpret_cast<const float*>(p);
#else
  float v;
  std::memcpy(&v, p, sizeof v);
  return v;
#endif
}

PTH_HD void store_f32(char* p, float v) {
#if defined(__CUDA_ARCH__)
  *reinterpret_cast<float*>(p) = v;
#else
  std::memcpy(p, &v, sizeof v);
#endif
}

// ------------------------------------------------------------- combine
// Byte strides of the split partials (same layout as KernelParams' ws_*).
struct PartialsView {
  const char* ws_o;
  int64_t s_o_split, s_o_token, s_o_head;
  const char* ws_lse;
  int64_t s_l_split, s_l_token, s_l_head;

  PTH_HD float lse(int32_t s, int64_t token, int32_t head) const {
    return load_f32(ws_lse + static_cast<int64_t>(s) * s_l_split +
                    token * s_l_token + static_cast<int64_t>(head) * s_l_head);
  }
  PTH_HD float o(int32_t s, int64_t token, int32_t head, int32_t d) const {
    return load_f32(ws_o + static_cast<int64_t>(s) * s_o_split +
                    token * s_o_token + static_cast<int64_t>(head) * s_o_head +
                    static_cast<int64_t>(d) *
                        static_cast<int64_t>(sizeof(float)));
  }
};

// Combines the S partials of row (token, head) for the columns of `lane`
// (d = lane + 32 j, j < kCols = D/32) into out[j]. Returns false if every split
// is empty (cannot happen for a query row of a request: key 0 is always
// visible); out[] is then zero.
// kCols = D / 32 is a template parameter so out[] and the sums stay in
// registers after unrolling.
template <int32_t kCols>
PTH_HD bool combine_row(const PartialsView& v, int32_t num_splits,
                        int64_t token, int32_t head, int32_t lane,
                        float out[kCols]) {
  // M = max over live splits. `x > m` is false for x = -inf, so empty splits
  // never set M; a NaN lse (a broken producer) is also never taken.
  float m = -INFINITY;
  for (int32_t s = 0; s < num_splits; ++s) {
    const float x = v.lse(s, token, head);
    if (x > m) m = x;
  }
  PTH_ATTN_UNROLL
  for (int32_t j = 0; j < kCols; ++j) out[j] = 0.0f;
  if (!(m > -INFINITY)) return false;
  float den = 0.0f;
  float num[kCols] = {};
  for (int32_t s = 0; s < num_splits; ++s) {
    const float x = v.lse(s, token, head);
    if (!(x > -INFINITY)) continue;  // skip BEFORE reading o_s
    const float w = exp2f(x - m);    // in (0, 1]
    den += w;
    PTH_ATTN_UNROLL
    for (int32_t j = 0; j < kCols; ++j) {
      num[j] = fmaf(w, v.o(s, token, head, lane + kWarpLanes * j), num[j]);
    }
  }
  PTH_ATTN_UNROLL
  for (int32_t j = 0; j < kCols; ++j) out[j] = num[j] / den;
  return true;
}

// Writes the combined row (token, head) for one lane's columns.
template <int32_t kCols>
PTH_HD void combine_row_store(const PartialsView& v, int32_t num_splits,
                              int64_t token, int32_t head, int32_t lane,
                              char* out_base, int64_t s_out_token,
                              int64_t s_out_head, int32_t out_dtype) {
  float o[kCols];
  combine_row<kCols>(v, num_splits, token, head, lane, o);
  const int64_t elem = out_dtype == PTH_ATTN_F32 ? 4 : 2;
  char* row =
      out_base + token * s_out_token + static_cast<int64_t>(head) * s_out_head;
  PTH_ATTN_UNROLL
  for (int32_t j = 0; j < kCols; ++j) {
    store_out(row + static_cast<int64_t>(lane + kWarpLanes * j) * elem,
              out_dtype, o[j]);
  }
}

}  // namespace pth_attn
