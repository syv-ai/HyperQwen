// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// dequant.cuh: int8 -> fp16 in registers, the V byte pairing and output
// column map, and the per-row power-of-two Q prescale.
//
// Decisions: ADR-0008 (magic conversion), ADR-0009 (V path), ADR-0010 (fp16
// MMA, Q prescale with the e >= -112 clamp). Proofs: docs/walkthrough/06
// sections 2, 5 and 6. Reference implementation: tests/sim/ptxsim/dequant.py,
// paths.py (v_words_to_b_fragments) and prescale.py; every function here is
// compared bit for bit with it by tests/cpp/test_primitives.cu.
//
// Everything is PTH_HD (host + device): the instructions underneath come from
// ptx.cuh, which emulates them on the host.

#pragma once

#include <cstdint>

#include "ptx.cuh"

namespace pth_attn {

// ---------------------------------------------------------------------------
// Magic-number conversion constants (docs/constants.d/T3.md)
// ---------------------------------------------------------------------------

// XOR with 0x80 per byte maps two's-complement x to x + 128 (monotone).
constexpr uint32_t kMagicSignFlip = 0x80808080u;
// 0x64 is the high byte of fp16 0x6400 = 1024, whose binade has spacing 1, so
// the fp16 with bits 0x64XX is exactly 1024 + XX.
constexpr uint32_t kMagicExpBytes = 0x64646464u;
// 1152 = 1024 + 128 in both halves: removes the 0x64 offset and the flip.
constexpr uint32_t kMagicBiasHalf2 = 0x64806480u;
// prmt selectors: halves (0x64:u.b0, 0x64:u.b1) and (0x64:u.b2, 0x64:u.b3).
// Nibbles from the low end: byte 0 <- u.b0 (0), byte 1 <- 0x64 (4), ...
constexpr uint32_t kSelBytes01 = 0x4140u;
constexpr uint32_t kSelBytes23 = 0x4342u;

// V pairing selectors (__byte_perm(w_t, w_t1, sel)): interleave token t and
// t+1 so each 16-bit slot holds one d: (t,d0)(t+1,d0)(t,d1)(t+1,d1) and the
// same for d2, d3.
constexpr uint32_t kVSelD01 = 0x5140u;
constexpr uint32_t kVSelD23 = 0x7362u;

static_assert(prmt_selector_is_copy_only(kSelBytes01) &&
                  prmt_selector_is_copy_only(kSelBytes23) &&
                  prmt_selector_is_copy_only(kVSelD01) &&
                  prmt_selector_is_copy_only(kVSelD23),
              "selectors must copy bytes (prmt == __byte_perm for them)");

// Two f16x2 registers. For cvt4: lo = values of bytes 0 and 1 (byte 0 in the
// low half), hi = bytes 2 and 3.
struct F16x2Pair {
  uint32_t lo;
  uint32_t hi;
};

// cvt4: four packed int8 values -> two f16x2 registers, exactly.
// 1 LOP3 + 2 PRMT + 2 HSUB2 (SASS: HADD2 with a negated immediate) per 4
// values, no I2F. Exact for all 256 inputs: after the PRMT each half is the
// fp16 number 1152 + x exactly; 1152 is an fp16 number; the exact difference
// x (|x| <= 128 < 2048) is representable, so the correctly rounded subtraction
// returns it (walkthrough 06 section 2.3, test_magic_exact_all_256).
PTH_HD F16x2Pair cvt4_s8_f16x2(uint32_t w) {
  // Plain XOR: compiles to LOP3.LUT with immLut 0x3c (a ^ b); the SASS gate
  // checks the count. An asm lop3 would hide the operation from the compiler
  // for no gain.
  const uint32_t u = w ^ kMagicSignFlip;
  return {sub_f16x2(prmt(u, kMagicExpBytes, kSelBytes01), kMagicBiasHalf2),
          sub_f16x2(prmt(u, kMagicExpBytes, kSelBytes23), kMagicBiasHalf2)};
}

// ---------------------------------------------------------------------------
// V path (ADR-0009)
// ---------------------------------------------------------------------------

// Token offsets of the four LDS.32 words of lane (g, q), relative to
// 2q + 16kt: the B fragment rows k = 2q, 2q+1, 2q+8, 2q+9 (PTX Figure 84).
PTH_HD constexpr int v_token_offset(int i) {
  // i = 0, 1, 2, 3 -> 0, 1, 8, 9
  return (i & 1) + 8 * (i >> 1);
}

// Pair two token rows: w_t = V[t][d..d+3], w_t1 = V[t+1][d..d+3] (one LDS.32
// each). Returns the two pre-conversion words: lo = (t,d0)(t+1,d0)(t,d1)
// (t+1,d1), hi = the same for d2, d3. cvt4 of each gives one half2 per d
// holding the token pair (t, t+1): a B-fragment register.
PTH_HD F16x2Pair v_pair_tokens(uint32_t w_t, uint32_t w_t1) {
  return {prmt(w_t, w_t1, kVSelD01), prmt(w_t, w_t1, kVSelD23)};
}

// The register part of the V path for one lane and one 32-wide d group:
// w[i] = the LDS.32 word of token 2q + v_token_offset(i) (+16kt) at bytes
// 32a + 4g .. +3. b[n_sub][0..1] = B fragment of PV n-tile 4a + n_sub:
// register 0 = tokens (2q, 2q+1), register 1 = tokens (2q+8, 2q+9), both at
// d = 32a + 4g + n_sub. 4 PRMT + 4 cvt4 per call.
PTH_HD void v_b_fragments(const uint32_t (&w)[4], uint32_t (&b)[4][2]) {
  for (int r = 0; r < 2; ++r) {
    const F16x2Pair p = v_pair_tokens(w[2 * r], w[2 * r + 1]);
    const F16x2Pair d01 = cvt4_s8_f16x2(p.lo);
    const F16x2Pair d23 = cvt4_s8_f16x2(p.hi);
    b[0][r] = d01.lo;
    b[1][r] = d01.hi;
    b[2][r] = d23.lo;
    b[3][r] = d23.hi;
  }
}

// d-permutation: PV n-tile j = 4a + b, fragment column n -> physical d =
// 32a + 4n + b. A bijection of 0..D-1 (test_v_d_permutation_is_bijection).
PTH_HD constexpr int v_ntile_col_to_d(int j, int n) {
  return 32 * (j >> 2) + 4 * n + (j & 3);
}

// Physical d of accumulator c_ci (ci in {0, 1}; c2, c3 repeat the columns for
// row g + 8) of PV n-tile j held by `lane` (C fragment column 2q + ci,
// PTX Figure 86).
PTH_HD constexpr int pv_out_col(int lane, int j, int ci) {
  return v_ntile_col_to_d(j, 2 * (lane & 3) + ci);
}

// First of the 8 contiguous d a lane holds for group a: listing c0 of n-tiles
// 4a..4a+3 then c1 of the same n-tiles gives d = 32a + 8q + 0..7, in order
// (walkthrough 06 section 5.3), so the epilogue stores 16 bytes per row.
PTH_HD constexpr int epilogue_d0(int lane, int a) {
  return 32 * a + 8 * (lane & 3);
}

// ---------------------------------------------------------------------------
// Q prescale (ADR-0010)
// ---------------------------------------------------------------------------

constexpr int kQPrescaleWindowLog2 = 14;  // absmax * 2^-e in [2^14, 2^15)
constexpr int kQPrescaleEMin = -112;      // -126 + 14: 2^-e, 2^e normal f32

// Row exponent e from the row's absmax (finite, >= 0), from the exponent
// bits (no log2): e = floor(log2 absmax) - 14, clamped at -112; 0 for an
// all-zero row. f32 subnormal absmax has exponent field 0, which gives
// e = -141 and the clamp; that equals the true floor(log2) - 14 clamped,
// because every subnormal is below 2^-126.
PTH_HD int q_prescale_exponent(float absmax) {
  const uint32_t bits = f32_as_u32(absmax) & kF32AbsMask;
  if (bits == 0) return 0;
  const int e = static_cast<int>((bits >> kF32MantBits) & kF32ExpFieldMask) -
                kF32ExpBias - kQPrescaleWindowLog2;
  return e < kQPrescaleEMin ? kQPrescaleEMin : e;
}

// 2^k as an f32 built from the exponent field. Valid for -126 <= k <= 127
// (normal f32); the clamp keeps every use in -113 <= k <= 112.
PTH_HD float pow2_f32(int k) {
  return u32_as_f32(static_cast<uint32_t>(k + kF32ExpBias) << kF32MantBits);
}

// Two Q elements (exact bf16 or fp16 values carried as f32) -> one f16x2
// register of x * 2^-e, x_lo in the low half. The f32 multiply by a power of
// two is exact unless the product leaves the normal f32 range, where it
// cannot change the fp16 result (it is below 2^-126, far under the fp16
// rounding boundary 2^-25); the only rounding is cvt.rn. Exact when
// |x * 2^-e| >= 2^-17 (bf16 inputs) or 2^-14 (fp16 inputs); otherwise the
// error is <= 2^(e - 25) in x units (test_prescale_exact_above_threshold_*).
PTH_HD uint32_t q_prescale_pack(float x_lo, float x_hi, int e) {
  const float s = pow2_f32(-e);
  return pack_f16x2_rn(x_lo * s, x_hi * s);
}

}  // namespace pth_attn
