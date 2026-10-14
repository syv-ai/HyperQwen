// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// tiling.cuh: which rows and tokens each warp of the attention kernel
// computes, and the two tensor-core loops over one compute sub-tile:
//
//   qk_subtile   S[16 rows x SUB tokens] = Q~ . K_int^T   (fp32 accumulators)
//   pv_subtile   O[16 rows x D]         += P'' . V_int     (fp32 accumulators)
//
// Both are the register probe's loops (research/probe/probe_tile.cu, ADR
// 0012) made reusable: the kernel (attn_kernel.cuh) and the one-warp unit
// test (attn_test_kernels.cu, pth_attn_test_warp_tile) call the same
// functions, so the unit test checks exactly the code the kernel runs.
//
// Warp layout (plan 3.2, ADR 0012): 4 warps as WM x WN. Warp w has
// warp_m = w % WM (its 16 query rows are 16 warp_m .. 16 warp_m + 15 of the
// CTA's row tile) and warp_n = w / WM (its tokens are slice warp_n of each
// 64-token load tile, 64 / WN tokens, walked in steps of SUB).
//
// Fragment facts used below (PTX ISA 9.7.16.5.8, mma.m16n8k16 .f16, and the
// T3 proofs in docs/walkthrough/06):
//   lane = 4 g + q;  A (16 x 16):  a0 = (row g, k 2q..2q+1), a1 = (row g+8,
//   same k), a2 = (row g, k 2q+8..+9), a3 = (row g+8, k 2q+8..+9);
//   B (16 x 8):  b0 = (k 2q..2q+1, col g), b1 = (k 2q+8..+9, col g);
//   C (16 x 8):  c0,c1 = (row g, cols 2q, 2q+1), c2,c3 = (row g+8, same).
//   QK: n = token, k = d (permuted by pi, ADR 0008); B comes from ldmatrix of
//   int8 K + cvt4. PV: n = d (d-permuted, ADR 0009), k = token; A comes
//   straight from the S accumulators (FA2 register reuse), B from LDS.32 +
//   byte pairing + cvt4.

#pragma once

#include <cstdint>

#include "dequant.cuh"
#include "ptx.cuh"
#include "smem_layout.cuh"
#include "tile_traits.cuh"

namespace pth_attn {

// Smallest WM in {1, 2, 4} whose 16 * WM MMA rows hold `rows` rows (bq * G);
// 0 if more than 64 rows (unsupported, plan C26). For the planner's bq =
// floor(16 wm / G) this returns the planner's wm (ADR 0013).
PTH_HD int32_t attn_wm_for_rows(int32_t rows) {
  return rows <= kMmaM ? 1
                       : (rows <= 2 * kMmaM ? 2 : (rows <= 4 * kMmaM ? 4 : 0));
}

// The geometry of one kernel instantiation: TileGeometry (tile_traits.cuh)
// with the shipped N, stages and CTAs/SM, and the given SUB.
template <int D, int WM, int SUB>
using AttnGeometry =
    TileGeometry<D, WM, kLoadTileTokens, kStages, SUB, ctas_per_sm(D)>;

// ---------------------------------------------------------------------------
// Stage readers: where token t's K/V bytes and scales sit inside one stage.
// Offsets are relative to the stage base. Two layouts (ADR 0006, ADR 0012
// Amendment 1):
//   SepStageReader   separate streams: StageLayout, K and V regions with
//                    the XOR swizzles of ADR 0007, ks / vs arrays;
//   CellStageReader  whole inline cells [K | ks | V | vs] at pitch 2D + 16
//                    (528 at D = 256), no swizzle: the odd number of 16-B
//                    chunks per row is what makes K ldmatrix and V LDS.32
//                    conflict-free (plan V4).
// ---------------------------------------------------------------------------
template <int D>
struct SepStageReader {
  using St = StageLayout<D, kLoadTileTokens>;
  static constexpr bool kCell = false;
  static constexpr int kBytes = St::kBytes;
  PTH_HD static constexpr uint32_t k(uint32_t t, uint32_t byte) {
    return St::kK + k_swz(t, byte, D);
  }
  PTH_HD static constexpr uint32_t v(uint32_t t, uint32_t byte) {
    return St::kV + v_swz(t, byte, D);
  }
  PTH_HD static constexpr uint32_t ks(uint32_t t) {
    return St::kKs + kScaleBytes * t;
  }
  PTH_HD static constexpr uint32_t vs(uint32_t t) {
    return St::kVs + kScaleBytes * t;
  }
};

template <int D>
struct CellStageReader {
  using Cs = CellStageLayout<D, kLoadTileTokens>;
  static constexpr bool kCell = true;
  static constexpr int kBytes = Cs::kBytes;
  PTH_HD static constexpr uint32_t k(uint32_t t, uint32_t byte) {
    return t * Cs::kPitch + Cs::kCellK + byte;
  }
  PTH_HD static constexpr uint32_t v(uint32_t t, uint32_t byte) {
    return t * Cs::kPitch + Cs::kCellV + byte;
  }
  PTH_HD static constexpr uint32_t ks(uint32_t t) {
    return t * Cs::kPitch + Cs::kCellKs;
  }
  PTH_HD static constexpr uint32_t vs(uint32_t t) {
    return t * Cs::kPitch + Cs::kCellVs;
  }
};

#if defined(__CUDACC__)

// Lane coordinates in the fragment layouts: lane = 4 g + q.
__device__ __forceinline__ int frag_g(int lane) { return lane >> 2; }
__device__ __forceinline__ int frag_q(int lane) { return lane & 3; }

// The scales of tokens t and t + 1 (one LDS.64 in the separate layout, two
// LDS.32 in the cell layout).
template <class R>
__device__ __forceinline__ float2 load_scale_pair(const uint8_t* stage, bool v,
                                                  int t) {
  if constexpr (R::kCell) {
    const uint32_t o0 = v ? R::vs(t) : R::ks(t);
    const uint32_t o1 = v ? R::vs(t + 1) : R::ks(t + 1);
    return make_float2(*reinterpret_cast<const float*>(stage + o0),
                       *reinterpret_cast<const float*>(stage + o1));
  } else {
    return *reinterpret_cast<const float2*>(stage + (v ? R::vs(t) : R::ks(t)));
  }
}

// S = Q~ . K^T for warp rows row0 .. row0 + 15 and SUB tokens t0 .. t0+SUB-1
// of a stage. `s` is zeroed here. q_smem / stage_smem are shared-state-space
// addresses (smem_u32) of the Q buffer and the stage.
template <int D, int SUB, class R = SepStageReader<D>>
__device__ __forceinline__ void qk_subtile(float (&s)[SUB / kMmaN][4],
                                           uint32_t q_smem, uint32_t stage_smem,
                                           int t0, int row0, int lane) {
  constexpr int kNt = SUB / kMmaN;        // n-tiles (8 tokens each)
  constexpr int kKSteps = D / kMmaK;      // k16 steps over d
  constexpr int kTg = kNt < 4 ? kNt : 4;  // token groups per ldmatrix.x4
  constexpr int kKsPerLd = 4 / kTg;       // k-steps per ldmatrix.x4
  static_assert(kNt % kTg == 0 && kKSteps % kKsPerLd == 0, "fragment tiling");
  #pragma unroll
  for (int n = 0; n < kNt; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
  // ldmatrix.x4: matrix mi = lane / 8 covers token group tg0 + mi % kTg at
  // k-step ks0 + mi / kTg; the lane supplies the address of row lane % 8.
  // For kTg = 4 this is T3's k_ldsm_x4_offset (one k-step, 4 n-tiles).
  const int mi = lane >> 3;
  #pragma unroll
  for (int ks0 = 0; ks0 < kKSteps; ks0 += kKsPerLd) {
    uint32_t a[kKsPerLd][4];
  #pragma unroll
    for (int i = 0; i < kKsPerLd; ++i) {
      ldmatrix_x4(a[i], q_smem + q_ldsm_x4_offset(lane, row0, ks0 + i, D));
    }
  #pragma unroll
    for (int tg0 = 0; tg0 < kNt; tg0 += kTg) {
      uint32_t kb[4];
      const int tok = t0 + (tg0 + mi % kTg) * kMmaN + (lane & 7);
      ldmatrix_x4(kb,
                  stage_smem + R::k(tok, (ks0 + mi / kTg) * kSmemChunkBytes));
  #pragma unroll
      for (int m4 = 0; m4 < 4; ++m4) {
        // Register m4 holds bytes 4q..4q+3 of one token at one k16 step;
        // cvt4 gives (b0, b1) = MMA k (2q, 2q+1) and (2q+8, 2q+9) under pi.
        const F16x2Pair f = cvt4_s8_f16x2(kb[m4]);
        const uint32_t b[2] = {f.lo, f.hi};
        mma_m16n8k16_f16f32(s[tg0 + m4 % kTg], a[m4 / kTg], b);
      }
    }
  }
}

// O += P'' . V for SUB tokens t0 .. t0+SUB-1 of a stage.
// pa[n][h] is the fp16 pair of n-tile n, row half h (rows g, g+8): the
// S accumulator layout, which is the A fragment layout of PV (k = token).
// `stage` is a generic pointer into shared memory (C++ loads, which NVVM
// emits as LDS: checked by the SASS gate, no generic LD).
template <int D, int SUB, class R = SepStageReader<D>>
__device__ __forceinline__ void pv_subtile(float (&o)[D / kMmaN][4],
                                           const uint32_t (&pa)[SUB / kMmaN][2],
                                           const uint8_t* stage, int t0,
                                           int lane) {
  const int g = lane >> 2, q = lane & 3;
  #pragma unroll
  for (int j = 0; j < SUB / kMmaK; ++j) {
    const uint32_t a[4] = {pa[2 * j][0], pa[2 * j][1], pa[2 * j + 1][0],
                           pa[2 * j + 1][1]};
    const int t16 = t0 + j * kMmaK;
  #pragma unroll
    for (int grp = 0; grp < D / kVGroupD; ++grp) {
      uint32_t w[4];
  #pragma unroll
      for (int i = 0; i < 4; ++i) {  // tokens 2q, 2q+1, 2q+8, 2q+9
        // = T3's v_lds32_offset for the separate layout.
        const int tok = t16 + 2 * q + v_token_offset(i);
        w[i] = *reinterpret_cast<const uint32_t*>(
            stage + R::v(tok, kVGroupD * grp + 4 * g));
      }
      uint32_t b[4][2];
      v_b_fragments(w, b);  // 4 PRMT + 4 cvt4 (ADR 0009)
      // n-tile 4 grp + n_sub holds d = 32 grp + 4 n + n_sub at column n.
  #pragma unroll
      for (int n_sub = 0; n_sub < 4; ++n_sub) {
        mma_m16n8k16_f16f32(o[4 * grp + n_sub], a, b[n_sub]);
      }
    }
  }
}

#endif  // __CUDACC__

}  // namespace pth_attn
