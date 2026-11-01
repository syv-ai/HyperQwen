// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Compile-time tile geometry of the attention kernel.
//
// Everything here is a constant that follows from a few hardware limits and
// three choices (load tile N = 64 tokens, 2 pipeline stages, 4 warps). The
// derivations are written next to each constant. The static_asserts turn the
// shared-memory inequality, the divisibility rules and the occupancy claims
// into compile errors, so an edit that breaks them cannot build.
//
// Header-only, depends on <cstdint> only, usable from host C++ (g++ test
// tests/cpp/test_tile_traits.cpp) and device code.
//
// Units: bytes (B), tokens, rows (one row = one (query token, query head)
// pair after GQA packing), registers (32-bit, per thread).

#pragma once

#include <cstdint>

// Byte layout of Q and the stages (StageLayout, SmemLayout), the per-block
// limit kSmemPerBlockLimit = 101376 and kScaleBytes / kF16Bytes come from
// here; this header only chooses N, stages and warps.
#include "smem_layout.cuh"

namespace pth_attn {

// ---------------------------------------------------------------------------
// Hardware limits (sm_86 and sm_89; sm_80 allows more and is not binding).
// CUDA C++ Programming Guide, "Technical Specifications per Compute
// Capability": 99 KB opt-in shared memory per block (kSmemPerBlockLimit in
// smem_layout.cuh), 100 KB per SM, 64 K 32-bit registers per SM, 255
// registers per thread. The runtime reserves 1 KB of shared memory per
// resident block.
// ---------------------------------------------------------------------------
inline constexpr int kSmemPerSm = 102400;  // 100 * 1024
inline constexpr int kSmemReservedPerBlock = 1024;
inline constexpr int kRegsPerSm = 65536;
inline constexpr int kMaxRegsPerThread = 255;
// Registers are allocated per warp in units of 256, i.e. a per-thread count
// rounded up to a multiple of 8 (CUDA occupancy calculator, sm_8x).
inline constexpr int kRegAllocUnitPerThread = 8;
inline constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
// MMA and data-format facts.
// ---------------------------------------------------------------------------
// mma.sync.m16n8k16 (PTX ISA 9.7.16.5.8): one warp computes 16 rows x 8
// columns, reducing over 16. M = 16 is why each warp owns 16 query rows.
inline constexpr int kMmaM = 16;
inline constexpr int kMmaN = 8;
inline constexpr int kMmaK = 16;
// The V path consumes d in groups of 32 (n-tile j = 4a + b maps to
// d = 32a + 4n + b), so D must be a multiple of 32.
inline constexpr int kVGroupD = 32;
inline constexpr int kAccElemBytes = 4;  // fp32 accumulators and partials

// ---------------------------------------------------------------------------
// The three choices.
// ---------------------------------------------------------------------------
// Load tile: tokens per pipeline stage. Largest multiple of 16 * WN = 64 with
// Q(64 rows) + 2 * N * (2D + 8) <= 101376 at D = 256, i.e. N <= 65.
inline constexpr int kLoadTileTokens = 64;
// Pipeline depth: 3 stages need 32768 + 3 * 33280 = 132608 B at D = 256.
inline constexpr int kStages = 2;
// Warps per CTA: arranged WM x WN = 4; O per warp is D/2 registers.
inline constexpr int kWarps = 4;
inline constexpr int kThreads = kWarps * kWarpSize;  // 128
// Largest compute sub-tile per warp and step by head size. At D = 256 a
// 64-token step spills in the real kernel (8 B at 255 registers on sm_86; a
// register probe saw the same: it needed 224 registers against 208 at 32), so
// D = 256 steps 32 tokens at a time.
constexpr int sub_tile_cap(int d) { return d == 256 ? 32 : 64; }
// Resident CTAs per SM by head size: the planner's
// C_target uses the same numbers (CTAS_PER_SM in planner.py).
constexpr int ctas_per_sm(int d) { return d == 256 ? 1 : (d == 128 ? 2 : 3); }

// ---------------------------------------------------------------------------
// Generic geometry: every derived number and every rule, for any candidate
// (N, stages, SUB, CTAs). Traits below picks the shipped candidate;
// tests/cpp/test_tile_traits.cpp instantiates unsupported ones to show why.
// ---------------------------------------------------------------------------
template <int D, int WM, int N, int STAGES, int SUB, int CTAS>
struct TileGeometry {
  static constexpr int kHeadSize = D;
  static constexpr int kWm = WM;            // warps along rows (M)
  static constexpr int kWn = kWarps / WM;   // warps along tokens (N)
  static constexpr int kRows = kMmaM * WM;  // rows per CTA row tile
  static constexpr int kN = N;
  static constexpr int kStagesN = STAGES;
  static constexpr int kTokensPerWarp = N / kWn;  // each warp's token slice
  static constexpr int kSub = SUB;                // tokens per compute step
  static constexpr int kSubSteps = kTokensPerWarp / SUB;
  static constexpr int kCtasPerSm = CTAS;

  // Shared memory, from smem_layout.cuh: one stage holds, for N
  // tokens of one KV head, K [N][D] int8, V [N][D] int8, ks [N] f32 and
  // vs [N] f32 = N * (2D + 8) B; Q is [rows][D] fp16. SmemLayout itself
  // asserts the 101376 B per-block limit and 128-B region alignment.
  using Layout = SmemLayout<D, N, kRows, STAGES>;
  static constexpr int kStageBytes = Layout::kStageBytes;
  static constexpr int kQBytes = Layout::kQBytes;
  static constexpr int kSmemBytes = Layout::kBytes;

  // In-CTA merge of the WN token slices (decode WN > 1): after the
  // last tile the stage buffers are free; warps n = 1..WN-1 stage their fp32
  // O (16 rows x D) and their row max and sum (2 floats per row).
  static constexpr int kMergeBytes =
      (kWn - 1) * WM * kMmaM * (D * kAccElemBytes + 2 * kAccElemBytes);

  // Per-thread accumulator registers (estimates the probe checks against
  // ptxas): O is 16 x D fp32 per warp over 32 lanes; S is 16 x SUB fp32.
  static constexpr int kORegs = kMmaM * D / kWarpSize;    // D / 2
  static constexpr int kSRegs = kMmaM * SUB / kWarpSize;  // SUB / 2

  // Register budget that keeps CTAS blocks resident: what ptxas is told via
  // __launch_bounds__(kThreads, CTAS), rounded down to the allocation unit.
  static constexpr int kRegBudgetRaw = kRegsPerSm / (CTAS * kThreads);
  static constexpr int kRegBudget =
      (kRegBudgetRaw / kRegAllocUnitPerThread) * kRegAllocUnitPerThread >
              kMaxRegsPerThread
          ? kMaxRegsPerThread
          : (kRegBudgetRaw / kRegAllocUnitPerThread) * kRegAllocUnitPerThread;

  static_assert(D % kMmaK == 0 && D % kVGroupD == 0,
                "D must be a multiple of 16 (QK k-steps) and 32 (V groups)");
  static_assert(WM == 1 || WM == 2 || WM == 4, "WM must be 1, 2 or 4");
  static_assert(kRows <= kMmaM * kWarps, "at most 64 rows per CTA row tile");
  static_assert(N % (kWn * kMmaK) == 0,
                "each warp's token slice must be whole PV k-steps of 16");
  static_assert(SUB % kMmaK == 0 && kTokensPerWarp % SUB == 0,
                "SUB must be a multiple of 16 that divides the warp's slice");
  static_assert(CTAS * (kSmemBytes + kSmemReservedPerBlock) <= kSmemPerSm,
                "CTAS blocks do not fit in 102400 B of shared memory per SM");
  static_assert(kMergeBytes <= STAGES * kStageBytes,
                "the WN merge must fit in the freed stage buffers");
};

// ---------------------------------------------------------------------------
// The shipped configuration per (D, WM, IoT). IoT (bf16 or fp16, the model's
// dtype) only changes Q staging and output stores, not the geometry.
// ---------------------------------------------------------------------------
template <int D, int WM>
constexpr int default_sub() {
  // min(the warp's token slice, the per-D cap): WM = 4 has one 64-token slice
  // per warp, WM = 2 a 32-token slice, WM = 1 a 16-token slice.
  constexpr int slice = kLoadTileTokens * WM / kWarps;
  return slice < sub_tile_cap(D) ? slice : sub_tile_cap(D);
}

template <int D, int WM, typename IoT>
struct Traits : TileGeometry<D, WM, kLoadTileTokens, kStages,
                             default_sub<D, WM>(), ctas_per_sm(D)> {
  static_assert(D == 64 || D == 128 || D == 256,
                "head sizes with a kernel instantiation: 64, 128, 256");
  static_assert(sizeof(IoT) == 2, "IoT is a 16-bit float type");
  using IoType = IoT;
  static constexpr int kOutElemBytes = sizeof(IoT);
};

}  // namespace pth_attn
