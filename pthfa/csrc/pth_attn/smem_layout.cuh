// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// smem_layout.cuh: the shared-memory layout of the Q buffer and the K/V
// pipeline stages, the XOR swizzles, the K permutation pi, and the
// shared-memory byte offset every lane gives each fragment load.
//
// Reference implementation: tests/sim/ptxsim/layouts.py and paths.py; every
// function here is compared with it for every input, for D = 64, 128, 256, by
// tests/cpp/test_primitives.cu.
//
// Ownership: the tile size N, the number of stages and the warp layout belong
// to tile_traits.cuh, which instantiates SmemLayout with them; this
// header only turns those numbers into byte offsets and enforces the per-block
// shared-memory limit.
//
// Offsets are 32-bit: shared memory is < 2^17 bytes. (Global-memory offsets
// are int64, see cache_geometry.cuh.)

#pragma once

#include <cstdint>

#include "ptx.cuh"

namespace pth_attn {

// sm_86 per-block shared-memory opt-in maximum (also the
// ptxas gate's limit). sm_80 allows more; the same layout is used there.
constexpr int kSmemPerBlockLimit = 101376;

// Bank geometry (CUDA Programming Guide 2.3): 32 banks x 4 B, so a 128-B line
// covers every bank once and a 16-B chunk (one ldmatrix row) covers 4 banks.
constexpr uint32_t kSmemLineBytes = 128;
constexpr uint32_t kSmemChunkBytes = 16;
constexpr uint32_t kSmemChunksPerLine = kSmemLineBytes / kSmemChunkBytes;  // 8

// Swizzle keys: K and Q `t & 7` (the 8 rows of one ldmatrix 8x8
// matrix hit 8 bank groups); V `2 * ((t >> 1) & 3)` (the 4 token pairs of one
// LDS.32 hit 4 even chunk offsets; even because chunk bit 0 already separates
// g = 0..3 from g = 4..7).
constexpr uint32_t kKSwizzleMask = 7;
constexpr uint32_t kVSwizzleMask = 3;
constexpr uint32_t kVSwizzleKeyScale = 2;
// Every key is < 8 (one of the 8 chunks of a line), which line_swizzle needs.
static_assert(kKSwizzleMask < kSmemChunksPerLine &&
                  kVSwizzleKeyScale * kVSwizzleMask < kSmemChunksPerLine,
              "swizzle keys must stay below 8 chunks per line");

constexpr int kScaleBytes = 4;  // one f32 scale per token (ks, vs)
constexpr int kF16Bytes = 2;    // Q is staged as fp16
constexpr int kPiChunk = 16;    // pi permutes inside each k16 MMA step

// ---------------------------------------------------------------------------
// Swizzles: byte offset within the K / V region or the Q buffer
// ---------------------------------------------------------------------------

// Permute the 8 16-B chunks of the 128-B line that holds dense offset `dense`
// by XOR with `key` (< 8): bits 0-3 (byte in chunk) and 7+ (line) are kept,
// bits 4-6 (chunk in line) are XOR-ed. Defined on lines, not on the chunks of
// a token row, so it is a permutation even when a row is 64 B (D = 64);
// for rows of >= 128 B it is identical to the plain `c ^ key`
// on the chunk index (checked by tests/sim/test_sim_swizzle.py).
//
// Written as one XOR: key * 16 < 128 only has bits 4-6, so `dense ^ (key *
// 16)` flips exactly the chunk-in-line bits. The equivalent arithmetic form
// `dense - chunk*16 + (chunk ^ key)*16` cost 488 more SASS instructions in
// the D=256 register probe (3384 -> 3872) because ptxas does not reduce it to
// an XOR.
PTH_HD constexpr uint32_t line_swizzle(uint32_t dense, uint32_t key) {
  return dense ^ (key * kSmemChunkBytes);
}

// K[t][byte] (int8, dense row pitch D) -> offset in the K region.
PTH_HD constexpr uint32_t k_swz(uint32_t t, uint32_t byte, uint32_t D) {
  return line_swizzle(t * D + byte, t & kKSwizzleMask);
}

// V[t][byte] (int8, dense row pitch D) -> offset in the V region.
PTH_HD constexpr uint32_t v_swz(uint32_t t, uint32_t byte, uint32_t D) {
  return line_swizzle(t * D + byte,
                      kVSwizzleKeyScale * ((t >> 1) & kVSwizzleMask));
}

// Q_smem[row][k] (fp16, pitch 2D bytes, k = position after pi) -> offset in
// the Q buffer. Key row & 7, like K: Q is read by ldmatrix 8 rows at a time.
PTH_HD constexpr uint32_t q_swz(uint32_t row, uint32_t k, uint32_t D) {
  return line_swizzle(row * kF16Bytes * D + kF16Bytes * k, row & kKSwizzleMask);
}

// ---------------------------------------------------------------------------
// The K permutation pi
// ---------------------------------------------------------------------------

// MMA index k (0..15) of a k16 step reads K at in-chunk offset d = pi(k).
// Derivation: lane q's ldmatrix register holds K
// bytes 4q..4q+3; cvt4 makes (b0,b1) = bytes (4q, 4q+1) and (b2,b3) = bytes
// (4q+2, 4q+3); the MMA uses b0..b3 at k = 2q, 2q+1, 2q+8, 2q+9. Writing
// k = 8h + 2q + j (h = k >> 3 selects b0b1 / b2b3, j = k & 1):
// d = 4q + 2h + j. pi = [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15].
PTH_HD constexpr int pi_k_to_d(int k) {
  return 4 * ((k & 7) >> 1) + 2 * (k >> 3) + (k & 1);
}

// Inverse: d = 4q + 2h + j -> k = 2q + 8h + j.
PTH_HD constexpr int pi_d_to_k(int d) {
  return 2 * (d >> 2) + 8 * ((d >> 1) & 1) + (d & 1);
}

// Where the prologue stores Q[row][d] (pi inside each 16-element chunk, then
// the Q swizzle).
PTH_HD constexpr uint32_t q_elem_offset(uint32_t row, uint32_t d, uint32_t D) {
  return q_swz(row,
               d - d % kPiChunk + pi_d_to_k(static_cast<int>(d % kPiChunk)), D);
}

// ---------------------------------------------------------------------------
// Per-lane addresses of the fragment loads (offsets within the region)
// ---------------------------------------------------------------------------

// ldmatrix.x4 of K (4 QK n-tiles of 8 tokens, k16 step s): lanes 8m..8m+7
// address tokens tok0 + 8m + 0..7 at 16-B chunk s. Register m of lane (g, q)
// then holds bytes 16s + 4q .. +3 of token tok0 + 8m + g.
PTH_HD constexpr uint32_t k_ldsm_x4_offset(uint32_t lane, uint32_t tok0,
                                           uint32_t s, uint32_t D) {
  return k_swz(tok0 + 8 * (lane >> 3) + (lane & 7), kSmemChunkBytes * s, D);
}

// ldmatrix.x4 of Q (A fragment of rows row0..row0+15, k16 step s): lane l
// addresses row row0 + l % 16 at 8-element chunk 2s + l / 16, so the four
// matrices are the register order (a0a1) (a2a3) (a4a5) (a6a7) of Figure 82.
PTH_HD constexpr uint32_t q_ldsm_x4_offset(uint32_t lane, uint32_t row0,
                                           uint32_t s, uint32_t D) {
  return q_swz(row0 + (lane & 15), 8 * (2 * s + (lane >> 4)), D);
}

// LDS.32 number i (0..3) of lane (g, q) for tokens t16..t16+15 and d group a:
// V[t16 + 2q + {0,1,8,9}[i]][32a + 4g .. +3].
PTH_HD constexpr uint32_t v_lds32_offset(uint32_t lane, uint32_t t16,
                                         uint32_t a, uint32_t i, uint32_t D) {
  return v_swz(t16 + 2 * (lane & 3) + (i & 1) + 8 * (i >> 1),
               32 * a + 4 * (lane >> 2), D);
}

// ---------------------------------------------------------------------------
// Stage and buffer layout
// ---------------------------------------------------------------------------

// One pipeline stage of N tokens of one KV head:
//   K [N][D] int8 swizzled | V [N][D] int8 swizzled | ks [N] f32 | vs [N] f32
template <int D, int N>
struct StageLayout {
  static constexpr int kK = 0;
  static constexpr int kV = kK + N * D;
  static constexpr int kKs = kV + N * D;
  static constexpr int kVs = kKs + kScaleBytes * N;
  static constexpr int kBytes = kVs + kScaleBytes * N;  // N * (2D + 8)

  // The bank analysis assumes K and V start on a 128-B line; LDS.64 of the
  // scales needs 8-B alignment; ldmatrix and cp.async 16 need 16 B.
  static_assert(kV % kSmemLineBytes == 0,
                "V region must start on a 128-B line");
  static_assert(kKs % kSmemChunkBytes == 0 && kVs % kSmemChunkBytes == 0,
                "scale regions must be 16-B aligned");
};

// Whole-cell stage (shipped at D = 256): token t's whole
// vLLM cell [K D | ks f32 | V D | vs f32] (2D + 8 B) copied verbatim to
// t * kPitch. kPitch = 2D + 16 is an odd multiple of 16 B, so consecutive
// cells start in different 16-B bank groups (1-way for K ldmatrix
// and V LDS.32 without a swizzle). The last 8 B of each row are padding the
// loader never writes. The kernels read it through tiling.cuh
// CellStageReader (K at kCellK, ks at kCellKs, and so on).
template <int D, int N>
struct CellStageLayout {
  static constexpr int kCellBytes = 2 * D + 2 * kScaleBytes;  // 520 at D = 256
  static constexpr int kPitch = 2 * D + kSmemChunkBytes;      // 528 at D = 256
  static constexpr int kCellK = 0;
  static constexpr int kCellKs = D;
  static constexpr int kCellV = D + kScaleBytes;
  static constexpr int kCellVs = 2 * D + kScaleBytes;
  static constexpr int kBytes = N * kPitch;  // 33792 at D = 256, N = 64

  static_assert(kPitch % kSmemChunkBytes == 0 &&
                    (kPitch / kSmemChunkBytes) % 2 == 1,
                "the cell pitch must be an odd multiple of 16 B");
  static_assert(kPitch >= kCellBytes, "a cell must fit its row");
  static_assert(kBytes % kSmemLineBytes == 0,
                "stages must stay 128-B aligned when stacked");
};

// Whole dynamic shared-memory buffer: Q (QRows x D fp16) then Stages stages.
// The kernel must declare the buffer 128-B aligned (extern __shared__
// __align__(128)) for the bank analysis to hold.
template <int D, int N, int QRows, int Stages>
struct SmemLayout {
  using Stage = StageLayout<D, N>;
  static constexpr int kQ = 0;
  static constexpr int kQBytes = QRows * kF16Bytes * D;
  static constexpr int kStageBytes = Stage::kBytes;
  static constexpr int kBytes = kQBytes + Stages * kStageBytes;

  PTH_HD static constexpr int stage_offset(int stage) {
    return kQBytes + stage * kStageBytes;
  }

  static_assert(kQBytes % kSmemLineBytes == 0 &&
                    kStageBytes % kSmemLineBytes == 0,
                "every region must start on a 128-B line");
  static_assert(kBytes <= kSmemPerBlockLimit,
                "shared memory exceeds the sm_86 per-block limit (101376 B)");
};

}  // namespace pth_attn
