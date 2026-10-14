// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// loader.cuh: global -> shared copy of one 64-token KV tile, and the two-stage
// cp.async pipeline discipline around it.
//
// Two variants (plan C1, ADR-0006):
//
//   Separate streams (D = 64, 128, and any layout that is not an inline
//   cell). K, V, k_scale and v_scale are four streams with their own views
//   and their own cp.async width w in {16, 8, 4}, picked on the host from
//   alignment (cache_geometry.cuh stream_width; vLLM's inline cell gives K 8,
//   V 4, scales 4; #56164's decoupled layout K/V 16). Destination: the
//   swizzled StageLayout of smem_layout.cuh (ADR-0007).
//
//   Whole cell (the shipped default at D = 256, ADR 0012 Amendment 1).
//   vLLM's inline cell [K D | ks | V D | vs] (2D + 8 B) is one contiguous run
//   per (token, head), so one stream with one width (8 B for vLLM's layout:
//   520 = 65 x 8) copies all four, into CellStageLayout (pitch 2D + 16). Only
//   valid when the four views are that inline cell (cell_layout_width()
//   checks it on the host). At 8-B alignment the D = 256 kernels copy cells
//   through registers instead (attn_kernel.cuh produce_cells_regstaged); the
//   cp.async cell path below serves the 4-B aligned case.
//
// Who copies what. Each of the 4 warps owns 16 consecutive tokens of the
// tile. Lane l (l < 16) resolves token l of its warp: one FastDivmod, one
// block_table read, then the int64 byte offset of each stream (all address
// math in cache_geometry.cuh: token_page, StreamGeometry::offset). A row is
// copied as chunks of w bytes; instruction `it` gives lane l chunk
// 32*it + l of the warp's 16-row run (stream_chunk), so each instruction
// moves 32*w contiguous bytes of one row (w = 4, 8) or covers two rows
// (w = 16, D = 256), and the row offset is fetched from the resolving lane
// with a shuffle. The two scale streams share ONE 4-byte instruction per
// warp: lanes 0..15 copy k_scale of the warp's tokens, lanes 16..31 copy
// v_scale of the partner warp w ^ 1's tokens (lanes 16..31 resolve those
// tokens themselves). Partner tokens, not the warp's own, because the vs
// region starts 256 B after ks (a multiple of the 128-B bank sweep):
// same-token pairing would put both halves in the same 16 banks (2-way),
// partner pairing is 1-way
// (test_scale_pairing_with_partner_warp_is_conflict_free). The resulting lane
// -> (token, byte) map is the simulator's golden loader map
// (tests/sim/ptxsim/layouts.py), checked by tests/cpu/test_loader_host.py.
//
// Tokens at or past `tok_end` (the CTA's KV range end) are zero-filled with
// cp.async's src-size operand (0 bytes read, w zero bytes written), so stale
// shared memory or a poisoned scale beyond the range never reaches the math
// (error-budget K6). block_table is not read for them.
//
// Host emulation: the per-lane code is written once as PTH_HD templates over
// two small policies, a broadcast (device: __shfl_sync; host: a lookup in the
// 32 lanes' values) and a copy (device: cp.async; host: memcpy plus zero
// fill). emulate_tile_* below runs it on the host so g++ tests compare the
// very same index logic with the simulator (tests/cpu/test_loader_host.py);
// the device path is compared on the GPU with the same expected images.

#pragma once

#include <cstdint>
#include <cstring>

#include "cache_geometry.cuh"
#include "ptx.cuh"
#include "smem_layout.cuh"
#include "tile_traits.cuh"

namespace pth_attn {

constexpr int kLoaderTokensPerWarp = kLoadTileTokens / kWarps;  // 16
// Lane l < 16 owns token l of its warp; lanes 16..31 own the partner warp's.
static_assert(2 * kLoaderTokensPerWarp == kWarpSize,
              "two 16-token groups per warp: own and partner tokens");
static_assert(kWarps % 2 == 0, "warps pair up as w and w ^ 1");
constexpr uint32_t kFullWarpMask = 0xFFFFFFFFu;

// Unroll factor of the chunk loops (ADR-0006). Full unrolling lets ptxas
// hoist every iteration's shuffled row offset: 255 registers and a 204-B
// spill at D = 256 in the test kernel, impossible next to the attention
// math (R2). Static sm_86 SASS of the test kernels, registers and
// instructions per LDGSTS in the loop bodies:
//   unroll 1: 38-40 regs, ~27 instr/LDGSTS;  unroll 2: 35-45 regs, ~21;
//   unroll 4: 36-64 regs, ~18;  unroll 8: 40-127 regs;  full: 254-255 + spill.
// 2 is the smallest register footprint with less loop overhead than 1. The
// G2 register gate on the real kernel and its timing decide whether to move.
constexpr int kLoaderUnroll = 2;
// Spelled so that g++ -Wall does not warn about an unknown pragma.
#if defined(__CUDACC__)
  #define PTH_LOADER_UNROLL _Pragma("unroll kLoaderUnroll")
#else
  #define PTH_LOADER_UNROLL
#endif

// ---------------------------------------------------------------------------
// Index logic (host + device)
// ---------------------------------------------------------------------------

// Chunk `32*it + lane` of a warp's run of 16 rows of kRow bytes in chunks of
// kW bytes: row (token within the warp) and byte. `active` is false past the
// run (only the cell variant has a partial last instruction).
struct LoaderChunk {
  int32_t token;  // 0..15, within the warp
  int32_t byte;   // byte offset within the row
  bool active;
};

template <int kRow, int kW>
PTH_HD LoaderChunk stream_chunk(int it, int lane) {
  static_assert(kRow % kW == 0, "a row is whole chunks");
  constexpr int kChunksPerRow = kRow / kW;
  const int i = kWarpSize * it + lane;
  return {i / kChunksPerRow, (i % kChunksPerRow) * kW,
          i < kLoaderTokensPerWarp * kChunksPerRow};
}

// Warp-instructions needed for one stream of one warp.
template <int kRow, int kW>
PTH_HD constexpr int stream_iters() {
  return (kLoaderTokensPerWarp * (kRow / kW) + kWarpSize - 1) / kWarpSize;
}

// Token (within the 64-token tile) that `lane` of `warp` resolves: its own
// warp's token l for l < 16, the partner warp's token l - 16 otherwise.
PTH_HD int lane_token(int warp, int lane) {
  const int w = lane < kLoaderTokensPerWarp ? warp : (warp ^ 1);
  return kLoaderTokensPerWarp * w + (lane & (kLoaderTokensPerWarp - 1));
}

// Scale stream of the paired instruction for `lane`.
PTH_HD int scale_stream(int lane) {
  return lane < kLoaderTokensPerWarp ? kStreamKScale : kStreamVScale;
}

// Byte offset of a scale slot in the separate-stream stage.
template <int D>
PTH_HD uint32_t scale_dst(int lane, int token_in_tile) {
  using St = StageLayout<D, kLoadTileTokens>;
  return static_cast<uint32_t>(
      (lane < kLoaderTokensPerWarp ? St::kKs : St::kVs) +
      kScaleBytes * token_in_tile);
}

// What one lane knows after resolving its token: the byte offsets (from
// each stream's base) of that token's rows, valid only if `valid`.
struct LaneRows {
  int64_t k;      // K row (separate) or whole cell (cell variant)
  int64_t v;      // V row
  int64_t scale;  // this lane's scale: k_scale (lane < 16) or v_scale
  bool valid;     // token < tok_end
};

// Resolves lane_token(warp, lane) of KV tile `tile` of request `req`, head
// `kv_head`. Reads block_table only for tokens inside the KV range.
PTH_HD LaneRows resolve_lane(const KernelParams& p, int32_t req,
                             int32_t kv_head, int32_t tile, int32_t tok_end,
                             int warp, int lane) {
  LaneRows r{0, 0, 0, false};
  const int32_t pos = tile * kLoadTileTokens + lane_token(warp, lane);
  if (pos >= tok_end) return r;
  uint32_t slot = 0;
  const int64_t page = token_page(p, req, static_cast<uint32_t>(pos), slot);
  r.k = p.stream[kStreamK].offset(page, slot, kv_head);
  r.v = p.stream[kStreamV].offset(page, slot, kv_head);
  r.scale = p.stream[scale_stream(lane)].offset(page, slot, kv_head);
  r.valid = true;
  return r;
}

// Tokens of `warp`'s 16 that lie inside the KV range: 0..16.
PTH_HD int warp_valid_tokens(int32_t tile, int32_t tok_end, int warp) {
  const int32_t first = tile * kLoadTileTokens + kLoaderTokensPerWarp * warp;
  const int32_t n = tok_end - first;
  return n < 0 ? 0 : (n > kLoaderTokensPerWarp ? kLoaderTokensPerWarp : n);
}

// Whole-cell width: the largest w in {16, 8, 4} that divides the K stream's
// width (which already divides the base and every used stride) and the cell
// size 2D + 8. Returns 0 if the four views are not one inline cell: V, ks and
// vs must start at K + D + 4, K + D and K + 2D + 4 with K's strides.
template <int D>
PTH_HD int32_t cell_layout_width(const KernelParams& p) {
  using Cs = CellStageLayout<D, kLoadTileTokens>;
  const StreamGeometry& k = p.stream[kStreamK];
  const int64_t field[kNumStreams] = {Cs::kCellK, Cs::kCellV, Cs::kCellKs,
                                      Cs::kCellVs};
  for (int s = 0; s < kNumStreams; ++s) {
    const StreamGeometry& g = p.stream[s];
    if (g.base != k.base + field[s] || g.s_block != k.s_block ||
        g.s_token != k.s_token || g.s_head != k.s_head) {
      return 0;
    }
  }
  const int32_t widths[] = {kGeomWidth16, kGeomWidth8, kGeomWidth4};
  for (int32_t w : widths) {
    if (w <= k.width && Cs::kCellBytes % w == 0) return w;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Per-lane issue code, shared by the device loader and the host emulation.
// Bcast: int64_t operator()(int64_t mine, int src_lane) const.
// Copy:  template <int W> void op(uint32_t dst, const char* src, uint32_t n)
//        copies n (0 or W) bytes and zero-fills the rest of the W.
// ---------------------------------------------------------------------------

// One K or V stream of one warp, width kW.
#if defined(__CUDACC__)
  #pragma nv_exec_check_disable
#endif
template <int D, int kW, bool kIsV, class Bcast, class Copy>
PTH_HD void issue_kv_stream(const StreamGeometry& g, int64_t my_row, int warp,
                            int lane, int valid, const Bcast& bcast,
                            const Copy& copy) {
  using St = StageLayout<D, kLoadTileTokens>;
  PTH_LOADER_UNROLL
  for (int it = 0; it < stream_iters<D, kW>(); ++it) {
    const LoaderChunk c = stream_chunk<D, kW>(it, lane);
    // Every lane takes part in the shuffle (a warp-collective); K/V runs are
    // whole instructions, so `active` is always true here.
    const int64_t row = bcast(my_row, c.token);
    const int t = kLoaderTokensPerWarp * warp + c.token;  // token in the tile
    const uint32_t dst =
        kIsV ? St::kV + v_swz(t, c.byte, D) : St::kK + k_swz(t, c.byte, D);
    const bool ok = c.token < valid;
    copy.template op<kW>(dst, ok ? g.base + row + c.byte : g.base,
                         ok ? static_cast<uint32_t>(kW) : 0u);
  }
}

// Runtime width switch (warp-uniform: the width is a kernel argument), so
// all three widths exist in the binary and the branch never diverges.
#if defined(__CUDACC__)
  #pragma nv_exec_check_disable
#endif
template <int D, bool kIsV, class Bcast, class Copy>
PTH_HD void issue_kv_stream_any_width(const StreamGeometry& g, int64_t my_row,
                                      int warp, int lane, int valid,
                                      const Bcast& bcast, const Copy& copy) {
  switch (g.width) {
    case kGeomWidth16:
      issue_kv_stream<D, kGeomWidth16, kIsV>(g, my_row, warp, lane, valid,
                                             bcast, copy);
      break;
    case kGeomWidth8:
      issue_kv_stream<D, kGeomWidth8, kIsV>(g, my_row, warp, lane, valid, bcast,
                                            copy);
      break;
    default:
      issue_kv_stream<D, kGeomWidth4, kIsV>(g, my_row, warp, lane, valid, bcast,
                                            copy);
      break;
  }
}

// Separate streams: one lane's share of loading `tile` into a stage.
#if defined(__CUDACC__)
  #pragma nv_exec_check_disable
#endif
template <int D, class BcastK, class BcastV, class Copy>
PTH_HD void issue_tile_separate_lane(const KernelParams& p,
                                     const LaneRows& mine, int32_t tile,
                                     int32_t tok_end, int warp, int lane,
                                     const BcastK& bcast_k,
                                     const BcastV& bcast_v, const Copy& copy) {
  const int valid = warp_valid_tokens(tile, tok_end, warp);
  issue_kv_stream_any_width<D, false>(p.stream[kStreamK], mine.k, warp, lane,
                                      valid, bcast_k, copy);
  issue_kv_stream_any_width<D, true>(p.stream[kStreamV], mine.v, warp, lane,
                                     valid, bcast_v, copy);
  // Both scale streams in one 4-B instruction (see the file comment). Scale
  // rows are one f32, so their width is always 4.
  const StreamGeometry& g = p.stream[scale_stream(lane)];
  copy.template op<kGeomWidth4>(
      scale_dst<D>(lane, lane_token(warp, lane)),
      mine.valid ? g.base + mine.scale : g.base,
      mine.valid ? static_cast<uint32_t>(kGeomWidth4) : 0u);
}

// Whole cell, width kW: one lane's share.
#if defined(__CUDACC__)
  #pragma nv_exec_check_disable
#endif
template <int D, int kW, class Bcast, class Copy>
PTH_HD void issue_cells(const StreamGeometry& k, int64_t my_cell, int warp,
                        int lane, int valid, const Bcast& bcast,
                        const Copy& copy) {
  using Cs = CellStageLayout<D, kLoadTileTokens>;
  PTH_LOADER_UNROLL
  for (int it = 0; it < stream_iters<Cs::kCellBytes, kW>(); ++it) {
    const LoaderChunk c = stream_chunk<Cs::kCellBytes, kW>(it, lane);
    // Clamp the source lane so the shuffle stays warp-collective in the
    // partial last instruction; the copy itself is predicated on `active`.
    const int src = c.active ? c.token : kLoaderTokensPerWarp - 1;
    const int64_t cell = bcast(my_cell, src);
    if (c.active) {
      const int t = kLoaderTokensPerWarp * warp + c.token;
      const bool ok = c.token < valid;
      copy.template op<kW>(static_cast<uint32_t>(t * Cs::kPitch + c.byte),
                           ok ? k.base + cell + c.byte : k.base,
                           ok ? static_cast<uint32_t>(kW) : 0u);
    }
  }
}

#if defined(__CUDACC__)
  #pragma nv_exec_check_disable
#endif
template <int D, class Bcast, class Copy>
PTH_HD void issue_tile_cell_lane(const KernelParams& p, const LaneRows& mine,
                                 int32_t cell_width, int32_t tile,
                                 int32_t tok_end, int warp, int lane,
                                 const Bcast& bcast, const Copy& copy) {
  const int valid = warp_valid_tokens(tile, tok_end, warp);
  const StreamGeometry& k = p.stream[kStreamK];
  if (cell_width == kGeomWidth8) {
    issue_cells<D, kGeomWidth8>(k, mine.k, warp, lane, valid, bcast, copy);
  } else {
    issue_cells<D, kGeomWidth4>(k, mine.k, warp, lane, valid, bcast, copy);
  }
}

// ---------------------------------------------------------------------------
// Host emulation (shim: cp.async -> memcpy + zero fill; shuffle -> lookup)
// ---------------------------------------------------------------------------

struct HostCopy {
  uint8_t* stage;  // host buffer of the stage image
  template <int W>
  void op(uint32_t dst, const char* src, uint32_t n) const {
    std::memcpy(stage + dst, src, n);
    std::memset(stage + dst + n, 0, W - n);
  }
};

struct HostBcast {
  const int64_t* lane_values;  // the 32 lanes' values of one field
  int64_t operator()(int64_t /*mine*/, int src_lane) const {
    return lane_values[src_lane];
  }
};

// Runs all 128 threads of one tile load in program order. `stage` must hold
// the stage image (StageLayout or CellStageLayout bytes); bytes the loader
// does not write are left as they are.
template <int D>
inline void emulate_tile(const KernelParams& p, bool whole_cell,
                         int32_t cell_width, uint8_t* stage, int32_t req,
                         int32_t kv_head, int32_t tile, int32_t tok_end) {
  const HostCopy copy{stage};
  for (int warp = 0; warp < kWarps; ++warp) {
    LaneRows rows[kWarpSize];
    int64_t k[kWarpSize], v[kWarpSize];
    for (int lane = 0; lane < kWarpSize; ++lane) {
      rows[lane] = resolve_lane(p, req, kv_head, tile, tok_end, warp, lane);
      k[lane] = rows[lane].k;
      v[lane] = rows[lane].v;
    }
    for (int lane = 0; lane < kWarpSize; ++lane) {
      if (whole_cell) {
        issue_tile_cell_lane<D>(p, rows[lane], cell_width, tile, tok_end, warp,
                                lane, HostBcast{k}, copy);
      } else {
        issue_tile_separate_lane<D>(p, rows[lane], tile, tok_end, warp, lane,
                                    HostBcast{k}, HostBcast{v}, copy);
      }
    }
  }
}

#if defined(__CUDACC__)
// ---------------------------------------------------------------------------
// Device loader
// ---------------------------------------------------------------------------

struct DeviceBcast {
  __device__ int64_t operator()(int64_t mine, int src_lane) const {
    return __shfl_sync(kFullWarpMask, mine, src_lane);
  }
};

struct DeviceCopy {
  uint32_t stage_smem;  // shared-state-space address of the stage
  template <int W>
  __device__ void op(uint32_t dst, const char* src, uint32_t n) const {
    cp_async_zfill<W>(stage_smem + dst, src, n);
  }
};

// Issues (does not wait for) the cp.async of KV tile `tile` into the stage at
// `stage_smem`. All 128 threads must call it (shuffles are warp-collective).
template <int D>
__device__ __forceinline__ void load_tile_separate(const KernelParams& p,
                                                   uint32_t stage_smem,
                                                   int32_t req, int32_t kv_head,
                                                   int32_t tile,
                                                   int32_t tok_end) {
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const LaneRows mine =
      resolve_lane(p, req, kv_head, tile, tok_end, warp, lane);
  issue_tile_separate_lane<D>(p, mine, tile, tok_end, warp, lane, DeviceBcast{},
                              DeviceBcast{}, DeviceCopy{stage_smem});
}

template <int D>
__device__ __forceinline__ void load_tile_cell(const KernelParams& p,
                                               int32_t cell_width,
                                               uint32_t stage_smem, int32_t req,
                                               int32_t kv_head, int32_t tile,
                                               int32_t tok_end) {
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const LaneRows mine =
      resolve_lane(p, req, kv_head, tile, tok_end, warp, lane);
  issue_tile_cell_lane<D>(p, mine, cell_width, tile, tok_end, warp, lane,
                          DeviceBcast{}, DeviceCopy{stage_smem});
}

// The two-stage pipeline over KV tiles [begin, end) (plan 3.4 step 3):
//
//   issue(begin -> stage 0); commit
//   for i in [begin, end):
//     wait_group 0      this thread's copies of tile i have landed
//     __syncthreads()   (a) every thread's copies of tile i have landed and
//                       are visible; (b) every thread has finished consume()
//                       of tile i - 1, so its stage, (i + 1 - begin) & 1, may
//                       be overwritten
//     issue(i + 1 -> stage (i + 1 - begin) & 1), if i + 1 < end; commit
//                       (an empty group when nothing was issued keeps the
//                       group count uniform; wait_group 0 never stalls on it)
//     consume(tile i in stage (i - begin) & 1)   overlaps the copy of i + 1
//
// So at most one group is in flight while a stage is consumed, and no stage
// is written while any thread may still read it. issue(tile, stage) and
// consume(tile, stage) are called by all threads.
template <class Issue, class Consume>
__device__ __forceinline__ void run_two_stage_pipeline(int32_t begin,
                                                       int32_t end,
                                                       const Issue& issue,
                                                       const Consume& consume) {
  if (begin >= end) return;
  issue(begin, 0);
  cp_async_commit();
  for (int32_t i = begin; i < end; ++i) {
    cp_async_wait<0>();
    __syncthreads();
    if (i + 1 < end) issue(i + 1, (i + 1 - begin) & 1);
    cp_async_commit();
    consume(i, (i - begin) & 1);
  }
}
#endif  // __CUDACC__

}  // namespace pth_attn
