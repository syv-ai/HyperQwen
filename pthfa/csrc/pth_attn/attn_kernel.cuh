// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_kernel.cuh: the tensor-core attention kernel (plan section 3; build
// ladder steps 12-19; owner T4 = P0InfraCI in Phase 3).
//
// One CTA of 128 threads computes one (q-tile x, KV head, split): the
// planner's grid (ADR 0016), the same as the scalar reference kernel
// (reference_kernel.cu, the executable specification it is tested against).
// Every address, row tile, causal limit and split range comes from
// cache_geometry.cuh; the KV tiles come from loader.cuh (separate streams,
// two-stage cp.async pipeline, ADR 0006); the fragment loops from tiling.cuh;
// the online softmax and P'' from softmax.cuh; the mask from mask.cuh.
//
//   prologue   find the row tile; issue the first KV tile; stage Q into
//              shared memory as fp16 with the per-row 2^-e prescale and the
//              K permutation pi (ADR 0008, 0010); per-row score factor
//              c_row = fp32(softmax_scale * log2 e) * 2^e.
//   main loop  per KV tile of this split: wait, barrier, issue the next tile
//              into the other stage, then per sub-tile of SUB tokens of the
//              warp's slice: S = Q~ K^T (qk_subtile), s = fp32(fp32(S ks)
//              c_row), causal mask on boundary sub-tiles, softmax_step (P''), O
//              += P'' V (pv_subtile).
//   merge      WN > 1: warps n > 0 park (O, m, l, c) in the freed stages and
//              warp n = 0 merges them (emulate.py `_merge_warps` order).
//   epilogue   o = fp32(O / l) * 2^C (rcp.approx); S = 1: out in its dtype,
//              16-B stores of 8 contiguous d (ADR 0009); S > 1: the fp32
//              partial and lse = m + lg2(l) (log2 units, ADR 0016); an empty
//              row writes o = 0, lse = -inf.
//
// The arithmetic is python/pth_testing/emulate.py's, which the bound of
// python/pth_testing/bound.py covers with Tiling(64, SUB, WN, S, bq).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "attn_args.h"
#include "cache_geometry.cuh"
#include "dequant.cuh"
#include "loader.cuh"
#include "mask.cuh"
#include "ptx.cuh"
#include "smem_layout.cuh"
#include "softmax.cuh"
#include "tile_traits.cuh"
#include "tiling.cuh"

namespace pth_attn {

// Everything a launch needs: the frozen KernelParams plus the output dtype
// (a runtime switch: it only touches the epilogue) and, for the whole-cell
// loader, its cp.async width (loader.cuh cell_layout_width: 8 or 4).
struct AttnParams {
  KernelParams k;
  int32_t out_dtype;   // PTH_ATTN_F32 / PTH_ATTN_F16 / PTH_ATTN_BF16
  int32_t cell_width;  // whole-cell instantiations only
};

// Shared memory of one instantiation: Q (T3's layout), two stages in the
// loader's layout (separate streams: StageLayout; whole cell:
// CellStageLayout), and the per-row score factors (one f32 per CTA row).
// The factors get their own region at 1 CTA per SM (there is room, and the
// producer warps may fill stage 1 early); with 2-3 CTAs per SM there is no
// room (D = 128: 2 x (50176 + 256 + 1024) > 102400), so they live in stage
// 1, which no copy touches before every thread has read its factors (the
// first loop barrier).
template <int D, int WM, int SUB, bool kCell>
struct AttnSmem {
  using TT = AttnGeometry<D, WM, SUB>;
  using R = std::conditional_t<kCell, CellStageReader<D>, SepStageReader<D>>;
  static constexpr int kQ = 0;
  static constexpr int kQBytes = TT::Layout::kQBytes;
  static constexpr int kStageBytes = R::kBytes;
  PTH_HD static constexpr int stage_offset(int s) {
    return kQBytes + s * kStageBytes;
  }
  static constexpr bool kOwnRowFactor = ctas_per_sm(D) == 1;
  static constexpr int kRowFactor =
      kOwnRowFactor ? kQBytes + kStages * kStageBytes : kQBytes + kStageBytes;
  static constexpr int kBytes = kQBytes + kStages * kStageBytes +
                                (kOwnRowFactor ? TT::kRows * kScaleBytes : 0);
  // WN merge area in the stages: per parked warp its O (D/8 x 4 x 32 floats),
  // then (m, l) per parked row, then one c per parked warp.
  static constexpr int kParked = (TT::kWn - 1) * WM;
  static constexpr int kMergeOFloats = D / kMmaN * 4 * kWarpSize;
  static constexpr int kMergeBytes =
      kParked * (kMergeOFloats + 2 * kMmaM + 1) * kAccElemBytes;
  static_assert(kQBytes % kSmemLineBytes == 0 &&
                    kStageBytes % kSmemLineBytes == 0,
                "regions must start on 128-B lines");
  static_assert(kBytes <= kSmemPerBlockLimit, "shared memory over the limit");
  static_assert(TT::kCtasPerSm * (kBytes + kSmemReservedPerBlock) <= kSmemPerSm,
                "CTAs per SM must still fit");
  static_assert(kMergeBytes <= kStages * kStageBytes,
                "WN merge must fit in the stages");
};

#if defined(__CUDACC__)

// Two IoT values (one 32-bit word, low half first) as floats.
template <typename QT>
__device__ __forceinline__ float2 io_pair_to_float2(uint32_t w);
template <>
__device__ __forceinline__ float2 io_pair_to_float2<__nv_bfloat16>(uint32_t w) {
  return make_float2(__uint_as_float(w << 16),
                     __uint_as_float(w & 0xffff0000u));
}
template <>
__device__ __forceinline__ float2 io_pair_to_float2<__half>(uint32_t w) {
  __half2 h;
  *reinterpret_cast<uint32_t*>(&h) = w;
  return __half22float2(h);
}

// Stores 8 floats (8 contiguous d) at `dst` in `dtype`, rounding to nearest
// even for the 16-bit types (as combine_kernel.cuh's store_out).
__device__ __forceinline__ void store8(char* dst, int32_t dtype,
                                       const float (&v)[8]) {
  if (dtype == PTH_ATTN_F32) {
    reinterpret_cast<float4*>(dst)[0] = make_float4(v[0], v[1], v[2], v[3]);
    reinterpret_cast<float4*>(dst)[1] = make_float4(v[4], v[5], v[6], v[7]);
    return;
  }
  uint32_t w[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (dtype == PTH_ATTN_BF16) {
      const __nv_bfloat162 b = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
      w[i] = *reinterpret_cast<const uint32_t*>(&b);
    } else {
      const __half2 h = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
      w[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
  }
  *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
}

// Stages the CTA's kRows rows of Q (rows past the tile's tokens are zero)
// and writes each row's score factor. Warp w stages rows w, w + 4, ...
template <int D, int kRows, typename QT>
__device__ __forceinline__ void stage_q(const KernelParams& p,
                                        const RowTile& rt, int32_t kvh,
                                        uint8_t* smem_q, float* row_factor,
                                        int warp, int lane) {
  constexpr int kPairsPerLane = D / 2 / kWarpSize;
  #pragma unroll 1
  for (int r = warp; r < kRows; r += kWarps) {
    int32_t tok = 0, h = 0;
    const bool valid = row_to_token_head(static_cast<uint32_t>(r), p.group,
                                         rt.num_tok, tok, h);
    const char* src =
        static_cast<const char*>(p.q) +
        q_offset(p, static_cast<int64_t>(rt.q_start) + rt.tok_begin + tok,
                 kvh * p.group_size + h);
    float2 v[kPairsPerLane];
    float amax = 0.f;
  #pragma unroll
    for (int i = 0; i < kPairsPerLane; ++i) {
      const int d = 2 * (lane + i * kWarpSize);
      v[i] = valid ? io_pair_to_float2<QT>(*reinterpret_cast<const uint32_t*>(
                         src + kF16Bytes * d))
                   : make_float2(0.f, 0.f);
      amax = fmaxf(amax, fmaxf(fabsf(v[i].x), fabsf(v[i].y)));
    }
  #pragma unroll
    for (int o = kWarpSize / 2; o > 0; o /= 2) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const int e = q_prescale_exponent(amax);
  #pragma unroll
    for (int i = 0; i < kPairsPerLane; ++i) {
      const int d = 2 * (lane + i * kWarpSize);
      // pi inside each 16-element chunk, then the Q swizzle (ADR 0007/0008);
      // d and d + 1 land in adjacent bytes, so the pair is one STS.32.
      *reinterpret_cast<uint32_t*>(smem_q + q_elem_offset(r, d, D)) =
          q_prescale_pack(v[i].x, v[i].y, e);
    }
    if (lane == 0) row_factor[r] = p.score_scale_log2 * pow2_f32(e);
  }
}

// Producer warps (ADR 0012 Amendment 1): at 1 CTA per SM (D = 256) four
// extra warps only issue the KV copies, so the issue (throttled to the copy
// rate) overlaps the math of the four compute warps. With 2-3 CTAs per SM
// (D = 128, 64) the other CTAs provide that overlap, and 256 threads would
// not fit their register budget (65536 / (2 * 256) = 128 registers).
// (Variable templates: usable in device code, unlike the host constexpr
// function ctas_per_sm.)
template <int D>
inline constexpr int kAttnProducerWarps = ctas_per_sm(D) == 1 ? kWarps : 0;
template <int D>
inline constexpr int kAttnThreads = (kWarps + kAttnProducerWarps<D>)*kWarpSize;
// All threads of a warp-specialised CTA (compute + producer warps).
constexpr int kAttnThreadsWs = 2 * kWarps * kWarpSize;
// Named barriers (id 0 is __syncthreads): consumers only; tile landed in
// stage s (producers arrive, consumers wait); stage s free (consumers
// arrive, producers wait).
constexpr int kBarConsumers = 1;
constexpr int kBarFull0 = 2;   // + stage
constexpr int kBarEmpty0 = 4;  // + stage

// bar.sync / bar.arrive on a named barrier with `n` participating threads
// (PTX ISA 9.7.13.1); both order prior memory accesses of the CTA.
__device__ __forceinline__ void named_bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// Barrier among the four compute warps only (producers never join it).
template <int kProducers>
__device__ __forceinline__ void consumer_sync() {
  if constexpr (kProducers > 0) {
    named_bar_sync(kBarConsumers, kWarps * kWarpSize);
  } else {
    __syncthreads();
  }
}

// Global loads that do not allocate in L1 (PTX ISA 9.7.9.8,
// ld.global.L1::no_allocate): the KV stream is read once, and L1 shares its
// SRAM with the shared-memory stages (ADR 0012 Amendment 1). 16 B for the
// aligned body of a cell, 8 B for its edge (produce_cells_regstaged).
struct NoAllocLoad16 {
  __device__ static uint4 load(const char* g) {
    uint4 r;
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(g));
    return r;
  }
};
struct NoAllocLoad8 {
  __device__ static uint2 load(const char* g) {
    uint2 r;
    asm volatile("ld.global.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
                 : "=r"(r.x), "=r"(r.y)
                 : "l"(g));
    return r;
  }
};

// Producer warp `pw` (0..3) of a warp-specialised CTA: copies its 16 tokens
// of every KV tile [tb, te) as whole inline cells (loader.cuh's cell lane
// map, CellStageLayout pitch 2D + 16) through registers: loads that skip L1,
// then shared stores once the stage is free. Tokens at or past the range end
// are zero-filled (as the cp.async loader's zfill). Measured against cp.async
// of the same bytes in ADR 0012 Amendment 1, and 8-B against 16-B loads in
// Amendment 2.
//
// An 8-B aligned cell of 2D + 8 bytes is one 8-B edge plus 2D bytes of 16-B
// aligned body. The edge is the cell's last 8 bytes if the cell
// starts 16-B aligned (e = 0) and its first 8 bytes otherwise (e = 8), so the
// body is bytes [e, e + 2D). Iteration `it` copies token `it`'s body, one
// 16-B chunk per lane (2D / 16 = 32 chunks = one warp instruction, 512
// contiguous bytes); one more instruction copies the 16 edges (lanes 0..15).
// e is per token, hence uniform within an iteration: the shared store is one
// 16-B store when the destination t * kPitch + e is 16-B aligned (kPitch is a
// multiple of 16) and two 8-B stores otherwise.
template <int D, class Sm>
__device__ __forceinline__ void produce_cells_regstaged(
    const KernelParams& p, int32_t req, int32_t kvh, int32_t tb, int32_t te,
    int32_t kv_end, uint8_t* smem, int pw, int lane) {
  using Cs = CellStageLayout<D, kLoadTileTokens>;
  constexpr int kBody = 2 * D;
  constexpr int kChunk = 16;
  constexpr int kEdge = Cs::kCellBytes - kBody;  // 8: the two f32 scales
  static_assert(kBody == kWarpSize * kChunk, "one body = one warp instruction");
  static_assert(kEdge == 8, "edge is one 8-B load");
  const StreamGeometry& kg = p.stream[kStreamK];
  uint4 body[kLoaderTokensPerWarp];
  uint2 edge;
  #pragma unroll 1
  for (int32_t j = tb; j < te; ++j) {
    const LaneRows mine = resolve_lane(p, req, kvh, j, kv_end, pw, lane);
    const int valid = warp_valid_tokens(j, kv_end, pw);
    // e of this lane's token (lanes 0..15 own tokens 0..15).
    const int e_mine = static_cast<int>(
        reinterpret_cast<uintptr_t>(kg.base + mine.k) & (kChunk - 1));
  #pragma unroll
    for (int it = 0; it < kLoaderTokensPerWarp; ++it) {
      const int64_t cell = __shfl_sync(kFullWarpMask, mine.k, it);
      const int e = __shfl_sync(kFullWarpMask, e_mine, it);
      body[it] = it < valid
                     ? NoAllocLoad16::load(kg.base + cell + e + kChunk * lane)
                     : make_uint4(0u, 0u, 0u, 0u);
    }
    const bool edge_lane = lane < kLoaderTokensPerWarp;
    const int edge_byte = e_mine == 0 ? kBody : 0;
    edge = edge_lane && lane < valid
               ? NoAllocLoad8::load(kg.base + mine.k + edge_byte)
               : make_uint2(0u, 0u);
    const int sj = (j - tb) & 1;
    if (j - tb >= kStages) named_bar_sync(kBarEmpty0 + sj, kAttnThreadsWs);
    uint8_t* st = smem + Sm::stage_offset(sj);
  #pragma unroll
    for (int it = 0; it < kLoaderTokensPerWarp; ++it) {
      const int e = __shfl_sync(kFullWarpMask, e_mine, it);
      uint8_t* dst = st + (kLoaderTokensPerWarp * pw + it) * Cs::kPitch + e +
                     kChunk * lane;
      if (e == 0) {
        *reinterpret_cast<uint4*>(dst) = body[it];
      } else {
        reinterpret_cast<uint2*>(dst)[0] = make_uint2(body[it].x, body[it].y);
        reinterpret_cast<uint2*>(dst)[1] = make_uint2(body[it].z, body[it].w);
      }
    }
    if (edge_lane) {
      *reinterpret_cast<uint2*>(
          st + (kLoaderTokensPerWarp * pw + lane) * Cs::kPitch + edge_byte) =
          edge;
    }
    named_bar_arrive(kBarFull0 + sj, kAttnThreadsWs);
  }
}

// One loader warp's share of copying KV tile `tile` into `stage`
// (loader.cuh; loader warp lw owns tokens 16 lw .. 16 lw + 15).
template <int D, bool kCell, class Sm>
__device__ __forceinline__ void issue_tile_to(const AttnParams& ap,
                                              const RowTile& rt, int32_t kvh,
                                              int32_t kv_end,
                                              uint32_t smem_base, int32_t tile,
                                              int stage, int lw, int lane) {
  const KernelParams& p = ap.k;
  const LaneRows mine = resolve_lane(p, rt.req, kvh, tile, kv_end, lw, lane);
  const DeviceCopy copy{smem_base +
                        static_cast<uint32_t>(Sm::stage_offset(stage))};
  if constexpr (kCell) {
    issue_tile_cell_lane<D>(p, mine, ap.cell_width, tile, kv_end, lw, lane,
                            DeviceBcast{}, copy);
  } else {
    issue_tile_separate_lane<D>(p, mine, tile, kv_end, lw, lane, DeviceBcast{},
                                DeviceBcast{}, copy);
  }
}

// Producer warp `pw` (0..3) of a warp-specialised CTA (ADR 0012 Amendment 1):
// it only copies, so the copy overlaps the consumers' math. Two stages; tile
// j may be written as soon as tile j - 2 has been released by the consumers.
// Shared by the 16-row-tile kernel (attn_fwd_body) and the decode kernel
// (attn_decode.cuh): both use the same stage layout and named barriers.
template <int D, bool kCell, class Sm>
__device__ __forceinline__ void run_producer_warp(
    const AttnParams& ap, const RowTile& rt, int32_t kvh, int32_t tb,
    int32_t te, int32_t kv_end, uint8_t* smem, int pw, int lane) {
  const KernelParams& p = ap.k;
  if (tb >= te) return;
  if constexpr (kCell) {
    // The inline cell at 8-B alignment (every standard vLLM layout):
    // register-staged loads that skip L1.
    if (ap.cell_width == kGeomWidth8) {
      produce_cells_regstaged<D, Sm>(p, rt.req, kvh, tb, te, kv_end, smem, pw,
                                     lane);
      return;
    }
  }
  // Separate streams (#56164's layout, 16-B rows), or a cell that is only 4-B
  // aligned (odd page padding): cp.async into the same stage layout
  // (loader.cuh).
  const uint32_t smem_base = smem_u32(smem);
  auto issue = [&](int32_t tile, int stage) {
    issue_tile_to<D, kCell, Sm>(ap, rt, kvh, kv_end, smem_base, tile, stage, pw,
                                lane);
  };
  issue(tb, 0);
  cp_async_commit();
  #pragma unroll 1
  for (int32_t i = tb; i < te; ++i) {
    if (i + 1 < te) {
      const int s1 = (i + 1 - tb) & 1;
      if (i + 1 - tb >= kStages) {
        named_bar_sync(kBarEmpty0 + s1, kAttnThreadsWs);
      }
      issue(i + 1, s1);
      cp_async_commit();
      cp_async_wait<1>();  // this thread's copies of tile i have landed
    } else {
      cp_async_wait<0>();
    }
    named_bar_arrive(kBarFull0 + ((i - tb) & 1), kAttnThreadsWs);
  }
}

template <int D, int WM, int SUB, typename QT, bool kCell>
__device__ __forceinline__ void attn_fwd_body(const AttnParams& ap) {
  using TT = AttnGeometry<D, WM, SUB>;
  using Sm = AttnSmem<D, WM, SUB, kCell>;
  using R = typename Sm::R;
  constexpr int kNt = SUB / kMmaN;
  extern __shared__ __align__(128) uint8_t smem[];
  const KernelParams& p = ap.k;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int warp_m = warp % WM, warp_n = warp / WM;
  const int32_t kvh = static_cast<int32_t>(blockIdx.y);
  const int32_t split = static_cast<int32_t>(blockIdx.z);

  RowTile rt{};
  if (!row_tile(p.query_start_loc, p.seq_lens, p.bq_div, p.bq, p.num_reqs,
                static_cast<int32_t>(blockIdx.x), rt)) {
    return;  // surplus q-tile of the padded grid
  }
  const int32_t kv_end = kv_range_end(rt);
  const int32_t min_limit = kv_min_limit(rt);
  int32_t tb = 0, te = 0;
  split_tile_range(split, p.num_splits, num_kv_tiles(kv_end, kLoadTileTokens),
                   tb, te);

  const uint32_t smem_base = smem_u32(smem);
  constexpr int kProducers = kAttnProducerWarps<D>;
  auto issue_tile = [&](int32_t tile, int stage, int lw) {
    issue_tile_to<D, kCell, Sm>(ap, rt, kvh, kv_end, smem_base, tile, stage, lw,
                                lane);
  };

  if constexpr (kProducers > 0) {
    if (warp >= kWarps) {
      run_producer_warp<D, kCell, Sm>(ap, rt, kvh, tb, te, kv_end, smem,
                                      warp - kWarps, lane);
      return;
    }
  } else {
    // No producer warps (2-3 CTAs per SM hide the issue): the first tile's
    // copy overlaps the Q staging.
    if (tb < te) issue_tile(tb, 0, warp);
    cp_async_commit();
  }

  float* row_factor = reinterpret_cast<float*>(smem + Sm::kRowFactor);
  stage_q<D, WM * kMmaM, QT>(p, rt, kvh, smem + Sm::kQ, row_factor, warp, lane);
  consumer_sync<kProducers>();

  const int g = frag_g(lane), q = frag_q(lane);
  float row_scale[2];
  int32_t limit[2];
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = warp_m * kMmaM + g + 8 * h;
    row_scale[h] = row_factor[r];
    int32_t tok = 0, hh = 0;
    limit[h] = row_to_token_head(static_cast<uint32_t>(r), p.group, rt.num_tok,
                                 tok, hh)
                   ? kv_limit(rt.q_len, rt.seq_len, rt.tok_begin + tok)
                   : -1;
  }

  WarpSoftmax<D> st;
  softmax_init(st);

  #pragma unroll 1
  for (int32_t i = tb; i < te; ++i) {
    const int buf = (i - tb) & 1;
    if constexpr (kProducers > 0) {
      named_bar_sync(kBarFull0 + buf, kAttnThreadsWs);  // tile i has landed
    } else {
      // Two-stage pipeline (loader.cuh run_two_stage_pipeline's discipline,
      // with the first issue hoisted above the Q staging).
      cp_async_wait<0>();
      __syncthreads();
      if (i + 1 < te) issue_tile(i + 1, (i + 1 - tb) & 1, warp);
      cp_async_commit();
    }
    const uint32_t stage = smem_base + Sm::stage_offset(buf);
    const uint8_t* stage_ptr = smem + Sm::stage_offset(buf);
  #pragma unroll 1
    for (int sub = 0; sub < TT::kSubSteps; ++sub) {
      const int t0 = warp_n * TT::kTokensPerWarp + sub * SUB;
      const int32_t abs0 = i * kLoadTileTokens + t0;
      float s[kNt][4];
      qk_subtile<D, SUB, R>(s, smem_base + Sm::kQ, stage, t0, warp_m * kMmaM,
                            lane);
  #pragma unroll
      for (int n = 0; n < kNt; ++n) {
        const float2 k2 =
            load_scale_pair<R>(stage_ptr, false, t0 + n * kMmaN + 2 * q);
  #pragma unroll
        for (int h = 0; h < 2; ++h) {
          s[n][2 * h] = (s[n][2 * h] * k2.x) * row_scale[h];
          s[n][2 * h + 1] = (s[n][2 * h + 1] * k2.y) * row_scale[h];
        }
      }
      if (subtile_needs_mask(abs0, SUB, min_limit)) {
        apply_causal_mask<SUB>(s, abs0, limit, lane);
      }
      uint32_t pa[kNt][2];
      softmax_step<D, SUB, R>(st, s, stage_ptr, t0, pa, lane);
      pv_subtile<D, SUB, R>(st.o, pa, stage_ptr, t0, lane);
    }
    if constexpr (kProducers > 0) {
      // Release the stage for tile i + 2 (the producers wait for exactly
      // these arrivals, so every arrival is matched).
      if (i + kStages < te) named_bar_arrive(kBarEmpty0 + buf, kAttnThreadsWs);
    }
  }
  if constexpr (kProducers == 0) cp_async_wait<0>();

  st.l[0] = quad_sum(st.l[0]);
  st.l[1] = quad_sum(st.l[1]);
  int big_c = st.c;

  if constexpr (TT::kWn > 1) {
    // All warps are done with the stages: park warps n > 0 there.
    consumer_sync<kProducers>();
    float* merge_o = reinterpret_cast<float*>(smem + Sm::stage_offset(0));
    float* merge_ml = merge_o + Sm::kParked * Sm::kMergeOFloats;
    int* merge_c = reinterpret_cast<int*>(merge_ml + Sm::kParked * 2 * kMmaM);
    if (warp_n > 0) {
      const int slot = (warp_n - 1) * WM + warp_m;
      float* dst = merge_o + slot * Sm::kMergeOFloats;
  #pragma unroll
      for (int j = 0; j < D / kMmaN; ++j) {
  #pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          dst[(j * 4 + cc) * kWarpSize + lane] = st.o[j][cc];
        }
      }
      if (q == 0) {
  #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = slot * kMmaM + g + 8 * h;
          merge_ml[2 * row] = st.m[h];
          merge_ml[2 * row + 1] = st.l[h];
        }
      }
      if (lane == 0) merge_c[slot] = st.c;
    }
    consumer_sync<kProducers>();
    if (warp_n > 0) return;
  #pragma unroll
    for (int n = 1; n < TT::kWn; ++n) {
      const int cn = merge_c[(n - 1) * WM + warp_m];
      big_c = cn > big_c ? cn : big_c;
    }
  #pragma unroll
    for (int h = 0; h < 2; ++h) {
      float big_m = st.m[h];
  #pragma unroll
      for (int n = 1; n < TT::kWn; ++n) {
        const int row = ((n - 1) * WM + warp_m) * kMmaM + g + 8 * h;
        big_m = fmaxf(big_m, merge_ml[2 * row]);
      }
      const float m_use = big_m == -INFINITY ? 0.f : big_m;
      // Warp 0 of the row first, then n = 1 .. WN-1 (emulate.py order).
      const float f0 = st.m[h] == -INFINITY ? 0.f : ex2_approx(st.m[h] - m_use);
      const float scale0 = f0 * pow2_clamped(st.c - big_c);
      float l_sum = f0 * st.l[h];
  #pragma unroll
      for (int j = 0; j < D / kMmaN; ++j) {
        st.o[j][2 * h] *= scale0;
        st.o[j][2 * h + 1] *= scale0;
      }
  #pragma unroll
      for (int n = 1; n < TT::kWn; ++n) {
        const int slot = (n - 1) * WM + warp_m;
        const int row = slot * kMmaM + g + 8 * h;
        const float mn = merge_ml[2 * row];
        const float fn = mn == -INFINITY ? 0.f : ex2_approx(mn - m_use);
        const float scale_n = fn * pow2_clamped(merge_c[slot] - big_c);
        l_sum = l_sum + fn * merge_ml[2 * row + 1];
        const float* src = merge_o + slot * Sm::kMergeOFloats;
  #pragma unroll
        for (int j = 0; j < D / kMmaN; ++j) {
          st.o[j][2 * h] += scale_n * src[(j * 4 + 2 * h) * kWarpSize + lane];
          st.o[j][2 * h + 1] +=
              scale_n * src[(j * 4 + 2 * h + 1) * kWarpSize + lane];
        }
      }
      st.l[h] = l_sum;
      st.m[h] = big_m;
    }
  }

  // ---- epilogue: 8 contiguous d per group a and row (ADR 0009) ----
  const float c_unit = pow2_clamped(big_c);
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = warp_m * kMmaM + g + 8 * h;
    int32_t tok = 0, hh = 0;
    if (!row_to_token_head(static_cast<uint32_t>(r), p.group, rt.num_tok, tok,
                           hh)) {
      continue;  // padding row: never stored
    }
    const int64_t t_abs = static_cast<int64_t>(rt.q_start) + rt.tok_begin + tok;
    const int32_t head = kvh * p.group_size + hh;
    const bool live = st.l[h] > 0.f;
    const float inv_l = live ? rcp_approx(st.l[h]) : 0.f;
    char* dst_row;
    int32_t dtype;
    int64_t elem;
    if (p.num_splits == 1) {
      dst_row = static_cast<char*>(p.out) + out_offset(p, t_abs, head);
      dtype = ap.out_dtype;
      elem = dtype == PTH_ATTN_F32 ? kAccElemBytes : kF16Bytes;
    } else {
      dst_row = static_cast<char*>(p.ws_o) + ws_o_offset(p, split, t_abs, head);
      dtype = PTH_ATTN_F32;
      elem = kAccElemBytes;
    }
  #pragma unroll
    for (int grp = 0; grp < D / kVGroupD; ++grp) {
      float v[8];
  #pragma unroll
      for (int b = 0; b < 4; ++b) {
        v[b] = (st.o[4 * grp + b][2 * h] * inv_l) * c_unit;
        v[4 + b] = (st.o[4 * grp + b][2 * h + 1] * inv_l) * c_unit;
      }
      store8(dst_row + epilogue_d0(lane, grp) * elem, dtype, v);
    }
    if (p.num_splits > 1 && q == 0) {
      const float lse = live ? st.m[h] + lg2_approx(st.l[h]) : -INFINITY;
      *reinterpret_cast<float*>(static_cast<char*>(p.ws_lse) +
                                ws_lse_offset(p, split, t_abs, head)) = lse;
    }
  }
}

// __grid_constant__ keeps the parameters in the constant bank (no local copy
// when a member's address is taken). __launch_bounds__(threads, CTAs/SM)
// keeps ptxas within the occupancy register budget (ADR 0012).
template <int D, int WM, int SUB, typename QT, bool kCell>
__global__ void __launch_bounds__(kAttnThreads<D>, ctas_per_sm(D))
    attn_fwd_kernel(const __grid_constant__ AttnParams ap) {
  attn_fwd_body<D, WM, SUB, QT, kCell>(ap);
}

// ---------------------------------------------------------------------------
// Host side: one launch descriptor, and the per-D dispatchers defined in
// instantiate_d{64,128,256}.cu (one translation unit per head size, so the
// three compile in parallel and each holds only its own kernels).
// ---------------------------------------------------------------------------
struct AttnLaunch {
  AttnParams ap;
  dim3 grid;        // (num_q_tiles, num_kv_heads, num_splits)
  int32_t wm;       // 1, 2 or 4 (attn_wm_for_rows(bq * G))
  int32_t rows;     // bq * G: the decode kernel's n-tiles are ceil(rows / 8)
  int32_t q_dtype;  // PTH_ATTN_F16 or PTH_ATTN_BF16
  bool cell;        // whole-cell loader (ap.cell_width set) or separate streams
};

// The whole-cell loader exists where it fits next to the kernel's
// occupancy: D = 256 (1 CTA per SM). At D = 128 its pitch-272 stages break
// 2 CTAs per SM (2 x (16384 + 34816 + 1024) > 102400); D = 64 would fit but
// has 3 CTAs per SM to overlap copies already (ADR 0012 Amendment 1).
template <int D>
inline constexpr bool kAttnHasCell = D == 256;

// Launches `kKernel` with kSmemBytes of dynamic shared memory and
// kBlockThreads threads per block. Above 48 KB the limit must be raised once
// per kernel and process; the thread-safe static does that once per
// instantiation, which is once per kernel because the kernel itself is the
// template argument (a function-pointer parameter would share one static
// between kernels).
template <auto kKernel, int kSmemBytes, int kBlockThreads, class Arg>
cudaError_t launch_with_smem(dim3 grid, cudaStream_t stream, const Arg& arg) {
  static const cudaError_t attr = cudaFuncSetAttribute(
      kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (attr != cudaSuccess) return attr;
  kKernel<<<grid, kBlockThreads, kSmemBytes, stream>>>(arg);
  return cudaGetLastError();
}

template <int D, int WM, typename QT, bool kCell>
cudaError_t launch_attn_one(const AttnLaunch& L, cudaStream_t stream) {
  constexpr int kSub = default_sub<D, WM>();
  return launch_with_smem<attn_fwd_kernel<D, WM, kSub, QT, kCell>,
                          AttnSmem<D, WM, kSub, kCell>::kBytes,
                          kAttnThreads<D>>(L.grid, stream, L.ap);
}

// Picks the q dtype and the loader of one (D, WM) instantiation.
template <int D, int WM>
cudaError_t launch_attn_wm(const AttnLaunch& L, cudaStream_t stream) {
  const bool bf16 = L.q_dtype == PTH_ATTN_BF16;
  if constexpr (kAttnHasCell<D>) {
    if (L.cell) {
      return bf16 ? launch_attn_one<D, WM, __nv_bfloat16, true>(L, stream)
                  : launch_attn_one<D, WM, __half, true>(L, stream);
    }
  }
  if (L.cell) return cudaErrorInvalidValue;
  return bf16 ? launch_attn_one<D, WM, __nv_bfloat16, false>(L, stream)
              : launch_attn_one<D, WM, __half, false>(L, stream);
}

cudaError_t launch_attn_d64(const AttnLaunch& L, cudaStream_t stream);
cudaError_t launch_attn_d128(const AttnLaunch& L, cudaStream_t stream);
cudaError_t launch_attn_d256(const AttnLaunch& L, cudaStream_t stream);

#endif  // __CUDACC__

}  // namespace pth_attn
