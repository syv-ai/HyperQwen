// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_combine.cu: launch_combine of attn_launch.h, the split combine.
//
// One warp per row (token, head), with the reference library's arithmetic
// in the same order (combine_kernel.cuh combine_row): M = max of
// the live lse_s, then for s = 0 .. S-1: w = exp2f(lse_s - M), den += w,
// num_j = fmaf(w, o_s[j], num_j), skipping empty splits before reading o_s;
// out = num / den. The results are bit-identical to pth_ref_combine's.
//
// The difference is speed: the reference walks the splits one at a time, so
// each split costs a full global-load latency (14.6 us per launch at S = 34,
// dev timing). Here the loads of kBatch splits are
// issued together before their FMAs consume them in order.
//
// A separate translation unit so that it compiles in parallel with the
// attention kernels (tools/build_attn_lib.sh).

#include <cuda_runtime.h>

#include <string>

#include "attn_launch.h"
#include "combine_kernel.cuh"

namespace {

// Rows per combine CTA: one warp per (token, head), 4 warps (as the
// reference's combine launch).
constexpr int32_t kCombineWarps = 4;

struct CombineParams {
  pth_attn::PartialsView v;
  char* out;
  int64_t s_out_token, s_out_head;
  const int32_t* query_start_loc;
  int32_t num_reqs, num_tokens, num_q_heads, num_splits, out_dtype;
};

// Splits whose loads are in flight together.
constexpr int32_t kBatch = 8;

// Minimum one CTA per SM: without it ptxas caps registers for occupancy and
// spills the batch (72 B at D = 256, sm_80; build/attn/ptxas_attn_combine.log).
template <int32_t kCols>
__global__ void __launch_bounds__(kCombineWarps* pth_attn::kWarpLanes, 1)
    attn_combine_kernel(const CombineParams c) {
  constexpr int32_t kLanes = pth_attn::kWarpLanes;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kCombineWarps + threadIdx.x / kLanes;
  const int64_t token = row / c.num_q_heads;
  const int32_t head = static_cast<int32_t>(row % c.num_q_heads);
  if (token >= c.num_tokens || token >= c.query_start_loc[c.num_reqs]) return;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kLanes;
  const pth_attn::PartialsView& v = c.v;
  const int32_t S = c.num_splits;
  float m = -INFINITY;
  for (int32_t s0 = 0; s0 < S; s0 += kBatch) {
    float x[kBatch];
#pragma unroll
    for (int32_t b = 0; b < kBatch; ++b) {
      x[b] = s0 + b < S ? v.lse(s0 + b, token, head) : -INFINITY;
    }
#pragma unroll
    for (int32_t b = 0; b < kBatch; ++b) m = x[b] > m ? x[b] : m;
  }
  float num[kCols] = {};
  float den = 0.0f;
  if (m > -INFINITY) {
    for (int32_t s0 = 0; s0 < S; s0 += kBatch) {
      float x[kBatch];
      float o[kBatch][kCols];
#pragma unroll
      for (int32_t b = 0; b < kBatch; ++b) {
        x[b] = s0 + b < S ? v.lse(s0 + b, token, head) : -INFINITY;
      }
#pragma unroll
      for (int32_t b = 0; b < kBatch; ++b) {
        // One row base per split; the lane's columns are constant offsets
        // from it (an int64 address per element would need 2 registers each).
        const float* ob =
            reinterpret_cast<const float*>(
                v.ws_o + static_cast<int64_t>(s0 + b) * v.s_o_split +
                token * v.s_o_token + static_cast<int64_t>(head) * v.s_o_head) +
            lane;
#pragma unroll
        for (int32_t j = 0; j < kCols; ++j) {
          // Empty splits are skipped BEFORE their o is read.
          o[b][j] = x[b] > -INFINITY ? ob[kLanes * j] : 0.0f;
        }
      }
#pragma unroll
      for (int32_t b = 0; b < kBatch; ++b) {
        if (!(x[b] > -INFINITY)) continue;
        const float w = exp2f(x[b] - m);
        den += w;
#pragma unroll
        for (int32_t j = 0; j < kCols; ++j) num[j] = fmaf(w, o[b][j], num[j]);
      }
    }
  }
  char* out_row =
      c.out + token * c.s_out_token + static_cast<int64_t>(head) * c.s_out_head;
  const int64_t elem = c.out_dtype == PTH_ATTN_F32 ? 4 : 2;
#pragma unroll
  for (int32_t j = 0; j < kCols; ++j) {
    pth_attn::store_out(
        out_row + static_cast<int64_t>(lane + kLanes * j) * elem, c.out_dtype,
        m > -INFINITY ? num[j] / den : 0.0f);
  }
}

}  // namespace

namespace pth_attn {

int32_t launch_combine(const PthAttnCombineArgs& a, cudaStream_t s) {
  clear_launch_error();
  PTH_ATTN_REQUIRE(
      a.out_dtype == PTH_ATTN_F32 || a.out_dtype == PTH_ATTN_F16 ||
          a.out_dtype == PTH_ATTN_BF16,
      "out_dtype must be PTH_ATTN_F32, PTH_ATTN_F16 or PTH_ATTN_BF16");
  PTH_ATTN_REQUIRE(is_head_size(a.head_size),
                   "head_size must be 64, 128 or 256");
  PTH_ATTN_REQUIRE(
      a.num_splits >= 1 && a.num_reqs >= 1 && a.num_tokens >= 1 &&
          a.num_q_heads >= 1,
      "num_splits, num_reqs, num_tokens, num_q_heads must be >= 1");
  PTH_ATTN_REQUIRE(a.ws_o && a.ws_lse && a.out && a.query_start_loc,
                   "ws_o, ws_lse, out and query_start_loc must be non-NULL");
  CombineParams c{};
  c.v.ws_o = reinterpret_cast<const char*>(a.ws_o);
  c.v.s_o_split = a.ws_o_stride_split;
  c.v.s_o_token = a.ws_o_stride_token;
  c.v.s_o_head = a.ws_o_stride_head;
  c.v.ws_lse = reinterpret_cast<const char*>(a.ws_lse);
  c.v.s_l_split = a.ws_lse_stride_split;
  c.v.s_l_token = a.ws_lse_stride_token;
  c.v.s_l_head = a.ws_lse_stride_head;
  c.out = static_cast<char*>(a.out);
  c.s_out_token = a.out_stride_token;
  c.s_out_head = a.out_stride_head;
  c.query_start_loc = a.query_start_loc;
  c.num_reqs = a.num_reqs;
  c.num_tokens = a.num_tokens;
  c.num_q_heads = a.num_q_heads;
  c.num_splits = a.num_splits;
  c.out_dtype = a.out_dtype;
  const int64_t rows = static_cast<int64_t>(a.num_tokens) * a.num_q_heads;
  const unsigned blocks =
      static_cast<unsigned>((rows + kCombineWarps - 1) / kCombineWarps);
  const unsigned threads = kCombineWarps * kWarpLanes;
  constexpr int32_t kLanes = kWarpLanes;
  switch (a.head_size) {
    case 64:
      attn_combine_kernel<64 / kLanes><<<blocks, threads, 0, s>>>(c);
      break;
    case 128:
      attn_combine_kernel<128 / kLanes><<<blocks, threads, 0, s>>>(c);
      break;
    default:
      attn_combine_kernel<256 / kLanes><<<blocks, threads, 0, s>>>(c);
      break;
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return launch_fail(
        PTH_ATTN_ECUDA,
        std::string("attn_combine_kernel launch: ") + cudaGetErrorString(err));
  }
  return PTH_ATTN_OK;
}

}  // namespace pth_attn
