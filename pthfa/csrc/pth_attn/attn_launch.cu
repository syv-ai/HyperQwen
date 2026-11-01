// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_launch.cu: launch_attention of attn_launch.h, and the last-error slot
// of the launchers.
//
// Responsibilities: check the arguments (every check names the field; the
// shared checks are the reference library's, ref_capi.cu build_attn), turn
// the struct into KernelParams (byte strides copied as given, cp.async widths
// and divisors built here), pick the instantiation and launch. Nothing is
// allocated and nothing is read back from the device.

#include "attn_launch.h"

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include "attn_decode.cuh"

namespace {

using pth_attn::AttnLaunch;
using pth_attn::FastDivmod;
using pth_attn::KernelParams;
using pth_attn::StreamGeometry;

thread_local std::string g_last_error;

// fp32(log2 e): scores in log2 units (as reference_kernel.cu kRefLog2e).
constexpr float kLog2e = 1.44269504088896340736f;
// q, out and the workspace move in 16-B vectors.
constexpr int64_t kVecAlign = 16;
// Logical dims of a cache stream view with free strides: [NB, BS, Hkv].
constexpr int32_t kStreamDims = 3;
// The default loader takes the whole cell wherever it applies.
constexpr bool kDefaultCell = true;

// Fills a stream's geometry and its cp.async width (the largest of 16/8/4
// dividing base, used strides and row bytes); width 0 = not even 4-B aligned.
StreamGeometry stream_from(const PthAttnStream& s, int32_t row_bytes,
                           const PthAttnArgs& a) {
  StreamGeometry g{};
  g.base = static_cast<const char*>(s.data);
  g.s_block = s.stride_block;
  g.s_token = s.stride_slot;
  g.s_head = s.stride_head;
  g.row_bytes = row_bytes;
  const int64_t sizes[kStreamDims] = {a.num_blocks, a.block_size,
                                      a.num_kv_heads};
  const int64_t strides[kStreamDims] = {s.stride_block, s.stride_slot,
                                        s.stride_head};
  g.width = pth_attn::stream_width(reinterpret_cast<uint64_t>(s.data), sizes,
                                   strides, kStreamDims, row_bytes);
  return g;
}

bool aligned(const void* p, int64_t a) {
  return reinterpret_cast<uintptr_t>(p) % static_cast<uintptr_t>(a) == 0;
}

int32_t build(const PthAttnArgs& a, int32_t loader, AttnLaunch* L) {
  PTH_ATTN_REQUIRE(
      loader >= pth_attn::kLoaderDefault && loader <= pth_attn::kLoaderCell,
      "options.loader must be 0 (default), 1 (separate) or 2 (cell)");
  PTH_ATTN_REQUIRE(a.q_dtype == PTH_ATTN_F16 || a.q_dtype == PTH_ATTN_BF16,
                   "q_dtype must be PTH_ATTN_F16 or PTH_ATTN_BF16");
  PTH_ATTN_REQUIRE(
      a.out_dtype == PTH_ATTN_F32 || a.out_dtype == PTH_ATTN_F16 ||
          a.out_dtype == PTH_ATTN_BF16,
      "out_dtype must be PTH_ATTN_F32, PTH_ATTN_F16 or PTH_ATTN_BF16");
  PTH_ATTN_REQUIRE(pth_attn::is_head_size(a.head_size),
                   "head_size must be 64, 128 or 256");
  PTH_ATTN_REQUIRE(a.num_kv_heads >= 1 && a.num_q_heads >= a.num_kv_heads &&
                       a.num_q_heads % a.num_kv_heads == 0,
                   "num_q_heads must be a positive multiple of num_kv_heads");
  PTH_ATTN_REQUIRE(a.block_size >= 1, "block_size must be >= 1");
  PTH_ATTN_REQUIRE(a.num_blocks >= 1, "num_blocks must be >= 1");
  PTH_ATTN_REQUIRE(a.num_reqs >= 1, "num_reqs must be >= 1");
  PTH_ATTN_REQUIRE(a.num_tokens >= 1, "num_tokens must be >= 1");
  PTH_ATTN_REQUIRE(a.bq >= 1, "bq must be >= 1");
  PTH_ATTN_REQUIRE(a.num_splits >= 1, "num_splits must be >= 1");
  PTH_ATTN_REQUIRE(a.num_q_tiles >= 0, "num_q_tiles must be >= 0");
  PTH_ATTN_REQUIRE(std::isfinite(a.softmax_scale) && a.softmax_scale > 0.0f,
                   "softmax_scale must be finite and > 0");
  PTH_ATTN_REQUIRE(a.window >= 0, "window must be >= 0 (0 = none)");
  PTH_ATTN_REQUIRE(a.causal == 0 || a.causal == 1, "causal must be 0 or 1");
  // The windowed kernels exist at D = 128 only.
  const bool windowed = a.window > 0 || a.causal == 0;
  PTH_ATTN_REQUIRE(!windowed || a.head_size == 128,
                   "sliding-window and non-causal attention need head size "
                   "D = 128");
  PTH_ATTN_REQUIRE(a.q && a.k_cache.data && a.v_cache.data && a.k_scale.data &&
                       a.v_scale.data && a.block_table && a.seq_lens &&
                       a.query_start_loc,
                   "q, the four cache streams, block_table, seq_lens and "
                   "query_start_loc must be non-NULL");
  const int32_t group = a.num_q_heads / a.num_kv_heads;
  const int32_t wm = pth_attn::attn_wm_for_rows(a.bq * group);
  PTH_ATTN_REQUIRE(wm != 0, "bq * G = " + std::to_string(a.bq * group) +
                                " rows exceed one 64-row tile");
  // 16-B vectors: q rows, out rows (S == 1) and the fp32 partials (S > 1).
  PTH_ATTN_REQUIRE(aligned(a.q, kVecAlign) &&
                       a.q_stride_token % kVecAlign == 0 &&
                       a.q_stride_head % kVecAlign == 0,
                   "q base and byte strides must be 16-B aligned");
  if (a.num_splits == 1) {
    PTH_ATTN_REQUIRE(a.out != nullptr,
                     "out must be non-NULL when num_splits == 1");
    PTH_ATTN_REQUIRE(aligned(a.out, kVecAlign) &&
                         a.out_stride_token % kVecAlign == 0 &&
                         a.out_stride_head % kVecAlign == 0,
                     "out base and byte strides must be 16-B aligned");
  } else {
    PTH_ATTN_REQUIRE(a.ws_o != nullptr && a.ws_lse != nullptr,
                     "ws_o and ws_lse must be non-NULL when num_splits > 1");
    PTH_ATTN_REQUIRE(aligned(a.ws_o, kVecAlign) &&
                         a.ws_o_stride_split % kVecAlign == 0 &&
                         a.ws_o_stride_token % kVecAlign == 0 &&
                         a.ws_o_stride_head % kVecAlign == 0,
                     "ws_o base and byte strides must be 16-B aligned");
  }

  KernelParams& k = L->ap.k;
  k = KernelParams{};
  k.q = a.q;
  k.out = a.out;
  k.s_q_token = a.q_stride_token;
  k.s_q_head = a.q_stride_head;
  k.s_out_token = a.out_stride_token;
  k.s_out_head = a.out_stride_head;
  k.ws_o = a.ws_o;
  k.ws_lse = a.ws_lse;
  k.s_wso_split = a.ws_o_stride_split;
  k.s_wso_token = a.ws_o_stride_token;
  k.s_wso_head = a.ws_o_stride_head;
  k.s_wsl_split = a.ws_lse_stride_split;
  k.s_wsl_token = a.ws_lse_stride_token;
  k.s_wsl_head = a.ws_lse_stride_head;
  k.stream[pth_attn::kStreamK] = stream_from(a.k_cache, a.head_size, a);
  k.stream[pth_attn::kStreamV] = stream_from(a.v_cache, a.head_size, a);
  k.stream[pth_attn::kStreamKScale] =
      stream_from(a.k_scale, static_cast<int32_t>(sizeof(float)), a);
  k.stream[pth_attn::kStreamVScale] =
      stream_from(a.v_scale, static_cast<int32_t>(sizeof(float)), a);
  for (int s = 0; s < pth_attn::kNumStreams; ++s) {
    PTH_ATTN_REQUIRE(k.stream[s].width != 0,
                     "cache stream " + std::to_string(s) +
                         " (K, V, k_scale, v_scale) is not 4-B aligned");
  }
  k.block_table = a.block_table;
  k.bt_row_stride = a.block_table_row_stride;
  k.seq_lens = a.seq_lens;
  k.query_start_loc = a.query_start_loc;
  k.block_size = FastDivmod::make(static_cast<uint32_t>(a.block_size));
  k.group = FastDivmod::make(static_cast<uint32_t>(group));
  k.bq_div = FastDivmod::make(static_cast<uint32_t>(a.bq));
  k.num_reqs = a.num_reqs;
  k.num_tokens = a.num_tokens;
  k.num_q_heads = a.num_q_heads;
  k.num_kv_heads = a.num_kv_heads;
  k.group_size = group;
  k.head_size = a.head_size;
  k.wm = wm;
  k.bq = a.bq;
  k.num_splits = a.num_splits;
  k.max_query_len = 0;  // not used: q_len is read on the device
  k.softmax_scale = a.softmax_scale;
  k.score_scale_log2 = a.softmax_scale * kLog2e;
  L->ap.out_dtype = a.out_dtype;
  // Whole-cell loader: only where it is instantiated (D = 256) and the four
  // views are one inline cell (loader.cuh cell_layout_width > 0).
  const int32_t cell_width =
      a.head_size == 256 ? pth_attn::cell_layout_width<256>(k) : 0;
  PTH_ATTN_REQUIRE(
      loader != pth_attn::kLoaderCell || cell_width > 0,
      "options.loader = cell needs D = 256 and the inline cell layout");
  L->cell =
      loader == pth_attn::kLoaderCell ||
      (loader == pth_attn::kLoaderDefault && kDefaultCell && cell_width > 0);
  L->ap.cell_width = cell_width;
  L->ap.window = a.window;
  L->ap.causal = a.causal;
  L->windowed = windowed;
  L->wm = wm;
  L->rows = a.bq * group;
  L->q_dtype = a.q_dtype;
  const int32_t tiles =
      a.num_q_tiles > 0 ? a.num_q_tiles : a.num_tokens / a.bq + a.num_reqs;
  L->grid =
      dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(a.num_kv_heads),
           static_cast<unsigned>(a.num_splits));
  return PTH_ATTN_OK;
}

}  // namespace

namespace pth_attn {

const char* last_launch_error() { return g_last_error.c_str(); }

int32_t launch_fail(int32_t code, const std::string& msg) {
  g_last_error = msg;
  return code;
}

void clear_launch_error() { g_last_error.clear(); }

int32_t launch_attention(const PthAttnArgs& args, int32_t loader,
                         cudaStream_t stream) {
  clear_launch_error();
  AttnLaunch L{};
  const int32_t rc = build(args, loader, &L);
  if (rc != PTH_ATTN_OK) return rc;
  cudaError_t err = cudaErrorInvalidValue;
  switch (args.head_size) {
    case 64:
      err = launch_attn_d64(L, stream);
      break;
    case 128:
      err = launch_attn_d128(L, stream);
      break;
    default:
      err = launch_attn_d256(L, stream);
      break;
  }
  if (err != cudaSuccess) {
    return launch_fail(PTH_ATTN_ECUDA, std::string("attention launch (D=") +
                                           std::to_string(args.head_size) +
                                           ", wm=" + std::to_string(L.wm) +
                                           ", rows=" + std::to_string(L.rows) +
                                           "): " + cudaGetErrorString(err));
  }
  return PTH_ATTN_OK;
}

}  // namespace pth_attn
