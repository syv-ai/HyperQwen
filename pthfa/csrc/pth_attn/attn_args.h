// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_args.h: the launch arguments of the attention kernel and the split
// combine (attn_launch.h). Plain C structs, so that the C ABI of the test
// libraries (attn_capi.h, ref_capi.h) and their ctypes mirrors
// (python/pth_testing/ref_capi.py) pass the very structs the op fills.
//
// Conventions:
// - Every stride is in BYTES, taken from the tensor view,
//   except block_table_row_stride, which is in int32 ELEMENTS (as
//   block_table.stride(0)).
// - Pointers are device pointers (host pointers for the reference library's
//   *_host entry points). Nothing is allocated, nothing is copied to the
//   host, and a launch does not synchronise.
// - abi_version and struct_bytes are checked at the C ABI only (a stale
//   ctypes mirror fails loudly); a change of a struct bumps
//   PTH_ATTN_ABI_VERSION.
//
// Semantics of an attention launch:
// - Bottom-right aligned GQA attention over the int8_per_token_head cache:
//   query token i of request r (q_len = qsl[r+1] - qsl[r], L = seq_lens[r],
//   both read on the device) sits at KV position p = L - q_len + i. Causal
//   with window 0 (the default kernels) it sees KV positions j <= p. Query
//   head h reads KV head h / G, G = num_q_heads / num_kv_heads.
// - Sliding window and non-causal (`window`, `causal` below): it
//   sees [lo, hi] with lo = window > 0 ? max(0, p - window + 1) : 0 and
//   hi = causal ? p : (window > 0 ? min(L - 1, p + window - 1) : L - 1)
//   (vLLM's Triton window_size = (window - 1, 0)). window > 0 or !causal
//   selects the windowed kernels, which exist for head_size 128 only.
// - Grid = (num_q_tiles, num_kv_heads, num_splits), the planner's grid: q-tile
//   x holds up to `bq` query tokens of one request (cache_geometry row_tile);
//   its KV range [range_begin, range_end) (range_begin = 0 for the causal
//   kernels) is cut into 64-token tiles from the tile holding range_begin,
//   and split s takes tiles [floor(s*T/S), floor((s+1)*T/S)) of those T
//   (planner.split_tile_range).
// - num_splits == 1: writes out[token, head, :] for every query token of
//   every request (out_dtype).
// - num_splits > 1: writes, for every query token, head and split s,
//   ws_o[s, token, head, :] = that split's softmax-normalised output (fp32)
//   and ws_lse[s, token, head] = m + log2(l) in log2 units of the scaled
//   scores (an empty split writes ws_o = 0, ws_lse = -inf). `out` is not
//   touched; the combine runs next.
// - Rows that belong to no request (tokens >= qsl[num_reqs]) are never
//   written, in out or in the workspace.
//
// Semantics of the combine:
//   M = max_s lse_s over splits with lse_s > -inf (fixed order s = 0..S-1);
//   w_s = exp2(lse_s - M), skipping -inf splits BEFORE touching ws_o;
//   out = (sum_s w_s * o_s) / (sum_s w_s), summed in order s = 0..S-1.
//   Deterministic for fixed inputs. Rows >= qsl[num_reqs] are not written.

#ifndef PTH_ATTN_ATTN_ARGS_H_
#define PTH_ATTN_ATTN_ARGS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PTH_ATTN_ABI_VERSION 2

// Element types of q and out.
typedef enum PthAttnDtype {
  PTH_ATTN_F32 = 0,   // out only (test builds)
  PTH_ATTN_F16 = 1,   // q or out
  PTH_ATTN_BF16 = 2,  // q or out
} PthAttnDtype;

typedef enum PthAttnStatus {
  PTH_ATTN_OK = 0,
  PTH_ATTN_EINVAL = -1,   // an argument check failed (see the last error)
  PTH_ATTN_ECUDA = -2,    // a CUDA runtime call or the launch failed
  PTH_ATTN_ENOCUDA = -3,  // a CUDA entry point of the host-only reference
} PthAttnStatus;

// One cache stream: a view of logical shape [num_blocks, block_size,
// num_kv_heads, row] (row = head_size int8 for K/V, one fp32 for the scales)
// with the innermost row contiguous and arbitrary BYTE strides outside it.
typedef struct PthAttnStream {
  const void* data;      // view.data_ptr()
  int64_t stride_block;  // bytes between kernel blocks (pages, padding incl.)
  int64_t stride_slot;   // bytes between slots of one block
  int64_t stride_head;   // bytes between KV heads of one slot
} PthAttnStream;

typedef struct PthAttnArgs {
  int32_t abi_version;   // = PTH_ATTN_ABI_VERSION (C ABI callers)
  int32_t struct_bytes;  // = sizeof(PthAttnArgs) as compiled by the caller

  // Queries [num_tokens, num_q_heads, head_size], last dim contiguous.
  const void* q;
  int32_t q_dtype;         // PTH_ATTN_F16 or PTH_ATTN_BF16
  int64_t q_stride_token;  // bytes
  int64_t q_stride_head;   // bytes

  // Output [num_tokens, num_q_heads, head_size] (num_splits == 1 only).
  void* out;
  int32_t out_dtype;         // PTH_ATTN_F32, PTH_ATTN_F16 or PTH_ATTN_BF16
  int64_t out_stride_token;  // bytes
  int64_t out_stride_head;   // bytes

  // Cache streams and their shared logical shape.
  PthAttnStream k_cache;  // int8 rows of head_size bytes
  PthAttnStream v_cache;  // int8 rows of head_size bytes
  PthAttnStream k_scale;  // one fp32 per (block, slot, head)
  PthAttnStream v_scale;  // one fp32 per (block, slot, head)
  int32_t num_blocks;     // kernel blocks in the views (bounds of block ids)
  int32_t block_size;     // tokens per kernel block, >= 1
  int32_t num_kv_heads;
  int32_t head_size;  // D: 64, 128 or 256

  // Paging and per-request lengths (device-resident int32, read every call).
  const int32_t* block_table;      // [>= num_reqs, *] kernel-block ids
  int64_t block_table_row_stride;  // ELEMENTS between rows
  const int32_t* seq_lens;         // [num_reqs] KV length incl. new tokens
  const int32_t* query_start_loc;  // [num_reqs + 1], starts at 0

  // Host integers (the launch descriptor).
  int32_t num_reqs;
  int32_t num_tokens;   // rows of q/out/workspace (may exceed qsl[num_reqs])
  int32_t num_q_heads;  // multiple of num_kv_heads
  int32_t bq;           // query tokens per row tile, >= 1
  int32_t num_q_tiles;  // grid.x; 0 = num_tokens / bq + num_reqs
  int32_t num_splits;   // S >= 1
  float softmax_scale;  // multiplies q . k before the softmax, > 0

  // Split partials (num_splits > 1 only; may be NULL otherwise).
  float* ws_o;                  // [>= S, >= num_tokens, num_q_heads, head_size]
  int64_t ws_o_stride_split;    // bytes
  int64_t ws_o_stride_token;    // bytes
  int64_t ws_o_stride_head;     // bytes
  float* ws_lse;                // [>= S, >= num_tokens, num_q_heads]
  int64_t ws_lse_stride_split;  // bytes
  int64_t ws_lse_stride_token;  // bytes
  int64_t ws_lse_stride_head;   // bytes

  // Visible range. window: 0 = none, else the sliding window in
  // tokens (>= 1). causal: 1 = bottom-right causal, 0 = non-causal. The
  // default (window 0, causal 1) is the causal attention above.
  int32_t window;
  int32_t causal;
} PthAttnArgs;

typedef struct PthAttnCombineArgs {
  int32_t abi_version;   // = PTH_ATTN_ABI_VERSION (C ABI callers)
  int32_t struct_bytes;  // = sizeof(PthAttnCombineArgs) as compiled by caller

  const float* ws_o;            // as in PthAttnArgs
  int64_t ws_o_stride_split;    // bytes
  int64_t ws_o_stride_token;    // bytes
  int64_t ws_o_stride_head;     // bytes
  const float* ws_lse;          // as in PthAttnArgs
  int64_t ws_lse_stride_split;  // bytes
  int64_t ws_lse_stride_token;  // bytes
  int64_t ws_lse_stride_head;   // bytes

  void* out;
  int32_t out_dtype;         // PTH_ATTN_F32, PTH_ATTN_F16 or PTH_ATTN_BF16
  int64_t out_stride_token;  // bytes
  int64_t out_stride_head;   // bytes

  const int32_t* query_start_loc;  // [num_reqs + 1]; rows >= qsl[num_reqs]
                                   // are skipped (read on the device)
  int32_t num_reqs;
  int32_t num_tokens;
  int32_t num_q_heads;
  int32_t head_size;
  int32_t num_splits;  // S >= 1
} PthAttnCombineArgs;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PTH_ATTN_ATTN_ARGS_H_
