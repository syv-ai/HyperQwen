// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Host-side interface of the pth_flash_attn op library.
//
// The op schema is frozen by plan section 4.1; the C++ signature below is its
// one-to-one translation under the libtorch stable ABI (Tensor! -> Tensor&,
// int -> int64_t, float -> double). launch.cu validates every argument, then
// launches the attention kernel and, for num_splits > 1, the combine kernel on
// the current CUDA stream; it allocates nothing and reads no device data.

#pragma once

#include <torch/csrc/stable/tensor.h>

#include <cstdint>

namespace pth_attn {

// pth_flash_attn_fwd: paged int8_per_token_head attention forward.
//
// out             [T, Hq, D] bf16|fp16, written in place.
// q               [T, Hq, D] same dtype as out.
// k_cache/v_cache [NB, BS, Hkv, D] int8 views with arbitrary strides.
// k_scale/v_scale [NB, BS, Hkv] float32 views with arbitrary strides.
// block_table     [num_reqs_or_more, max_blocks] int32 kernel-block ids.
// seq_lens        [num_reqs] int32, KV length including the new tokens.
// query_start_loc [num_reqs + 1] int32, rebased to this slice.
// max_query_len   host upper bound of q_len (the device value may be smaller).
// wm              warps along M, one of {1, 2, 4} (planner output).
// num_splits      KV splits S >= 1; S > 1 needs the workspace below.
// softmax_scale   scale applied to Q.K before the softmax.
// ws_o            [S', T', Hq, D] float32 partials, S' >= S, T' >= T.
// ws_lse          [S', T', Hq] float32 log2-sum-exp of each partial.
void pth_flash_attn_fwd(
    torch::stable::Tensor& out, const torch::stable::Tensor& q,
    const torch::stable::Tensor& k_cache, const torch::stable::Tensor& v_cache,
    const torch::stable::Tensor& k_scale, const torch::stable::Tensor& v_scale,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens,
    const torch::stable::Tensor& query_start_loc, int64_t max_query_len,
    int64_t wm, int64_t num_splits, double softmax_scale,
    torch::stable::Tensor& ws_o, torch::stable::Tensor& ws_lse);

namespace detail {

// Byte extent of a strided view: one past the last byte any element of the
// view can touch, measured from the view's data pointer. It is
//   element_size + sum_i (size_i - 1) * stride_i * element_size
// for a view with every size_i >= 1, and 0 for an empty view.
//
// The kernel forms every address as base + sum(index_i * byte_stride_i) in
// int64 (plan rule 5). If this extent fits in int64, no partial sum of that
// expression can overflow either, because every term is non-negative and at
// most its counterpart here. That is what "int64-safe strides" means.
//
// Returns false (and leaves *extent unspecified) on overflow or on a negative
// size/stride, which the address math above does not support. Kept in this
// header, dependency-free, so a g++ unit test can drive the overflow cases:
// no real tensor that small CI machines can allocate reaches them.
inline bool view_extent_bytes(const int64_t* sizes, const int64_t* strides,
                              int64_t ndim, int64_t element_size,
                              int64_t* extent) {
  if (element_size <= 0) return false;
  for (int64_t i = 0; i < ndim; ++i) {
    if (sizes[i] < 0 || strides[i] < 0) return false;
    if (sizes[i] == 0) {
      *extent = 0;
      return true;
    }
  }
  int64_t total = element_size;
  for (int64_t i = 0; i < ndim; ++i) {
    int64_t term = 0;
    if (__builtin_mul_overflow(sizes[i] - 1, strides[i], &term) ||
        __builtin_mul_overflow(term, element_size, &term) ||
        __builtin_add_overflow(total, term, &total)) {
      return false;
    }
  }
  *extent = total;
  return true;
}

}  // namespace detail
}  // namespace pth_attn
