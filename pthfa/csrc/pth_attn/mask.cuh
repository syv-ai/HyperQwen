// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// mask.cuh: bottom-right causal masking of one warp's score fragments
// (cache_geometry.cuh kv_limit).
//
// Row r of a CTA row tile is query token tok of its request; it may see KV
// positions j <= limit(r) = seq_len - q_len + tok (inclusive). Padding rows
// (past the tile's tokens) get limit -1: they see nothing and are never
// stored. A thread holds rows g and g + 8 of its warp, so two limits.
//
// Masking is applied only to sub-tiles that reach past the CTA's smallest
// limit (kv_min_limit, the tile's first token): every earlier position is
// visible to every real row. Tokens at or past the CTA's KV range end lie
// past every row's limit, so the same test also hides the zero-filled tail
// of the last tile.
//
// Windowed kernels (attn_kernel.cuh kWindow): row r sees
// [lo(r), hi(r)] (cache_geometry.cuh kv_window_lo/hi); padding rows get
// lo = 0, hi = -1. A sub-tile needs the mask iff it starts below the CTA's
// largest lo (the tile's last token) or ends past its smallest hi (the tile's
// first token); the first test also hides the zero-filled tokens below the
// range begin. Two compares per score instead of one; the causal kernels do
// not instantiate any of it.

#pragma once

#include <cmath>
#include <cstdint>

#include "tile_traits.cuh"

namespace pth_attn {

// True if the sub-tile [tok0, tok0 + sub) needs the mask for a CTA whose
// smallest row limit is min_limit (positions <= min_limit are visible to all
// real rows). int64 so tok0 + sub cannot overflow near 2^31.
PTH_HD bool subtile_needs_mask(int32_t tok0, int32_t sub, int32_t min_limit) {
  return static_cast<int64_t>(tok0) + sub - 1 > min_limit;
}

// The windowed counterpart: positions in [max_lo, min_hi] are visible to all
// real rows.
PTH_HD bool subtile_needs_window_mask(int32_t tok0, int32_t sub, int32_t max_lo,
                                      int32_t min_hi) {
  return tok0 < max_lo || static_cast<int64_t>(tok0) + sub - 1 > min_hi;
}

// True if KV position `pos` is outside the inclusive window [lo, hi].
PTH_HD bool outside_window(int32_t pos, int32_t lo, int32_t hi) {
  return pos < lo || pos > hi;
}

#if defined(__CUDACC__)

// s[n][2h + i] is the score of row (g + 8h) and KV position
// tok0 + 8n + 2q + i; set it to -inf where the position exceeds the row's
// limit.
template <int SUB>
__device__ __forceinline__ void apply_causal_mask(float (&s)[SUB / kMmaN][4],
                                                  int32_t tok0,
                                                  const int32_t (&limit)[2],
                                                  int lane) {
  const int32_t base = tok0 + 2 * (lane & 3);
  #pragma unroll
  for (int n = 0; n < SUB / kMmaN; ++n) {
  #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int32_t pos = base + n * kMmaN;
      if (pos > limit[h]) s[n][2 * h] = -INFINITY;
      if (pos + 1 > limit[h]) s[n][2 * h + 1] = -INFINITY;
    }
  }
}

// The windowed counterpart of apply_causal_mask: -inf where the position is
// outside the row's window [lo[h], hi[h]].
template <int SUB>
__device__ __forceinline__ void apply_window_mask(float (&s)[SUB / kMmaN][4],
                                                  int32_t tok0,
                                                  const int32_t (&lo)[2],
                                                  const int32_t (&hi)[2],
                                                  int lane) {
  const int32_t base = tok0 + 2 * (lane & 3);
  #pragma unroll
  for (int n = 0; n < SUB / kMmaN; ++n) {
  #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int32_t pos = base + n * kMmaN;
      if (outside_window(pos, lo[h], hi[h])) s[n][2 * h] = -INFINITY;
      if (outside_window(pos + 1, lo[h], hi[h])) s[n][2 * h + 1] = -INFINITY;
    }
  }
}

#endif  // __CUDACC__

}  // namespace pth_attn
