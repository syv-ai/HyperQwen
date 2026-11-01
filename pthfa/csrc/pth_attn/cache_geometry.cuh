// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// cache_geometry.cuh: where every byte the kernel reads or writes lives.
//
// This header answers four questions, each with integer arithmetic only and the
// same code on the host (g++ unit tests) and on the device:
//
//   1. Paging.  KV token `pos` of request `req` lives in block
//      block_table[req][pos / BS] at slot pos % BS. The division is a
//      FastDivmod (multiply-high and shift) because BS is a run-time value
//      that need not be a power of two (for example BS = 816).
//   2. Bytes.   The byte address of K, V, k_scale, v_scale for (page, slot,
//      KV head) is base + page*s_block + slot*s_token + head*s_head, in int64,
//      with every stride read from the tensor view the op receives and never
//      derived from a shape (vLLM #49716). One StreamGeometry
//      per stream, so NHD, HND, padded pages, the inline [K|ks|V|vs] cell and
//      #56164's decoupled layout are all the same code.
//   3. Copy width. Per stream, the widest cp.async size w in {16, 8, 4} that
//      divides the base address, every byte stride that is ever multiplied by
//      a non-zero index, and the row bytes.
//   4. Which rows and tokens a CTA owns. Row tile -> request and query tokens
//      (BQ packing), row -> (query token, query head) (token-major),
//      the bottom-right causal limit of every row (or, for the windowed
//      kernels, its visible window [lo, hi]), the CTA's KV range,
//      the tile-balanced split partition (equal to planner.split_tile_range),
//      and which KV tiles need a mask at all.
//
// Tests: tests/cpp/test_geometry.cpp (g++, exhaustive/property tests, stride
// fuzz with negative controls; also compiled by nvcc for sm_80/86/89) and
// tests/cpu/test_geometry_vs_reference.py (the same functions, loaded through
// ctypes, against pth_testing.reference and the planner).
//
// Header-only, depends on <cstdint> only, so it builds without CUDA or torch.

#pragma once

#include <cstdint>

// One definition shared verbatim by every csrc/pth_attn header: host+device
// and force-inlined under nvcc; plain `inline` under g++ so
// non-template functions can sit in a header included by several TUs (ODR).
#ifndef PTH_HD
  #if defined(__CUDACC__)
    #define PTH_HD __host__ __device__ __forceinline__
  #else
    #define PTH_HD inline
  #endif
#endif

namespace pth_attn {

// ------------------------------------------------------------------ constants
// Rows of one m16n8k16 MMA slice; a warp owns 16 query rows, so a CTA with
// `wm` warps along M holds 16*wm rows.
constexpr int32_t kGeomRowsPerWarp = 16;
// cp.async copy sizes, widest first (PTX ISA cp.async: 4, 8 or 16 bytes).
constexpr int32_t kGeomWidth16 = 16;
constexpr int32_t kGeomWidth8 = 8;
constexpr int32_t kGeomWidth4 = 4;
// FastDivmod domain: exact for every numerator n < 2^31 and every
// divisor 1 <= d <= 2^31. Positions are int32 sequence positions (< 2^31), so
// the domain covers every possible input; the proof is at FastDivmod.
constexpr uint32_t kFastDivmodLog2Domain = 31;
constexpr uint32_t kFastDivmodMaxNumerator =
    (uint32_t{1} << kFastDivmodLog2Domain) - 1u;
constexpr uint32_t kFastDivmodMaxDivisor = uint32_t{1} << kFastDivmodLog2Domain;
// Bits of the hardware multiply-high: umulhi returns bits 32..63 of a 64-bit
// product.
constexpr uint32_t kMulHiBits = 32;
// A stream view's outer dimensions with free byte strides: [NB, BS, Hkv].
constexpr int32_t kGeomOuterDims = 3;

// High 32 bits of the 64-bit product a*b (one IMAD.HI on the device).
PTH_HD uint32_t umulhi32(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
  return __umulhi(a, b);
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> kMulHiBits);
#endif
}

// ---------------------------------------------------------------- FastDivmod
// Division by a run-time divisor d as one multiply-high and one shift.
//
// Construction (Granlund & Montgomery 1994; the CUTLASS FastDivmod idea, no
// code copied): l = ceil(log2 d), p = 31 + l,
// m = ceil(2^p / d), shift = p - 32, and n / d = umulhi(n, m) >> shift.
//
// Proof of exactness for 0 <= n < 2^31, 2 <= d <= 2^31 (d = 1 is a separate,
// warp-uniform branch because p - 32 would be -1):
//   Write m*d = 2^p + e. Since m is the smallest integer >= 2^p/d,
//   0 <= e < d <= 2^l. Write n = q*d + r with 0 <= r < d. Then
//     n*m / 2^p = n/d + n*e/(d*2^p) = q + (r + n*e/2^p) / d.
//   n*e < 2^31 * 2^l = 2^p, so n*e/2^p < 1 and r + n*e/2^p < r + 1 <= d.
//   Hence floor(n*m / 2^p) = q. Finally umulhi(n, m) >> shift equals
//   floor(n*m / 2^32) >> (p - 32) = floor(n*m / 2^p) (nested floors of a
//   division by powers of two compose). m fits in 32 bits: for d a power of
//   two m = 2^31; otherwise d > 2^(l-1) gives m <= 2^p/(2^(l-1)+1) + 1 < 2^32.
// The g++ test checks the proof's premises (m < 2^32, 0 <= e < d,
// e*(2^31 - 1) < 2^p) for every d in [1, 2^16], every n < 2^20 for d up to
// 1056 and at powers of two, and the top of the domain for every d; and shows
// that one bit less of precision (p = 30 + l) is wrong (negative control).
struct FastDivmod {
  uint32_t divisor;     // d, 1 <= d <= 2^31 (0 marks an invalid object)
  uint32_t multiplier;  // ceil(2^p / d); unused when d == 1
  uint32_t shift;       // p - 32; unused when d == 1

  // Builds the divisor; d outside [1, 2^31] yields divisor 0 (valid() false).
  PTH_HD static FastDivmod make(uint32_t d) {
    FastDivmod f{0u, 0u, 0u};
    if (d == 0u || d > kFastDivmodMaxDivisor) return f;
    f.divisor = d;
    if (d == 1u) return f;
    uint32_t l = 0;  // ceil(log2 d), 1..31
    while ((uint64_t{1} << l) < d) ++l;
    const uint32_t p = kFastDivmodLog2Domain + l;  // 32..62: 2^p fits uint64
    f.multiplier = static_cast<uint32_t>(((uint64_t{1} << p) + (d - 1u)) / d);
    f.shift = p - kMulHiBits;
    return f;
  }

  PTH_HD bool valid() const { return divisor != 0u; }

  // n / d for 0 <= n <= kFastDivmodMaxNumerator.
  PTH_HD uint32_t div(uint32_t n) const {
    return divisor == 1u ? n : (umulhi32(n, multiplier) >> shift);
  }

  // q = n / d, r = n % d for 0 <= n <= kFastDivmodMaxNumerator.
  PTH_HD void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = div(n);
    r = n - q * divisor;
  }
};

// ------------------------------------------------------------------- streams
enum StreamId : int32_t {
  kStreamK = 0,
  kStreamV = 1,
  kStreamKScale = 2,
  kStreamVScale = 3,
  kNumStreams = 4,
};

// One cache stream (K, V, k_scale or v_scale) as the op receives it: a view of
// logical shape [NB, BS, Hkv, row] (row = D int8 for K/V, one f32 for the
// scales) with arbitrary byte strides over [NB, BS, Hkv].
struct StreamGeometry {
  const char* base;   // view.data_ptr(): the element [0, 0, 0, 0]
  int64_t s_block;    // bytes between kernel blocks (pages, padding included)
  int64_t s_token;    // bytes between slots of one block
  int64_t s_head;     // bytes between KV heads of one slot
  int32_t width;      // cp.async bytes per instruction: 16, 8 or 4
  int32_t row_bytes;  // contiguous bytes per (token, head): D, or 4 for scales

  // Byte offset of (page, slot, KV head) from base, in int64. Every product
  // is formed in int64 (a page id times an 848,640-B stride exceeds 2^31 at
  // page 2,531); the op checks that the view's whole byte extent fits int64
  // (ops.h view_extent_bytes), and every term here is non-negative and at
  // most its counterpart in that extent, so no partial sum overflows.
  PTH_HD int64_t offset(int64_t page, uint32_t slot, int32_t kv_head) const {
    return page * s_block + static_cast<int64_t>(slot) * s_token +
           static_cast<int64_t>(kv_head) * s_head;
  }

  PTH_HD const char* address(int64_t page, uint32_t slot,
                             int32_t kv_head) const {
    return base + offset(page, slot, kv_head);
  }
};

// The largest w in {16, 8, 4} dividing the base address, `row_bytes`, and the
// byte stride of every dimension whose size exceeds 1. A size-1 dimension's
// index is always 0, so its stride never reaches an address (torch may give
// such a dimension any stride); counting it would reject valid views. Returns
// 0 when even 4 does not divide (the op rejects such a stream; launch.cu's
// 4-byte check).
//
// Why all of them: copy k of a row goes to base + offsets + k*w, and the
// offsets range over every combination of indices, so w must divide each
// stride individually, not just one particular sum.
PTH_HD int32_t stream_width(uint64_t base_addr, const int64_t* sizes,
                            const int64_t* byte_strides, int32_t ndim,
                            int64_t row_bytes) {
  const int32_t widths[] = {kGeomWidth16, kGeomWidth8, kGeomWidth4};
  for (int32_t w : widths) {
    bool ok = base_addr % static_cast<uint64_t>(w) == 0u && row_bytes % w == 0;
    for (int32_t i = 0; ok && i < ndim; ++i) {
      if (sizes[i] > 1 && byte_strides[i] % w != 0) ok = false;
    }
    if (ok) return w;
  }
  return 0;
}

// Builds a StreamGeometry from a view's data pointer, sizes and element
// strides of [NB, BS, Hkv] (the innermost row must be contiguous: launch.cu
// checks stride 1). Strides are converted to bytes here and nowhere else.
PTH_HD StreamGeometry make_stream(const void* data, const int64_t* sizes3,
                                  const int64_t* elem_strides3,
                                  int64_t elem_bytes, int32_t row_bytes) {
  StreamGeometry s{};
  s.base = static_cast<const char*>(data);
  s.s_block = elem_strides3[0] * elem_bytes;
  s.s_token = elem_strides3[1] * elem_bytes;
  s.s_head = elem_strides3[2] * elem_bytes;
  const int64_t bytes[] = {s.s_block, s.s_token, s.s_head};
  s.row_bytes = row_bytes;
  s.width = stream_width(reinterpret_cast<uint64_t>(data), sizes3, bytes,
                         kGeomOuterDims, row_bytes);
  return s;
}

// ------------------------------------------------------------- kernel params
// Everything one launch reads, frozen as the kernel library's interface.
// Pointers are untyped bytes plus byte strides, so this header needs no CUDA or
// torch types; the kernel casts at the point of use.
struct KernelParams {
  // Queries and outputs: [T, Hq, D] with byte strides (last dim contiguous).
  const void* q;
  void* out;
  int64_t s_q_token, s_q_head;
  int64_t s_out_token, s_out_head;
  // Split partials (S > 1): ws_o [S', T', Hq, D] f32, ws_lse [S', T', Hq] f32.
  void* ws_o;
  void* ws_lse;
  int64_t s_wso_split, s_wso_token, s_wso_head;
  int64_t s_wsl_split, s_wsl_token, s_wsl_head;
  // The four cache streams (indexed by StreamId).
  StreamGeometry stream[kNumStreams];
  // Paging and per-request lengths, all read on the device every launch.
  const int32_t* block_table;      // [>= num_reqs, max_blocks] kernel-block ids
  int64_t bt_row_stride;           // elements between block_table rows
  const int32_t* seq_lens;         // [num_reqs] KV length incl. the new tokens
  const int32_t* query_start_loc;  // [num_reqs + 1], rebased to this slice
  // Divisors built on the host.
  FastDivmod block_size;  // BS (kernel block, tokens)
  FastDivmod group;       // G = Hq / Hkv (row -> (token, head))
  FastDivmod bq_div;      // BQ (q-tile -> request search)
  // Host integers (from the descriptor; never from a device length).
  int32_t num_reqs;
  int32_t num_tokens;  // T of this launch (padded)
  int32_t num_q_heads;
  int32_t num_kv_heads;
  int32_t group_size;  // G
  int32_t head_size;   // D (also a template parameter of the kernel)
  int32_t wm;          // warps along M: 1, 2 or 4
  int32_t bq;          // query tokens per row tile: bq_capacity(wm, G)
  int32_t num_splits;  // S
  int32_t max_query_len;
  float softmax_scale;
  // fp32(fp32(softmax_scale) * fp32(log2 e)); the kernel multiplies it by the
  // row's exact 2^e prescale.
  float score_scale_log2;
};

// Page of KV token `pos` of request `req`, and its slot in that page.
// block_table entries are int32 kernel-block ids; the page is widened to
// int64 before any multiplication (StreamGeometry::offset).
PTH_HD int64_t token_page(const KernelParams& p, int32_t req, uint32_t pos,
                          uint32_t& slot) {
  uint32_t blk = 0;
  p.block_size.divmod(pos, blk, slot);
  return static_cast<int64_t>(
      p.block_table[static_cast<int64_t>(req) * p.bt_row_stride + blk]);
}

// Byte offsets of query/output rows and of the split partials.
PTH_HD int64_t q_offset(const KernelParams& p, int64_t token, int32_t head) {
  return token * p.s_q_token + static_cast<int64_t>(head) * p.s_q_head;
}
PTH_HD int64_t out_offset(const KernelParams& p, int64_t token, int32_t head) {
  return token * p.s_out_token + static_cast<int64_t>(head) * p.s_out_head;
}
PTH_HD int64_t ws_o_offset(const KernelParams& p, int32_t split, int64_t token,
                           int32_t head) {
  return static_cast<int64_t>(split) * p.s_wso_split + token * p.s_wso_token +
         static_cast<int64_t>(head) * p.s_wso_head;
}
PTH_HD int64_t ws_lse_offset(const KernelParams& p, int32_t split,
                             int64_t token, int32_t head) {
  return static_cast<int64_t>(split) * p.s_wsl_split + token * p.s_wsl_token +
         static_cast<int64_t>(head) * p.s_wsl_head;
}

// --------------------------------------------------------- GQA row packing
// Query tokens of one request per row tile: floor(16*wm / G). The
// op receives wm (planner output) but not the planner's bq, so the kernel
// uses this capacity; it is >= the planner's bq, so the planner's grid
// (T // bq + num_reqs row tiles) still covers every tile.
PTH_HD int32_t bq_capacity(int32_t wm, int32_t group_size) {
  return kGeomRowsPerWarp * wm / group_size;
}

// Row r of a row tile is (query token r / G, head-in-group r % G):
// token-major packing. Returns false for padding rows (rows past
// num_tok * G); their causal limit is -1 and they are never stored.
PTH_HD bool row_to_token_head(uint32_t row, const FastDivmod& group,
                              int32_t num_tok, int32_t& tok,
                              int32_t& head_in_group) {
  uint32_t t = 0, g = 0;
  group.divmod(row, t, g);
  tok = static_cast<int32_t>(t);
  head_in_group = static_cast<int32_t>(g);
  return static_cast<int64_t>(t) < num_tok;
}

// ------------------------------------------------------- row tile -> request
// q-tiles are numbered as in vLLM's unified Triton kernel: request r owns the
// tile indices [qsl[r] / bq + r, qsl[r+1] / bq + r + 1). That is at least
// ceil(q_len / bq) tiles per request (floor((a+q)/b) - floor(a/b) >=
// floor(q/b), plus 1), and the last request ends at qsl[R] / bq + R <=
// T // bq + R, the planner's grid.x. Surplus tiles find no tokens and exit.
struct RowTile {
  int32_t req;        // request index
  int32_t tok_begin;  // first query token of the tile, within the request
  int32_t num_tok;    // query tokens in the tile (1..bq)
  int32_t q_len;      // query tokens of the request (from the device)
  int32_t seq_len;    // KV length of the request incl. new tokens (device)
  int32_t q_start;    // global index of the request's first query token
};

// First tile index of request r.
PTH_HD int32_t first_q_tile(const int32_t* query_start_loc,
                            const FastDivmod& bq_div, int32_t r) {
  return static_cast<int32_t>(
             bq_div.div(static_cast<uint32_t>(query_start_loc[r]))) +
         r;
}

// The largest r in [0, num_reqs) with first_q_tile(r) <= x (binary search;
// first_q_tile is strictly increasing in r).
PTH_HD int32_t find_request(const int32_t* query_start_loc,
                            const FastDivmod& bq_div, int32_t num_reqs,
                            int32_t x) {
  int32_t lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo + 1) / 2;
    if (first_q_tile(query_start_loc, bq_div, mid) <= x) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// Fills `t` for q-tile x; returns false if the tile holds no query token (the
// CTA exits; for S > 1 its partials are never read because only tokens that
// exist are combined).
PTH_HD bool row_tile(const int32_t* query_start_loc, const int32_t* seq_lens,
                     const FastDivmod& bq_div, int32_t bq, int32_t num_reqs,
                     int32_t x, RowTile& t) {
  if (num_reqs <= 0) return false;
  const int32_t r = find_request(query_start_loc, bq_div, num_reqs, x);
  t.req = r;
  t.q_start = query_start_loc[r];
  t.q_len = query_start_loc[r + 1] - t.q_start;
  t.seq_len = seq_lens[r];
  const int64_t begin =
      static_cast<int64_t>(x - first_q_tile(query_start_loc, bq_div, r)) * bq;
  if (begin >= t.q_len) return false;
  t.tok_begin = static_cast<int32_t>(begin);
  t.num_tok = t.q_len - t.tok_begin < bq ? t.q_len - t.tok_begin : bq;
  return true;
}

// ----------------------------------------------------------- causal geometry
// Bottom-right alignment: query token i of a request with q_len
// query tokens and seq_len KV tokens (new ones included) sees KV positions
// j <= seq_len - q_len + i. Precondition: seq_len >= q_len (vLLM appends the
// query tokens to the cache before attention).
PTH_HD int32_t kv_limit(int32_t q_len, int32_t seq_len, int32_t tok) {
  return seq_len - q_len + tok;
}

// One past the last KV position any row of the tile sees: the limit of its
// last token, + 1. Tiles at or beyond it are never loaded.
PTH_HD int32_t kv_range_end(const RowTile& t) {
  return kv_limit(t.q_len, t.seq_len, t.tok_begin + t.num_tok - 1) + 1;
}

// The smallest limit of the tile (its first token): tiles entirely at or
// below it need no mask.
PTH_HD int32_t kv_min_limit(const RowTile& t) {
  return kv_limit(t.q_len, t.seq_len, t.tok_begin);
}

// ceil(range_end / tile_tokens): KV tiles of a row tile.
PTH_HD int32_t num_kv_tiles(int32_t range_end, int32_t tile_tokens) {
  return (range_end + tile_tokens - 1) / tile_tokens;
}

// Tiles [begin, end) of split `split` out of `num_splits` over `num_tiles`
// tiles: [floor(s*T/S), floor((s+1)*T/S)), identical to
// planner.split_tile_range. Any two splits differ by at
// most one tile; empty splits (T < S) get begin == end. Products in int64.
PTH_HD void split_tile_range(int32_t split, int32_t num_splits,
                             int32_t num_tiles, int32_t& begin, int32_t& end) {
  begin = static_cast<int32_t>(static_cast<int64_t>(split) * num_tiles /
                               num_splits);
  end = static_cast<int32_t>(static_cast<int64_t>(split + 1) * num_tiles /
                             num_splits);
}

// True iff some (row, token) pair of KV tile `tile` is masked, i.e. the tile's
// last token exceeds the tile's smallest row limit. Exact (not just
// conservative): the last token against the first row is the worst pair.
// Covers tokens past the range end too (they exceed every limit).
PTH_HD bool tile_needs_mask(int32_t tile, int32_t tile_tokens,
                            int32_t min_limit) {
  return static_cast<int64_t>(tile) * tile_tokens + tile_tokens - 1 > min_limit;
}

// --------------------------------------------------------- window geometry
// The windowed kernels (sliding window and/or non-causal; the
// kWindow instantiations of attn_kernel.cuh) give row token `tok`, at
// absolute position p = kv_limit(q_len, seq_len, tok), the inclusive visible
// range [lo, hi]:
//   lo = window > 0 ? max(0, p - window + 1) : 0
//   hi = causal ? p : (window > 0 ? min(seq_len - 1, p + window - 1)
//                                 : seq_len - 1)
// window = 0 means no window. Causal with window 0 is the causal geometry
// above (lo = 0, hi = kv_limit). This is vLLM's Triton mask with
// window_size = (window - 1, 0): non-causal k < seq_len and |p - k| < window,
// causal 0 <= p - k < window. Both bounds are non-decreasing in tok, so a row
// tile's extremes come from its first and last token. No intermediate value
// overflows int32 (each comparison is a difference of in-range positions).
PTH_HD int32_t kv_window_lo(int32_t q_len, int32_t seq_len, int32_t tok,
                            int32_t window) {
  const int32_t p = kv_limit(q_len, seq_len, tok);
  return window > 0 && p >= window ? p - window + 1 : 0;
}

PTH_HD int32_t kv_window_hi(int32_t q_len, int32_t seq_len, int32_t tok,
                            int32_t window, bool causal) {
  const int32_t p = kv_limit(q_len, seq_len, tok);
  if (causal) return p;
  return window > 0 && seq_len - 1 - p >= window ? p + window - 1 : seq_len - 1;
}

// First KV position any row of the tile sees: lo of its first token. The
// loader zero-fills the tokens of the first tile below it.
PTH_HD int32_t kv_window_range_begin(const RowTile& t, int32_t window) {
  return kv_window_lo(t.q_len, t.seq_len, t.tok_begin, window);
}

// One past the last KV position any row of the tile sees: hi of its last
// token, + 1.
PTH_HD int32_t kv_window_range_end(const RowTile& t, int32_t window,
                                   bool causal) {
  return kv_window_hi(t.q_len, t.seq_len, t.tok_begin + t.num_tok - 1, window,
                      causal) +
         1;
}

// The largest lo of the tile (its last token): positions at or above it,
// and at or below kv_window_min_hi, are visible to every real row.
PTH_HD int32_t kv_window_max_lo(const RowTile& t, int32_t window) {
  return kv_window_lo(t.q_len, t.seq_len, t.tok_begin + t.num_tok - 1, window);
}

// The smallest hi of the tile (its first token).
PTH_HD int32_t kv_window_min_hi(const RowTile& t, int32_t window, bool causal) {
  return kv_window_hi(t.q_len, t.seq_len, t.tok_begin, window, causal);
}

// The KV tile holding range_begin: the windowed CTA's tiles start there
// (rounded down, so the first tile may start below the range; the loader
// zero-fills and the mask hides those tokens).
PTH_HD int32_t kv_first_tile(int32_t range_begin, int32_t tile_tokens) {
  return range_begin / tile_tokens;
}

// Tiles [begin, end) of split `split` for a windowed CTA: split_tile_range
// over the tiles from kv_first_tile(range_begin) to ceil(range_end / N),
// offset by the first tile. With range_begin = 0 it is the causal partition.
PTH_HD void window_split_tile_range(int32_t split, int32_t num_splits,
                                    int32_t range_begin, int32_t range_end,
                                    int32_t tile_tokens, int32_t& begin,
                                    int32_t& end) {
  const int32_t first = kv_first_tile(range_begin, tile_tokens);
  split_tile_range(split, num_splits,
                   num_kv_tiles(range_end, tile_tokens) - first, begin, end);
  begin += first;
  end += first;
}

// True iff some (row, token) pair of KV tile `tile` lies outside its row's
// window. Exact by the argument of tile_needs_mask, once per side: the tile's
// first token against the last row (largest lo) and its last token against
// the first row (smallest hi) are the worst pairs. Covers the zero-filled
// tokens below the range begin and past the range end.
PTH_HD bool tile_needs_window_mask(int32_t tile, int32_t tile_tokens,
                                   int32_t max_lo, int32_t min_hi) {
  const int64_t first = static_cast<int64_t>(tile) * tile_tokens;
  return first < max_lo || first + tile_tokens - 1 > min_hi;
}

}  // namespace pth_attn
