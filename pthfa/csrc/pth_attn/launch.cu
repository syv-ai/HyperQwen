// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// The shipped op pth_flash_attn_fwd (docs/api.md): validation, then
// the two kernels of the library (attention, split combine) on the current
// CUDA stream.
//
// Layering, and why: the kernels, their launch parameters and their per-launch
// argument checks live in the launcher (attn_launch.h: attn_launch.cu,
// attn_combine.cu), which the whole kernel test suite (GPU grid, graph
// replay, stress, gates) drives through the test library's C ABI. This file
// translates tensors into the launcher's structs and nothing else, so the op
// runs the very code those tests proved; there is no second launcher that
// could drift from it. The checks below are the frozen interface (they are
// mirrored by the Python fake in ops.py and stay compiled on CPU so CI can
// test them); the launcher re-checks what it needs and its message is
// surfaced.
//
// What the op never does: allocate, read device data on the host, or
// synchronise. Everything it needs per call is a host integer (the planner's
// launch descriptor) or a device tensor. That is what makes a launch capturable
// in a CUDA graph.
//
// Every message starts with "pth_flash_attn_fwd: " so tests and users can
// match it; STD_TORCH_CHECK prepends the C++ function, file and line.

#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cmath>
#include <cstdint>

#include "attn_launch.h"
#include "ops.h"

// The stable-ABI call that returns the raw cudaStream_t of a device's current
// stream. torch declares it under USE_CUDA; the signature is repeated here
// (AOTITorchError is int32_t) instead of defining that macro for a whole
// translation unit. torch's Stream::nativeHandle() would do too, but it needs
// torch 2.13 and the extension targets the 2.11 floor.
extern "C" int32_t aoti_torch_get_current_cuda_stream(int32_t device_index,
                                                      void** ret_stream);

namespace pth_attn {
namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// Head sizes with a kernel instantiation.
constexpr int64_t kHeadSize64 = 64;
constexpr int64_t kHeadSize128 = 128;
constexpr int64_t kHeadSize256 = 256;

// Rows of one m16n8k16 MMA slice: each warp owns 16 query rows, so a launch
// with wm warps along M packs at most 16 * wm rows of one KV head (G <= 16*wm).
constexpr int64_t kRowsPerWarp = 16;

// GQA packing limit: one CTA row tile holds 64 rows = 4 warps * 16.
constexpr int64_t kMaxGroupSize = 64;

// Cache streams (K, V, k_scale, v_scale) are copied with cp.async of width
// w in {16, 8, 4} bytes chosen on the host; 4 is the minimum (and
// the V half of a 520-byte cell starts at byte 260, which is only 4-aligned).
constexpr int64_t kCacheStreamAlignBytes = 4;

// q, out and the fp32 partials move in 16-byte vectors; their base and outer
// strides must allow that.
constexpr int64_t kVectorAlignBytes = 16;

// Compute capability major served by the sm_80/86/89 cubins.
constexpr int kSupportedCcMajor = 8;

bool is_supported_head_size(int64_t d) {
  return d == kHeadSize64 || d == kHeadSize128 || d == kHeadSize256;
}

const char* dtype_name(ScalarType t) {
  switch (t) {
    case ScalarType::BFloat16:
      return "bfloat16";
    case ScalarType::Half:
      return "float16";
    case ScalarType::Float:
      return "float32";
    case ScalarType::Char:
      return "int8";
    case ScalarType::Int:
      return "int32";
    case ScalarType::Long:
      return "int64";
    default:
      return "an unsupported dtype";
  }
}

void check_dtype(const Tensor& t, ScalarType want, const char* name) {
  STD_TORCH_CHECK(t.scalar_type() == want, "pth_flash_attn_fwd: ", name,
                  " must be ", dtype_name(want), ", got ",
                  dtype_name(t.scalar_type()));
}

void check_dim(const Tensor& t, int64_t want, const char* name) {
  STD_TORCH_CHECK(t.dim() == want, "pth_flash_attn_fwd: ", name, " must be ",
                  want, "-D, got ", t.dim(), "-D");
}

void check_size(const Tensor& t, int64_t dim, int64_t want, const char* name,
                const char* what) {
  STD_TORCH_CHECK(t.size(dim) == want, "pth_flash_attn_fwd: ", name, ".size(",
                  dim, ") must equal ", what, " = ", want, ", got ",
                  t.size(dim));
}

void check_inner_stride_one(const Tensor& t, const char* name) {
  // The innermost dimension is moved as contiguous bytes (one row of D
  // elements, or consecutive int32 entries); every outer stride is free.
  STD_TORCH_CHECK(t.stride(t.dim() - 1) == 1, "pth_flash_attn_fwd: ", name,
                  " must have stride 1 in its last dimension, got ",
                  t.stride(t.dim() - 1));
}

void check_same_device(const Tensor& t, const Tensor& ref, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() == ref.is_cuda() &&
                      t.get_device_index() == ref.get_device_index(),
                  "pth_flash_attn_fwd: ", name,
                  " must be on the same device as q");
}

// Every reachable byte offset must fit in int64 (see ops.h for the proof that
// this bound covers the kernel's address sums).
void check_int64_extent(const Tensor& t, const char* name) {
  int64_t extent = 0;
  const bool ok = detail::view_extent_bytes(
      t.sizes().data(), t.strides().data(), t.dim(),
      static_cast<int64_t>(t.element_size()), &extent);
  STD_TORCH_CHECK(ok, "pth_flash_attn_fwd: ", name,
                  " has negative strides or a byte extent that overflows "
                  "int64");
}

// Base pointer and the byte strides of all but the last dimension must be
// multiples of `align`. The last dimension has stride 1 (checked separately)
// and its row length is a multiple of 16 bytes for every supported D.
void check_alignment(const Tensor& t, int64_t align, const char* name) {
  const auto addr = reinterpret_cast<uintptr_t>(t.data_ptr());
  STD_TORCH_CHECK(addr % static_cast<uintptr_t>(align) == 0,
                  "pth_flash_attn_fwd: ", name, " data pointer must be ", align,
                  "-byte aligned");
  const int64_t elem = static_cast<int64_t>(t.element_size());
  for (int64_t i = 0; i + 1 < t.dim(); ++i) {
    STD_TORCH_CHECK((t.stride(i) * elem) % align == 0,
                    "pth_flash_attn_fwd: ", name, " byte stride of dim ", i,
                    " (", t.stride(i) * elem, ") must be a multiple of ",
                    align);
  }
}

// Checks that depend only on metadata (shapes, dtypes, strides, devices,
// scalars). The Python fake in ops.py mirrors exactly this list so that
// torch.compile fails at trace time with the same reason.
void check_metadata(const Tensor& out, const Tensor& q, const Tensor& k_cache,
                    const Tensor& v_cache, const Tensor& k_scale,
                    const Tensor& v_scale, const Tensor& block_table,
                    const Tensor& seq_lens, const Tensor& query_start_loc,
                    int64_t max_query_len, int64_t wm, int64_t num_splits,
                    double softmax_scale, const Tensor& ws_o,
                    const Tensor& ws_lse, int64_t window, bool causal) {
  // q and out: [T, Hq, D], 16-bit float, rows of D contiguous.
  check_dim(q, 3, "q");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16 ||
                      q.scalar_type() == ScalarType::Half,
                  "pth_flash_attn_fwd: q must be bfloat16 or float16, got ",
                  dtype_name(q.scalar_type()));
  const int64_t num_tokens = q.size(0);
  const int64_t num_q_heads = q.size(1);
  const int64_t head_size = q.size(2);
  STD_TORCH_CHECK(is_supported_head_size(head_size),
                  "pth_flash_attn_fwd: head size D = q.size(2) must be 64, "
                  "128 or 256, got ",
                  head_size);
  check_inner_stride_one(q, "q");

  check_dim(out, 3, "out");
  check_dtype(out, q.scalar_type(), "out (same dtype as q)");
  check_size(out, 0, num_tokens, "out", "T");
  check_size(out, 1, num_q_heads, "out", "Hq");
  check_size(out, 2, head_size, "out", "D");
  check_inner_stride_one(out, "out");

  // Caches: int8 views [NB, BS, Hkv, D] with arbitrary outer strides.
  check_dim(k_cache, 4, "k_cache");
  check_dim(v_cache, 4, "v_cache");
  check_dtype(k_cache, ScalarType::Char, "k_cache");
  check_dtype(v_cache, ScalarType::Char, "v_cache");
  const int64_t num_blocks = k_cache.size(0);
  const int64_t block_size = k_cache.size(1);
  const int64_t num_kv_heads = k_cache.size(2);
  check_size(k_cache, 3, head_size, "k_cache", "D");
  check_size(v_cache, 0, num_blocks, "v_cache", "NB");
  check_size(v_cache, 1, block_size, "v_cache", "BS");
  check_size(v_cache, 2, num_kv_heads, "v_cache", "Hkv");
  check_size(v_cache, 3, head_size, "v_cache", "D");
  check_inner_stride_one(k_cache, "k_cache");
  check_inner_stride_one(v_cache, "v_cache");

  // Scales: float32 views [NB, BS, Hkv] with arbitrary strides.
  check_dim(k_scale, 3, "k_scale");
  check_dim(v_scale, 3, "v_scale");
  check_dtype(k_scale, ScalarType::Float, "k_scale");
  check_dtype(v_scale, ScalarType::Float, "v_scale");
  check_size(k_scale, 0, num_blocks, "k_scale", "NB");
  check_size(k_scale, 1, block_size, "k_scale", "BS");
  check_size(k_scale, 2, num_kv_heads, "k_scale", "Hkv");
  check_size(v_scale, 0, num_blocks, "v_scale", "NB");
  check_size(v_scale, 1, block_size, "v_scale", "BS");
  check_size(v_scale, 2, num_kv_heads, "v_scale", "Hkv");

  // GQA: G = Hq / Hkv query heads share one KV head and are packed as rows.
  STD_TORCH_CHECK(num_kv_heads >= 1 && num_q_heads % num_kv_heads == 0,
                  "pth_flash_attn_fwd: Hq = ", num_q_heads,
                  " must be a positive multiple of Hkv = ", num_kv_heads);
  const int64_t group = num_q_heads / num_kv_heads;
  STD_TORCH_CHECK(wm == 1 || wm == 2 || wm == 4,
                  "pth_flash_attn_fwd: wm must be 1, 2 or 4, got ", wm);
  STD_TORCH_CHECK(group <= kMaxGroupSize,
                  "pth_flash_attn_fwd: G = Hq/Hkv = ", group,
                  " exceeds the supported maximum ", kMaxGroupSize);
  STD_TORCH_CHECK(group <= kRowsPerWarp * wm,
                  "pth_flash_attn_fwd: G = Hq/Hkv = ", group,
                  " must be <= 16*wm = ", kRowsPerWarp * wm);

  // Per-request metadata: int32, consecutive entries contiguous.
  check_dim(block_table, 2, "block_table");
  check_dim(seq_lens, 1, "seq_lens");
  check_dim(query_start_loc, 1, "query_start_loc");
  check_dtype(block_table, ScalarType::Int, "block_table");
  check_dtype(seq_lens, ScalarType::Int, "seq_lens");
  check_dtype(query_start_loc, ScalarType::Int, "query_start_loc");
  check_inner_stride_one(block_table, "block_table");
  check_inner_stride_one(seq_lens, "seq_lens");
  check_inner_stride_one(query_start_loc, "query_start_loc");
  const int64_t num_reqs = seq_lens.size(0);
  check_size(query_start_loc, 0, num_reqs + 1, "query_start_loc",
             "num_reqs + 1");
  STD_TORCH_CHECK(block_table.size(0) >= num_reqs,
                  "pth_flash_attn_fwd: block_table has ", block_table.size(0),
                  " rows, fewer than num_reqs = ", num_reqs);

  // Host scalars.
  STD_TORCH_CHECK(max_query_len >= 0 && max_query_len <= num_tokens,
                  "pth_flash_attn_fwd: max_query_len must be in [0, T = ",
                  num_tokens, "], got ", max_query_len);
  STD_TORCH_CHECK(num_splits >= 1,
                  "pth_flash_attn_fwd: num_splits must be >= 1, got ",
                  num_splits);
  STD_TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
                  "pth_flash_attn_fwd: softmax_scale must be finite and > 0, "
                  "got ",
                  softmax_scale);
  STD_TORCH_CHECK(window >= 0, "pth_flash_attn_fwd: window must be >= 0, got ",
                  window);
  // The windowed kernels exist at D = 128 only.
  STD_TORCH_CHECK(!((window > 0 || !causal) && head_size != kHeadSize128),
                  "pth_flash_attn_fwd: sliding-window and non-causal attention "
                  "need head size D = 128, got ",
                  head_size);

  // Devices: everything the kernel reads or writes lives with q.
  check_same_device(out, q, "out");
  check_same_device(k_cache, q, "k_cache");
  check_same_device(v_cache, q, "v_cache");
  check_same_device(k_scale, q, "k_scale");
  check_same_device(v_scale, q, "v_scale");
  check_same_device(block_table, q, "block_table");
  check_same_device(seq_lens, q, "seq_lens");
  check_same_device(query_start_loc, q, "query_start_loc");

  // Workspace: read only when S > 1; may be empty when S == 1 (docs/api.md).
  // Views of at least [S, T, Hq, D] / [S, T, Hq]: the reservation is made at
  // maximum size at load and sliced per call, so the leading two
  // sizes may exceed S and T while Hq and D must match exactly.
  if (num_splits > 1) {
    check_dim(ws_o, 4, "ws_o");
    check_dim(ws_lse, 3, "ws_lse");
    check_dtype(ws_o, ScalarType::Float, "ws_o");
    check_dtype(ws_lse, ScalarType::Float, "ws_lse");
    STD_TORCH_CHECK(ws_o.size(0) >= num_splits && ws_o.size(1) >= num_tokens,
                    "pth_flash_attn_fwd: ws_o must have size(0) >= num_splits "
                    "= ",
                    num_splits, " and size(1) >= T = ", num_tokens, ", got [",
                    ws_o.size(0), ", ", ws_o.size(1), ", ...]");
    check_size(ws_o, 2, num_q_heads, "ws_o", "Hq");
    check_size(ws_o, 3, head_size, "ws_o", "D");
    STD_TORCH_CHECK(
        ws_lse.size(0) >= num_splits && ws_lse.size(1) >= num_tokens,
        "pth_flash_attn_fwd: ws_lse must have size(0) >= num_splits = ",
        num_splits, " and size(1) >= T = ", num_tokens, ", got [",
        ws_lse.size(0), ", ", ws_lse.size(1), ", ...]");
    check_size(ws_lse, 2, num_q_heads, "ws_lse", "Hq");
    check_inner_stride_one(ws_o, "ws_o");
    check_inner_stride_one(ws_lse, "ws_lse");
    check_same_device(ws_o, q, "ws_o");
    check_same_device(ws_lse, q, "ws_lse");
  }
}

// Checks that need data pointers or overflow arithmetic on real strides. The
// fake cannot reproduce them (fake tensors have no addresses), so they live
// only here.
void check_addresses(const Tensor& out, const Tensor& q, const Tensor& k_cache,
                     const Tensor& v_cache, const Tensor& k_scale,
                     const Tensor& v_scale, const Tensor& block_table,
                     const Tensor& seq_lens, const Tensor& query_start_loc,
                     int64_t num_splits, const Tensor& ws_o,
                     const Tensor& ws_lse) {
  check_int64_extent(out, "out");
  check_int64_extent(q, "q");
  check_int64_extent(k_cache, "k_cache");
  check_int64_extent(v_cache, "v_cache");
  check_int64_extent(k_scale, "k_scale");
  check_int64_extent(v_scale, "v_scale");
  check_int64_extent(block_table, "block_table");
  check_int64_extent(seq_lens, "seq_lens");
  check_int64_extent(query_start_loc, "query_start_loc");

  check_alignment(k_cache, kCacheStreamAlignBytes, "k_cache");
  check_alignment(v_cache, kCacheStreamAlignBytes, "v_cache");
  check_alignment(k_scale, kCacheStreamAlignBytes, "k_scale");
  check_alignment(v_scale, kCacheStreamAlignBytes, "v_scale");
  check_alignment(q, kVectorAlignBytes, "q");
  check_alignment(out, kVectorAlignBytes, "out");

  if (num_splits > 1) {
    STD_TORCH_CHECK(ws_o.data_ptr() != nullptr && ws_lse.data_ptr() != nullptr,
                    "pth_flash_attn_fwd: workspace (ws_o, ws_lse) must be "
                    "allocated when num_splits > 1");
    check_int64_extent(ws_o, "ws_o");
    check_int64_extent(ws_lse, "ws_lse");
    check_alignment(ws_o, kVectorAlignBytes, "ws_o");
  }
}

// The cubins in this library are sm_80/86/89 (major 8); cubins of one major
// run on every later minor of that major, and on no other major.
void check_device_capability(const Tensor& q) {
  int major = 0;
  const cudaError_t err = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, q.get_device_index());
  STD_TORCH_CHECK(err == cudaSuccess,
                  "pth_flash_attn_fwd: cannot query the compute capability: ",
                  cudaGetErrorString(err));
  STD_TORCH_CHECK(major == kSupportedCcMajor,
                  "pth_flash_attn_fwd: needs compute capability 8.x "
                  "(sm_80/86/89), device has major ",
                  major);
}

// Smallest wm in {1, 2, 4} with 16 wm >= rows (planner.wm_for_rows, the
// library's own derivation from bq * G).
int64_t wm_for_rows(int64_t rows) {
  int64_t wm = 1;
  while (wm < 4 && rows > kRowsPerWarp * wm) wm *= 2;
  return wm;
}

// Query tokens per row tile. The op schema has no bq, so it is derived here
// from wm, G and max_query_len: balanced bq = min(planner bq, q), planner
// bq = floor(16 wm / G), so a q = 1 decode (G = 6) is one 6-row n-tile, not
// the two of the capacity 2.
// The shorter tile is taken only when it keeps the planned wm; a prefill launch
// (wm = 4) whose longest request is short keeps the capacity, because its wm
// was chosen for the capacity's rows. Needs G <= 16 wm (check_metadata).
// Any bq is correct for any q_len: the grid is T / bq + num_reqs tiles, which
// covers every request. The planner's split count only sized the workspace.
int32_t tile_tokens(int64_t wm, int64_t group, int64_t max_query_len) {
  const int64_t capacity = kRowsPerWarp * wm / group;
  if (max_query_len >= 1 && max_query_len < capacity &&
      wm_for_rows(max_query_len * group) == wm) {
    return static_cast<int32_t>(max_query_len);
  }
  return static_cast<int32_t>(capacity);
}

// The library derives its wm from bq * G. It must be the wm the caller planned
// for, else the kernel family and the workspace views were sized for another
// tile height. tile_tokens() keeps this true for every planner output; the
// check guards a hand-made call.
void check_wm_matches_rows(int64_t wm, int64_t group, int32_t bq) {
  const int64_t rows = static_cast<int64_t>(bq) * group;
  const int64_t derived = wm_for_rows(rows);
  STD_TORCH_CHECK(derived == wm, "pth_flash_attn_fwd: wm = ", wm,
                  " is not the tile height of G = ", group, " (rows ", rows,
                  " need wm = ", derived, ")");
}

int32_t dtype_code(ScalarType t) {
  return t == ScalarType::Half ? PTH_ATTN_F16 : PTH_ATTN_BF16;
}

// Strides arrive in elements; the launcher takes bytes.
int64_t byte_stride(const Tensor& t, int64_t dim) {
  return t.stride(dim) * static_cast<int64_t>(t.element_size());
}

PthAttnStream stream_of(const Tensor& t) {
  PthAttnStream s{};
  s.data = t.data_ptr();
  s.stride_block = byte_stride(t, 0);
  s.stride_slot = byte_stride(t, 1);
  s.stride_head = byte_stride(t, 2);
  return s;
}

cudaStream_t current_cuda_stream(int32_t device_index) {
  void* stream = nullptr;
  const int32_t err = aoti_torch_get_current_cuda_stream(device_index, &stream);
  STD_TORCH_CHECK(err == 0,
                  "pth_flash_attn_fwd: cannot get the current CUDA stream of "
                  "device ",
                  device_index);
  return static_cast<cudaStream_t>(stream);
}

// Turns the checked tensors into the library's structs and enqueues the
// attention kernel and, for S > 1, the combine on the current stream.
void launch_library(Tensor& out, const Tensor& q, const Tensor& k_cache,
                    const Tensor& v_cache, const Tensor& k_scale,
                    const Tensor& v_scale, const Tensor& block_table,
                    const Tensor& seq_lens, const Tensor& query_start_loc,
                    int64_t max_query_len, int64_t wm, int64_t num_splits,
                    double softmax_scale, Tensor& ws_o, Tensor& ws_lse,
                    int64_t window, bool causal) {
  const int64_t num_kv_heads = k_cache.size(2);
  const int64_t group = q.size(1) / num_kv_heads;
  const int32_t bq = tile_tokens(wm, group, max_query_len);
  check_wm_matches_rows(wm, group, bq);

  PthAttnArgs a{};
  a.q = q.data_ptr();
  a.q_dtype = dtype_code(q.scalar_type());
  a.q_stride_token = byte_stride(q, 0);
  a.q_stride_head = byte_stride(q, 1);
  a.out = out.data_ptr();
  a.out_dtype = dtype_code(out.scalar_type());
  a.out_stride_token = byte_stride(out, 0);
  a.out_stride_head = byte_stride(out, 1);
  a.k_cache = stream_of(k_cache);
  a.v_cache = stream_of(v_cache);
  a.k_scale = stream_of(k_scale);
  a.v_scale = stream_of(v_scale);
  a.num_blocks = static_cast<int32_t>(k_cache.size(0));
  a.block_size = static_cast<int32_t>(k_cache.size(1));
  a.num_kv_heads = static_cast<int32_t>(num_kv_heads);
  a.head_size = static_cast<int32_t>(q.size(2));
  a.block_table = static_cast<const int32_t*>(block_table.data_ptr());
  a.block_table_row_stride = block_table.stride(0);
  a.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
  a.query_start_loc = static_cast<const int32_t*>(query_start_loc.data_ptr());
  a.num_reqs = static_cast<int32_t>(seq_lens.size(0));
  a.num_tokens = static_cast<int32_t>(q.size(0));
  a.num_q_heads = static_cast<int32_t>(q.size(1));
  a.bq = bq;
  a.num_q_tiles = 0;  // the library's T / bq + num_reqs (planner q_tiles_upper)
  a.num_splits = static_cast<int32_t>(num_splits);
  a.softmax_scale = static_cast<float>(softmax_scale);
  // A window wider than any int32 sequence is the same as INT32_MAX.
  a.window = window > INT32_MAX ? INT32_MAX : static_cast<int32_t>(window);
  a.causal = causal ? 1 : 0;
  if (num_splits > 1) {
    a.ws_o = static_cast<float*>(ws_o.data_ptr());
    a.ws_o_stride_split = byte_stride(ws_o, 0);
    a.ws_o_stride_token = byte_stride(ws_o, 1);
    a.ws_o_stride_head = byte_stride(ws_o, 2);
    a.ws_lse = static_cast<float*>(ws_lse.data_ptr());
    a.ws_lse_stride_split = byte_stride(ws_lse, 0);
    a.ws_lse_stride_token = byte_stride(ws_lse, 1);
    a.ws_lse_stride_head = byte_stride(ws_lse, 2);
  }

  // The launch goes to q's device and its current stream: the kernels use
  // neither the default stream nor the thread's current device.
  const int32_t device = static_cast<int32_t>(q.get_device_index());
  const torch::stable::accelerator::DeviceGuard guard(device);
  const cudaStream_t stream = current_cuda_stream(device);

  int32_t rc = launch_attention(a, kLoaderDefault, stream);
  STD_TORCH_CHECK(rc == PTH_ATTN_OK, "pth_flash_attn_fwd: attention launch: ",
                  last_launch_error());
  if (num_splits == 1) {
    return;
  }
  PthAttnCombineArgs c{};
  c.ws_o = a.ws_o;
  c.ws_o_stride_split = a.ws_o_stride_split;
  c.ws_o_stride_token = a.ws_o_stride_token;
  c.ws_o_stride_head = a.ws_o_stride_head;
  c.ws_lse = a.ws_lse;
  c.ws_lse_stride_split = a.ws_lse_stride_split;
  c.ws_lse_stride_token = a.ws_lse_stride_token;
  c.ws_lse_stride_head = a.ws_lse_stride_head;
  c.out = a.out;
  c.out_dtype = a.out_dtype;
  c.out_stride_token = a.out_stride_token;
  c.out_stride_head = a.out_stride_head;
  c.query_start_loc = a.query_start_loc;
  c.num_reqs = a.num_reqs;
  c.num_tokens = a.num_tokens;
  c.num_q_heads = a.num_q_heads;
  c.head_size = a.head_size;
  c.num_splits = a.num_splits;
  rc = launch_combine(c, stream);
  STD_TORCH_CHECK(rc == PTH_ATTN_OK,
                  "pth_flash_attn_fwd: combine launch: ", last_launch_error());
}

}  // namespace

void pth_flash_attn_fwd(Tensor& out, const Tensor& q, const Tensor& k_cache,
                        const Tensor& v_cache, const Tensor& k_scale,
                        const Tensor& v_scale, const Tensor& block_table,
                        const Tensor& seq_lens, const Tensor& query_start_loc,
                        int64_t max_query_len, int64_t wm, int64_t num_splits,
                        double softmax_scale, Tensor& ws_o, Tensor& ws_lse,
                        int64_t window, bool causal) {
  check_metadata(out, q, k_cache, v_cache, k_scale, v_scale, block_table,
                 seq_lens, query_start_loc, max_query_len, wm, num_splits,
                 softmax_scale, ws_o, ws_lse, window, causal);
  check_addresses(out, q, k_cache, v_cache, k_scale, v_scale, block_table,
                  seq_lens, query_start_loc, num_splits, ws_o, ws_lse);
  // The same function is registered for the CPU key (torch_bindings.cpp) so
  // that every check above is testable on CPU-only CI. The kernels exist for
  // CUDA tensors only: a valid CPU call ends here with its own message, so
  // no caller mistakes it for a computed result.
  STD_TORCH_CHECK(q.is_cuda(),
                  "pth_flash_attn_fwd: arguments are valid, but the op "
                  "computes on CUDA tensors only (the CPU registration "
                  "exists to test the argument checks)");
  check_device_capability(q);
  // An empty slice (no decode or no prefill requests) has nothing to launch;
  // the library requires at least one request and one token.
  if (q.size(0) == 0 || seq_lens.size(0) == 0) {
    return;
  }
  launch_library(out, q, k_cache, v_cache, k_scale, v_scale, block_table,
                 seq_lens, query_start_loc, max_query_len, wm, num_splits,
                 softmax_scale, ws_o, ws_lse, window, causal);
}

}  // namespace pth_attn
