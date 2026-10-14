// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_launch.h: the host launcher of the attention kernels (attn_launch.cu)
// and of the split combine (attn_combine.cu).
//
// The op (launch.cu) and the C ABI of the test library (attn_capi.cu, driven
// by the kernel test ladder through ctypes) both launch through these
// functions, so the op runs exactly the code the tests prove.
//
// A launcher checks its arguments (every check names the field), turns them
// into the kernel parameters, picks the instantiation and enqueues it on
// `stream`. It allocates nothing, reads nothing back from the device and does
// not synchronise. It returns PTH_ATTN_OK, PTH_ATTN_EINVAL (an argument check
// failed) or PTH_ATTN_ECUDA (the launch failed); the message of a failure is
// last_launch_error() on the calling thread.
//
// Requirements beyond the semantics of attn_args.h:
// - wm is derived: the smallest wm in {1, 2, 4} with bq * G <= 16 wm
//   (= the planner's wm for the planner's bq, ADR 0013); bq * G > 64 is
//   EINVAL.
// - Alignment: q, out and the workspace need 16-B aligned base pointers and
//   outer byte strides (16-B vector loads/stores); the four cache streams
//   need 4-B alignment (the narrowest cp.async). EINVAL otherwise.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "attn_args.h"

namespace pth_attn {

// KV loader of launch_attention: the shipped choice (ADR 0012 Amendment 1:
// the whole-cell loader wherever it applies), or one forced by a test:
// separate streams, or the whole cell (D = 256 with the inline cell layout
// only; EINVAL otherwise).
constexpr int32_t kLoaderDefault = 0;
constexpr int32_t kLoaderSeparate = 1;
constexpr int32_t kLoaderCell = 2;

int32_t launch_attention(const PthAttnArgs& args, int32_t loader,
                         cudaStream_t stream);
int32_t launch_combine(const PthAttnCombineArgs& args, cudaStream_t stream);

// The calling thread's message of the last failed launch ("" after a
// successful one). Valid until the next launch on the same thread.
const char* last_launch_error();
// Records `msg` as the calling thread's last error and returns `code`.
int32_t launch_fail(int32_t code, const std::string& msg);
// Clears the last error; every launcher and C entry point starts with it.
void clear_launch_error();

// Head sizes with a kernel instantiation (instantiate_d{64,128,256}.cu).
inline bool is_head_size(int32_t d) { return d == 64 || d == 128 || d == 256; }

}  // namespace pth_attn

// Returns PTH_ATTN_EINVAL with `msg` from the enclosing launcher unless
// `cond` holds.
#define PTH_ATTN_REQUIRE(cond, msg)                                    \
  do {                                                                 \
    if (!(cond)) return pth_attn::launch_fail(PTH_ATTN_EINVAL, (msg)); \
  } while (0)
