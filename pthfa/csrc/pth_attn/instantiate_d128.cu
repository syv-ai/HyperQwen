// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Instantiations of the attention kernel for head size D = 128: WM in
// {1, 2, 4} x q dtype in {bf16, fp16}, each with its shipped SUB
// (tile_traits.cuh Traits), and the same six as the windowed kernel
// (attn_fwd_window_kernel). One translation unit per D
// (attn_kernel.cuh).

#include "attn_kernel.cuh"

namespace pth_attn {

cudaError_t launch_attn_d128(const AttnLaunch& L, cudaStream_t stream) {
  switch (L.wm) {
    case 1:
      return launch_attn_wm<128, 1>(L, stream);
    case 2:
      return launch_attn_wm<128, 2>(L, stream);
    case 4:
      return launch_attn_wm<128, 4>(L, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace pth_attn
