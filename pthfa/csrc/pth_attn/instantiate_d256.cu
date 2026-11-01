// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Instantiations of the attention kernels for head size D = 256: the decode
// kernel (attn_decode.cuh) for row tiles of up to 32 rows, the 16-row-tile
// kernel at WM = 4 for the rest (33..64 rows), each x q dtype in {bf16, fp16}
// with its shipped SUB (tile_traits.cuh Traits). WM = 1 and 2 cover at most
// 16 and 32 rows, which the decode kernel always takes, so they have no
// 16-row-tile instantiation here. No windowed kernel at D = 256 (the
// launcher refuses window > 0 or non-causal there). One translation unit
// per D (attn_kernel.cuh).

#include "attn_decode.cuh"

namespace pth_attn {

cudaError_t launch_attn_d256(const AttnLaunch& L, cudaStream_t stream) {
  if (L.rows <= kDecMaxRows) return launch_dec<256>(L, stream);
  if (L.wm == 4) return launch_attn_wm<256, 4>(L, stream);
  return cudaErrorInvalidValue;
}

}  // namespace pth_attn
