// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Registers the pth_flash_attn op library through the libtorch stable ABI
// (ADR 0003): only the C shim and header-only types are used, so the built
// .so keeps working across torch releases >= TORCH_TARGET_VERSION (setup.py)
// without a rebuild.
//
// The registered implementation is launch.cu: argument checks, then the
// attention and combine kernels on the current CUDA stream.

#include <torch/csrc/stable/library.h>

#include "ops.h"

// Schema text is plan section 4.1 verbatim. `Tensor!` marks the three tensors
// the op writes (out and the two workspace views); everything else is
// read-only. In-tree this block becomes STABLE_TORCH_LIBRARY_FRAGMENT(_C, m).
STABLE_TORCH_LIBRARY(pth_flash_attn, m) {
  m.def(
      "pth_flash_attn_fwd(Tensor! out, Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor k_scale, Tensor v_scale, Tensor block_table, "
      "Tensor seq_lens, Tensor query_start_loc, int max_query_len, int wm, "
      "int num_splits, float softmax_scale, Tensor! ws_o, Tensor! ws_lse) "
      "-> ()");
}

STABLE_TORCH_LIBRARY_IMPL(pth_flash_attn, CUDA, m) {
  m.impl("pth_flash_attn_fwd", TORCH_BOX(&pth_attn::pth_flash_attn_fwd));
}

// The CPU registration exists only so that argument validation is testable on
// CPU-only CI (ADR 0003). The function never computes on CPU: after the checks
// it raises, so no caller mistakes a CPU call for a computed result.
STABLE_TORCH_LIBRARY_IMPL(pth_flash_attn, CPU, m) {
  m.impl("pth_flash_attn_fwd", TORCH_BOX(&pth_attn::pth_flash_attn_fwd));
}
