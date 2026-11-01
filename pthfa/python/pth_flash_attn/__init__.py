# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""pth_flash_attn: paged int8 attention op (sm_80/86/89) with an optional vLLM backend.

Importing the package has no side effects: it neither loads the CUDA op
library nor touches a GPU, so CPU-only code (``planner``, the vLLM plugin
hook) imports cleanly on machines without CUDA. The op library is loaded
explicitly by ``pth_flash_attn.ops``.
"""
