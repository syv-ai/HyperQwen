# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Builds ``pth_flash_attn._C``, the stable-ABI CUDA op library.

Build in the dev image with ``tools/build_in_image.sh``; it runs
``python setup.py build_ext --inplace`` and the library lands at
``python/pth_flash_attn/_C.abi3.so``. The ``.so`` has no ``PyInit`` function:
load it with ``torch.ops.load_library`` (``pth_flash_attn/ops.py`` does).

Arch selection: the cubin set is ``tools/build_flags.ARCHS`` (8.0;8.6;8.9).
torch's standard build knob ``TORCH_CUDA_ARCH_LIST`` is honoured when set, so a
single-arch build (as ``compile_cost.py`` does for its per-arch table) uses the
same code path; it is a build-time setting only, never read at run time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))

import build_flags  # noqa: E402  (tools/ is not a package; path set above)

# torch derives the -gencode flags from TORCH_CUDA_ARCH_LIST; without it torch
# would target the arch of a visible GPU or a long default list.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", build_flags.TORCH_ARCH_LIST)

# Function-local statics of the kernel launchers (`static const cudaError_t
# attr = cudaFuncSetAttribute(...)` in attn_decode.cuh / attn_kernel.cuh) are
# vague-linkage symbols, which g++ binds as STB_GNU_UNIQUE: the dynamic loader
# then merges them across every DSO of the process, even RTLD_LOCAL ones. A
# process that also loads libpth_attn.so (the C-ABI test library, same
# sources) would find the attribute "already set", skip it for this DSO's own
# kernel copy, and fail the launch with "invalid argument" (observed in
# tests/gpu/test_backend_op.py). -fno-gnu-unique keeps one copy per DSO.
_NO_GNU_UNIQUE = "-fno-gnu-unique"

ext = CUDAExtension(
    name="pth_flash_attn._C",
    sources=list(build_flags.EXTENSION_SOURCES),
    include_dirs=[str(build_flags.CSRC_DIR)],
    define_macros=list(build_flags.DEFINES),
    extra_compile_args={
        "cxx": [*build_flags.CXX_FLAGS, _NO_GNU_UNIQUE],
        "nvcc": [*build_flags.NVCC_FLAGS, "-Xcompiler", _NO_GNU_UNIQUE],
    },
    # Stable ABI: no libtorch_python, no Python C API, tagged abi3.
    py_limited_api=True,
)

setup(
    # Only the shipped package; python/pth_testing holds the test oracles.
    packages=find_packages(where="python", include=["pth_flash_attn*"]),
    package_dir={"": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    # abi3 wheel tag from vLLM's minimum Python (3.10); the .so uses no Python
    # C API at all, so the tag only states the supported floor.
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
