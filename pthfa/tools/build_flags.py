# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Single source of truth for how the CUDA sources are compiled (ADR 0003).

``setup.py`` builds the extension with these flags, and the measurement tools
(``ptxas_gate.py``, ``sass_check.py``, ``compile_cost.py``) compile single
translation units with the very same flags, so a register count or compile time
they report is the one the shipped ``.so`` gets.

Every value here has a row in ``docs/constants.d/T1.md``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Compute capabilities with a cubin in the library (plan sections 4.7, 6.1).
# sm_80 (A100), sm_86 (GA10x, the RTX 3080 target), sm_89 (Ada). No PTX is
# embedded: sm_90+ must be rejected, not JIT-compiled (ADR 0003).
ARCHS: tuple[str, ...] = ("80", "86", "89")

# torch's own spelling of ARCHS for TORCH_CUDA_ARCH_LIST ("8.0;8.6;8.9").
TORCH_ARCH_LIST = ";".join(f"{a[0]}.{a[1]}" for a in ARCHS)

# Stable-ABI floor: the .so only uses C-shim APIs present in torch 2.11, the
# same floor vLLM's _C_stable_libtorch uses (vLLM CMakeLists.txt v0.30.0), so
# the op can later move in-tree unchanged. Encoding: major << 56 | minor << 48.
TORCH_TARGET_VERSION = "0x020B000000000000ULL"

CXX_STD = "-std=c++17"
OPT_LEVEL = "-O3"

# Preprocessor definitions shared by host and device compilation.
DEFINES: tuple[tuple[str, str], ...] = (("TORCH_TARGET_VERSION", TORCH_TARGET_VERSION),)

# Flags for g++ (host-only translation units such as torch_bindings.cpp).
CXX_FLAGS: tuple[str, ...] = (OPT_LEVEL, CXX_STD)

# Flags for nvcc, excluding -gencode (that comes from ARCHS). No fast-math:
# the error bound (plan section 3.8) assumes IEEE fp32 except where a kernel
# opts in explicitly (ex2.approx is written as inline PTX, not via a flag).
NVCC_FLAGS: tuple[str, ...] = (OPT_LEVEL, CXX_STD)

CSRC_DIR = REPO_ROOT / "csrc" / "pth_attn"

# Translation units of the extension, in link order. launch.cu is the op; the
# other five are the kernel library it launches through attn_launch.h, the
# very sources tools/build_attn_lib.sh builds (with the test-only C ABI on
# top) for the kernel test ladder, so the production op runs the code the
# kernel tests proved (ADR 0021).
EXTENSION_SOURCES: tuple[str, ...] = (
    "csrc/pth_attn/torch_bindings.cpp",
    "csrc/pth_attn/launch.cu",
    "csrc/pth_attn/attn_launch.cu",
    "csrc/pth_attn/attn_combine.cu",
    "csrc/pth_attn/instantiate_d64.cu",
    "csrc/pth_attn/instantiate_d128.cu",
    "csrc/pth_attn/instantiate_d256.cu",
)


def define_flags() -> list[str]:
    """Returns ``-DNAME=VALUE`` flags for DEFINES."""
    return [f"-D{name}={value}" for name, value in DEFINES]


def gencode_flags(arch: str) -> list[str]:
    """Returns the nvcc flags that emit SASS (no PTX) for one ``sm_<arch>``."""
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


def torch_include_dirs() -> list[str]:
    """Returns the include directories torch's CUDAExtension would pass."""
    from torch.utils.cpp_extension import include_paths

    return [str(p) for p in include_paths(device_type="cuda")]


def nvcc_compile_command(
    nvcc: str, source: Path, arch: str, extra: list[str]
) -> list[str]:
    """Builds an nvcc command line equal to the extension build for one arch.

    Args:
        nvcc: Path of the nvcc executable.
        source: The ``.cu`` translation unit.
        arch: Compute capability without dot, e.g. ``"86"``.
        extra: Mode flags such as ``["-cubin", "-o", out]`` or ``["-c", ...]``.

    Returns:
        The argument vector, ready for ``subprocess.run``.
    """
    includes = [f"-I{d}" for d in [str(CSRC_DIR), *torch_include_dirs()]]
    return [
        nvcc,
        *NVCC_FLAGS,
        *define_flags(),
        *includes,
        *gencode_flags(arch),
        *extra,
        str(source),
    ]
