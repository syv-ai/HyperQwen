#!/bin/bash
# Build the fp8-KV FlashAttention-2 libraries (fa2_fp8kv.so, fa2_fp8kv_prefill.so) against this venv's
# torch and install them with the fa2_fp8kv_vllm plugin into the venv. Optional: only FA2_FP8KV=1 uses it.
# Needs nvcc 13 (the Docker image has one; a host venv has pip's nvidia-cuda-nvcc 13.0.88), a host C++
# compiler and git. No GPU.
# usage: bash fa2/install.sh        (idempotent: the compile is skipped while both libraries are newer
#        than every source and were built for the same torch, nvcc, arch, CUTLASS and flashinfer)
# env:   PY (venv python), TORCH_CUDA_ARCH_LIST (8.6), MAX_JOBS (2), CUDA_HOME, CUTLASS_PATH (an
#        include dir; default: a CUTLASS_COMMIT checkout under FA2_CACHE), FA2_CACHE (the CUTLASS
#        checkout and the build tree; default ~/.cache/hyperqwen-fa2)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY=${PY:-$REPO/venv/bin/python}
fail() { echo "fa2/install.sh: $*" >&2; exit 1; }
. "$HERE/PIN"
CACHE=${FA2_CACHE:-$HOME/.cache/hyperqwen-fa2}
WORK=$CACHE/build
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-2}

"$PY" -c 'import torch, flashinfer' >/dev/null 2>&1 || fail "cannot import torch and flashinfer with $PY (README: Setup)"
"$PY" -c 'import setuptools' >/dev/null 2>&1 || fail "setuptools is missing from $PY (pip install setuptools)"
command -v g++ >/dev/null || fail "no host C++ compiler (apt install build-essential)"

# CUTLASS: include/ only, fetched at the pinned commit and verified. A cache at another commit is refetched.
if [ -z "${CUTLASS_PATH:-}" ]; then
  CUT=$CACHE/cutlass
  if [ "$(git -C "$CUT" rev-parse HEAD 2>/dev/null)" != "$CUTLASS_COMMIT" ]; then
    command -v git >/dev/null || fail "git is needed to fetch CUTLASS (or set CUTLASS_PATH to an include dir)"
    echo "== fetching CUTLASS $CUTLASS_COMMIT into $CUT"
    rm -rf "$CUT" && mkdir -p "$CUT"
    git -C "$CUT" init -q
    git -C "$CUT" remote add origin https://github.com/NVIDIA/cutlass.git
    git -C "$CUT" sparse-checkout set include
    git -C "$CUT" fetch -q --depth=1 --filter=blob:none origin "$CUTLASS_COMMIT" || fail "cannot fetch CUTLASS $CUTLASS_COMMIT"
    git -C "$CUT" checkout -q --detach FETCH_HEAD
    [ "$(git -C "$CUT" rev-parse HEAD)" = "$CUTLASS_COMMIT" ] || fail "CUTLASS checkout is not $CUTLASS_COMMIT"
  fi
  export CUTLASS_PATH=$CUT/include
fi
[ -f "$CUTLASS_PATH/cute/tensor.hpp" ] || fail "$CUTLASS_PATH/cute/tensor.hpp not found: CUTLASS_PATH must be an include dir"

# nvcc: CUDA_HOME if set, else a system nvcc of major >= 13, else the venv's pip CUDA 13 toolkit. The pip
# layout has lib/ (no unversioned libcudart.so) where torch.utils.cpp_extension wants lib64/, so it gets a symlink shim. Pin 13.0.x
# (docker/requirements.txt): a newer pip nvcc/nvvm beside the 13.0 headers fails to compile.
nvcc_major() { "$1" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+)\..*/\1/p'; }
if [ -z "${CUDA_HOME:-}" ]; then
  SYS_NVCC=$(command -v nvcc || true)
  if [ -n "$SYS_NVCC" ] && [ "$(nvcc_major "$SYS_NVCC")" -ge 13 ]; then
    CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$SYS_NVCC")")")
  else
    CU13=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13
    [ -x "$CU13/bin/nvcc" ] || fail "no nvcc 13: the one on PATH is ${SYS_NVCC:-missing} and $CU13/bin/nvcc does not exist (pip install -r docker/requirements.txt, or set CUDA_HOME)"
    CUDA_HOME=$CACHE/cuda13-home
    rm -rf "$CUDA_HOME" && mkdir -p "$CUDA_HOME"
    for d in bin include nvvm cccl; do [ -e "$CU13/$d" ] && ln -s "$CU13/$d" "$CUDA_HOME/$d"; done
    # pip ships libcudart.so.13 only; the link step wants -lcudart
    mkdir "$CUDA_HOME/lib64" && ln -s "$CU13/lib"/* "$CUDA_HOME/lib64/"
    ln -sf "$(cd "$CU13/lib" && ls libcudart.so.* | head -1)" "$CUDA_HOME/lib64/libcudart.so"
  fi
fi
export CUDA_HOME
# the venv's bin first: torch's cpp_extension needs its ninja
export PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
echo "== nvcc: $(nvcc --version | tail -1)  (CUDA_HOME=$CUDA_HOME)"

STAMP_NOW=$("$PY" - <<'PY'
import subprocess, torch
from importlib.metadata import version
nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
print(f"torch {torch.__version__} | {nvcc} | flashinfer {version('flashinfer-python')}")
PY
)
STAMP_NOW="$STAMP_NOW | arch $TORCH_CUDA_ARCH_LIST | cutlass $CUTLASS_COMMIT"
SRC=$WORK/src
LIB=$SRC/build-pipeline/fa2_fp8kv.so
PREFILL=$SRC/build-prefill/fa2_fp8kv_prefill.so
if [ -f "$LIB" ] && [ -f "$PREFILL" ] && [ "$(cat "$WORK/stamp" 2>/dev/null)" = "$STAMP_NOW" ] \
   && [ -z "$(find "$HERE/headers" "$HERE"/*.cu "$HERE/build.py" -newer "$WORK/stamp" -print -quit)" ]; then
  echo "== libraries are current ($STAMP_NOW)"
else
  # build.py writes its build dirs next to itself: build from a copy, so fa2/ stays clean
  rm -rf "$WORK" && mkdir -p "$SRC"
  cp -r "$HERE/headers" "$HERE"/*.cu "$HERE/build.py" "$SRC/"
  echo "== compiling for sm_${TORCH_CUDA_ARCH_LIST/./} (single-threaded nvcc, a few minutes on a fast CPU)"
  "$PY" "$SRC/build.py" --pipeline --prefill
  [ -f "$LIB" ] && [ -f "$PREFILL" ] || fail "build.py finished without $LIB and $PREFILL"
  echo "$STAMP_NOW" > "$WORK/stamp"
fi

# The package is staged with the libraries inside it, so pip's own build never touches fa2/.
PKG=$WORK/pkg
rm -rf "$PKG" && mkdir -p "$PKG"
cp -r "$HERE/python" "$HERE/pyproject.toml" "$HERE/LICENSE" "$PKG/"
cp "$LIB" "$PREFILL" "$PKG/python/fa2_fp8kv_vllm/"
"$PY" -m pip install -q --no-deps --no-build-isolation "$PKG" || fail "pip install of the fa2_fp8kv_vllm package failed"
"$PY" -c 'import fa2_fp8kv_vllm.paths as p; p.check()' || fail "installed package has no libraries"
"$PY" -c 'from importlib.metadata import entry_points
assert any(e.value == "fa2_fp8kv_vllm:register" for e in entry_points(group="vllm.general_plugins")), "entry point missing"'
ls -l "$("$PY" -c 'import fa2_fp8kv_vllm.paths as p; print(p.PACKAGE)')"/*.so
echo "fa2_fp8kv installed (FA2_FP8KV=1 selects it; unset, it is a no-op)"
