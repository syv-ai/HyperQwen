#!/bin/bash
# Build the pth-flash-attn op library (_C.abi3.so: SASS for sm_80/86/89, no PTX) against this venv's torch
# and install it with the pth_flash_attn package and its vllm.general_plugins entry point into the venv.
# Optional: only PTH_FA=1 selects the backend (--attention-backend CUSTOM). Unselected, the entry point
# records a class path under AttentionBackendEnum.CUSTOM and imports no CUDA code (pthfa/README.md).
# Needs nvcc 13 and a host C++ compiler (the Docker image has both). No GPU, no network.
# usage: bash pthfa/install.sh    (idempotent: skipped while the installed copy carries this PIN's commit,
#        source receipt, arch list, torch and nvcc)
# env:   PY (venv python), MAX_JOBS (4), CUDA_HOME (default: the nvcc on PATH, else /usr/local/cuda, if its
#        CUDA major is torch's; else the venv's pip nvcc), PTHFA_BUILD (build dir; default a temp dir, removed
#        afterwards)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY=${PY:-$REPO/venv/bin/python}
fail() { echo "pthfa/install.sh: $*" >&2; exit 1; }
. "$HERE/PIN"
export TORCH_CUDA_ARCH_LIST MAX_JOBS=${MAX_JOBS:-4}

"$PY" -c 'import torch, setuptools' >/dev/null 2>&1 || fail "cannot import torch and setuptools with $PY (README: Setup)"
command -v g++ >/dev/null || fail "no host C++ compiler (apt install build-essential)"
# nvcc: CUDA_HOME if set, else the system one if its CUDA major is torch's (torch.utils.cpp_extension refuses
# any other), else the venv's pip toolkit (nvidia-cuda-nvcc, docker/requirements.txt) behind a temp CUDA_HOME
# with the lib64/libcudart.so the link step wants, as fa2/install.sh does.
cuda_ver() { "$1" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p'; }
TORCH_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda)')
if [ -z "${CUDA_HOME:-}" ]; then
  SYS_NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc); SYS_VER=$(cuda_ver "$SYS_NVCC")
  if [ "${SYS_VER%%.*}" = "${TORCH_CUDA%%.*}" ]; then
    CUDA_HOME=$(dirname "$(dirname "$SYS_NVCC")")
  else
    CU=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu${TORCH_CUDA%%.*}
    PIP_VER=$(cuda_ver "$CU/bin/nvcc")
    [ "${PIP_VER%%.*}" = "${TORCH_CUDA%%.*}" ] || fail "torch is built for CUDA $TORCH_CUDA, the nvcc on PATH is $SYS_NVCC (CUDA ${SYS_VER:-not found}) and the venv has no $CU/bin/nvcc of CUDA ${TORCH_CUDA%%.*} (pip install -r docker/requirements.txt, or set CUDA_HOME to a CUDA $TORCH_CUDA toolkit)"
    CUDA_HOME=$(mktemp -d); trap 'rm -rf "$CUDA_HOME"' EXIT
    ln -s "$CU/bin" "$CU/include" "$CU/nvvm" "$CUDA_HOME/"
    # pip ships libcudart.so.13 only; the link step wants -lcudart
    mkdir "$CUDA_HOME/lib64" && ln -s "$CU/lib"/* "$CUDA_HOME/lib64/"
    ln -sf "$(cd "$CU/lib" && ls libcudart.so.* | head -1)" "$CUDA_HOME/lib64/libcudart.so"
  fi
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || fail "no nvcc under CUDA_HOME=$CUDA_HOME (set CUDA_HOME to a CUDA 13 toolkit)"
export CUDA_HOME PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
echo "== nvcc: $(nvcc --version | tail -1)  (CUDA_HOME=$CUDA_HOME)"

# torch 2.11 is the library's stable-ABI floor; PIN names the version it was built and measured with.
TORCH_NOW=$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')
case "$TORCH_NOW" in "$TORCH_VERSION"*) ;; *) echo "pthfa/install.sh: note: torch $TORCH_NOW, PIN says $TORCH_VERSION" >&2 ;; esac
SOURCE_SHA=$(sha256sum "$HERE/SHA256SUMS"); SOURCE_SHA=${SOURCE_SHA%% *}
STAMP_NOW="$PTH_COMMIT | sources $SOURCE_SHA | arch $TORCH_CUDA_ARCH_LIST | torch $TORCH_NOW | $(nvcc --version | tail -1)"
PKG_DIR=$("$PY" -c 'import importlib.util as u; s = u.find_spec("pth_flash_attn"); print(s.submodule_search_locations[0] if s else "")' 2>/dev/null || true)
if [ -n "$PKG_DIR" ] && [ -f "$PKG_DIR/_C.abi3.so" ] && [ "$(cat "$PKG_DIR/.pthfa-stamp" 2>/dev/null)" = "$STAMP_NOW" ]; then
  echo "== pth_flash_attn already installed for this PIN ($PKG_DIR); nothing to do"
else
  # Build from a copy, so the compile never writes into pthfa/ (build/, *.egg-info, the in-place .so).
  B=${PTHFA_BUILD:-$(mktemp -d)}
  rm -rf "$B/src" && mkdir -p "$B" && cp -r "$HERE" "$B/src"
  echo "== building pth_flash_attn ${PTH_COMMIT:0:12} for $TORCH_CUDA_ARCH_LIST with MAX_JOBS=$MAX_JOBS"
  START=$(date +%s)
  "$PY" -m pip install -q --no-deps --no-build-isolation "$B/src" || fail "pip install of pthfa/ failed"
  echo "== built and installed in $(( $(date +%s) - START )) s"
  [ -n "${PTHFA_BUILD:-}" ] || rm -rf "$B"
  PKG_DIR=$("$PY" -c 'import importlib.util as u; print(u.find_spec("pth_flash_attn").submodule_search_locations[0])')
  printf '%s\n' "$STAMP_NOW" > "$PKG_DIR/.pthfa-stamp"
fi

# The installed copy, checked without a GPU: SASS for exactly the PIN's archs and no PTX, the op library
# loads, the entry point is there and registers CUSTOM, and nothing selects it unless asked.
LIB="$PKG_DIR/_C.abi3.so"
# cuobjdump: the first on PATH (CUDA_HOME/bin leads it), else Triton's bundled one, skipping any older than
# the nvcc that built the library (a CUDA 12.0 cuobjdump cannot read CUDA 13.0 cubins).
NVCC_VER=$(cuda_ver nvcc) CUOBJDUMP='' OLD=''
for c in "$(command -v cuobjdump || true)" "$(ls "$("$PY" -c 'import triton, os; print(os.path.dirname(triton.__file__))' 2>/dev/null)"/backends/nvidia/bin/cuobjdump 2>/dev/null || true)"; do
  [ -n "$c" ] || continue
  v=$(cuda_ver "$c")
  if [ "$(printf '%s\n' "$NVCC_VER" "$v" | sort -V | head -1)" = "$NVCC_VER" ]; then CUOBJDUMP=$c; break; fi
  OLD="$OLD $c (CUDA ${v:-unknown})"
done
[ -n "$CUOBJDUMP" ] || [ -z "$OLD" ] || fail "no cuobjdump of CUDA $NVCC_VER or newer to check $LIB, only:$OLD (put a newer one first on PATH)"
if [ -n "$CUOBJDUMP" ]; then
  want=$(printf '%s\n' ${TORCH_CUDA_ARCH_LIST//;/ } | sed 's/\.//; s/^/sm_/' | sort -u | tr '\n' ' ')
  have=$("$CUOBJDUMP" --list-elf "$LIB" | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
  [ "$have" = "$want" ] || fail "$LIB has cubins for [$have], PIN wants [$want]"
  "$CUOBJDUMP" --list-ptx "$LIB" 2>/dev/null | grep -q . && fail "$LIB embeds PTX: sm_90+ would JIT it instead of refusing"
  echo "== $LIB: $(stat -c %s "$LIB") bytes, $("$CUOBJDUMP" --list-elf "$LIB" | wc -l) cubins for [$have], no PTX"
else
  echo "== cuobjdump not found: arch list of $LIB not checked" >&2
fi
CUDA_VISIBLE_DEVICES= "$PY" "$HERE/tests/gate_test.py" || fail "gate test failed on the installed copy"
echo "pth_flash_attn installed (PTH_FA=1 selects it; unset, nothing selects it)"
