#!/bin/bash
# CPU test for the toolchain pthfa/install.sh picks: the nvcc that builds the library (CUDA_HOME if set, else the
# system one if its CUDA major is torch's, else the venv's pip nvcc) and the cuobjdump that checks its cubins (the
# first on PATH, else Triton's, skipping any older than that nvcc). Runs the script with a PATH of only stub tools
# and the coreutils it uses: stub nvcc and cuobjdump print a CUDA release (cuobjdump refuses a library from a
# newer nvcc, as a CUDA 12.0 one does on CUDA 13.0 cubins); a stub venv python answers the script's probes, and
# its `-m pip install` fails as torch.utils.cpp_extension does on a CUDA-major mismatch (or the link on a missing
# lib64/libcudart.so), else records the nvcc and writes a stand-in _C.abi3.so. No GPU, no CUDA, no network.
#
#   bash bench/test_pthfa_install_toolchain.sh
#   PTHFA_INSTALL_SH=<checkout>/pthfa/install.sh bash bench/test_pthfa_install_toolchain.sh
#       runs the same cases against another copy of the script (the negative control: the script before it
#       compared CUDA versions fails the cases with an older system CUDA on PATH)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
SH=${PTHFA_INSTALL_SH:-$REPO/pthfa/install.sh}
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
ENV=$(command -v env)
mkdir "$T/tools"
for u in bash cat cp date dirname grep head ln ls mkdir mktemp readlink rm sed sh sha256sum sort stat tail tr wc; do
  ln -s "$(command -v $u)" "$T/tools/$u"; done
# install.sh only asks whether a host compiler exists; the stub pip never compiles
printf '#!/bin/sh\nexit 1\n' > "$T/tools/g++"; chmod +x "$T/tools/g++"

nvcc_stub() { mkdir -p "$(dirname "$1")"; printf '#!/bin/sh\necho "Cuda compilation tools, release %s, V%s.0"\necho "Build cuda_%s.r%s/compiler.0_0"\n' \
  "$2" "$2" "$2" "$2" > "$1"; chmod +x "$1"; }
cuobjdump_stub() { mkdir -p "$(dirname "$1")"; cat > "$1" <<EOF
#!/bin/sh
case "\$1" in
  --version) echo "Cuda compilation tools, release $2, V$2.0" ;;
  --list-elf) lib=\$(cat "\$2")
    [ "\$(printf '%s\n' "\$lib" $2 | sort -V | head -1)" = "\$lib" ] \\
      || { echo "cuobjdump fatal   : Support for 'sm_4' has been removed. Please use CUDA toolkit (null) or earlier." >&2; exit 1; }
    echo "\$0" >> "\$STUB_LOG.cuobjdump"
    for a in 80 86 89; do echo "ELF file    1: _C.abi3.1.sm_\$a.cubin"; done ;;
esac
EOF
  chmod +x "$1"; }
# toolkit <dir> <cuda> [cuobjdump]: a CUDA install (bin/nvcc, lib64/libcudart.so, and cuobjdump if asked)
toolkit() { nvcc_stub "$1/bin/nvcc" "$2"; mkdir -p "$1/lib64" && : > "$1/lib64/libcudart.so"
  [ $# -lt 3 ] || cuobjdump_stub "$1/bin/cuobjdump" "$2"; }
# pip_nvcc <cuda>: the nvidia-cuda-nvcc wheel's layout in the venv (no cuobjdump, lib/ with libcudart.so.13 only)
pip_nvcc() { local d=$C/site/nvidia/cu${1%%.*}; nvcc_stub "$d/bin/nvcc" "$1"; mkdir -p "$d/include" "$d/nvvm" "$d/lib"
  : > "$d/lib/libcudart.so.${1%%.*}"; }
triton_cuobjdump() { cuobjdump_stub "$C/site/triton/backends/nvidia/bin/cuobjdump" "$1"; }
# newcase <name>: $C with a stub venv; torch reports CUDA 13.0. $C/sys is the "system" toolkit, first on PATH.
newcase() { C=$T/$1; mkdir -p "$C/tmp" "$C/venv/bin" "$C/site/triton"; cat > "$C/venv/bin/python" <<'EOF'
#!/bin/bash
case "$*" in
  "-m pip install "*)
    echo install >> "$STUB_LOG.installs"
    v=$("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')
    [ "${v%%.*}" = "${STUB_TORCH_CUDA%%.*}" ] || { echo "RuntimeError: ('The detected CUDA version (%s) mismatches the version that was used to compilePyTorch (%s). Please make sure to use the same CUDA versions.', '$v', '$STUB_TORCH_CUDA')" >&2; exit 1; }
    [ -e "$CUDA_HOME/lib64/libcudart.so" ] || { echo "ld: cannot find -lcudart" >&2; exit 1; }
    readlink -f "$CUDA_HOME/bin/nvcc" > "$STUB_LOG.nvcc"
    mkdir -p "$STUB_SITE/pth_flash_attn" && echo "$v" > "$STUB_SITE/pth_flash_attn/_C.abi3.so" ;;
  *gate_test.py | *"import torch, setuptools"*) ;;
  *torch.version.cuda*) echo "$STUB_TORCH_CUDA" ;;
  *torch.__version__*) echo 2.13.0 ;;
  *sysconfig*) echo "$STUB_SITE" ;;
  *pth_flash_attn*) [ -d "$STUB_SITE/pth_flash_attn" ] && echo "$STUB_SITE/pth_flash_attn" || echo ;;
  *triton*) echo "$STUB_SITE/triton" ;;
  *) echo "stub python: unexpected $*" >&2; exit 2 ;;
esac
EOF
  chmod +x "$C/venv/bin/python"; }
# run [VAR=value...]: runs the script on $C; output in $C/out, exit code in $C/rc.
run() { "$ENV" -i HOME="$C" TMPDIR="$C/tmp" PATH="$C/sys/bin:$T/tools" PY="$C/venv/bin/python" STUB_SITE="$C/site" \
    STUB_LOG="$C/log" STUB_TORCH_CUDA=13.0 "$@" "$T/tools/bash" "$SH" > "$C/out" 2>&1; echo $? > "$C/rc"; }
# result: exit code | the nvcc that built | the cuobjdump that read the cubins | temp dirs left behind
result() { echo "$(cat "$C/rc") | $(cat "$C/log.nvcc" 2>/dev/null || echo none) | $(tail -1 "$C/log.cuobjdump" 2>/dev/null || echo none) | $(find "$C/tmp" -mindepth 1 -maxdepth 1 | wc -l)"; }
said() { grep -qF -- "$1" "$C/out" && echo yes || echo no; }
TRITON=site/triton/backends/nvidia/bin/cuobjdump

echo "== older system CUDA (12.0) first on PATH, torch on CUDA 13.0 (a bare-metal venv)"
newcase old-system; toolkit "$C/sys" 12.0 cuobjdump; pip_nvcc 13.0; triton_cuobjdump 13.1; run
check "0 | $C/site/nvidia/cu13/bin/nvcc | $C/$TRITON | 0" "$(result)" "the venv's pip nvcc builds, Triton's cuobjdump checks"
newcase old-system-cuda-home; toolkit "$C/sys" 12.0 cuobjdump; toolkit "$C/cuda13" 13.0; triton_cuobjdump 13.1
run CUDA_HOME="$C/cuda13"
check "0 | $C/cuda13/bin/nvcc | $C/$TRITON | 0" "$(result)" "CUDA_HOME at a 13.0 toolkit: it builds, Triton's cuobjdump checks"
newcase no-nvcc; toolkit "$C/sys" 12.0 cuobjdump; triton_cuobjdump 13.1; run
check "1 | none | none no" "$(result | cut -d'|' -f1-3 | sed 's/ $//') $(said "== building")" "no CUDA 13 nvcc anywhere: refused before the build"
check "yes yes" "$(said "the nvcc on PATH is $C/sys/bin/nvcc (CUDA 12.0)") $(said "set CUDA_HOME to a CUDA 13.0 toolkit")" \
  "the refusal names the nvcc found, its CUDA and CUDA_HOME"
newcase old-cuobjdump; toolkit "$C/sys" 12.0 cuobjdump; toolkit "$C/cuda13" 13.0; mkdir -p "$C/site/triton"
run CUDA_HOME="$C/cuda13"
check "1 | $C/cuda13/bin/nvcc | none" "$(result | cut -d'|' -f1-3 | sed 's/ $//')" "only a CUDA 12.0 cuobjdump: refused after the build"
check yes "$(said "only: $C/sys/bin/cuobjdump (CUDA 12.0)")" "the refusal names that cuobjdump and its CUDA"

echo "== CUDA_HOME, a matching system CUDA and no cuobjdump: picked as before"
newcase cuda-home; toolkit "$C/sys" 13.0 cuobjdump; pip_nvcc 13.0; toolkit "$C/cuda13" 13.0 cuobjdump; triton_cuobjdump 13.1
run CUDA_HOME="$C/cuda13"
check "0 | $C/cuda13/bin/nvcc | $C/cuda13/bin/cuobjdump | 0" "$(result)" "CUDA_HOME wins over the system and the venv's nvcc"
newcase image; toolkit "$C/sys" 13.0; pip_nvcc 13.0; triton_cuobjdump 13.1; run
check "0 | $C/sys/bin/nvcc | $C/$TRITON | 0" "$(result)" "Docker image layout (system 13.0 nvcc, no cuobjdump; pip nvcc too): system nvcc, Triton's cuobjdump"
check yes "$(said "(CUDA_HOME=$C/sys)")" "the image layout's CUDA_HOME is the system one's"
newcase system13; toolkit "$C/sys" 13.0 cuobjdump; pip_nvcc 13.0; triton_cuobjdump 13.1; run
check "0 | $C/sys/bin/nvcc | $C/sys/bin/cuobjdump | 0" "$(result)" "a system CUDA 13.0 with cuobjdump: both from it"
newcase system13-newer; toolkit "$C/sys" 13.3 cuobjdump; pip_nvcc 13.0; triton_cuobjdump 13.1; run
check "0 | $C/sys/bin/nvcc | $C/sys/bin/cuobjdump | 0" "$(result)" "a newer system CUDA 13.3 (torch only warns on the minor): both from it"
newcase no-cuobjdump; toolkit "$C/sys" 13.0; mkdir -p "$C/site/triton"; run
check "0 | $C/sys/bin/nvcc | none | 0" "$(result)" "no cuobjdump anywhere: installs"
check yes "$(said "cuobjdump not found: arch list of")" "and says the arch list was not checked"

echo "== source receipt invalidates the installed Python adapter"
newcase source-receipt; toolkit "$C/sys" 13.0 cuobjdump
ORIGINAL_SH=$SH
mkdir "$C/recipe"
cp "$SH" "$C/recipe/install.sh"
cp "$(dirname "$SH")/PIN" "$(dirname "$SH")/SHA256SUMS" "$C/recipe/"
SH=$C/recipe/install.sh
run
check 0 "$(cat "$C/rc")" "initial install succeeds"
run
check 0 "$(cat "$C/rc")" "identical receipt remains installable"
check 1 "$(wc -l < "$C/log.installs" | tr -d ' ')" "identical source skips rebuilding"
printf '\n' >> "$C/recipe/SHA256SUMS"
run
check 0 "$(cat "$C/rc")" "changed source receipt installs successfully"
check 2 "$(wc -l < "$C/log.installs" | tr -d ' ')" "changed receipt rebuilds instead of retaining stale Python"
SH=$ORIGINAL_SH

# This is the existing PTH CPU CI entry point; also exercise the optional adapter.
python3 "$HERE/test_fa2_slab_plan.py" || FAILS=$((FAILS+1))
python3 "$HERE/test_fa2_slab_dispatch.py" || FAILS=$((FAILS+1))

echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
