#!/bin/bash
# CPU test for #401: single-user/alternative.sh defaults GPU_UTIL to 0.95 on native Linux and to 0.93 under WSL2,
# where vLLM sees less than the nominal 24 GiB (22.75 GiB free on a 3090) and 0.95 fails to start. An explicit
# GPU_UTIL wins in both. WSL is faked with WSL_DISTRO_NAME. Copies the checkout to a temp dir and runs the
# launcher with PRINT_ARGV=1 (launcher_common.sh) against a stub vllm. No GPU, no model, no network.
#
#   bash bench/test_alt_wsl_gpu_util.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
check() { if [ "$1" = "$2" ]; then echo "  PASS  $3"; else echo "  FAIL  $3 (want '$1', got '$2')"; FAILS=$((FAILS+1)); fi; }
# util_of <env assignments...>: the argument after --gpu-memory-utilization in the PRINT_ARGV=1 argv.
util_of() {
  (cd "$T" && env -u GPU_UTIL -u WSL_DISTRO_NAME VLLM_API_KEY=k PRINT_ARGV=1 "$@" bash single-user/alternative.sh 2>/dev/null </dev/null) \
    | sed -n '/^--gpu-memory-utilization$/{n;p;}'; }
# The sandbox itself may be WSL; only the faked variable matters when /proc says otherwise, so skip the native rows there.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "  skip  native rows: this host reports WSL in /proc/sys/kernel/osrelease"
else
  check 0.95 "$(util_of)"                    "native Linux: GPU_UTIL defaults to 0.95"
  check 0.90 "$(util_of GPU_UTIL=0.90)"      "native Linux: an explicit GPU_UTIL wins"
fi
check 0.93 "$(util_of WSL_DISTRO_NAME=Ubuntu)"                "WSL2: GPU_UTIL defaults to 0.93"
check 0.90 "$(util_of WSL_DISTRO_NAME=Ubuntu GPU_UTIL=0.90)"  "WSL2: an explicit GPU_UTIL wins"
check 0.97 "$(util_of WSL_DISTRO_NAME=Ubuntu GPU_UTIL=0.97)"  "WSL2: an explicit GPU_UTIL above the default is kept"
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
