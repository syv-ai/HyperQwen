#!/bin/bash
# CPU test: every launcher turns vLLM 0.34's extensible KV cache off (--disable-extensible-kv-cache, once) unless
# EXTRA_ARGS names it, in which case the launcher adds nothing and EXTRA_ARGS decides; and
# batch's fp8 default GPU_UTIL is 0.93 (docs/vllm-0.34.md, "The extensible KV cache"). Copies
# the checkout to a temp dir and runs the launchers with PRINT_ARGV=1 (launcher_common.sh) against a stub vllm. No
# GPU, no model, no network.
#
#   bash bench/test_extensible_kv_default.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
check() { if [ "$1" = "$2" ]; then echo "  PASS  $3"; else echo "  FAIL  $3 (want '$1', got '$2')"; FAILS=$((FAILS+1)); fi; }
# argv_of <launcher> <env assignments...>: the PRINT_ARGV=1 argv, one argument per line.
argv_of() { local sc=$1; shift
  (cd "$T" && env -u GPU_UTIL -u EXTRA_ARGS -u WSL_DISTRO_NAME -u LSA VLLM_API_KEY=k PRINT_ARGV=1 "$@" bash "$sc" 2>/dev/null </dev/null); }
count() { grep -cx -- "$1" || :; }
for sc in single-user/start_qwen.sh batch/start_qwen.sh single-user/alternative.sh; do
  echo "== $sc"
  check 1 "$(argv_of "$sc" | count --disable-extensible-kv-cache)" "default: --disable-extensible-kv-cache, once"
  a=$(argv_of "$sc" EXTRA_ARGS=--enable-extensible-kv-cache)
  check "0 1" "$(echo "$a" | count --disable-extensible-kv-cache) $(echo "$a" | count --enable-extensible-kv-cache)" \
    "EXTRA_ARGS=--enable-extensible-kv-cache: the launcher adds no --disable"
  a=$(argv_of "$sc" EXTRA_ARGS=--enable_extensible_kv_cache)
  check "0 1" "$(echo "$a" | count --disable-extensible-kv-cache) $(echo "$a" | count --enable-extensible-kv-cache)" \
    "EXTRA_ARGS with the underscore spelling: normalized, and no --disable added"
  check 1 "$(argv_of "$sc" EXTRA_ARGS=--disable-extensible-kv-cache | count --disable-extensible-kv-cache)" \
    "EXTRA_ARGS=--disable-extensible-kv-cache: once, not twice"
done
# The native rows need a host that is not WSL; WSL changes batch's GPU_UTIL default on purpose.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "  skip  native batch GPU_UTIL rows: this host reports WSL in /proc/sys/kernel/osrelease"
else
  echo "== batch/start_qwen.sh GPU_UTIL"
  util() { argv_of batch/start_qwen.sh "$@" | sed -n '/^--gpu-memory-utilization$/{n;p;}'; }
  check 0.93 "$(util)"               "native, KV=fp8: GPU_UTIL defaults to 0.93"
  check 0.94 "$(util GPU_UTIL=0.94)" "native: an explicit GPU_UTIL wins"
fi
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
