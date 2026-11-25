#!/bin/bash
# CPU test for KV_OFFLOAD_GB (launcher_common.sh, qwen_kv_offload_args): each launcher turns it into one
# --kv-offloading-size flag, an explicit flag or --kv-transfer-config in EXTRA_ARGS wins, 0 and unset add
# nothing, and the allocator default the launchers pick for a KV connector (expandable_segments:False)
# follows the knob the same way it follows the flag.
# Copies the checkout to a temp dir and runs the three launchers against a stub vllm that prints its
# argv and PYTORCH_CUDA_ALLOC_CONF. No GPU, no model, no network.
#
#   bash bench/test_kv_offload_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\necho "ALLOC=${PYTORCH_CUDA_ALLOC_CONF-unset}"\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
# run <launcher> <env assignments...>: the stub's output (ALLOC=... then the argv, one per line).
run() { local sc=$1; shift
  (cd "$T" && env -u PRINT_ARGV -u KV_OFFLOAD_GB -u EXTRA_ARGS -u PYTORCH_CUDA_ALLOC_CONF -u WSL_DISTRO_NAME \
    VLLM_API_KEY=k "$@" bash "$sc" 2>/dev/null </dev/null); }
# tier_of: every value passed to --kv-offloading-size, space-separated, "none" for no flag.
tier_of() { local v; v=$(run "$@" | sed -n -e '/^--kv-offloading-size$/{n;p;}' -e 's/^--kv-offloading-size=//p' | paste -sd ' ' -); echo "${v:-none}"; }
alloc_of() { run "$@" | sed -n 's/^ALLOC=//p'; }
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$what" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
NATIVE=1; grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && NATIVE=0
for sc in single-user/start_qwen.sh batch/start_qwen.sh single-user/alternative.sh; do
  echo "== $sc"
  check none "$(tier_of $sc)"                                               "KV_OFFLOAD_GB unset"
  check none "$(tier_of $sc KV_OFFLOAD_GB=0)"                               "KV_OFFLOAD_GB=0"
  check 12   "$(tier_of $sc KV_OFFLOAD_GB=12)"                              "KV_OFFLOAD_GB=12"
  check 4    "$(tier_of $sc KV_OFFLOAD_GB=12 EXTRA_ARGS='--kv-offloading-size 4')" "KV_OFFLOAD_GB=12, flag 4 in EXTRA_ARGS"
  check 4    "$(tier_of $sc KV_OFFLOAD_GB=12 EXTRA_ARGS='--kv_offloading_size 4')" "KV_OFFLOAD_GB=12, flag 4 spelled --kv_offloading_size"
  check none "$(tier_of $sc KV_OFFLOAD_GB=12 EXTRA_ARGS='--kv-transfer-config {}')" "KV_OFFLOAD_GB=12, --kv-transfer-config in EXTRA_ARGS"
  check 4    "$(tier_of $sc KV_OFFLOAD_GB=12 EXTRA_ARGS='--kv_offloading_size=4')" "explicit equals-form size wins exactly once"
  check none "$(tier_of $sc KV_OFFLOAD_GB=12 EXTRA_ARGS='--kv_transfer_config.kv_connector=Example')" "dotted connector overrides the tier knob"
  for flag in --kv-transfer-config --kv-offloading-size; do
    revision="--revision=example$flag"
    check 12 "$(tier_of $sc KV_OFFLOAD_GB=12 "EXTRA_ARGS=$revision")" "$flag in a revision preserves the requested tier"
    [ $NATIVE = 1 ] && check expandable_segments:True "$(alloc_of $sc "EXTRA_ARGS=$revision")" "$flag in a revision preserves the native allocator"
  done
  check expandable_segments:False "$(alloc_of $sc KV_OFFLOAD_GB=12)"        "allocator with KV_OFFLOAD_GB=12"
  [ $NATIVE = 1 ] && check expandable_segments:True "$(alloc_of $sc)"       "allocator with no tier (native)"
done
sc=single-user/alternative.sh
retention_of() { run "$sc" "$@" | sed -n '/^--prefix-cache-retention-interval$/{n;p;}'; }
check 0 "$(retention_of EXTRA_ARGS='--revision=example--kv-transfer-config')" "alternative: unrelated revision keeps boundaries-only retention"
check None "$(retention_of EXTRA_ARGS='--kv_transfer_config.kv_connector=Example')" "alternative: dotted connector selects dense retention"

# Standalone resolution also receives raw EXTRA_ARGS, before launcher normalization.
resolve_lsa() {
  env -u KV_OFFLOAD_GB REPO="$T" CTX=long SPEC=mtp LSA=1 PTH_FA=0 FA2_FP8KV=0 EXTRA_ARGS="$1" \
    bash "$T/resolve_config.sh" single >/dev/null 2>&1
  echo "$?"
}
check 0 "$(resolve_lsa '--revision=example--kv-offloading-size')" "standalone: a revision allows the LSA profile"
check 1 "$(resolve_lsa $'--revision main\t--kv_transfer_config.kv_connector=Example')" "standalone: a real dotted connector refuses the LSA profile"
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
