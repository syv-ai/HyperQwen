#!/bin/bash
# CPU test for FA2_FP8KV (single-user/start_qwen.sh): fp8 KV on FLASH_ATTN through the fa2/ plugin
# on SPEC=dflash2 CTX=long, where it defaults to 1. FA2_FP8KV=0 is the int8 + Triton arm and nothing
# is exported. On, it is FLASH_ATTN + fp8, the drafter selects FLASH_ATTN too, FA2_FP8KV=1 is exported
# (the plugin's gate) and the split-KV verify env is 0. Unset anywhere else resolves to 0 with no
# refusal and no change to the argv or the environment. A KV offload tier turns the default off with
# one note and refuses an explicit 1; any other SPEC/CTX with an explicit 1 and a missing plugin
# install (default or explicit) are refused, and so is an explicit 1 in the launchers that have no
# such arm. Copies the checkout to a temp dir and runs the launchers against stub venv/bin/vllm (prints
# FA2_FP8KV, VLLM_SPEC_DECODE_ATTN, then the argv) and stub venv/bin/python (stands for the install
# probe: exits $PROBE_RC and records its arguments). No GPU, no model, no network.
#
#   bash bench/test_fa2_fp8kv_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin" "$T/draft"
printf '#!/bin/sh\necho "ENV FA2_FP8KV=${FA2_FP8KV-unset} VLLM_SPEC_DECODE_ATTN=${VLLM_SPEC_DECODE_ATTN-unset}"\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"
printf '#!/bin/sh\nprintf "%%s\\n" "$@" > "%s/probe.args"\nexit ${PROBE_RC:-0}\n' "$T" > "$T/venv/bin/python"
chmod +x "$T/venv/bin/vllm" "$T/venv/bin/python"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
# run <launcher> <env assignments...>: stdout (the stub's output) in $T/out, stderr in $T/err, exit code echoed.
run() { local sc=$1; shift
  rm -f "$T/probe.args"
  (cd "$T" && env -u PRINT_ARGV -u KV_OFFLOAD_GB -u EXTRA_ARGS -u SPEC_ATTN -u FA2_FP8KV -u PROBE_RC -u DFLASH_TOKENS \
    -u CTX -u SPEC -u PYTORCH_CUDA_ALLOC_CONF -u WSL_DISTRO_NAME VLLM_API_KEY=k DRAFT="$T/draft" "$@" \
    bash "$sc" >"$T/out" 2>"$T/err" </dev/null); echo $?; }
env_line() { grep -m1 "^ENV " "$T/out"; }
arg_after() { sed -n "/^$1\$/{n;p;}" "$T/out"; }
err_has() { grep -qF -- "$1" "$T/err" && echo yes || echo no; }
LONG="SPEC=dflash2 CTX=long"; SC=single-user/start_qwen.sh

echo "== FA2_FP8KV=0: the int8 + Triton arm"
check 0 "$(run $SC $LONG FA2_FP8KV=0)" "FA2_FP8KV=0: starts"
check "ENV FA2_FP8KV=0 VLLM_SPEC_DECODE_ATTN=1" "$(env_line)" "only the user's own 0 reaches the server, split-KV verify on"
check "TRITON_ATTN" "$(arg_after --attention-backend)" "backend TRITON_ATTN"
check "int8_per_token_head" "$(arg_after --kv-cache-dtype)" "kv dtype int8_per_token_head"
check "no" "$(arg_after --speculative-config | grep -q attention_backend && echo yes || echo no)" "spec config has no attention_backend"
check "no" "$(test -e "$T/probe.args" && echo yes || echo no)" "no install probe when off"
check yes "$(err_has 'FA2_FP8KV=0 (explicit)')" "effective-config says 0, explicit"
check 0 "$(run $SC $LONG FA2_FP8KV=0 PROBE_RC=1)" "off ignores a failing probe"
check 0 "$(run $SC $LONG FA2_FP8KV=0 KV_OFFLOAD_GB=12)" "off starts with a tier"
check "12" "$(arg_after --kv-offloading-size)" "  and the tier is on"
check no "$(err_has 'FA2_FP8KV defaults to 0')" "  with no default note"

echo "== unset on SPEC=dflash2 CTX=long: default on"
check 0 "$(run $SC $LONG)" "unset: starts"; cp "$T/out" "$T/on.unset"
check yes "$(err_has 'FA2_FP8KV=1 (default)')" "effective-config says 1, from the default"
check 0 "$(run $SC $LONG FA2_FP8KV=1)" "FA2_FP8KV=1: starts"
check "yes" "$(cmp -s "$T/out" "$T/on.unset" && echo yes)" "unset and explicit 1 are byte-identical (env line and argv)"
check yes "$(err_has 'FA2_FP8KV=1 (explicit)')" "effective-config says 1, explicit"
check "ENV FA2_FP8KV=1 VLLM_SPEC_DECODE_ATTN=0" "$(env_line)" "FA2_FP8KV=1 exported, split-KV verify env 0"
check "FLASH_ATTN" "$(arg_after --attention-backend)" "backend FLASH_ATTN"
check "fp8" "$(arg_after --kv-cache-dtype)" "kv dtype fp8"
SC_CFG=$(arg_after --speculative-config)
check "yes" "$(printf %s "$SC_CFG" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("yes" if d.get("attention_backend")=="FLASH_ATTN" and d["method"]=="dflash" and d["num_speculative_tokens"]==7 else "no")')" "spec config is valid JSON with attention_backend FLASH_ATTN"
check "-c
import fa2_fp8kv_vllm.paths as p; p.check()" "$(cat "$T/probe.args")" "the probe is the contract's command"
check "131072" "$(arg_after --max-model-len)" "MAX_LEN default unchanged"
check "--kv-cache-memory=5583457484" "$(grep -m1 -- '--kv-cache-memory' "$T/out")" "KV_MEM default unchanged"
check 0 "$(run $SC $LONG FA2_FP8KV=1 KV_OFFLOAD_GB=0)" "KV_OFFLOAD_GB=0 is not a tier"
check 0 "$(run $SC $LONG KV_OFFLOAD_GB=0)" "  nor is it with the default"
check "ENV FA2_FP8KV=1 VLLM_SPEC_DECODE_ATTN=0" "$(env_line)" "  and the default stays on"
check 0 "$(run $SC $LONG SPEC_ATTN=1)" "SPEC_ATTN=1 starts, with a note"
check yes "$(err_has 'SPEC_ATTN=1 has no effect')" "  and the note says so"
check "ENV FA2_FP8KV=1 VLLM_SPEC_DECODE_ATTN=0" "$(env_line)" "  and the verify env stays 0"
check 0 "$(run $SC $LONG FA2_FP8KV=)" "empty FA2_FP8KV starts"
check "ENV FA2_FP8KV=1 VLLM_SPEC_DECODE_ATTN=0" "$(env_line)" "  and counts as unset: default on"

echo "== KV offload tier: the default resolves to 0 with one note, int8 + Triton"
for tier in "KV_OFFLOAD_GB=12" "EXTRA_ARGS=--kv-offloading-size 4" "EXTRA_ARGS=--kv-transfer-config {}"; do
  check 0 "$(run $SC $LONG "$tier")" "$tier: starts"
  check "ENV FA2_FP8KV=unset VLLM_SPEC_DECODE_ATTN=1" "$(env_line)" "  nothing exported, split-KV verify on"
  check "TRITON_ATTN" "$(arg_after --attention-backend)" "  backend TRITON_ATTN"
  check "int8_per_token_head" "$(arg_after --kv-cache-dtype)" "  kv dtype int8_per_token_head"
  check 1 "$(grep -c 'FA2_FP8KV defaults to 0' "$T/err")" "  one note"
  check yes "$(err_has 'LBNHC KV layout does not fit the OffloadingConnector')" "  and it says why"
  check yes "$(err_has 'FA2_FP8KV=0 (default, off:')" "  effective-config says 0, from the default"
  check "no" "$(test -e "$T/probe.args" && echo yes || echo no)" "  no install probe"
done
check 0 "$(run $SC $LONG KV_OFFLOAD_GB=12 PROBE_RC=1)" "a tier ignores a missing plugin: the default is off"

echo "== unset elsewhere: 0, no refusal, nothing exported"
for cfg in "SPEC=mtp CTX=long" "SPEC=mtp CTX=fast" "SPEC=dflash2 CTX=fast" "SPEC=dflash2 CTX=huge" "SPEC=off CTX=long" ""; do
  check 0 "$(run $SC $cfg PROBE_RC=1)" "${cfg:-defaults}: starts, a missing plugin is irrelevant"
  check "unset" "$(env_line | sed 's/^ENV FA2_FP8KV=\([^ ]*\) .*/\1/')" "  FA2_FP8KV stays unset"
  check yes "$(err_has 'FA2_FP8KV=0 (default)')" "  effective-config says 0, from the default"
  check no "$(err_has 'FA2_FP8KV defaults to 0')" "  no note"
  cp "$T/out" "$T/e.unset"; run $SC $cfg FA2_FP8KV=0 PROBE_RC=1 >/dev/null; sed -i "/^ENV /d" "$T/out" "$T/e.unset"
  check yes "$(cmp -s "$T/out" "$T/e.unset" && echo yes)" "  argv equals FA2_FP8KV=0"
done
for sc in batch/start_qwen.sh single-user/alternative.sh; do
  check 0 "$(run $sc PROBE_RC=1)" "$sc: unset starts"
  check "unset" "$(env_line | sed 's/^ENV FA2_FP8KV=\([^ ]*\) .*/\1/')" "  FA2_FP8KV stays unset"
done

echo "== refusals"
refused() { local what=$1 msg=$2; shift 2
  check 1 "$(run "$@")" "$what: refused"; check yes "$(err_has "$msg")" "  with its reason"; }
refused "SPEC=mtp CTX=long"       "runs only on SPEC=dflash2 CTX=long (got SPEC=mtp CTX=long)"  $SC SPEC=mtp CTX=long FA2_FP8KV=1
refused "default SPEC and CTX"    "runs only on SPEC=dflash2 CTX=long (got SPEC=mtp CTX=fast)"  $SC FA2_FP8KV=1
refused "SPEC=dflash2 CTX=fast"   "runs only on SPEC=dflash2 CTX=long (got SPEC=dflash2 CTX=fast)" $SC SPEC=dflash2 CTX=fast FA2_FP8KV=1
refused "SPEC=dflash2 CTX=huge"   "runs only on SPEC=dflash2 CTX=long (got SPEC=dflash2 CTX=huge)" $SC SPEC=dflash2 CTX=huge FA2_FP8KV=1
refused "SPEC=off CTX=long"       "runs only on SPEC=dflash2 CTX=long (got SPEC=off CTX=long)"  $SC SPEC=off CTX=long FA2_FP8KV=1
refused "KV_OFFLOAD_GB=12"        "KV_OFFLOAD_GB=12: the FA2 backend supports only the LBNHC KV layout" $SC $LONG FA2_FP8KV=1 KV_OFFLOAD_GB=12
refused "--kv-offloading-size"    "KV connector in EXTRA_ARGS" $SC $LONG FA2_FP8KV=1 EXTRA_ARGS="--kv-offloading-size 4"
refused "--kv-transfer-config"    "KV connector in EXTRA_ARGS" $SC $LONG FA2_FP8KV=1 EXTRA_ARGS="--kv-transfer-config {}"
refused "FA2_FP8KV=true"          "FA2_FP8KV=true: this knob is read as 1 = on" $SC $LONG FA2_FP8KV=true
refused "plugin not installed"    "bash fa2/install.sh" $SC $LONG FA2_FP8KV=1 PROBE_RC=1
check "-c
import fa2_fp8kv_vllm.paths as p; p.check()" "$(cat "$T/probe.args")" "  the failing probe was the contract's command"
check yes "$(err_has 'FA2_FP8KV=0')" "  and the hint offers FA2_FP8KV=0"
mv "$T/venv/bin/python" "$T/venv/bin/python.off"
refused "no venv python"          "bash fa2/install.sh" $SC $LONG FA2_FP8KV=1
mv "$T/venv/bin/python.off" "$T/venv/bin/python"

echo "== default on, plugin not installed: refused, no silent fallback"
refused "unset, probe fails"      "bash fa2/install.sh, or set FA2_FP8KV=0 for the int8 + Triton arm" $SC $LONG PROBE_RC=1
check yes "$(err_has 'FA2_FP8KV=1 (default)')" "  and it says the 1 came from the default"
check "" "$(cat "$T/out")" "  and nothing booted"
mv "$T/venv/bin/python" "$T/venv/bin/python.off"
refused "unset, no venv python"   "bash fa2/install.sh, or set FA2_FP8KV=0 for the int8 + Triton arm" $SC $LONG
mv "$T/venv/bin/python.off" "$T/venv/bin/python"

echo "== launchers with no FA2 arm"
for sc in batch/start_qwen.sh single-user/alternative.sh; do
  check 0 "$(run $sc FA2_FP8KV=0)" "$sc: FA2_FP8KV=0 starts"
  refused "$sc FA2_FP8KV=1" "FA2_FP8KV=1 runs only on single-user/start_qwen.sh SPEC=dflash2 CTX=long" $sc FA2_FP8KV=1
done
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
