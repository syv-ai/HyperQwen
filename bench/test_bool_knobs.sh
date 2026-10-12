#!/bin/bash
# CPU test for #310. QWEN_BOOL_KNOBS (launcher_common.sh) must name exactly the boolean knobs the series and the
# KVarN patches register as os.environ.get(X, ...) == "1", plus FA2_FP8KV (the fa2/ plugin's gate, read in
# Python outside the patches), and each launcher must refuse to start when one of them
# is set to anything but 0 or 1 ("true" would silently mean off). Copies the checkout, without its own venv, to a
# temp dir and runs the three launchers with PRINT_ARGV=1 against a stub vllm. No GPU, no model, no network.
#
#   bash bench/test_bool_knobs.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }

echo "== the knob list matches the patches"
want=$(grep -hE '^\+' "$REPO"/patches/*.patch "$REPO"/kvarn/*.patch \
  | grep -oE '"(VLLM|KVARN)_[A-Z0-9_]+"[^)]*\)[^=]*== *"1"' | grep -oE '^"[A-Z0-9_]+"' | tr -d '"' | { cat; echo FA2_FP8KV; } | sort -u | tr '\n' ' ')
have=$(bash -c 'source "$1/launcher_common.sh"; printf "%s\n" $QWEN_BOOL_KNOBS' _ "$REPO" | sort -u | tr '\n' ' ')
check "$want" "$have" "QWEN_BOOL_KNOBS names every knob a patch reads as == \"1\", and FA2_FP8KV ($(echo $want | wc -w) knobs)"

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
# rc_of <launcher> <env assignments...>: the launcher's exit code under PRINT_ARGV=1; stderr kept in $T/err.
rc_of() { local sc=$1; shift
  (cd "$T" && env -u PRINT_ARGV -u VLLM_DFLASH2_LOOKUP -u KVARN_FUSED_DECODE VLLM_API_KEY=k PRINT_ARGV=1 "$@" \
     bash "$sc" >/dev/null 2>"$T/err" </dev/null); echo $?; }
for sc in single-user/start_qwen.sh batch/start_qwen.sh single-user/alternative.sh; do
  echo "== $sc"
  check 0 "$(rc_of "$sc")"                                            "no knob set: starts"
  check 0 "$(rc_of "$sc" VLLM_DFLASH2_LOOKUP=1)"                      "VLLM_DFLASH2_LOOKUP=1: starts"
  check 0 "$(rc_of "$sc" VLLM_DFLASH2_LOOKUP=0 KVARN_FUSED_DECODE=)"  "=0 and empty: start"
  check 1 "$(rc_of "$sc" VLLM_DFLASH2_LOOKUP=true)"                   "VLLM_DFLASH2_LOOKUP=true: refused"
  grep -q "VLLM_DFLASH2_LOOKUP=true" "$T/err" && check 1 1 "the refusal names the knob" || check 1 0 "the refusal names the knob"
  check 1 "$(rc_of "$sc" KVARN_FUSED_DECODE=yes)"                     "KVARN_FUSED_DECODE=yes: refused"
  check 1 "$(rc_of "$sc" FA2_FP8KV=true)"                             "FA2_FP8KV=true: refused"
done
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
