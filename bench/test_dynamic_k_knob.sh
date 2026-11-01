#!/bin/bash
# CPU test for DYNAMIC_K (single-user/start_qwen.sh): set, it lands in the --speculative-config JSON as
# num_speculative_tokens_per_batch_size with every other key unchanged, under SPEC=dflash2 and SPEC=mtp; unset or
# empty changes nothing; SPEC=off ignores it with a warning; CTX=huge with prefix caching on (PREFIX_CACHE unset or 1) refuses it.
# Copies the checkout to a temp dir and runs the launcher against a stub vllm that prints its argv. No GPU, no
# model, no network.
#
#   bash bench/test_dynamic_k_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
K='[[1,3,7],[4,512,3]]'; KEY=num_speculative_tokens_per_batch_size; D=DRAFT=/app/models/stub-drafter
# launch <env assignments...>: the stub's argv on stdout; the launcher's stderr in $T/err, its exit code in $T/rc.
launch() { (cd "$T" && env -u DYNAMIC_K -u EXTRA_ARGS -u PRINT_ARGV -u PREFIX_CACHE -u DFLASH_TOKENS -u DRAFT_METHOD \
    VLLM_API_KEY=k "$@" bash single-user/start_qwen.sh 2>"$T/err" </dev/null); echo $? >"$T/rc"; }
# cfg <env assignments...>: the --speculative-config value, empty when there is none.
cfg() { launch "$@" | sed -n '/^--speculative-config$/{n;p;}'; }
# field <key>: that key of the JSON on stdin, compact; "absent" without it, "invalid" if stdin is not JSON.
field() { python3 -I -c 'import json, sys
try: d = json.loads(sys.stdin.read())
except ValueError: print("invalid"); sys.exit()
k = sys.argv[1]; print(json.dumps(d[k], separators=(",", ":")) if k in d else "absent")' "$1"; }
# same <without> <with>: True when the two JSON configs are equal once the second loses $KEY.
same() { printf '%s\n%s\n' "$1" "$2" | python3 -I -c 'import json, sys
a, b = map(json.loads, sys.stdin.read().splitlines()); b.pop(sys.argv[1], None); print(a == b)' "$KEY" 2>/dev/null; }
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$what" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }

echo "== SPEC=dflash2 CTX=fast"
c0=$(cfg SPEC=dflash2 CTX=fast $D)
check absent "$(field $KEY <<<"$c0")"                        "DYNAMIC_K unset: $KEY"
check yes "$([ "$(launch SPEC=dflash2 CTX=fast $D)" = "$(launch SPEC=dflash2 CTX=fast $D DYNAMIC_K=)" ] && echo yes || echo no)" \
  "DYNAMIC_K empty: argv as unset"
c1=$(cfg SPEC=dflash2 CTX=fast $D DYNAMIC_K="$K")
check "$K"     "$(field $KEY <<<"$c1")"                      "DYNAMIC_K=$K: $KEY"
check '"dflash"' "$(field method <<<"$c1")"                  "DYNAMIC_K=$K: method"
check 7        "$(field num_speculative_tokens <<<"$c1")"    "DYNAMIC_K=$K: num_speculative_tokens"
check True     "$(same "$c0" "$c1")"                         "DYNAMIC_K=$K: every other key as unset"

echo "== SPEC=mtp"
m0=$(cfg SPEC=mtp CTX=fast)
m1=$(cfg SPEC=mtp CTX=fast DYNAMIC_K="$K")
check "$K"     "$(field $KEY <<<"$m1")"                      "DYNAMIC_K=$K: $KEY"
check '"mtp"'  "$(field method <<<"$m1")"                    "DYNAMIC_K=$K: method"
check True     "$(same "$m0" "$m1")"                         "DYNAMIC_K=$K: every other key as unset"

echo "== SPEC=off"
out=$(launch SPEC=off CTX=fast DYNAMIC_K="$K")
check 0  "$(cat "$T/rc")"                                    "DYNAMIC_K=$K: starts"
check no "$(grep -qx -- --speculative-config <<<"$out" && echo yes || echo no)" "DYNAMIC_K=$K: --speculative-config in argv"
check yes "$(grep -q DYNAMIC_K "$T/err" && echo yes || echo no)" "DYNAMIC_K=$K: stderr names DYNAMIC_K"

echo "== SPEC=dflash2 CTX=huge"
out=$(launch SPEC=dflash2 CTX=huge PREFIX_CACHE=1 $D DYNAMIC_K="$K")
check 1  "$(cat "$T/rc")"                                    "PREFIX_CACHE=1 DYNAMIC_K=$K: exit code"
check yes "$(grep -q 'DYNAMIC_K refused' "$T/err" && echo yes || echo no)" "PREFIX_CACHE=1 DYNAMIC_K=$K: stderr names DYNAMIC_K"
check ""  "$out"                                             "PREFIX_CACHE=1 DYNAMIC_K=$K: vllm not started"
out=$(launch SPEC=dflash2 CTX=huge $D DYNAMIC_K="$K")
check 1  "$(cat "$T/rc")"                                    "PREFIX_CACHE unset DYNAMIC_K=$K: exit code"
check ""  "$out"                                             "PREFIX_CACHE unset DYNAMIC_K=$K: vllm not started"
check "$K" "$(cfg SPEC=dflash2 CTX=huge PREFIX_CACHE=0 $D DYNAMIC_K="$K" | field $KEY)" "PREFIX_CACHE=0, DYNAMIC_K=$K: $KEY"
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
