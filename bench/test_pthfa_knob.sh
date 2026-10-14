#!/bin/bash
# CPU test for PTH_FA (single-user/start_qwen.sh, resolve_config.sh): unset, 0 and empty give the same argv and
# stderr on every single-user arm; PTH_FA=1 on SPEC=mtp CTX=long puts the target AND the MTP drafter on the
# CUSTOM backend with an int8_per_token_head cache and changes nothing else in the argv; every other arm, a
# conflicting EXTRA_ARGS, a KV connector, a missing install and a value other than 0/1 are refused; batch mode
# warns and ignores it. Copies the checkout, without its own venv, to a temp dir and runs the launchers against a
# stub vllm (prints its argv) and a stub venv python (answers the install probe from PTHFA_STUB_INSTALLED, and
# passes anything else to python3). No GPU, no model, no network.
#
#   bash bench/test_pthfa_knob.sh
#   PTHFA_BASE_TREE=<checkout without pthfa> bash bench/test_pthfa_knob.sh
#       also compares every single-user and batch arm with PTH_FA unset against that tree's launchers, argv and
#       stderr, byte for byte (the negative control against the tree this branch is cut from)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# stage <src> <dst>: a copy of a checkout with the stub venv and an empty DFlash2 drafter dir.
stage() {
  mkdir -p "$2"
  (cd "$1" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$2"
  mkdir -p "$2/venv/bin" "$2/models/Qwen3.8-27B-DFlash2-W4A16"
  : > "$2/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"
  printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$2/venv/bin/vllm"
  printf '#!/bin/sh\ncase "$*" in *pth_flash_attn*) [ "${PTHFA_STUB_INSTALLED:-1}" = 1 ]; exit $? ;; esac\nexec python3 "$@"\n' > "$2/venv/bin/python"
  chmod +x "$2/venv/bin/vllm" "$2/venv/bin/python"
}
stage "$REPO" "$T/new"
# run <tree> <launcher> <env...>: argv on stdout (stub vllm), stderr in $T/err, exit code in $T/rc.
run() { local tree=$1 sc=$2; shift 2
  (cd "$tree" && env -u PRINT_ARGV -u PTH_FA -u KV_OFFLOAD_GB -u EXTRA_ARGS -u SPEC -u CTX -u PYTORCH_CUDA_ALLOC_CONF \
     -u WSL_DISTRO_NAME -u CUDA_HOME VLLM_API_KEY=k "$@" bash "$sc" 2>"$T/err" </dev/null); echo $? > "$T/rc"; }
rc() { cat "$T/rc"; }
# after <flag>: the argument that follows <flag> in the argv on stdin ("none" if absent; every value if repeated).
after() { local v; v=$(sed -n "/^$1\$/{n;p;}" | tr '\n' ' ' | sed 's/ $//'); echo "${v:-none}"; }
ARMS=("SPEC=mtp CTX=fast" "SPEC=mtp CTX=long" "SPEC=mtp CTX=huge" "SPEC=dflash2 CTX=fast" "SPEC=dflash2 CTX=long"
      "SPEC=dflash2 CTX=huge" "SPEC=off CTX=long")

echo "== unset, PTH_FA=0 and PTH_FA= give the same argv and stderr on every single-user arm"
for arm in "${ARMS[@]}"; do
  run "$T/new" single-user/start_qwen.sh $arm > "$T/a"; cp "$T/err" "$T/a.err"; ra=$(rc)
  run "$T/new" single-user/start_qwen.sh $arm PTH_FA=0 > "$T/b"; cp "$T/err" "$T/b.err"
  run "$T/new" single-user/start_qwen.sh $arm PTH_FA= > "$T/c"; cp "$T/err" "$T/c.err"
  same=1; cmp -s "$T/a" "$T/b" && cmp -s "$T/a" "$T/c" && cmp -s "$T/a.err" "$T/b.err" && cmp -s "$T/a.err" "$T/c.err" || same=0
  check "0 1" "$ra $same" "$arm: boots, and unset = 0 = empty"
done

echo "== PTH_FA=1, SPEC=mtp CTX=long"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long > "$T/off"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long PTH_FA=1 > "$T/on"; cp "$T/err" "$T/on.err"
check 0 "$(rc)" "boots"
check CUSTOM "$(after --attention-backend < "$T/on")" "--attention-backend CUSTOM, once"
check int8_per_token_head "$(after --kv-cache-dtype < "$T/on")" "--kv-cache-dtype int8_per_token_head, once"
check fp8 "$(after --kv-cache-dtype < "$T/off")" "(control: the arm without it runs fp8)"
spec=$(after --speculative-config < "$T/on")
check "mtp CUSTOM" "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d["method"], d.get("attention_backend"))' "$spec" 2>&1)" \
  "the drafter's backend is in the speculative config"
spec_off=$(after --speculative-config < "$T/off")
check "$spec_off" "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.pop("attention_backend"); print(json.dumps(d, separators=(",", ":")))' "$spec")" \
  "the speculative config is otherwise the arm's own"
# Everything but the backend, the cache dtype and the speculative config is the arm's own argv.
strip() { awk 'skip { skip = 0; next } /^--(attention-backend|kv-cache-dtype|speculative-config)$/ { skip = 1; next } { print }'; }
check same "$(cmp -s <(strip < "$T/on") <(strip < "$T/off") && echo same || echo differs)" \
  "every other argument unchanged (compilation config, graph sizes, max len, seqs)"
grep -q '^\[effective-config\] PTH_FA=1' "$T/on.err" && check 1 1 "the effective config names it" || check 1 0 "the effective config names it"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--tensor-parallel-size 2 --disable-custom-all-reduce" > "$T/tp"
check "0 2" "$(rc) $(after --tensor-parallel-size < "$T/tp")" "TP=2 through EXTRA_ARGS boots"

echo "== refused (exit 1, with the reason on stderr)"
refused() { local why=$1; shift; run "$T/new" single-user/start_qwen.sh "$@" >/dev/null
  if [ "$(rc)" = 1 ] && grep -q "PTH_FA" "$T/err"; then check 1 1 "$why"; else check "1 (PTH_FA named)" "$(rc)" "$why"; fi; }
refused "SPEC=mtp CTX=fast"                   SPEC=mtp CTX=fast PTH_FA=1
refused "SPEC=mtp CTX=huge"                   SPEC=mtp CTX=huge PTH_FA=1
refused "SPEC=dflash2 CTX=long"               SPEC=dflash2 CTX=long PTH_FA=1
refused "SPEC=off CTX=long"                   SPEC=off CTX=long PTH_FA=1
refused "EXTRA_ARGS --attention-backend"      SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--attention-backend TRITON_ATTN"
refused "EXTRA_ARGS --kv-cache-dtype"         SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--kv-cache-dtype=fp8"
refused "EXTRA_ARGS --speculative-config"     SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--speculative-config {}"
refused "KV_OFFLOAD_GB=8"                     SPEC=mtp CTX=long PTH_FA=1 KV_OFFLOAD_GB=8
refused "EXTRA_ARGS --kv-offloading-size"     SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--kv-offloading-size 4"
refused "no pth_flash_attn in the venv"       SPEC=mtp CTX=long PTH_FA=1 PTHFA_STUB_INSTALLED=0
refused "PTH_FA=true"                         SPEC=mtp CTX=long PTH_FA=true
refused "PTH_FA=2"                            SPEC=mtp CTX=long PTH_FA=2
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long PTH_FA=1 KV_OFFLOAD_GB=0 >/dev/null
check 0 "$(rc)" "KV_OFFLOAD_GB=0 is no tier: boots"

echo "== batch mode ignores it, and says so"
run "$T/new" batch/start_qwen.sh > "$T/b0"
run "$T/new" batch/start_qwen.sh PTH_FA=1 > "$T/b1"
check "0 same" "$(rc) $(cmp -s "$T/b0" "$T/b1" && echo same || echo differs)" "batch argv unchanged by PTH_FA=1"
grep -q "PTH_FA=1 is set but batch mode ignores it" "$T/err" && check 1 1 "batch warns" || check 1 0 "batch warns"

if [ -n "${PTHFA_BASE_TREE:-}" ]; then
  echo "== negative control: PTH_FA unset against $PTHFA_BASE_TREE, argv and stderr byte for byte"
  stage "$PTHFA_BASE_TREE" "$T/base"
  for arm in "${ARMS[@]}" "PREFIX_CACHE=1 VISION=1 SPEC=mtp CTX=long" "SPEC=mtp CTX=long EXTRA_ARGS=--tensor-parallel-size=2"; do
    run "$T/base" single-user/start_qwen.sh $arm > "$T/x"; sed "s|$T/base|TREE|g" "$T/err" > "$T/x.err"
    run "$T/new" single-user/start_qwen.sh $arm > "$T/y"; sed "s|$T/new|TREE|g" "$T/err" > "$T/y.err"
    sed -i "s|$T/base|TREE|g" "$T/x"; sed -i "s|$T/new|TREE|g" "$T/y"
    check "same same" "$(cmp -s "$T/x" "$T/y" && echo same || echo differs) $(cmp -s "$T/x.err" "$T/y.err" && echo same || echo differs)" "single $arm"
  done
  for kv in fp8 kvarn int4pth; do
    run "$T/base" batch/start_qwen.sh KV=$kv > "$T/x"; sed "s|$T/base|TREE|g" "$T/err" > "$T/x.err"
    run "$T/new" batch/start_qwen.sh KV=$kv > "$T/y"; sed "s|$T/new|TREE|g" "$T/err" > "$T/y.err"
    sed -i "s|$T/base|TREE|g" "$T/x"; sed -i "s|$T/new|TREE|g" "$T/y"
    check "same same" "$(cmp -s "$T/x" "$T/y" && echo same || echo differs) $(cmp -s "$T/x.err" "$T/y.err" && echo same || echo differs)" "batch KV=$kv"
  done
fi
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
