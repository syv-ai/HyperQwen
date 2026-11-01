#!/bin/bash
# CPU test for PTH_FA (single-user/start_qwen.sh, resolve_config.sh): unset, 0 and empty give the same argv,
# VLLM_* environment and stderr on every single-user arm; PTH_FA=1 on SPEC=mtp CTX=long puts the target AND the
# MTP drafter on the CUSTOM backend with an int8_per_token_head cache, and on SPEC=dflash2 CTX=long the target AND
# the DFlash2 drafter (the arm's speculative config plus "attention_backend":"CUSTOM", no kv_cache_dtype); neither
# changes anything else in the argv or the VLLM_* environment; every other arm, a conflicting EXTRA_ARGS, a KV
# connector, a missing install, a value other than 0/1, FA2_FP8KV=1 alongside it and, on dflash2, a drafting setup
# other than the measured one (DRAFT_METHOD, DFLASH_TOKENS, LOOKUP, DYNAMIC_K) are refused, each with its own
# message; batch mode warns and ignores it. On a compute capability 8.6 host (a stub nvidia-smi), where
# SPEC=dflash2 CTX=long defaults to PTH_FA=1 with the package installed and the measured setup: an unset or empty
# PTH_FA gives the argv and VLLM_* environment of an explicit PTH_FA=1; PTH_FA=0, a missing install (one note),
# an explicit FA2_FP8KV and every conflicting or unmeasured setting keep the arm's own default, the FA2 one, with
# the argv, environment and stderr of PTH_FA=0 (no refusal); an explicit FA2_FP8KV=1 with PTH_FA=1 is refused
# before either plugin is probed; on 8.0, 8.9, 9.0 or no answer, and on every other arm, nothing changes. On
# SPEC=mtp CTX=long the default works the same way with that arm's gates (no explicit FA2_FP8KV, no KV tier or
# connector, no backend flags in EXTRA_ARGS, no DYNAMIC_K, no VISION=1, DRAFT_TOKENS unset or 3), against fp8 on
# FlashInfer, and with PTHFA_BASE_TREE every default-off mtp launch is the base tree's byte for byte. Every
# run gets a stub nvidia-smi that answers nothing unless it passes another. Copies
# the checkout, without its own venv, to a temp dir and runs the launchers against a stub vllm (prints its argv,
# then "@@ENV" and its VLLM_* environment) and a stub venv python (answers the install probes from
# PTHFA_STUB_INSTALLED and FA2_STUB_INSTALLED, and passes anything else to python3). No GPU, no model, no network.
#
#   bash bench/test_pthfa_knob.sh
#   PTHFA_BASE_TREE=<checkout without pthfa> bash bench/test_pthfa_knob.sh
#       also compares every single-user and batch arm with PTH_FA unset against that tree's launchers, argv,
#       VLLM_* environment and stderr, byte for byte (the negative control against the tree this branch is cut from)
#   PTHFA_PARENT_TREE=<checkout with an earlier PTH_FA knob> bash bench/test_pthfa_knob.sh
#       also compares SPEC=mtp CTX=long PTH_FA=1 against that tree's launcher: argv and VLLM_* environment byte for
#       byte, stderr byte for byte except the [effective-config] PTH_FA=1 line, whose wording names the models here;
#       and, if that tree accepts SPEC=dflash2 CTX=long PTH_FA=1 (target only), the dflash2 argv and VLLM_*
#       environment against it: the same but for ,"attention_backend":"CUSTOM" at the end of the speculative config
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
  printf '#!/bin/sh\nprintf "%%s\\n" "$@"\necho @@ENV\nenv | grep "^VLLM_" | sort\n' > "$2/venv/bin/vllm"
  printf '#!/bin/sh\ncase "$*" in *pth_flash_attn*) [ "${PTHFA_STUB_INSTALLED:-1}" = 1 ]; exit $? ;; *fa2_fp8kv*) [ "${FA2_STUB_INSTALLED:-1}" = 1 ]; exit $? ;; esac\nexec python3 "$@"\n' > "$2/venv/bin/python"
  chmod +x "$2/venv/bin/vllm" "$2/venv/bin/python"
}
stage "$REPO" "$T/new"
# smi <cap>: PATH=<dir>:$PATH, where <dir>/nvidia-smi answers <cap> to resolve_config.sh's compute_cap query;
# "none" answers nothing and fails, as on a host without a GPU driver.
smi() { mkdir -p "$T/smi-$1"
  if [ "$1" = none ]; then printf '#!/bin/sh\nexit 9\n'; else printf '#!/bin/sh\necho %s\n' "$1"; fi > "$T/smi-$1/nvidia-smi"
  chmod +x "$T/smi-$1/nvidia-smi"; echo "PATH=$T/smi-$1:$PATH"; }
SNONE=$(smi none)
# run <tree> <launcher> <env...>: argv and VLLM_* environment on stdout (stub vllm), stderr in $T/err, exit code
# in $T/rc. The host's own nvidia-smi is never asked: every run gets the "none" stub unless it passes another.
# FA2_FP8KV=0: an unset knob resolves from the GPU (1 on SPEC=dflash2 CTX=long on compute capability 8.6),
# so every run pins it but those that pass FA2_FP8KV= with a stub nvidia-smi (the 8.6 block below).
run() { local tree=$1 sc=$2; shift 2
  (cd "$tree" && env -u PRINT_ARGV -u PTH_FA -u KV_OFFLOAD_GB -u EXTRA_ARGS -u SPEC -u CTX -u PYTORCH_CUDA_ALLOC_CONF \
     -u WSL_DISTRO_NAME -u CUDA_HOME -u FA2_FP8KV -u DRAFT_METHOD -u DFLASH_TOKENS -u LOOKUP -u VISION -u DYNAMIC_K \
     -u DRAFT -u DRAFT_TOKENS -u CUDA_VISIBLE_DEVICES \
     VLLM_API_KEY=k FA2_FP8KV=0 "$SNONE" "$@" bash "$sc" 2>"$T/err" </dev/null); echo $? > "$T/rc"; }
rc() { cat "$T/rc"; }
# after <flag>: the argument that follows <flag> in the argv on stdin ("none" if absent; every value if repeated).
after() { local v; v=$(sed -n "/^$1\$/{n;p;}" | tr '\n' ' ' | sed 's/ $//'); echo "${v:-none}"; }
ARMS=("SPEC=mtp CTX=fast" "SPEC=mtp CTX=long" "SPEC=mtp CTX=huge" "SPEC=dflash2 CTX=fast" "SPEC=dflash2 CTX=long"
      "SPEC=dflash2 CTX=huge" "SPEC=off CTX=long")

echo "== unset, PTH_FA=0 and PTH_FA= give the same argv, VLLM_* environment and stderr on every single-user arm"
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
  "every other argument and VLLM_* variable unchanged (compilation config, graph sizes, max len, seqs)"
grep -q '^\[effective-config\] PTH_FA=1 .*target and MTP drafter' "$T/on.err" && check 1 1 "the effective config names it" \
  || check 1 0 "the effective config names it"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--tensor-parallel-size 2 --disable-custom-all-reduce" > "$T/tp"
check "0 2" "$(rc) $(after --tensor-parallel-size < "$T/tp")" "TP=2 through EXTRA_ARGS boots"

echo "== PTH_FA=1, SPEC=dflash2 CTX=long: the target and the DFlash2 drafter"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long > "$T/doff"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long PTH_FA=1 > "$T/don"; cp "$T/err" "$T/don.err"
check 0 "$(rc)" "boots"
check CUSTOM "$(after --attention-backend < "$T/don")" "--attention-backend CUSTOM, once"
check int8_per_token_head "$(after --kv-cache-dtype < "$T/don")" "--kv-cache-dtype int8_per_token_head, once"
check "TRITON_ATTN int8_per_token_head" "$(after --attention-backend < "$T/doff") $(after --kv-cache-dtype < "$T/doff")" \
  "(control: the arm without it runs TRITON_ATTN on the same cache dtype)"
dspec=$(after --speculative-config < "$T/don")
check "dflash CUSTOM" "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d["method"], d.get("attention_backend"))' "$dspec" 2>&1)" \
  "the drafter's backend is in the speculative config"
check "None" "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("kv_cache_dtype"))' "$dspec" 2>&1)" \
  "no kv_cache_dtype in it: the drafter's cache keeps the target's int8_per_token_head"
dspec_off=$(after --speculative-config < "$T/doff")
check "$dspec_off" "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.pop("attention_backend"); print(json.dumps(d, separators=(",", ":")))' "$dspec")" \
  "the speculative config is otherwise the arm's own"
check "${dspec_off%\}}"',"attention_backend":"CUSTOM"}' "$dspec" \
  "the speculative config is the arm's own with ,\"attention_backend\":\"CUSTOM\" appended, byte for byte"
check "dflash None" "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d["method"], d.get("attention_backend"))' "$dspec_off" 2>&1)" \
  "(control: the arm's own speculative config names no drafter backend)"
check "1 1" "$(sed -n 's/^VLLM_SPEC_DECODE_ATTN=//p' < "$T/don") $(sed -n 's/^VLLM_SPEC_DECODE_ATTN=//p' < "$T/doff")" \
  "VLLM_SPEC_DECODE_ATTN stays the arm's 1"
check same "$(cmp -s <(strip < "$T/don") <(strip < "$T/doff") && echo same || echo differs)" \
  "every other argument and VLLM_* variable unchanged (compilation config, graph sizes, max len, seqs)"
grep -q '^\[effective-config\] PTH_FA=1 .*: target and DFlash2 drafter)$' "$T/don.err" \
  && check 1 1 "the effective config names it, target and drafter" || check 1 0 "the effective config names it, target and drafter"
grep -q '^\[start_qwen\] PTH_FA=1: .*target and DFlash2$' "$T/don.err" && grep -q '^  drafter on CUSTOM: ' "$T/don.err" \
  && check 1 1 "the launcher says target and drafter on CUSTOM" || check 1 0 "the launcher says target and drafter on CUSTOM"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long PTH_FA=1 EXTRA_ARGS="--tensor-parallel-size 2 --disable-custom-all-reduce" > "$T/tp"
check "0 2" "$(rc) $(after --tensor-parallel-size < "$T/tp")" "TP=2 through EXTRA_ARGS boots"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long PTH_FA=1 DRAFT_METHOD=dflash DFLASH_TOKENS=7 LOOKUP=1 FA2_FP8KV=0 DYNAMIC_K= > "$T/dexp"
check "0 same" "$(rc) $(cmp -s "$T/dexp" "$T/don" && echo same || echo differs)" \
  "the measured drafting setup spelled out (DRAFT_METHOD=dflash DFLASH_TOKENS=7 LOOKUP=1, FA2_FP8KV=0, DYNAMIC_K empty) boots, same argv"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long PTH_FA=1 VISION=1 > /dev/null
check 0 "$(rc)" "VISION=1 is accepted (untested on this arm, gotcha 63)"

echo "== refused (exit 1, with the refusal's own message on stderr)"
# refused <why> <message fragment> <env...>: exit 1 and the fragment (fixed string) on stderr. The fragment is the
# refusal's own text, never just "PTH_FA": the effective-config block prints PTH_FA=1 before any refusal.
refused() { local why=$1 msg=$2; shift 2; run "$T/new" single-user/start_qwen.sh "$@" >/dev/null
  if [ "$(rc)" = 1 ] && grep -qF -- "$msg" "$T/err"; then check 1 1 "$why"; else check "1 ($msg)" "$(rc)" "$why"; fi; }
FA2_MTP="FA2_FP8KV=1 runs only on SPEC=dflash2 CTX=long"
GATE="is a SPEC=mtp CTX=long or SPEC=dflash2 CTX=long setting"
XARGS="sets --attention-backend, --kv-cache-dtype and the speculative config itself"
KVC="PTH_FA=1 with a KV connector"
INST="PTH_FA=1 needs the pth_flash_attn package"
FA2="PTH_FA=1 and FA2_FP8KV=1 both pick the dflash2 backend"
refused "SPEC=mtp CTX=fast"                   "$GATE"  SPEC=mtp CTX=fast PTH_FA=1
refused "SPEC=mtp CTX=huge"                   "$GATE"  SPEC=mtp CTX=huge PTH_FA=1
refused "SPEC=dflash2 CTX=fast"               "$GATE"  SPEC=dflash2 CTX=fast PTH_FA=1
refused "SPEC=dflash2 CTX=huge"               "$GATE"  SPEC=dflash2 CTX=huge PTH_FA=1
refused "SPEC=off CTX=long"                   "$GATE"  SPEC=off CTX=long PTH_FA=1
refused "EXTRA_ARGS --attention-backend"      "$XARGS" SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--attention-backend TRITON_ATTN"
refused "EXTRA_ARGS --kv-cache-dtype"         "$XARGS" SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--kv-cache-dtype=fp8"
refused "EXTRA_ARGS --speculative-config"     "$XARGS" SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--speculative-config {}"
refused "KV_OFFLOAD_GB=8"                     "$KVC"   SPEC=mtp CTX=long PTH_FA=1 KV_OFFLOAD_GB=8
refused "EXTRA_ARGS --kv-offloading-size"     "$KVC"   SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS="--kv-offloading-size 4"
refused "no pth_flash_attn in the venv"       "$INST"  SPEC=mtp CTX=long PTH_FA=1 PTHFA_STUB_INSTALLED=0
refused "FA2_FP8KV=1 alongside it"            "$FA2_MTP" SPEC=mtp CTX=long PTH_FA=1 FA2_FP8KV=1
refused "dflash2: EXTRA_ARGS --attention-backend"  "$XARGS" SPEC=dflash2 CTX=long PTH_FA=1 EXTRA_ARGS="--attention-backend TRITON_ATTN"
refused "dflash2: EXTRA_ARGS --kv-cache-dtype"     "$XARGS" SPEC=dflash2 CTX=long PTH_FA=1 EXTRA_ARGS="--kv-cache-dtype=int8_per_token_head"
refused "dflash2: EXTRA_ARGS --speculative-config" "$XARGS" SPEC=dflash2 CTX=long PTH_FA=1 EXTRA_ARGS="--speculative-config {}"
refused "dflash2: KV_OFFLOAD_GB=8"                 "$KVC"   SPEC=dflash2 CTX=long PTH_FA=1 KV_OFFLOAD_GB=8
refused "dflash2: EXTRA_ARGS --kv-transfer-config" "$KVC"   SPEC=dflash2 CTX=long PTH_FA=1 EXTRA_ARGS="--kv-transfer-config {}"
refused "dflash2: no pth_flash_attn in the venv"   "$INST"  SPEC=dflash2 CTX=long PTH_FA=1 PTHFA_STUB_INSTALLED=0
refused "dflash2: FA2_FP8KV=1 alongside it"        "$FA2"   SPEC=dflash2 CTX=long PTH_FA=1 FA2_FP8KV=1
refused "dflash2: DRAFT_METHOD=dspark" "DRAFT_METHOD=dspark is a" SPEC=dflash2 CTX=long PTH_FA=1 DRAFT_METHOD=dspark
refused "dflash2: DFLASH_TOKENS=15" "is measured at DFLASH_TOKENS=7 (an 8-token verify block); DFLASH_TOKENS=15" \
  SPEC=dflash2 CTX=long PTH_FA=1 DFLASH_TOKENS=15
refused "dflash2: DFLASH_TOKENS=3"  "is measured at DFLASH_TOKENS=7 (an 8-token verify block); DFLASH_TOKENS=3" \
  SPEC=dflash2 CTX=long PTH_FA=1 DFLASH_TOKENS=3
refused "dflash2: LOOKUP=0"         "LOOKUP=0 is untested on this" SPEC=dflash2 CTX=long PTH_FA=1 LOOKUP=0
refused "dflash2: DYNAMIC_K"        "measured with a fixed verify block of 8 tokens; DYNAMIC_K=[[1,3,7],[4,512,3]]" \
  SPEC=dflash2 CTX=long PTH_FA=1 DYNAMIC_K='[[1,3,7],[4,512,3]]'
refused "PTH_FA=true"                         "PTH_FA=true: set it to 0 or 1" SPEC=mtp CTX=long PTH_FA=true
refused "PTH_FA=2"                            "PTH_FA=2: set it to 0 or 1"    SPEC=mtp CTX=long PTH_FA=2
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long PTH_FA=1 KV_OFFLOAD_GB=0 >/dev/null
check 0 "$(rc)" "KV_OFFLOAD_GB=0 is no tier: boots"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long PTH_FA=1 KV_OFFLOAD_GB=0 >/dev/null
check 0 "$(rc)" "dflash2: KV_OFFLOAD_GB=0 is no tier: boots"

echo "== compute capability 8.6 host (stub nvidia-smi), FA2_FP8KV unset: PTH_FA defaults to 1 on SPEC=dflash2 CTX=long"
S86=$(smi 8.6); S89=$(smi 8.9); S80=$(smi 8.0); S90=$(smi 9.0)
OVR="PTH_FA=1 overrides the 8.6 default"
NOTE="PTH_FA defaults to 1 on SPEC=dflash2 CTX=long on compute capability 8.6, but the pth_flash_attn package is not in the venv"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV= PTH_FA=1 "$S86" > "$T/f1"; cp "$T/err" "$T/f1.err"
check "0 same" "$(rc) $(cmp -s "$T/f1" "$T/don" && echo same || echo differs)" \
  "PTH_FA=1 boots with the argv and VLLM_* environment of PTH_FA=1 FA2_FP8KV=0"
grep -qF "[effective-config] CTX=long SPEC=dflash2 FA2_FP8KV=0 ($OVR)" "$T/f1.err" \
  && check 1 1 "PTH_FA=1: the effective config says it overrides the default" || check 1 0 "PTH_FA=1: the effective config says it overrides the default"
grep -qF "[start_qwen] PTH_FA=1: FA2_FP8KV=0, $OVR" "$T/f1.err" \
  && check 1 1 "PTH_FA=1: the launcher says so" || check 1 0 "PTH_FA=1: the launcher says so"
grep -qF "[effective-config] PTH_FA=1 (explicit; " "$T/f1.err" \
  && check 1 1 "PTH_FA=1: the effective config says PTH_FA=1 (explicit)" || check 1 0 "PTH_FA=1: the effective config says PTH_FA=1 (explicit)"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV= "$S86" > "$T/d1"; cp "$T/err" "$T/d1.err"
check "0 same" "$(rc) $(cmp -s "$T/d1" "$T/f1" && echo same || echo differs)" \
  "no knob: the argv and VLLM_* environment of PTH_FA=1 (target and drafter on CUSTOM)"
check same "$(cmp -s <(sed 's/^\(\[effective-config\] PTH_FA=1 (\)explicit; /\1default; /' "$T/f1.err") "$T/d1.err" && echo same || echo differs)" \
  "no knob: the stderr of PTH_FA=1 but for (default) in its effective-config line"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV= PTH_FA= "$S86" > "$T/d1e"; cp "$T/err" "$T/d1e.err"
check "same same" "$(cmp -s "$T/d1e" "$T/d1" && echo same || echo differs) $(cmp -s "$T/d1e.err" "$T/d1.err" && echo same || echo differs)" \
  "PTH_FA= (empty) is the same as unset"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV= PTH_FA=0 "$S86" > "$T/f0"; cp "$T/err" "$T/f0.err"
check "0 FLASH_ATTN fp8" "$(rc) $(after --attention-backend < "$T/f0") $(after --kv-cache-dtype < "$T/f0")" \
  "PTH_FA=0 (opt out): the FA2 default runs (fp8 KV on FLASH_ATTN)"
grep -qF '[effective-config] CTX=long SPEC=dflash2 FA2_FP8KV=1 (default)' "$T/f0.err" \
  && check 1 1 "PTH_FA=0: the effective config says FA2_FP8KV=1 (default)" || check 1 0 "PTH_FA=0: the effective config says FA2_FP8KV=1 (default)"
check "0 0" "$(grep -c 'PTH_FA' "$T/f0.err") $(grep -cF "$NOTE" "$T/f0.err")" "PTH_FA=0: no PTH_FA line, no note"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV= PTHFA_STUB_INSTALLED=0 "$S86" > "$T/f00"; cp "$T/err" "$T/f00.err"
check "0 same" "$(rc) $(cmp -s "$T/f00" "$T/f0" && echo same || echo differs)" \
  "no package: no refusal, the argv and VLLM_* environment of PTH_FA=0 (the FA2 default)"
check "1 same" "$(grep -cF "$NOTE" "$T/f00.err") $(cmp -s <(grep -vF "$NOTE" "$T/f00.err") "$T/f0.err" && echo same || echo differs)" \
  "no package: the stderr of PTH_FA=0 plus one note"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV=0 PTH_FA=1 "$S86" > "$T/f10"; cp "$T/err" "$T/f10.err"
check "0 same 0" "$(rc) $(cmp -s "$T/f10" "$T/don" && echo same || echo differs) $(grep -c "$OVR" "$T/f10.err")" \
  "PTH_FA=1 FA2_FP8KV=0 (explicit): boots, the same argv, no override line"
refused "8.6: FA2_FP8KV=1 alongside it"                 "$FA2" SPEC=dflash2 CTX=long PTH_FA=1 FA2_FP8KV=1 "$S86"
refused "8.6: FA2_FP8KV=1 alongside it, no fa2 plugin" "$FA2" SPEC=dflash2 CTX=long PTH_FA=1 FA2_FP8KV=1 FA2_STUB_INSTALLED=0 "$S86"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long FA2_FP8KV= PTH_FA=1 "$S86" > "$T/fm"
check "0 same" "$(rc) $(cmp -s "$T/fm" "$T/on" && echo same || echo differs)" "mtp PTH_FA=1: the same argv and VLLM_* environment"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV= PTH_FA=1 "$S89" > "$T/f9"; cp "$T/err" "$T/f9.err"
check "0 same 0 1" "$(rc) $(cmp -s "$T/f9" "$T/don" && echo same || echo differs) $(grep -c "$OVR" "$T/f9.err") $(grep -cF \
  '[effective-config] CTX=long SPEC=dflash2 FA2_FP8KV=0 (default, off: compute capability 8.9)' "$T/f9.err")" \
  "8.9: PTH_FA=1 boots, the same argv, the default is the FA2 block's own off (no override line)"
run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long FA2_FP8KV= "$S89" > "$T/f90"
check "0 TRITON_ATTN" "$(rc) $(after --attention-backend < "$T/f90")" "8.9, no knob: int8 on TRITON_ATTN (control)"

echo "== no PTH_FA default where the launch is not the measured one: the same exit code, argv, environment and stderr as PTH_FA=0"
# keeps <why> <env...>: on the 8.6 stub with FA2_FP8KV unset (both overridable in <env>), <env> gives the exit code,
# argv, VLLM_* environment and stderr of <env> PTH_FA=0: the arm's own default, without a word and without a refusal.
keeps() { local why=$1 r1 r0; shift
  run "$T/new" single-user/start_qwen.sh FA2_FP8KV= "$S86" "$@" > "$T/k1"; cp "$T/err" "$T/k1.err"; r1=$(rc)
  run "$T/new" single-user/start_qwen.sh FA2_FP8KV= "$S86" "$@" PTH_FA=0 > "$T/k0"; cp "$T/err" "$T/k0.err"; r0=$(rc)
  check "$r0 same same 0" "$r1 $(cmp -s "$T/k1" "$T/k0" && echo same || echo differs) $(cmp -s "$T/k1.err" "$T/k0.err" \
    && echo same || echo differs) $(grep -c 'PTH_FA=1' "$T/k1.err")" "$why (exit $r1)"; }
L2="SPEC=dflash2 CTX=long"
keeps "explicit FA2_FP8KV=1: the FA2 arm"                $L2 FA2_FP8KV=1
keeps "explicit FA2_FP8KV=0: int8 on TRITON_ATTN"         $L2 FA2_FP8KV=0
keeps "KV_OFFLOAD_GB=8"                                   $L2 KV_OFFLOAD_GB=8
keeps "EXTRA_ARGS --kv-transfer-config"                   $L2 EXTRA_ARGS="--kv-transfer-config {}"
keeps "EXTRA_ARGS --kv-offloading-size"                   $L2 EXTRA_ARGS="--kv-offloading-size 4"
keeps "EXTRA_ARGS --attention-backend"                    $L2 EXTRA_ARGS="--attention-backend TRITON_ATTN"
keeps "EXTRA_ARGS --kv-cache-dtype"                       $L2 EXTRA_ARGS="--kv-cache-dtype=fp8"
keeps "EXTRA_ARGS --speculative-config"                   $L2 EXTRA_ARGS="--speculative-config {}"
keeps "DYNAMIC_K"                                         $L2 DYNAMIC_K='[[1,3,7],[4,512,3]]'
keeps "DFLASH_TOKENS=15"                                  $L2 DFLASH_TOKENS=15
keeps "DRAFT_METHOD=dspark"                               $L2 DRAFT_METHOD=dspark
keeps "LOOKUP=0"                                          $L2 LOOKUP=0
keeps "VISION=1"                                          $L2 VISION=1
keeps "DRAFT set"                                         $L2 DRAFT="$T/new/models/Qwen3.8-27B-DFlash2-W4A16"
keeps "nvidia-smi 8.0"                                    $L2 "$S80"
keeps "nvidia-smi 8.9"                                    $L2 "$S89"
keeps "nvidia-smi 9.0"                                    $L2 "$S90"
keeps "no nvidia-smi answer"                              $L2 "$SNONE"
for arm in "SPEC=mtp CTX=fast" "SPEC=mtp CTX=huge" "SPEC=dflash2 CTX=fast" "SPEC=dflash2 CTX=huge" "SPEC=off CTX=long"; do
  keeps "8.6, $arm" $arm
done

echo "== compute capability 8.6 host: PTH_FA defaults to 1 on SPEC=mtp CTX=long too (target and MTP drafter)"
NOTEM="PTH_FA defaults to 1 on SPEC=mtp CTX=long on compute capability 8.6, but the pth_flash_attn package is not in the venv"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long FA2_FP8KV= PTH_FA=1 "$S86" > "$T/m1"; cp "$T/err" "$T/m1.err"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long FA2_FP8KV= "$S86" > "$T/md"; cp "$T/err" "$T/md.err"
check "0 same" "$(rc) $(cmp -s "$T/md" "$T/m1" && echo same || echo differs)" \
  "no knob: the argv and VLLM_* environment of PTH_FA=1 (CUSTOM, the MTP drafter's backend in the speculative config)"
check same "$(cmp -s <(sed 's/^\(\[effective-config\] PTH_FA=1 (\)explicit; /\1default; /' "$T/m1.err") "$T/md.err" && echo same || echo differs)" \
  "no knob: the stderr of PTH_FA=1 but for (default) in its effective-config line"
grep -q '^\[effective-config\] PTH_FA=1 (default; .*target and MTP drafter)$' "$T/md.err" \
  && check 1 1 "no knob: the effective config says PTH_FA=1 (default), target and MTP drafter" \
  || check 1 0 "no knob: the effective config says PTH_FA=1 (default), target and MTP drafter"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long FA2_FP8KV= PTH_FA=0 "$S86" > "$T/m0"; cp "$T/err" "$T/m0.err"
check "0 fp8 none" "$(rc) $(after --kv-cache-dtype < "$T/m0") $(after --attention-backend < "$T/m0")" \
  "PTH_FA=0 (opt out): fp8 KV, no --attention-backend (FlashInfer, the arm's own default)"
run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long FA2_FP8KV= PTHFA_STUB_INSTALLED=0 "$S86" > "$T/m00"; cp "$T/err" "$T/m00.err"
check "0 same 1 same" "$(rc) $(cmp -s "$T/m00" "$T/m0" && echo same || echo differs) $(grep -cF "$NOTEM" "$T/m00.err") $(cmp -s \
  <(grep -vF "$NOTEM" "$T/m00.err") "$T/m0.err" && echo same || echo differs)" "no package: the launch of PTH_FA=0 and one note"
for v in "PREFIX_CACHE=1" "PREFIX_CACHE=1 DRAFT_TOKENS=3" "VISION=0" "EXTRA_ARGS=--tensor-parallel-size=2"; do
  run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long FA2_FP8KV= $v PTH_FA=1 "$S86" > "$T/x"
  run "$T/new" single-user/start_qwen.sh SPEC=mtp CTX=long FA2_FP8KV= $v "$S86" > "$T/y"; cp "$T/err" "$T/y.err"
  check "0 same 1" "$(rc) $(cmp -s "$T/x" "$T/y" && echo same || echo differs) $(grep -c '^\[effective-config\] PTH_FA=1 (default; ' "$T/y.err")" \
    "$v: the default applies, the argv of PTH_FA=1"
done
L3="SPEC=mtp CTX=long"
keeps "mtp: explicit FA2_FP8KV=1 (refused by the FA2 resolution, as without the default)" $L3 FA2_FP8KV=1
keeps "mtp: explicit FA2_FP8KV=0"                         $L3 FA2_FP8KV=0
keeps "mtp: VISION=1"                                     $L3 VISION=1
keeps "mtp: PREFIX_CACHE=1 VISION=1"                      $L3 PREFIX_CACHE=1 VISION=1
keeps "mtp: KV_OFFLOAD_GB=8"                              $L3 KV_OFFLOAD_GB=8
keeps "mtp: EXTRA_ARGS --kv-offloading-size"              $L3 EXTRA_ARGS="--kv-offloading-size 4"
keeps "mtp: EXTRA_ARGS --attention-backend"               $L3 EXTRA_ARGS="--attention-backend FLASHINFER"
keeps "mtp: EXTRA_ARGS --kv-cache-dtype"                  $L3 EXTRA_ARGS="--kv-cache-dtype=fp8"
keeps "mtp: EXTRA_ARGS --speculative-config"              $L3 EXTRA_ARGS="--speculative-config {}"
keeps "mtp: DYNAMIC_K"                                    $L3 DYNAMIC_K='[[1,2,3],[3,512,2]]'
keeps "mtp: DRAFT_TOKENS=4"                               $L3 DRAFT_TOKENS=4
keeps "mtp: nvidia-smi 8.0"                               $L3 "$S80"
keeps "mtp: nvidia-smi 8.9"                               $L3 "$S89"
keeps "mtp: nvidia-smi 9.0"                               $L3 "$S90"
keeps "mtp: no nvidia-smi answer"                         $L3 "$SNONE"

if [ -n "${PTHFA_BASE_TREE:-}" ]; then
  echo "== against $PTHFA_BASE_TREE (FA2_FP8KV unset): no knob off 8.6, PTH_FA=0 on 8.6, every default-off launch and dflash2 are byte for byte the base tree's"
  stage "$PTHFA_BASE_TREE" "$T/base86"
  # vsbase <why> <new-side env> -- <env...>: the base tree with <env> against this tree with <env> plus <new-side env>.
  vsbase() { local why=$1 extra=$2; shift 2
    run "$T/base86" single-user/start_qwen.sh FA2_FP8KV= "$@" > "$T/x"; sed "s|$T/base86|TREE|g" "$T/err" > "$T/x.err"
    run "$T/new" single-user/start_qwen.sh FA2_FP8KV= "$@" $extra > "$T/y"; sed "s|$T/new|TREE|g" "$T/err" > "$T/y.err"
    sed -i "s|$T/base86|TREE|g" "$T/x"; sed -i "s|$T/new|TREE|g" "$T/y"
    check "same same" "$(cmp -s "$T/x" "$T/y" && echo same || echo differs) $(cmp -s "$T/x.err" "$T/y.err" && echo same || echo differs)" "$why"; }
  for c in 8.0 8.9 9.0 none; do cap=$(smi "$c")
    for arm in "SPEC=dflash2 CTX=long" "SPEC=mtp CTX=long"; do vsbase "no knob, nvidia-smi $c, $arm" "" $arm "$cap"; done
  done
  for arm in "SPEC=dflash2 CTX=long" "SPEC=mtp CTX=long"; do vsbase "PTH_FA=0, nvidia-smi 8.6, $arm" "" $arm PTH_FA=0 "$S86"; done
  # With no knob on 8.6 every other arm, and SPEC=dflash2 CTX=long with its default, is the base tree's launch byte for byte.
  for arm in "SPEC=dflash2 CTX=long" "SPEC=mtp CTX=fast" "SPEC=mtp CTX=huge" "SPEC=dflash2 CTX=fast" "SPEC=dflash2 CTX=huge" \
             "SPEC=off CTX=long"; do
    vsbase "no knob, nvidia-smi 8.6, $arm" "" $arm "$S86"
  done
  # SPEC=mtp CTX=long where a gate fails: the base tree's launch (fp8 on FlashInfer, or its refusal) byte for byte.
  for g in "VISION=1" "PREFIX_CACHE=1 VISION=1" "FA2_FP8KV=0" "FA2_FP8KV=1" "DRAFT_TOKENS=4" "KV_OFFLOAD_GB=8" \
           "EXTRA_ARGS=--kv-offloading-size=4" "EXTRA_ARGS=--attention-backend=FLASHINFER" "EXTRA_ARGS=--kv-cache-dtype=fp8"; do
    vsbase "no knob, nvidia-smi 8.6, SPEC=mtp CTX=long $g" "" SPEC=mtp CTX=long $g "$S86"
  done
  vsbase "no knob, nvidia-smi 8.6, SPEC=mtp CTX=long DYNAMIC_K" "" SPEC=mtp CTX=long DYNAMIC_K='[[1,2,3],[3,512,2]]' "$S86"
  # SPEC=mtp CTX=long with no knob on 8.6 is the base tree's explicit PTH_FA=1: argv and environment byte for byte,
  # stderr but for (default) in the PTH_FA=1 effective-config line.
  run "$T/base86" single-user/start_qwen.sh FA2_FP8KV= SPEC=mtp CTX=long PTH_FA=1 "$S86" > "$T/x"; sed "s|$T/base86|TREE|g" "$T/err" > "$T/x.err"
  run "$T/new" single-user/start_qwen.sh FA2_FP8KV= SPEC=mtp CTX=long "$S86" > "$T/y"; sed "s|$T/new|TREE|g" "$T/err" > "$T/y.err"
  sed -i "s|$T/base86|TREE|g" "$T/x"; sed -i "s|$T/new|TREE|g" "$T/y"
  check "same same" "$(cmp -s "$T/x" "$T/y" && echo same || echo differs) $(cmp -s <(sed 's/^\(\[effective-config\] PTH_FA=1 (\)explicit; /\1default; /' \
    "$T/x.err") "$T/y.err" && echo same || echo differs)" "no knob, nvidia-smi 8.6, SPEC=mtp CTX=long: the base tree's PTH_FA=1"
  run "$T/base86" single-user/start_qwen.sh FA2_FP8KV= SPEC=mtp CTX=long "$S86" > "$T/x"; sed "s|$T/base86|TREE|g" "$T/err" > "$T/x.err"
  run "$T/new" single-user/start_qwen.sh FA2_FP8KV= SPEC=mtp CTX=long PTHFA_STUB_INSTALLED=0 "$S86" > "$T/y"
  sed "s|$T/new|TREE|g" "$T/err" > "$T/y.err"; sed -i "s|$T/base86|TREE|g" "$T/x"; sed -i "s|$T/new|TREE|g" "$T/y"
  check "same same 1" "$(cmp -s "$T/x" "$T/y" && echo same || echo differs) $(cmp -s "$T/x.err" <(grep -vF "$NOTEM" "$T/y.err") && echo same || echo differs) \
$(grep -cF "$NOTEM" "$T/y.err")" "no knob, nvidia-smi 8.6, SPEC=mtp CTX=long, no package: the base tree's launch, its stderr plus the note"
  # The base tree has the dflash2 default too, so without the package both print its note: byte for byte.
  vsbase "no knob, nvidia-smi 8.6, SPEC=dflash2 CTX=long, no package: the base tree's launch and note" "" \
    SPEC=dflash2 CTX=long PTHFA_STUB_INSTALLED=0 "$S86"
fi

echo "== batch mode ignores it, and says so"
run "$T/new" batch/start_qwen.sh > "$T/b0"
run "$T/new" batch/start_qwen.sh PTH_FA=1 > "$T/b1"
check "0 same" "$(rc) $(cmp -s "$T/b0" "$T/b1" && echo same || echo differs)" "batch argv unchanged by PTH_FA=1"
grep -q "PTH_FA=1 is set but batch mode ignores it" "$T/err" && check 1 1 "batch warns" || check 1 0 "batch warns"

if [ -n "${PTHFA_BASE_TREE:-}" ]; then
  echo "== negative control: PTH_FA unset against $PTHFA_BASE_TREE, argv, VLLM_* environment and stderr byte for byte"
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
if [ -n "${PTHFA_PARENT_TREE:-}" ]; then
  echo "== mtp unchanged: PTH_FA=1 against $PTHFA_PARENT_TREE, argv and VLLM_* environment byte for byte, stderr but the PTH_FA=1 effective-config line"
  stage "$PTHFA_PARENT_TREE" "$T/parent"
  for arm in "SPEC=mtp CTX=long PTH_FA=1" "PREFIX_CACHE=1 VISION=1 SPEC=mtp CTX=long PTH_FA=1" \
             "SPEC=mtp CTX=long PTH_FA=1 EXTRA_ARGS=--tensor-parallel-size=2"; do
    run "$T/parent" single-user/start_qwen.sh $arm > "$T/x"; sed "s|$T/parent|TREE|g" "$T/err" > "$T/x.err"; rx=$(rc)
    run "$T/new" single-user/start_qwen.sh $arm > "$T/y"; sed "s|$T/new|TREE|g" "$T/err" > "$T/y.err"; ry=$(rc)
    sed -i "s|$T/parent|TREE|g" "$T/x"; sed -i "s|$T/new|TREE|g" "$T/y"
    ex=$(grep -c '^\[effective-config\] PTH_FA=1 ' "$T/x.err"); ey=$(grep -c '^\[effective-config\] PTH_FA=1 ' "$T/y.err")
    check "0 0 same 1 1 same" "$rx $ry $(cmp -s "$T/x" "$T/y" && echo same || echo differs) $ex $ey $(cmp -s \
      <(grep -v '^\[effective-config\] PTH_FA=1 ' "$T/x.err") <(grep -v '^\[effective-config\] PTH_FA=1 ' "$T/y.err") \
      && echo same || echo differs)" "single $arm"
  done
  run "$T/parent" single-user/start_qwen.sh SPEC=dflash2 CTX=long PTH_FA=1 > "$T/x"; rx=$(rc)
  if [ "$rx" = 0 ]; then
    run "$T/new" single-user/start_qwen.sh SPEC=dflash2 CTX=long PTH_FA=1 > "$T/y"; ry=$(rc)
    sed -i "s|$T/parent|TREE|g" "$T/x"; sed -i "s|$T/new|TREE|g" "$T/y"
    px=$(after --speculative-config < "$T/x"); py=$(after --speculative-config < "$T/y")
    check "0 same" "$ry $(cmp -s <(strip < "$T/x") <(strip < "$T/y") && echo same || echo differs)" \
      "dflash2: every other argument and VLLM_* variable as the parent's"
    check "${px%\}}"',"attention_backend":"CUSTOM"}' "$py" "dflash2: the speculative config is the parent's plus the drafter's backend"
    check "$(after --attention-backend < "$T/x") $(after --kv-cache-dtype < "$T/x")" \
      "$(after --attention-backend < "$T/y") $(after --kv-cache-dtype < "$T/y")" "dflash2: the target's flags as the parent's"
  else
    echo "  SKIP  dflash2 against the parent: it refuses SPEC=dflash2 CTX=long PTH_FA=1 (an mtp-only knob)"
  fi
fi
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
