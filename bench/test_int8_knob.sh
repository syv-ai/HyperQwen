#!/bin/bash
# CPU test for INT8_ACT / INT8_LAYERS on vLLM 0.34 (launcher_common.sh, qwen_int8_exports and qwen_int8_args):
# int8 adds one --hf-overrides built from the checkpoint's config (scripts/int8_overrides.py) with the
# INT8_LAYERS regex in it, empty adds nothing, fp8 and an explicit int8 on a checkpoint it does not fit are
# refused, batch's unset INT8_ACT (int8 by default) falls back to W4A16 with a note on such a checkpoint or
# beside an --hf-overrides in EXTRA_ARGS, and the venv's NVRTC builtins directory reaches LD_LIBRARY_PATH.
# Copies the checkout to a temp dir and runs the three launchers against a stub vllm that prints
# LD_LIBRARY_PATH and its argv. No GPU, no weights, no network.
#
#   bash bench/test_int8_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin" "$T/venv/lib/python3.12/site-packages/nvidia/cu13/lib" "$T/ckpt" "$T/gptq"
touch "$T/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc-builtins.so.13.0"
printf '#!/bin/sh\necho "LDP=${LD_LIBRARY_PATH-unset}"\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
cat > "$T/ckpt/config.json" <<'EOF'
{"quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized", "ignore": [],
 "config_groups": {"group_0": {"format": "pack-quantized", "targets": ["Linear"], "input_activations": null,
  "weights": {"num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 128, "dynamic": false}}}}}
EOF
echo '{"quantization_config": {"quant_method": "gptq", "bits": 4}}' > "$T/gptq/config.json"
FAILS=0
# run <launcher> <env assignments...>: the stub's output, then "RC=<exit>". MODEL defaults to the fitting checkpoint.
run() { local sc=$1; shift
  (cd "$T" && env -u PRINT_ARGV -u INT8_ACT -u INT8_LAYERS -u EXTRA_ARGS -u LD_LIBRARY_PATH -u WSL_DISTRO_NAME \
    VLLM_API_KEY=k MODEL="$T/ckpt" "$@" bash "$sc" 2>"$T/err" </dev/null; echo "RC=$?"); }
# what <launcher> <env...>: "RC=<n> ov=<targets of the added group, or none>".
what() { local out ov; out=$(run "$@")
  # Each --hf-overrides value on its own line; only one carrying a quantization_config is the launcher's.
  ov=$(printf '%s\n' "$out" | sed -n '/^--hf-overrides$/{n;p;}' | python3 -c 'import json,sys
got=[]
for line in sys.stdin.read().splitlines():
    q=json.loads(line).get("quantization_config") if line.strip() else None
    if q:
        g=q["config_groups"]
        got+=[t for v in g.values() if v.get("input_activations") for t in v["targets"]] or ["no-act-group"]
print(",".join(got) or "none")')
  echo "$(printf '%s\n' "$out" | sed -n 's/^\(RC=[0-9]*\)$/\1/p') ov=$ov"; }
ldp() { run "$@" | sed -n 's/^LDP=//p'; }
check() { local want=$1 got=$2 desc=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$desc" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$desc" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
for sc in single-user/start_qwen.sh batch/start_qwen.sh single-user/alternative.sh; do
  echo "== $sc"
  check "RC=0 ov=re:.*(?:mlp).*"   "$(what $sc INT8_ACT=int8 INT8_LAYERS=mlp)"     "INT8_ACT=int8 INT8_LAYERS=mlp"
  check "RC=0 ov=re:.*(?:.).*"     "$(what $sc INT8_ACT=int8 INT8_LAYERS=.)"       "INT8_ACT=int8 INT8_LAYERS=."
  check "RC=0 ov=none"             "$(what $sc INT8_ACT=)"                         "INT8_ACT= (W4A16)"
  check "RC=2 ov=none"             "$(what $sc INT8_ACT=fp8)"                      "INT8_ACT=fp8 refused"
  check "RC=2 ov=none"             "$(what $sc INT8_ACT=int8 INT8_LAYERS=)"        "INT8_ACT=int8 with INT8_LAYERS empty refused"
  check "RC=2 ov=none"             "$(what $sc INT8_ACT=int8 MODEL=$T/gptq)"       "INT8_ACT=int8 on a GPTQ checkpoint refused"
  check "RC=2 ov=none"             "$(what $sc INT8_ACT=int8 MODEL=$T/nowhere)"    "INT8_ACT=int8 without a config.json refused"
  check "RC=2 ov=none"             "$(what $sc INT8_ACT=int8 EXTRA_ARGS='--hf-overrides {}')" "INT8_ACT=int8 beside --hf-overrides refused"
  check "$T/venv/lib/python3.12/site-packages/nvidia/cu13/lib" "$(ldp $sc INT8_ACT=)" "NVRTC builtins on LD_LIBRARY_PATH"
  check "/x:$T/venv/lib/python3.12/site-packages/nvidia/cu13/lib" "$(ldp $sc INT8_ACT= LD_LIBRARY_PATH=/x)" "appended after an existing LD_LIBRARY_PATH"
done
echo "== defaults"
check "RC=0 ov=re:.*(?:mlp).*"   "$(what batch/start_qwen.sh)"                     "batch, INT8_ACT unset: int8 on the MLP"
check "RC=0 ov=none"             "$(what batch/start_qwen.sh MODEL=$T/gptq)"       "batch default on a GPTQ checkpoint: W4A16"
grep -q "serving W4A16" "$T/err" && check yes yes "  and it says so" || check yes no "  and it says so"
check "RC=0 ov=none"             "$(what batch/start_qwen.sh EXTRA_ARGS='--hf-overrides {}')" "batch default beside --hf-overrides: W4A16"
check "RC=0 ov=none"             "$(what single-user/start_qwen.sh)"               "single-user, INT8_ACT unset: W4A16"
check "RC=0 ov=none"             "$(what single-user/alternative.sh)"              "alternative.sh, INT8_ACT unset: W4A16"
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
