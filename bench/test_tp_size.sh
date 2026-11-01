#!/bin/bash
# CPU test for how the launchers read EXTRA_ARGS (launcher_common.sh, qwen_normalize_extra_args and
# qwen_tp_size): --flag names with _ for -, -tp=N and any whitespace, as vLLM 0.31's parser takes
# them. The helpers run directly, so those rows hold on every platform. The launchers then run
# against a stub vllm that prints its argv and PYTORCH_CUDA_ALLOC_CONF: the TP size reaches the
# allocator default (expandable_segments:False at TP>1, #163; native only, WSL defaults it off
# anyway) and single-user's skip of the single-card KV_MEM pin (#40), and resolve_config.sh's
# shadow warning sees an underscore flag. Argparse abbreviations (--tensor 2) are not read.
# No GPU, no model, no network.
#
#   bash bench/test_tp_size.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin" "$T/draft"
printf '#!/bin/sh\necho "ALLOC=${PYTORCH_CUDA_ALLOC_CONF-unset}"\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$what" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
norm_of() ( cd "$T" && source ./launcher_common.sh 2>/dev/null; EXTRA_ARGS=$1; qwen_normalize_extra_args 2>/dev/null; printf %s "$EXTRA_ARGS" )
tp_of() ( cd "$T" && source ./launcher_common.sh 2>/dev/null; EXTRA_ARGS=$1; qwen_normalize_extra_args 2>/dev/null; qwen_tp_size 2>/dev/null )
CLEAN=(-u PRINT_ARGV -u KV_OFFLOAD_GB -u EXTRA_ARGS -u PYTORCH_CUDA_ALLOC_CONF -u WSL_DISTRO_NAME -u KV_MEM -u CTX
       -u MAX_LEN -u MAX_SEQS -u SPEC -u PREFIX_CACHE -u GPU_UTIL -u DFLASH_TOKENS -u DFLASH_MAX_LEN -u DRAFT)
# run <launcher> <env assignments...>: the stub's output (ALLOC=... then the argv, one per line).
run() { local sc=$1; shift; (cd "$T" && env "${CLEAN[@]}" VLLM_API_KEY=k "$@" bash "$sc" 2>/dev/null </dev/null); }
alloc_of() { run "$@" | sed -n 's/^ALLOC=//p'; }
kvmem_of() { run "$@" | grep -q '^--kv-cache-memory=' && echo pinned || echo none; }
shadow_of() { local sc=$1; shift
  (cd "$T" && env "${CLEAN[@]}" VLLM_API_KEY=k "$@" bash "$sc" 2>&1 >/dev/null </dev/null) | grep -c "EXTRA_ARGS carries --max-model-len"; }
NATIVE=1; grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && NATIVE=0

echo "== qwen_tp_size"
for tp in '-tp 2' '-tp=2' '--tensor-parallel-size 2' '--tensor-parallel-size=2' '--tensor_parallel_size 2' \
          '--tensor_parallel_size=2' '-tp 1 -tp=2' '--tensor-parallel-size    2' $'-tp\t2' $'-tp 1\n-tp 2' \
          $'--tensor-parallel-size 2\r' '--max-num-seqs 8 -tp 2 --enforce-eager'; do
  check 2 "$(tp_of "$tp")" "EXTRA_ARGS=$(printf %q "$tp")"
done
for tp in '' '-tp 2 -tp 1' '--max-num-seqs 2' '--data-parallel-size 2' '--pipeline-parallel-size 2'; do
  check 1 "$(tp_of "$tp")" "EXTRA_ARGS=$(printf %q "$tp")"
done

echo "== qwen_normalize_extra_args"
check '--tensor-parallel-size=2 --disable-custom-all-reduce' "$(norm_of '--tensor_parallel_size=2 --disable_custom_all_reduce')" "underscores in --flag names"
check '--compilation-config.cudagraph_mode=FULL' "$(norm_of '--compilation_config.cudagraph_mode=FULL')" "dotted key: up to the first dot only"
check '--hf-overrides={"x_y":1}' "$(norm_of '--hf_overrides={"x_y":1}')"                         "value after = untouched"
check '--kv-transfer-config {"kv_role":"kv_both"}' "$(norm_of '--kv-transfer-config {"kv_role":"kv_both"}')" "value word untouched"
check '-tp=2 --max-num-seqs 4' "$(norm_of $'-tp=2   --max_num_seqs\t 4')"                     "whitespace runs become one space"
check '--served-model-name *' "$(norm_of '--served_model_name *')"                               "a * is not globbed"

for sc in single-user/start_qwen.sh batch/start_qwen.sh; do
  echo "== $sc"
  if [ $NATIVE = 1 ]; then
    for tp in '-tp=2' '--tensor_parallel_size 2' $'-tp\t2'; do
      check expandable_segments:False "$(alloc_of $sc EXTRA_ARGS="$tp")" "allocator with EXTRA_ARGS=$(printf %q "$tp")"
    done
    check expandable_segments:True "$(alloc_of $sc)"                                            "allocator with no TP flag"
    check expandable_segments:True "$(alloc_of $sc EXTRA_ARGS='-tp 2 -tp 1')"                   "allocator with EXTRA_ARGS='-tp 2 -tp 1'"
    check expandable_segments:True "$(alloc_of $sc EXTRA_ARGS='--disable_custom_all_reduce --tensor_parallel_size 2')" \
                                                                                                "allocator with the all-reduce opt-out in underscores"
    check expandable_segments:True "$(alloc_of $sc EXTRA_ARGS='--enforce_eager -tp=2')"         "allocator with --enforce_eager"
  fi
  check 1 "$(shadow_of $sc EXTRA_ARGS='--max_model_len 8192')"                                  "shadow warning for --max_model_len"
done
echo "== single-user/start_qwen.sh SPEC=dflash2: the single-card KV_MEM pin"
sc=single-user/start_qwen.sh; D=(SPEC=dflash2 DRAFT="$T/draft")
check pinned "$(kvmem_of $sc "${D[@]}")"                                     "KV pin with no TP flag"
check none "$(kvmem_of $sc "${D[@]}" EXTRA_ARGS='-tp=2')"                   "KV pin with EXTRA_ARGS='-tp=2'"
check none "$(kvmem_of $sc "${D[@]}" EXTRA_ARGS='--tensor_parallel_size 2')" "KV pin with EXTRA_ARGS='--tensor_parallel_size 2'"
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
