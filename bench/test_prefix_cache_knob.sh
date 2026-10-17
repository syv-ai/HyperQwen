#!/bin/bash
# CPU test for PREFIX_CACHE (launcher_common.sh, qwen_prefix_cache): in each launcher unset, empty and 1
# mean prefix caching on (--enable-prefix-caching --mamba-cache-mode align), 0 means off
# (--no-enable-prefix-caching), anything else refuses to boot, and the last caching flag in EXTRA_ARGS
# (dashed or underscored, as vLLM reads either) decides the engine's state, which the launcher reports
# in one stderr line. The launcher's caching-on safeguards (single-user: prefix match unit and PIECEWISE
# capture at CTX=huge, retention interval for a draft profile, the pinned DFlash2 verify block;
# alternative.sh: the dflash2 retention interval) follow that effective state, not the raw knob.
# Copies the checkout to a temp dir and runs the three launchers against a stub vllm that prints its
# argv and the environment it reads. No GPU, no model, no network.
#
#   bash bench/test_prefix_cache_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin" "$T/draft"
printf '#!/bin/sh\necho "ALLOC=${PYTORCH_CUDA_ALLOC_CONF-unset}"\necho "ADAPTIVE=${VLLM_DFLASH2_LOOKUP_ADAPTIVE-unset}"\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
CLEAN="-u PRINT_ARGV -u KV_OFFLOAD_GB -u EXTRA_ARGS -u PYTORCH_CUDA_ALLOC_CONF -u WSL_DISTRO_NAME -u PREFIX_CACHE
  -u CTX -u SPEC -u KV -u KV_MEM -u DRAFT -u LOOKUP -u PREFIX_RETENTION -u VLLM_PREFIX_CACHE_RETENTION_INTERVAL
  -u DFLASH_TOKENS -u LOOKUP_ADAPTIVE -u VLLM_DFLASH2_LOOKUP_ADAPTIVE -u CUDAGRAPH_MODE"
# run <launcher> <env assignments...>: the stub's output (ALLOC=, ADAPTIVE=, then the argv, one per line).
# shellcheck disable=SC2086
run() { local sc=$1; shift
  (cd "$T" && env $CLEAN VLLM_API_KEY=k "$@" bash "$sc" 2>/dev/null </dev/null); }
# flag_of: the last prefix-caching flag in the argv, which is the one vLLM honours; "none" for neither.
flag_of() { local v; v=$(run "$@" | grep -E '^--(no-)?enable-prefix-caching$' | tail -n1); echo "${v:-none}"; }
# arg_of <flag> <launcher> <env...>: the value after the last <flag> in the argv, "none" for no flag.
arg_of() { local f=$1 v; shift; v=$(run "$@" | sed -n "/^$f\$/{n;p;}" | tail -n1); echo "${v:-none}"; }
piecewise_of() { run "$@" | grep -q 'cudagraph_mode":"PIECEWISE' && echo yes || echo no; }
adaptive_of() { run "$@" | sed -n 's/^ADAPTIVE=//p'; }
# status_of: the boot line the launcher prints on stderr about the state it picked.
# shellcheck disable=SC2086
status_of() { local sc=$1; shift
  (cd "$T" && env $CLEAN VLLM_API_KEY=k "$@" bash "$sc" 2>&1 >/dev/null </dev/null) | grep '^prefix caching: '; }
# argv0_of: the first stdout line under PRINT_ARGV=1, which must be the vllm binary, not a status line.
# shellcheck disable=SC2086
argv0_of() { local sc=$1; shift
  (cd "$T" && env $CLEAN VLLM_API_KEY=k PRINT_ARGV=1 "$@" bash "$sc" 2>/dev/null </dev/null) | sed -n 1p; }
# refused_of: "refused" when the launcher exits non-zero and its stderr names PREFIX_CACHE, else its exit code.
# shellcheck disable=SC2086
refused_of() { local sc=$1 err rc; shift
  err=$(cd "$T" && env $CLEAN VLLM_API_KEY=k "$@" bash "$sc" 2>&1 >/dev/null </dev/null); rc=$?
  if [ $rc != 0 ] && printf %s "$err" | grep -q PREFIX_CACHE; then echo refused; else echo "exit $rc"; fi; }
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$what" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
ON=--enable-prefix-caching; OFF=--no-enable-prefix-caching
for sc in single-user/start_qwen.sh batch/start_qwen.sh single-user/alternative.sh; do
  echo "== $sc"
  check $ON  "$(flag_of $sc)"                                        "PREFIX_CACHE unset"
  check $ON  "$(flag_of $sc PREFIX_CACHE=)"                          "PREFIX_CACHE= (empty)"
  check $ON  "$(flag_of $sc PREFIX_CACHE=1)"                         "PREFIX_CACHE=1"
  check $OFF "$(flag_of $sc PREFIX_CACHE=0)"                         "PREFIX_CACHE=0"
  check $ON  "$(flag_of $sc PREFIX_CACHE=0 EXTRA_ARGS=$ON)"          "PREFIX_CACHE=0, $ON in EXTRA_ARGS"
  check $OFF "$(flag_of $sc PREFIX_CACHE=1 EXTRA_ARGS=$OFF)"         "PREFIX_CACHE=1, $OFF in EXTRA_ARGS"
  check refused "$(refused_of $sc PREFIX_CACHE=true)"                "PREFIX_CACHE=true"
  check align "$(arg_of --mamba-cache-mode $sc)"                     "mamba cache mode, PREFIX_CACHE unset"
  check none  "$(arg_of --mamba-cache-mode $sc PREFIX_CACHE=0)"      "mamba cache mode, PREFIX_CACHE=0"
  check "prefix caching: on (PREFIX_CACHE unset, the default 1)" "$(status_of $sc)"  "status line, PREFIX_CACHE unset"
  check "prefix caching: off (PREFIX_CACHE=0)" "$(status_of $sc PREFIX_CACHE=0)"    "status line, PREFIX_CACHE=0"
  check "prefix caching: on (EXTRA_ARGS overrides PREFIX_CACHE=0)" "$(status_of $sc PREFIX_CACHE=0 EXTRA_ARGS=$ON)" \
                                                                     "status line, PREFIX_CACHE=0, $ON in EXTRA_ARGS"
  check "prefix caching: off (EXTRA_ARGS overrides PREFIX_CACHE unset, the default 1)" "$(status_of $sc EXTRA_ARGS=$OFF)" \
                                                                     "status line, $OFF in EXTRA_ARGS"
done
SU=single-user/start_qwen.sh
echo "== $SU, caching-on safeguards"
check venv/bin/vllm "$(argv0_of $SU)"                                         "PRINT_ARGV=1: first stdout line"
check yes  "$(piecewise_of $SU CTX=huge SPEC=mtp)"                              "CTX=huge SPEC=mtp: PIECEWISE capture"
check 128  "$(arg_of --prefix-match-unit $SU CTX=huge SPEC=mtp)"                "CTX=huge SPEC=mtp: prefix match unit"
check no   "$(piecewise_of $SU CTX=huge SPEC=mtp PREFIX_CACHE=0)"               "CTX=huge SPEC=mtp PREFIX_CACHE=0: PIECEWISE capture"
check none "$(arg_of --prefix-match-unit $SU CTX=huge SPEC=mtp PREFIX_CACHE=0)" "CTX=huge SPEC=mtp PREFIX_CACHE=0: prefix match unit"
check no   "$(piecewise_of $SU CTX=huge SPEC=mtp EXTRA_ARGS=$OFF)"             "CTX=huge SPEC=mtp, $OFF in EXTRA_ARGS: PIECEWISE capture"
check none "$(arg_of --prefix-match-unit $SU CTX=huge SPEC=mtp EXTRA_ARGS=$OFF)" "CTX=huge SPEC=mtp, $OFF in EXTRA_ARGS: prefix match unit"
# vLLM reads _ as - in flag names, so the underscore spellings decide the state too.
U_ON=--enable_prefix_caching; U_OFF=--no_enable_prefix_caching
check yes  "$(piecewise_of $SU CTX=huge SPEC=mtp PREFIX_CACHE=0 EXTRA_ARGS=$U_ON)"             "CTX=huge SPEC=mtp PREFIX_CACHE=0, $U_ON in EXTRA_ARGS: PIECEWISE capture"
check 128  "$(arg_of --prefix-match-unit $SU CTX=huge SPEC=mtp PREFIX_CACHE=0 EXTRA_ARGS=$U_ON)" "CTX=huge SPEC=mtp PREFIX_CACHE=0, $U_ON in EXTRA_ARGS: prefix match unit"
check no   "$(piecewise_of $SU CTX=huge SPEC=mtp EXTRA_ARGS=$U_OFF)"                           "CTX=huge SPEC=mtp, $U_OFF in EXTRA_ARGS: PIECEWISE capture"
check "prefix caching: on (EXTRA_ARGS overrides PREFIX_CACHE=0)" "$(status_of $SU PREFIX_CACHE=0 EXTRA_ARGS=$U_ON)" "status line, PREFIX_CACHE=0, $U_ON in EXTRA_ARGS"
check "prefix caching: off (EXTRA_ARGS overrides PREFIX_CACHE unset, the default 1)" "$(status_of $SU EXTRA_ARGS=$U_OFF)" "status line, $U_OFF in EXTRA_ARGS"
R=--prefix-cache-retention-interval
check None "$(arg_of $R $SU CTX=fast SPEC=mtp)"                                 "CTX=fast SPEC=mtp: retention interval"
check none "$(arg_of $R $SU CTX=fast SPEC=mtp PREFIX_CACHE=0)"                  "CTX=fast SPEC=mtp PREFIX_CACHE=0: retention interval"
check None "$(arg_of $R $SU CTX=fast SPEC=mtp PREFIX_CACHE=0 EXTRA_ARGS=$ON)"   "CTX=fast SPEC=mtp PREFIX_CACHE=0, $ON in EXTRA_ARGS: retention interval"
check none "$(arg_of $R $SU CTX=fast SPEC=mtp EXTRA_ARGS=$OFF)"                 "CTX=fast SPEC=mtp, $OFF in EXTRA_ARGS: retention interval"
check 0     "$(adaptive_of $SU CTX=huge SPEC=dflash2 DFLASH_TOKENS=15 DRAFT="$T/draft")"                "CTX=huge dflash2 DFLASH_TOKENS=15: VLLM_DFLASH2_LOOKUP_ADAPTIVE"
check unset "$(adaptive_of $SU CTX=huge SPEC=dflash2 DFLASH_TOKENS=15 DRAFT="$T/draft" PREFIX_CACHE=0)" "CTX=huge dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=0: VLLM_DFLASH2_LOOKUP_ADAPTIVE"
check unset "$(adaptive_of $SU CTX=huge SPEC=dflash2 DFLASH_TOKENS=15 DRAFT="$T/draft" EXTRA_ARGS=$OFF)" "CTX=huge dflash2 DFLASH_TOKENS=15, $OFF in EXTRA_ARGS: VLLM_DFLASH2_LOOKUP_ADAPTIVE"
AL=single-user/alternative.sh
echo "== $AL (SPEC=dflash2, its default), retention interval"
check 0    "$(arg_of $R $AL)"                                  "PREFIX_CACHE unset: retention interval"
check none "$(arg_of $R $AL PREFIX_CACHE=0)"                   "PREFIX_CACHE=0: retention interval"
check 0    "$(arg_of $R $AL PREFIX_CACHE=0 EXTRA_ARGS=$ON)"    "PREFIX_CACHE=0, $ON in EXTRA_ARGS: retention interval"
check none "$(arg_of $R $AL EXTRA_ARGS=$OFF)"                  "$OFF in EXTRA_ARGS: retention interval"
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
