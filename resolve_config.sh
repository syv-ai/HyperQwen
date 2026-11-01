#!/bin/bash
# resolve_config.sh — one validated effective-configuration resolver for the
# launchers (automation backlog item 6, finding F13).
#
# Sourced by batch/start_qwen.sh and single-user/start_qwen.sh right after
# REPO is set:
#   source "$REPO/resolve_config.sh"
#   resolve_effective_config batch|single
#
# What it does, in order:
#   1. Refuses (exit 1) unknown values for the enumerated controls — CTX and
#      SPEC in single-user mode, KV in batch mode. A typo'd profile must never
#      boot the wrong geometry.
#   2. Warns (stderr, continues) about controls that are set but silently
#      ignored: CTX/SPEC in batch mode, KV in single-user mode, and launcher
#      flags shadowed by EXTRA_ARGS (EXTRA_ARGS expands last, so it wins
#      without a word — the warning makes that visible).
#   3. Prints the redacted effective configuration to stderr: every control
#      with its resolved value, secrets as presence-only (VLLM_API_KEY is
#      never printed, only set/unset + length; api_key.txt as present/absent).
#
# Fail-closed: refusal paths call exit, which terminates a sourcing launcher
# even without `set -e`. This file needs nothing but bash and REPO set.
# Run standalone for tests/CI: `bash resolve_config.sh single`.

# _refuse prints the reason and stops the launcher. Separate function so the
# refusal shape (stderr + nonzero exit) has exactly one definition.
_refuse() {
  echo "resolve_config: refusing: $1" >&2
  exit 1
}

# _warn prints a non-fatal notice. Ignored/shadowed controls warn rather than
# refuse: EXTRA_ARGS shadowing is a documented override door (e.g. TP size),
# and cross-mode knobs come along for the ride in a shared .env.
_warn() {
  echo "resolve_config: WARNING: $1" >&2
}

# pthfa_installed: the pth_flash_attn package and its op library are in the venv. The launcher's
# PTH_FA=1 install check and the PTH_FA default below use this one probe.
pthfa_installed() {
  "$REPO/venv/bin/python" -c 'import importlib.util as u, os, sys
s = u.find_spec("pth_flash_attn")
sys.exit(0 if s and os.path.isfile(os.path.join(s.submodule_search_locations[0], "_C.abi3.so")) else 1)' 2>/dev/null
}

resolve_effective_config() {
  local mode=${1:?usage: resolve_effective_config batch|single}
  [ "$mode" = "batch" ] || [ "$mode" = "single" ] \
    || _refuse "unknown mode $mode (want batch|single)"
  [ -n "${REPO:-}" ] || _refuse "REPO is unset"

  # 1. Validate the enumerated controls. Defaults mirror the launchers.
  local ctx=${CTX:-} spec=${SPEC:-} kv=${KV:-}
  if [ "$mode" = "single" ]; then
    ctx=${ctx:-fast}; spec=${spec:-mtp}
    case "$ctx" in fast|long|huge) ;;
      *) _refuse "unknown CTX=$ctx (want fast|long|huge)" ;;
    esac
    case "$spec" in mtp|dflash2|off|none) ;;
      *) _refuse "unknown SPEC=$spec (want mtp|dflash2|off|none)" ;;
    esac
    [ -z "${KV:-}" ] || _warn "KV=$KV is set but single-user mode ignores it (KV is a batch-mode control)"
    # FA2_FP8KV (fp8 KV on the FA2 plugin, fa2/) is one arm, and this is the one place that resolves
    # it. Unset or empty means on for SPEC=dflash2 CTX=long on an 8.6 GPU and off everywhere else (left unset, so
    # every other arm's environment is untouched). The FA2 backend supports only the LBNHC
    # KV layout and vLLM 0.31's OffloadingConnector needs LBHNC or BLHNC
    # (offloading_connector.py:225-228), and the CPU tier on it is untested, so a KV offload tier
    # turns the default off with a note and an explicit 1 into a refusal. So does a GPU that is not
    # compute capability 8.6 (or no answer from nvidia-smi); an explicit 1 there is the user's call.
    # PTH_FA=1 (the int8 kernel, pthfa/) is the other dflash2 CTX=long backend: with it the 8.6 default
    # is not applied (FA2_FP8KV stays 0, and the effective-config line says why), and an explicit
    # FA2_FP8KV=1 alongside it is refused here, before the launcher probes either plugin.
    local _fa2_tier=
    if [ -n "${KV_OFFLOAD_GB:-}" ] && [ "$KV_OFFLOAD_GB" != 0 ]; then _fa2_tier="KV_OFFLOAD_GB=$KV_OFFLOAD_GB"
    else case " ${EXTRA_ARGS:-} " in
      *"--kv-offloading-size"*|*"--kv-transfer-config"*) _fa2_tier="a KV connector in EXTRA_ARGS (--kv-offloading-size/--kv-transfer-config)" ;;
    esac; fi
    # The GPU's compute capability, asked at most once: the PTH_FA and FA2_FP8KV defaults below both
    # depend on it. nvidia-smi ignores CUDA_VISIBLE_DEVICES and orders GPUs differently from CUDA, so
    # with no selection every GPU it lists must agree.
    local _cap= _cap_asked=
    _ask_cap() { [ -n "$_cap_asked" ] && return 0; _cap_asked=1
      _cap=$(nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "${CUDA_VISIBLE_DEVICES%%,*}"} \
        --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' \r' | sort -u); }
    # PTH_FA (the int8 kernel, pthfa/) unset or empty means 1 on SPEC=dflash2 CTX=long and SPEC=mtp CTX=long
    # when the launch is the measured one: a compute capability 8.6 GPU (the only one measured), the package
    # installed, no backend the user picked (FA2_FP8KV set to 0 or 1), no KV offload tier or connector, no
    # --attention-backend, --kv-cache-dtype or --speculative-config in EXTRA_ARGS, no DYNAMIC_K and no VISION;
    # on dflash2 the default DFlash2 drafter at DFLASH_TOKENS=7 with lookup on, on mtp DRAFT_TOKENS unset or 3.
    # Anything else keeps the arm's own default (FA2_FP8KV on dflash2 on 8.6, fp8 on FlashInfer on mtp) without
    # a word: a default must not turn a launch that works into a refusal of PTH_FA=1. A missing install says so
    # once. PTH_FA=0 opts out.
    PTH_FA_SRC=explicit
    if [ -z "${PTH_FA:-}" ]; then
      PTH_FA_SRC=default
      local _pth_arm=
      case "$spec/$ctx" in
        dflash2/long)
          [ -z "${FA2_FP8KV:-}" ] && [ "${DRAFT_METHOD:-dflash}" = dflash ] && [ "${DFLASH_TOKENS:-7}" = 7 ] \
            && [ "${LOOKUP:-1}" = 1 ] && [ "${VISION:-0}" != 1 ] && [ -z "${DRAFT:-}" ] \
            && [ -f "$REPO/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors" ] \
            && _pth_arm="SPEC=dflash2 CTX=long" ;;
        mtp/long)
          [ -z "${FA2_FP8KV:-}" ] && [ "${DRAFT_TOKENS:-3}" = 3 ] && [ "${VISION:-0}" != 1 ] \
            && _pth_arm="SPEC=mtp CTX=long" ;;
      esac
      { [ -z "${DYNAMIC_K:-}" ] && [ -z "$_fa2_tier" ]; } || _pth_arm=
      case " ${EXTRA_ARGS:-} " in
        *" --attention-backend"*|*" --kv-cache-dtype"*|*" --speculative-config"*) _pth_arm= ;;
      esac
      if [ -n "$_pth_arm" ]; then
        _ask_cap
        if [ "$_cap" = 8.6 ] && pthfa_installed; then
          export PTH_FA=1
        elif [ "$_cap" = 8.6 ]; then
          echo "resolve_config: PTH_FA defaults to 1 on $_pth_arm on compute capability 8.6, but the pth_flash_attn package is not in the venv, so the arm's own default runs (bash pthfa/install.sh to use it; PTH_FA=0 silences this)" >&2
        fi
      fi
    fi
    FA2_FP8KV_SRC=explicit
    if [ -z "${FA2_FP8KV:-}" ]; then
      FA2_FP8KV_SRC=default
      if [ "$spec" = dflash2 ] && [ "$ctx" = long ]; then
        if [ -n "$_fa2_tier" ]; then
          FA2_FP8KV_SRC="default, off: $_fa2_tier"
          echo "resolve_config: FA2_FP8KV defaults to 0 with $_fa2_tier: the FA2 arm's LBNHC KV layout does not fit the OffloadingConnector, so this stays on int8 + Triton (set FA2_FP8KV=0 to silence this)" >&2
        else
          # The fa2/ build is sm_86 only and measured only on an RTX 3090, so the default is on only
          # when the GPU the server gets is compute capability 8.6.
          _ask_cap
          local _fa2_cap=$_cap
          if [ "$_fa2_cap" = 8.6 ] && [ "${PTH_FA:-0}" = 1 ]; then
            FA2_FP8KV_SRC="PTH_FA=1 overrides the 8.6 default"
          elif [ "$_fa2_cap" = 8.6 ]; then
            export FA2_FP8KV=1
          else
            _fa2_cap=$(printf %s "${_fa2_cap:-unknown}" | tr '\n' /)
            FA2_FP8KV_SRC="default, off: compute capability $_fa2_cap"
            echo "resolve_config: FA2_FP8KV defaults to 0 on compute capability $_fa2_cap: the FA2 arm is built for and measured on 8.6 (RTX 3090) only, so this stays on int8 + Triton (FA2_FP8KV=1 opts in on 8.9; set FA2_FP8KV=0 to silence this)" >&2
          fi
        fi
      fi
    fi
    if [ "${FA2_FP8KV:-0}" = 1 ]; then
      [ "$spec" = dflash2 ] && [ "$ctx" = long ] \
        || _refuse "FA2_FP8KV=1 runs only on SPEC=dflash2 CTX=long (got SPEC=$spec CTX=$ctx)"
      [ "${PTH_FA:-0}" != 1 ] \
        || _refuse "PTH_FA=1 and FA2_FP8KV=1 both pick the dflash2 backend (the int8 kernel against the FA2 fp8 port); unset FA2_FP8KV (PTH_FA=1 overrides its 8.6 default) or set PTH_FA=0"
      [ -z "$_fa2_tier" ] \
        || _refuse "FA2_FP8KV=1 with $_fa2_tier: the FA2 backend supports only the LBNHC KV layout and vLLM's OffloadingConnector needs LBHNC or BLHNC; the CPU tier is untested on it (FA2_FP8KV=0, or no KV offload tier)"
    fi
  else
    kv=${kv:-fp8}
    case "$kv" in fp8|kvarn|int4pth) ;;
      *) _refuse "unknown KV=$kv (want fp8|kvarn|int4pth)" ;;
    esac
    [ -z "${CTX:-}" ] || _warn "CTX=$CTX is set but batch mode ignores it (CTX is a single-user control)"
    [ -z "${SPEC:-}" ] || _warn "SPEC=$SPEC is set but batch mode ignores it (SPEC is a single-user control)"
    [ -z "${PTH_FA:-}" ] || [ "$PTH_FA" = 0 ] \
      || _warn "PTH_FA=$PTH_FA is set but batch mode ignores it (PTH_FA is a single-user control, CTX=long with SPEC=mtp or dflash2)"
  fi

  # 2. EXTRA_ARGS shadow warnings. The launchers expand EXTRA_ARGS after their
  # own flags, so any of these in EXTRA_ARGS silently wins; say so.
  local _shadow="--port | --host | --max-model-len | --max-num-seqs | --gpu-memory-utilization | --kv-cache-dtype | --attention-backend | --served-model-name | --api-server-count | --block-size | --mamba-ssm-cache-dtype | --tool-strict-level "
  local _f
  for _f in $_shadow; do
    case " ${EXTRA_ARGS:-} " in
      *" $_f"*|*" ${_f%= }="*) _warn "EXTRA_ARGS carries ${_f% } which overrides the launcher's own flag (EXTRA_ARGS expands last)" ;;
    esac
  done
  case " ${EXTRA_ARGS:-} " in
    *" --api-key "*|*" --api-key="*)
      _warn "EXTRA_ARGS appears to carry --api-key inline; prefer VLLM_API_KEY or api_key.txt (value not shown)" ;;
  esac

  # 3. Redacted effective configuration. Secrets are presence-only: the key
  # value and api_key.txt contents never reach any output.
  local _model=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
  local _port=${PORT:-18020}
  local _key_status="unset"
  [ -n "${VLLM_API_KEY:-}" ] && _key_status="set(${#VLLM_API_KEY} chars)"
  local _keyfile="absent"
  [ -f "$REPO/api_key.txt" ] && _keyfile="present"
  {
    echo "[effective-config] MODE=$mode"
    [ "$mode" = "single" ] \
      && echo "[effective-config] CTX=$ctx SPEC=$spec FA2_FP8KV=${FA2_FP8KV:-0} (${FA2_FP8KV_SRC:-explicit})" \
      || echo "[effective-config] KV=$kv"
    if [ "$mode" = "single" ] && [ "${PTH_FA:-0}" = 1 ]; then
      local _pth_on=
      case "$spec/$ctx" in
        mtp/long) _pth_on=": target and MTP drafter" ;;
        dflash2/long) _pth_on=": target and DFlash2 drafter" ;;
      esac
      echo "[effective-config] PTH_FA=1 (${PTH_FA_SRC:-explicit}; int8_per_token_head KV on the CUSTOM backend, pthfa/$_pth_on)"
    fi
    echo "[effective-config] MODEL=$_model PORT=$_port"
    echo "[effective-config] MAX_LEN=${MAX_LEN:-<profile default>} MAX_SEQS=${MAX_SEQS:-<profile default>}"
    echo "[effective-config] VLLM_API_KEY=$_key_status api_key.txt=$_keyfile"
  } >&2
}

# Standalone mode (tests, CI dry-run): resolve with the ambient environment.
# When sourced, BASH_SOURCE[0] differs from $0 and only the function loads.
if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
  resolve_effective_config "${1:?usage: $0 batch|single}"
fi
