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
  else
    kv=${kv:-fp8}
    case "$kv" in fp8|kvarn|int4pth) ;;
      *) _refuse "unknown KV=$kv (want fp8|kvarn|int4pth)" ;;
    esac
    [ -z "${CTX:-}" ] || _warn "CTX=$CTX is set but batch mode ignores it (CTX is a single-user control)"
    [ -z "${SPEC:-}" ] || _warn "SPEC=$SPEC is set but batch mode ignores it (SPEC is a single-user control)"
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
      && echo "[effective-config] CTX=$ctx SPEC=$spec" \
      || echo "[effective-config] KV=$kv"
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
