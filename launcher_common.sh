#!/bin/bash
# launcher_common.sh - code shared by the three server launchers
# (single-user/start_qwen.sh, batch/start_qwen.sh, single-user/alternative.sh).
#
# Sourced after REPO is set. It defines functions and sets nothing by itself.
#
#   resolve_bind_host   after resolve_vllm_key (resolve_api_key.sh). Sets BIND_HOST, the
#                       --host the launchers pass. An explicit HOST wins. Else a
#                       server with a key listens on every interface (0.0.0.0,
#                       as before), and one without a key listens on 127.0.0.1
#                       only, because with no key nothing on the network is
#                       protected (#204). Inside a container the default stays
#                       0.0.0.0, since a published port cannot reach a loopback
#                       bind; there the port mapping is what limits exposure.
#
#   qwen_int8_exports   <act> <layers>, where the launcher sets INT8_ACT and INT8_LAYERS
#                       (each launcher keeps its own defaults). Exports
#                       VLLM_MARLIN_INPUT_DTYPE=<act> and, only with it,
#                       VLLM_MARLIN_INT8_INCLUDE_RE=<layers>. The engine reads the
#                       regex only when the dtype is set
#                       (patches/marlin-int8-layer-select.patch); alone it only
#                       changes vLLM's compile cache key. "Off" for these is UNSET,
#                       not empty. vllm/envs.py registers VLLM_MARLIN_INPUT_DTYPE
#                       through env_with_choices(..., None, ["int8", "fp8"]), which
#                       rejects "" outright -- `ValueError: Invalid value '' ...
#                       Valid options: ['int8', 'fp8']` -- so exporting the empty
#                       string killed the engine at startup instead of turning the
#                       feature off. That is the documented way to disable it
#                       (issue #20), so export only when non-empty.
#
#   qwen_kv_offload_args  KV_OFFLOAD_GB=N becomes --kv-offloading-size N at the front of
#                       EXTRA_ARGS: a CPU tier of N GiB of pinned RAM behind the GPU
#                       prefix cache (vLLM's OffloadingConnector). An evicted prefix comes
#                       back by a PCIe load instead of a recomputed prefill, and a
#                       preempted request resumes from it. An explicit
#                       --kv-offloading-size or --kv-transfer-config in EXTRA_ARGS wins.
#                       Called before anything reads EXTRA_ARGS, so the allocator and
#                       retention defaults the launchers pick for a tier see this one.
#
#   qwen_kv_offload_shm_check  after the stale-region sweep. vLLM backs the tier with a
#                       file in /dev/shm (vllm_offload_<engine_id>.mmap) and pins all of
#                       it at boot, so a tmpfs with less room than the tier fails the
#                       boot. Warns, does not refuse: df cannot see a per-user tmpfs
#                       quota, which is the other limit (docs/gotchas.md).
#
#   qwen_exec           the launcher's last call, in place of exec. With PRINT_ARGV=1 it
#                       prints the argv, one argument per line, and exits 0 instead
#                       of starting vLLM. Every check, default and warning before it
#                       has already run, so this is a dry run of the launcher that
#                       needs no GPU. The environment is not printed: it holds
#                       VLLM_API_KEY.

resolve_bind_host() {
  if [ -n "${HOST:-}" ]; then
    BIND_HOST=$HOST
    if [ -z "${VLLM_API_KEY:-}" ]; then
      case "$HOST" in 127.*|localhost|::1) ;; *)
        echo "WARNING: no API key and HOST=$HOST: anything that can reach this port can use the server. Set VLLM_API_KEY or api_key.txt (openssl rand -hex 24)." >&2 ;;
      esac
    fi
  elif [ -n "${VLLM_API_KEY:-}" ]; then
    BIND_HOST=0.0.0.0
  elif [ -f /.dockerenv ]; then
    BIND_HOST=0.0.0.0
    echo "WARNING: no API key: this container listens on 0.0.0.0, so whatever the published port reaches is open. Set VLLM_API_KEY (make keygen) or publish the port on 127.0.0.1 only." >&2
  else
    BIND_HOST=127.0.0.1
    echo "no API key: binding 127.0.0.1 only. To serve other machines, set VLLM_API_KEY (openssl rand -hex 24 > api_key.txt) or HOST=0.0.0.0." >&2
  fi
}

# if, not && lists: alternative.sh runs under set -e, and a function whose last
# && list fails returns 1 and stops the launcher.
qwen_int8_exports() {
  if [ -n "$1" ]; then
    export VLLM_MARLIN_INPUT_DTYPE=$1
    if [ -n "$2" ]; then export VLLM_MARLIN_INT8_INCLUDE_RE=$2; fi
  fi
}

qwen_exec() {
  if [ "${PRINT_ARGV:-0}" = 1 ]; then
    printf '%s\n' "$@"
    exit 0
  fi
  exec "$@"
}

qwen_kv_offload_args() {
  if [ -n "${KV_OFFLOAD_GB:-}" ] && [ "$KV_OFFLOAD_GB" != 0 ]; then
    case " ${EXTRA_ARGS:-} " in
      *"--kv-offloading-size"*|*"--kv-transfer-config"*)
        echo "KV_OFFLOAD_GB=$KV_OFFLOAD_GB ignored: EXTRA_ARGS already configures a KV connector" >&2 ;;
      *) EXTRA_ARGS="--kv-offloading-size $KV_OFFLOAD_GB ${EXTRA_ARGS:-}" ;;
    esac
  fi
}

qwen_kv_offload_shm_check() {
  local gib need avail
  gib=$(printf %s " ${EXTRA_ARGS:-}" | sed -En 's/.* --kv-offloading-size[= ]([0-9.]+).*/\1/p')
  [ -n "$gib" ] || return 0
  need=$(awk -v g="$gib" 'BEGIN { printf "%d", g * 1073741824 }')
  avail=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -n1 | tr -d ' ')
  if [ -n "$avail" ] && [ "$avail" -lt "$need" ]; then
    echo "WARNING: the KV offload tier needs ${gib} GiB in /dev/shm and $((avail / 1073741824)) GiB is free:" \
         "the boot will fail allocating it. Grow the tmpfs (fstab: tmpfs /dev/shm tmpfs" \
         "defaults,nosuid,nodev,size=<N>G 0 0, then mount -o remount /dev/shm) or lower the tier." >&2
  fi
}
