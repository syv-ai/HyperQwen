#!/usr/bin/env bash
# Single-user serving wrapper: boot the launcher, wait for /health, run the
# post-boot serving warmup, then wait on the server.
#
# Why: start_qwen.sh ends in `exec vllm serve`, so a systemd unit has no
# after-boot hook. /health can return 200 before the serving path has actually
# been exercised, and two things are still owed at that point:
#
#   * The transient allocation of the first real batch. On a tightly packed
#     24 GB config that bill is due either here, under controlled conditions,
#     or inside a user's first prompt (gotcha 35).
#   * The Triton variants the boot-time profile does not cover. #48's prewarm
#     closed the CTX=huge set; gotcha 45's scope note records three kernels
#     (_k_stats_kernel, _k_quant_kernel, _prefill_attn_kernel) still compiling
#     in-request on the current production profile (CTX=fast, INT8_ACT=int8
#     PREFILL_ATTN=int8). The real home for those is a prewarm patch; until
#     one exists, a warmup pass absorbs them.
#
#   WARMUP=1             wait for /health, run bench/warmup.sh, then serve
#   WARMUP=0 (default)   just wait on the server (what the unit does today)
#
# The warmup is advisory: a health timeout or a failed warmup logs and falls
# through to serving, because a server that answers /health is more useful than
# a restart loop. The wrapper's exit code is the server's, so Restart=on-failure
# behaves as before. To enable it in the unit:
#   Environment=WARMUP=1
#   ExecStart=/bin/bash %h/qwen-serving/single-user/qwen-server.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$DIR")"
PORT=${PORT:-18020}
WARMUP=${WARMUP:-0}
WARMUP_ATTEMPTS=${WARMUP_ATTEMPTS:-120}
WARMUP_INTERVAL=${WARMUP_INTERVAL:-5}

# start_qwen.sh ends in `exec vllm serve`, so its pid becomes the server's
# and `wait` below tracks it.
bash "$DIR/start_qwen.sh" &
SERVER_PID=$!

# Forward the signal and block until the engine is actually gone: it holds
# ~23 GiB, and the unit's ExecStartPre gate waits for the card to fall under
# 1000 MiB before the next boot. A non-blocking wait here lets the wrapper exit
# first and races that gate.
cleanup() {
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$SERVER_PID" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" = "1" ]; then
    echo "[server] waiting for server health..."
    HEALTHY=0
    for i in $(seq 1 "$WARMUP_ATTEMPTS"); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[server] server exited before becoming healthy" >&2
            # Reap the child and exit with its status (nonzero restarts the unit).
            wait "$SERVER_PID"
            exit $?
        fi
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
            HEALTHY=1
            echo "[server] HTTP 200 OK"
            break
        fi
        sleep "$WARMUP_INTERVAL"
    done
    if [ "$HEALTHY" != "1" ]; then
        # Do not tear down a boot that is merely slow: systemd's
        # TimeoutStartSec bounds a boot that never completes.
        echo "[server] health timeout after $((WARMUP_ATTEMPTS * WARMUP_INTERVAL))s, serving anyway" >&2
    # warmup.sh inherits this wrapper's environment (MODEL, HOST, API key,
    # VLLM_MODEL) and re-derives the same defaults as start_qwen.sh, so it
    # targets the server just started; only PORT needs pinning to the port
    # checked above.
    elif PORT="$PORT" bash "$REPO/bench/warmup.sh"; then
        echo "[server] Warmup OK"
        echo "[server] server ready for traffic"
    else
        # warmup.sh finds its venv relative to its tree and reads the served
        # name from /v1/models; a deploy dir that has drifted, or a transient
        # 401, must not turn a healthy engine into a restart loop.
        echo "[server] warmup failed, serving anyway" >&2
    fi
fi

wait "$SERVER_PID"
