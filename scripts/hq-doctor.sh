#!/bin/bash
# hq-doctor.sh — read-only status for HyperQwen. Prints nothing secret:
# VLLM_API_KEY is presence-only (length, never value).
#
#   bash scripts/hq-doctor.sh            # host side (venv or WSL checkout)
#   docker compose run --rm single verify # deep checks instead (patches, model)
#
# Checks, in order: effective config (resolve_config.sh), GPU, port occupant,
# /health, last server error in logs. Exit 0 = server healthy, 1 = not (or
# no server), 2 = script environment problem.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RC=0
say()  { printf '%s\n' "$1"; }
warn() { printf 'doctor: WARNING: %s\n' "$1"; }

say "== config"
if [ -f "$REPO/resolve_config.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO/resolve_config.sh" || { say "doctor: cannot source resolve_config.sh"; exit 2; }
  # Mode is advisory here: report the single-user resolution; batch users
  # get the same port/key/GPU sections below regardless.
  resolve_effective_config single 2>&1 | sed 's/^/  /' || warn "unvalidated config (see above)"
else
  warn "resolve_config.sh missing"
fi
if [ -f "$REPO/.env" ]; then say "  .env: present"; else warn ".env missing (copy .env.example; see make keygen)"; fi
if grep -q '^VLLM_API_KEY=.\+' "$REPO/.env" 2>/dev/null; then
  LEN=$(grep '^VLLM_API_KEY=' "$REPO/.env" | tail -1 | cut -d= -f2 | wc -c)
  say "  VLLM_API_KEY: set (~$((LEN - 1)) chars, value not shown)"
else
  warn "no VLLM_API_KEY entry in .env (the exported environment or api_key.txt may still supply one)"
fi

say "== gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,power.limit --format=csv,noheader 2>/dev/null | sed 's/^/  /' || warn "nvidia-smi failed"
else
  warn "nvidia-smi not on PATH (WSL: run inside the distro; Windows: drivers)"
fi

PORT=${PORT:-18020}
say "== port $PORT"
if command -v ss >/dev/null 2>&1; then
  # Read the whole listing: grep -q can SIGPIPE the producer under pipefail.
  if ss -ltn 2>/dev/null | grep ":$PORT " >/dev/null; then
    ss -ltnp 2>/dev/null | grep ":$PORT " | sed 's/^/  occupant: /' || true
  else
    say "  free (nothing listening)"
  fi
else
  warn "ss missing: cannot preflight port (bind failure would surface minutes late, after compile)"
fi

say "== health"
CRC=0
CODE=$(curl -s --connect-timeout 5 --max-time 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null) || { CRC=$?; CODE=000; }
say "  GET /health -> HTTP $CODE"
if [ "$CODE" = 200 ]; then
  say "  healthy (note: /health 200 can precede first-prompt readiness by ~1 min on a cold boot)"
  exit 0
fi

say "== last error (docker logs, if any)"
if command -v docker >/dev/null 2>&1; then
  for c in hyperqwen-single-1 hyperqwen-batch-1; do
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -x "$c" >/dev/null; then
      say "  -- $c --"
      docker logs --tail 25 "$c" 2>&1 | grep -iE 'error|fail|refus|traceback|killed|terminated|uv[as]|out of memory|no such|not found' | tail -8 | sed 's/^/  /' || true
    fi
  done
else
  warn "docker not on PATH"
fi
RC=1
if [ "$CODE" = 000 ] && [ "$CRC" = 28 ]; then
  say "doctor: /health check on :$PORT timed out (busy, unreachable or stuck; watch make logs, do NOT restart mid-compile)"
elif [ "$CODE" = 000 ]; then
  say "doctor: no server on :$PORT (start: make up-single; follow: make logs)"
else
  say "doctor: server on :$PORT answers HTTP $CODE (not ready; watch make logs, do NOT restart mid-compile)"
fi
exit $RC
