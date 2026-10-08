#!/bin/bash
# resolve_api_key.sh - one key-precedence resolver for the scripts that serve the
# API and the scripts that call it (issue #113: bench/warmup.sh re-derived the
# launcher's chain, and a mismatch means every request 401s).
#
# Sourced after REPO is set. It defines functions and sets nothing by itself, so
# each caller opts into the side it needs:
#
#   resolve_vllm_key    server side. vLLM binds `--api-key` from VLLM_API_KEY, so
#                       this is the key the server will require. api_key.txt is
#                       the documented file fallback and an exported
#                       VLLM_API_KEY wins. No placeholder: with no key anywhere,
#                       the server binds none, and a placeholder here would make
#                       it demand a key nobody configured.
#
#   resolve_client_key  client side. A client presents OPENAI_API_KEY, so it must
#                       equal the key the server bound. An explicit
#                       OPENAI_API_KEY wins (the client may point at a server on
#                       another host, where the local file is not the right key),
#                       else the server key, else api_key.txt, else a harmless
#                       placeholder - a value to send to a server that bound no
#                       key and ignores it. This reads the server side but does
#                       not export it, so a client process gains no variable the
#                       launcher did not set.

resolve_vllm_key() {
  if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$REPO/api_key.txt" ]; then
    export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
  fi
}

resolve_client_key() {
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -n "${VLLM_API_KEY:-}" ]; then
      export OPENAI_API_KEY="$VLLM_API_KEY"
    elif [ -f "$REPO/api_key.txt" ]; then
      export OPENAI_API_KEY="$(cat "$REPO/api_key.txt")"
    else
      export OPENAI_API_KEY="EMPTY"
    fi
  fi
}
