#!/bin/bash
# Install the KVarN KV-cache port into this repo's vLLM 0.30.0 venv:
# copies the new modules into site-packages/vllm and applies the upstream hunks.
# usage: bash kvarn/install.sh            (idempotent: re-copying files is fine, and
#        patches/apply.sh --kvarn skips a patch whose every hunk is already in the tree)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY=${PY:-$REPO/venv/bin/python}
SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] || { echo "cannot import vllm with $PY (README: Setup)"; exit 1; }
cp -r "$HERE/files/vllm/." "$SP/"
# patches/apply.sh --kvarn applies kvarn-0.30.0, kvarn-v2-runner-0.30.0 (lets SPEC=dflash2 run with
# CTX=huge: KVarN KV + prefix caching, 240k), kvarn-recycled-pages-0.30.0 (the runner tells KVarN
# which pages moved to another KV-cache group each step, so a late flush never lands on another
# request's mamba state, #208) and kvarn-fp16-dequant-0.30.0 (registers KVARN_FP16_DEQUANT in
# envs.py; the modules above read it through vllm.envs), in that order, at --fuzz 0, after the
# whole patches/ series. It checks each patch with an exact reverse dry-run first, so a rerun is a
# no-op, a venv installed before the fp16 patch existed only gets that patch, and a partly applied
# patch fails by name.
bash "$REPO/patches/apply.sh" --kvarn "$SP"
find "$SP" -type d -name __pycache__ -path "*kvarn*" -prune -exec rm -rf {} + 2>/dev/null || true
"$PY" - <<'PY'
from typing import get_args
from vllm.config.cache import CacheDType
assert "kvarn_k4v2_g128" in get_args(CacheDType)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
print("KVarN backend:", AttentionBackendEnum.KVARN.get_class().get_name())
from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
c = KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", 256)
print("tile bytes", c.tile_bytes, "-> per token per head", c.tile_bytes_aligned // c.group, "B (fp8: 256 B)")
PY
echo "kvarn installed"
