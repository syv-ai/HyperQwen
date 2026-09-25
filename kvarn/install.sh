#!/bin/bash
# Install the KVarN KV-cache port into this repo's vLLM 0.30.0 venv:
# copies the new modules into site-packages/vllm and applies the upstream hunks.
# usage: bash kvarn/install.sh            (idempotent-ish: re-copying files is fine;
#        the patch is applied with --forward so a second run is a no-op)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY=${PY:-$REPO/venv/bin/python}
SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] || { echo "cannot import vllm with $PY (README: Setup)"; exit 1; }
cp -r "$HERE/files/vllm/." "$SP/"
# --fuzz 0 and a FAILED grep: both files are exported from the fork branch at their position after the
# whole patches/ series, so a hunk that needs slack is a hunk cut against a tree this is not. `patch -N`
# exits 1 for "already applied" too (a rerun), so the exit code cannot be the signal; the text can.
apply_kvarn() { local out; out=$(patch -p1 -N --fuzz 0 -r /dev/null -d "$SP" < "$HERE/$1" 2>&1) || true; echo "$out"
  case "$out" in *FAILED*) echo "ERROR: $1 has a hunk that does not apply to this vLLM tree (re-cut it against the pin)" >&2; exit 1 ;; esac; }
apply_kvarn kvarn-0.30.0.patch
# V2-runner port: lets SPEC=dflash2 run with CTX=huge (KVarN KV + prefix caching, 240k).
# Depends on hunks from both the patches/ set and kvarn-0.30.0.patch, hence applied last.
apply_kvarn kvarn-v2-runner-0.30.0.patch
find "$SP" -type d -name __pycache__ -path "*kvarn*" -prune -exec rm -rf {} + 2>/dev/null || true
"$PY" - "$SP" "$HERE" <<'PY'
import sys
from pathlib import Path
from typing import get_args
from vllm.config.cache import CacheDType
assert "kvarn_k4v2_g128" in get_args(CacheDType)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
print("KVarN backend:", AttentionBackendEnum.KVARN.get_class().get_name())
from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
c = KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", 256)
print("tile bytes", c.tile_bytes, "-> per token per head", c.tile_bytes_aligned // c.group, "B (fp8: 256 B)")

# Check the result, not the exit code. `patch --forward` cannot tell "already
# applied" (a legitimate rerun) from "does not apply" (a vLLM tree this patch
# was not cut against) -- both exit non-zero -- which is why the calls above
# end in `|| true`. That makes a stale hunk fail SILENTLY: you get a partial
# port and no warning. Every hunk here adds a port(kvarn-v2) marker, so
# counting them per file says exactly which ones did not land.
sp, here = Path(sys.argv[1]), Path(sys.argv[2])
want, current = {}, None
for line in (here / "kvarn-v2-runner-0.30.0.patch").read_text().splitlines():
    if line.startswith("+++ b/"):
        current = line[len("+++ b/"):].strip()
        want.setdefault(current, 0)
    elif current and line.startswith("+") and "port(kvarn-v2)" in line:
        want[current] += 1
short = []
for rel, expected in sorted(want.items()):
    target = sp / rel
    found = target.read_text().count("port(kvarn-v2)") if target.is_file() else 0
    if found < expected:
        short.append(f"  {rel}: {found}/{expected} markers")
if short:
    print("\nERROR: kvarn-v2-runner-0.30.0.patch did not apply completely:", file=sys.stderr)
    print("\n".join(short), file=sys.stderr)
    print(
        "\nThis vLLM tree differs from the one the patch was cut against.\n"
        "Re-cut the failing hunks before using SPEC=dflash2 CTX=huge -- a\n"
        "partially applied port runs, but silently drops correctness fixes.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"kvarn-v2 runner port complete ({sum(want.values())} markers across {len(want)} files)")
PY
echo "kvarn installed"
