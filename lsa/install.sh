#!/bin/bash
# Install the LSA plugin (hyperqwen_lsa, pure Python + Triton, no compile) into the venv and fetch the
# distilled map into models/lsa/, checked against PIN. Optional: only LSA=1 uses either; unset, the
# plugin's register() is a no-op.
# usage: bash lsa/install.sh            (idempotent)
#        bash lsa/install.sh --map-only (the map alone, e.g. into a container's models volume)
#        bash lsa/install.sh --no-map   (the plugin alone, e.g. in an image whose models are a volume)
# env:   PY (venv python), MODELS (default: <repo>/models)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY=${PY:-$REPO/venv/bin/python}
MODELS=${MODELS:-$REPO/models}
fail() { echo "lsa/install.sh: $*" >&2; exit 1; }
. "$HERE/PIN"

if [ "${1:-}" != --map-only ]; then
  "$PY" -c 'import vllm, triton' >/dev/null 2>&1 || fail "cannot import vllm and triton with $PY"
  "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip >/dev/null || fail "$PY has no pip"
  "$PY" -m pip install -q --no-deps --no-build-isolation "$HERE" || fail "pip install of hyperqwen_lsa failed"
  "$PY" -c 'from importlib.metadata import entry_points
assert any(e.value == "hyperqwen_lsa:register" for e in entry_points(group="vllm.general_plugins")), "entry point missing"' \
    || fail "the vllm.general_plugins entry point is missing after install"
  echo "hyperqwen_lsa installed (LSA=1 selects it; unset, it is a no-op)"
fi

[ "${1:-}" = --no-map ] && exit 0
MAP=$MODELS/lsa/$LSA_MAP_NAME
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
if [ -f "$MAP" ] && [ "$(sha "$MAP")" = "$LSA_MAP_SHA256" ]; then
  echo "map is current: $MAP"
  exit 0
fi
mkdir -p "$MODELS/lsa"
URL=$LSA_REPO/releases/download/$LSA_MAP_RELEASE/$LSA_MAP_NAME
echo "== fetching $URL"
curl -fsSL --retry 3 -o "$MAP.tmp" "$URL" || fail "cannot download $URL"
[ "$(sha "$MAP.tmp")" = "$LSA_MAP_SHA256" ] || { rm -f "$MAP.tmp"; fail "checksum mismatch for $LSA_MAP_NAME (PIN says $LSA_MAP_SHA256)"; }
mv "$MAP.tmp" "$MAP"
echo "map installed: $MAP"
