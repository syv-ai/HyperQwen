#!/bin/bash
# Build xgrammar from mlc-ai/xgrammar#989 plus the Qwen template-newline fix on top of it (xgrammar/README.md) and
# install it over the release vLLM pins. Without it, a maxLength on a tool's string parameter puts Qwen's XML tool
# grammar on a slow path under TOOL_STRICT=parameter: ~78 ms per token mask, decode ~170 -> ~7 tok/s.
# Needs git, a C++17 compiler and network (pip fetches the build backend). No GPU.
# usage: bash xgrammar/install.sh        env: PY (venv python, default venv/bin/python)
set -e
PY=${PY:-venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
read -r REPO COMMIT < "$HERE/PIN"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
git -C "$W" init -q src
git -C "$W/src" fetch -q "$REPO" "$COMMIT"
# the release tags live upstream, not on the fork; the wheel's version comes from git describe
git -C "$W/src" fetch -q --tags https://github.com/mlc-ai/xgrammar
git -C "$W/src" checkout -q "$COMMIT"
[ "$(git -C "$W/src" rev-parse HEAD)" = "$COMMIT" ] || { echo "xgrammar: fetched $(git -C "$W/src" rev-parse HEAD), PIN says $COMMIT"; exit 1; }
git -C "$W/src" submodule update -q --init --recursive --depth 1   # dlpack, picojson
"$PY" -m pip wheel -q --no-deps -w "$W/dist" "$W/src"
"$PY" -m pip install -q --no-deps --force-reinstall "$W"/dist/xgrammar-*.whl
"$PY" "$HERE/check.py"
"$PY" -c "import vllm.v1.structured_output.backend_xgrammar"
