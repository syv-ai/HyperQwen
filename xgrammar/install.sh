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
# apache-tvm-ffi 0.1.16 (2026-11-11) exports cxx_std_20 from its CMake targets, so the bindings' tvm_ffi.cc compiles as
# C++20 while xgrammar's own CMakeLists pins C++17 for the rest. libstdc++'s std::variant differs between the two, and
# the -flto -Werror link fails on -Werror=odr. Hold the build environment (PIP_BUILD_CONSTRAINT on pip >= 25.3, which
# no longer reads PIP_CONSTRAINT for the isolated build env; PIP_CONSTRAINT on older pip) and the venv's runtime copy
# to the last C++17 series. Drop this with the directory, or when PIN moves to an xgrammar that builds as C++20.
TVM_FFI='apache-tvm-ffi<0.1.16'
echo "$TVM_FFI" > "$W/constraints.txt"
PIP_BUILD_CONSTRAINT="$W/constraints.txt" PIP_CONSTRAINT="$W/constraints.txt" "$PY" -m pip wheel -q --no-deps -w "$W/dist" "$W/src"
"$PY" -m pip install -q "$TVM_FFI"
"$PY" -m pip install -q --no-deps --force-reinstall "$W"/dist/xgrammar-*.whl
"$PY" "$HERE/check.py"
"$PY" -c "import vllm.v1.structured_output.backend_xgrammar"
