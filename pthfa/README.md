# pthfa: paged int8 per-token-head attention as a vLLM backend plugin

`pthfa/` is the source of pth-flash-attn (Apache-2.0) at one pinned commit (`pthfa/PIN`, a commit id): a CUDA op library that reads vLLM's `int8_per_token_head` KV cache directly (no dequantize
pass) on sm_80/86/89, and a `vllm.general_plugins` entry point that registers it as the backend `CUSTOM`.
`PTH_FA=1` (single-user, `SPEC=mtp CTX=long`) selects it for the target and the MTP drafter; nothing else
does. It is a separate extension, not a patch to vLLM, which is why it lives here and not in `patches/`:
no vLLM file changes, and `patches/check_vllm_series.sh` is untouched by it.

## What is vendored, and from where

`PIN` names the commit. `csrc/` and `python/` are the output of `tools/export_upstream.py --out` at that
commit: the extension's translation units, every header they include, and the `pth_flash_attn` package,
with the test-only handshake hooks stripped (upstream's `tools/prove_export_sass.sh` proves the stripped copy
compiles to the same SASS). `setup.py`, `pyproject.toml`, `tools/build_flags.py`, `LICENSE` and `NOTICE` are
the files at that commit, unchanged. `SHA256SUMS` lists all 32 files; the files are never edited here.

Source rather than a prebuilt `.so`: the same reasoning as #153, where a binary was declined in favour of a
source port CI can check. `bash pthfa/check_vendor.sh` checks the files against `SHA256SUMS` and that nothing
under `pthfa/` is unlisted; `--source <pth-flash-attn checkout>` re-exports `PIN`'s commit and diffs.

## How it is built and installed

`bash pthfa/install.sh` compiles the library against the venv's torch (`TORCH_CUDA_ARCH_LIST` from `PIN`,
SASS only, no PTX, so an sm_90+ card is refused rather than JIT-compiled) and pip-installs the package into
the venv. It needs nvcc 13 and g++, no GPU and no network. It checks the installed copy: cubins for exactly
sm_80/86/89 and no PTX, then `tests/gate_test.py`. A rerun with the same pin, arch list, torch and nvcc is a
no-op. The Dockerfile runs it in its own layer before `COPY . .`, so an edit elsewhere does not recompile it,
and runs the gate test again on the patched tree.

## What it does when nothing selects it

The entry point records a class path under `AttentionBackendEnum.CUSTOM` and imports no CUDA code; vLLM
loads general plugins at debug log level, so the boot log does not change. `CUSTOM` is in none of vLLM's
automatic-selection lists, so only an explicit `--attention-backend CUSTOM` (or `"attention_backend":"CUSTOM"`
in a speculative config) reaches it. If another plugin registered `CUSTOM` first, this one leaves it alone and
says so. `tests/gate_test.py` checks each of these.

## Limits

sm_80, sm_86 and sm_89 only; head sizes 64, 128 and 256; `--kv-cache-dtype int8_per_token_head` only;
causal decoder self-attention without sliding window, ALiBi, attention sinks or logits soft cap; no DCP/PCP.
An explicit selection outside these fails at model load with the reason (`docs/gotchas.md`, item 63). The
library reads no environment variables.

## Tests

- `pthfa/tests/gate_test.py` (CPU): the items above, against the venv's vLLM.
- `bench/test_pthfa_knob.sh` (CPU): `PTH_FA` in the launchers, against a stub `vllm`.
- `bench/test_pthfa_attn.py` (GPU): the op against an fp32 reference on the cache it reads, including a pool
  past block id 2,383 (#86/#109) and CUDA-graph replay; `SKIP(by design)` off sm_8x or without the package.

## Re-vendoring

From a pth-flash-attn checkout at the new commit: `tools/export_upstream.py --out <dir>`, copy `csrc/` and
`python/` and the five build files here, set `PTH_COMMIT` in `PIN`, regenerate `SHA256SUMS`
(`find . -path ./tests -prune -o -type f ! -name README.md ! -name PIN ! -name SHA256SUMS ! -name check_vendor.sh ! -name install.sh -print | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS`),
then `bash pthfa/check_vendor.sh --source <checkout>`.
