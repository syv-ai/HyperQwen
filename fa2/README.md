# fa2/: fp8-KV FlashAttention-2 for sm86

Source of the fp8 e4m3 KV-cache attention path for `FLASH_ATTN` (Q stays bf16, the fp8 KV becomes bf16 in
shared memory, head dims 128 and 256, split-KV and GQA packing for decode and verify). It is carried here
and checked in CI against the flash-attention commit vLLM pins. Tracked in syv-ai/HyperQwen#153, where the
maintainer decided against adopting the prebuilt plugin binary and asked for a source port that CI checks
against pinned upstream, like `patches/`.

This directory holds the sources, the check, the build and the adapter. Nothing uses it unless `FA2_FP8KV=1`
(the launcher knob, a later change): the installed plugin's `register()` is a no-op otherwise.

## Provenance

- Plugin: [AntonProkopyev/fa2-fp8kv-sm86](https://github.com/AntonProkopyev/fa2-fp8kv-sm86) at `5c3a7d8`
  (Apache-2.0). Its code is the `7646a2d` build; the two commits after it change docs only.
- Carried on top, in its own commit: the LSE and index-widening fix from that repo's PR #2
  (`headers/flash_fwd_kernel.h` in full, and the combine-grid `int64_t` in `fp8_attn.cu`). Not carried: the
  2048 `max_q` cap removal and the `max_query_tokens` op from the same PR. If PR #2 merges, rebase onto it.
- Layout differs from the plugin: `headers/` and the two `.cu` files are as in the plugin, the adapter is under
  `python/`, the patch is `fa2-fp8kv.patch`, and `pyproject.toml`'s `package-dir` points at `python/`.

## What differs from upstream FlashAttention

`csrc/flash_attn/src` of vllm-project/flash-attention:

- Four modified headers: `flash.h`, `flash_fwd_kernel.h`, `mask.h`, `utils.h`.
- Twelve headers byte-identical to upstream.
- Two new entrypoints with no upstream file: `fp8_attn.cu` (`torch.ops.fa2_fp8kv.forward`) and `prefill.cu`
  (`fa2_fp8kv_prefill.gather`).

`fa2-fp8kv.patch` is that diff (`headers/<f>` maps to `csrc/flash_attn/src/<f>`). The adapter
(`python/`) and `build.py` are this project's own and have no upstream counterpart.

## Pins (`PIN`)

| | |
|---|---|
| `FA_COMMIT` | `9cd61de`, the flash-attention commit vLLM 0.31.0 pins (`cmake/external_projects/vllm_flash_attn.cmake`) |
| `FA_PLUGIN_BASE` | `28e862d`, the commit the plugin's patch was cut against; `csrc/flash_attn/src` is identical at both |
| `CUTLASS_COMMIT` | `62750a2`, the CUTLASS the build needs |
| `VLLM_VERSION` | `0.31.0`, must equal the `vllm==` pin in `docker/requirements.txt` |

## Check

No GPU. Needs a git checkout (not an archive) of flash-attention at `FA_COMMIT`:

```bash
git init fa && git -C fa remote add origin https://github.com/vllm-project/flash-attention
git -C fa fetch --depth=1 origin "$(sed -n 's/^FA_COMMIT=//p' fa2/PIN)" && git -C fa checkout --detach FETCH_HEAD
python3 fa2/check_upstream.py --fa-dir fa
```

It checks the checkout is at `FA_COMMIT`, regenerates the diff from git blobs and compares it with
`fa2-fp8kv.patch`, applies the patch to a temporary tree and compares all 18 mapped files byte for byte.
After a deliberate change to a header or `.cu` file, regenerate the patch with
`python3 fa2/check_upstream.py --fa-dir fa --emit > fa2/fa2-fp8kv.patch.tmp && mv fa2/fa2-fp8kv.patch.tmp fa2/fa2-fp8kv.patch`.

The `patch-integrity` workflow runs this, and first checks that the pinned vLLM tag still pins `FA_COMMIT`.
When a vLLM bump moves that pin, re-run the check against the new commit and update `PIN`.

## Build and install

`bash fa2/install.sh` into the repo venv (`PY=...` for another interpreter). No GPU. It needs nvcc 13
(a system one, `CUDA_HOME`, or the pip `nvidia-cuda-nvcc==13.0.88` already in the venv per
`docker/requirements.txt`), a host C++ compiler and git. It:

1. fetches CUTLASS `CUTLASS_COMMIT` (`include/` only) into `~/.cache/hyperqwen-fa2` and checks the commit;
   `CUTLASS_PATH` points at an include dir you already have instead;
2. runs `build.py --pipeline --prefill` unchanged, from a copy under that cache, for
   `TORCH_CUDA_ARCH_LIST` (default 8.6) with `MAX_JOBS` (default 2). `fa2/` itself stays clean;
3. installs the `fa2_fp8kv_vllm` package with `fa2_fp8kv.so` and `fa2_fp8kv_prefill.so` inside it, and the
   `vllm.general_plugins` entry point, with `pip install --no-deps --no-build-isolation`.

Measured in the Docker build on this repo's builder: about 3 minutes of nvcc, `fa2_fp8kv.so` 9.7 MB and
`fa2_fp8kv_prefill.so` 0.2 MB.

Re-running skips the compile while both libraries are newer than every source and were built for the same
torch, nvcc, arch, CUTLASS and flashinfer (a stamp file). `FA2_FP8KV_LIBRARY` and `FA2_FP8KV_PREFILL_LIBRARY`
override the library paths. The Docker image runs the same script before `COPY . .`.

`FA2_FP8KV=1` is the gate. Unset (or any other value) `register()` does nothing, the libraries are not
loaded and `FLASH_ATTN` is the stock backend. Set, `FLASH_ATTN` becomes `Fp8FlashAttentionBackend`: for an fp8
cache on sm86/89/120 it narrows the KV layout to `LBNHC` and runs the fp8 op, for any other cache dtype it
behaves as the stock backend. `python -c 'import fa2_fp8kv_vllm.paths as p; p.check()'` exits non-zero
when a library is missing.

## GPU checks

`tests/` holds the plugin author's numerical checks, copied from `5c3a7d8`. `check_full.py`, `check_window.py`
and `check_replay.py` take `--library`/a path argument that now defaults to the installed library; `check_full.py`
also has three extra cases at the end of its default list for this model's shapes (`hq-*`). Run them on the
GPU, from the repo root, with the venv python:

```bash
export FA2_FP8KV=1                              # check_backend and check_mixed_prefill load the libraries through the gate
venv/bin/python fa2/tests/check_full.py          # add --full for the 262k-key and 2048-query cases
venv/bin/python fa2/tests/check_window.py        # noncausal 2048 window, CUDA-graph replay, discarded pages
venv/bin/python fa2/tests/check_replay.py        # ragged batches and changing metadata under graph replay
venv/bin/python fa2/tests/check_backend.py       # KV write, attention and graph replay through the adapter
venv/bin/python fa2/tests/check_mixed_prefill.py --device cuda
```

For this model the target is D256 with 24 q and 4 KV heads, and the drafter D128 with 8 KV heads and the
noncausal 2048 window (TP=1). `check_window.py` (D128, 4 KV heads) and `check_replay.py` (D256) match them most
directly, `check_full.py` covers both head dims, and the `hq-*` cases use the 4 and 8 KV head counts (the
drafter's q-head count there is a guess). `check_mixed_prefill.py --device cpu` needs no GPU and runs in the
image build, with `tests/gate_test.py` (the gate and the layout narrowing).

## Licences

`LICENSE` (Apache-2.0) and `NOTICE` are the plugin's. The vendored FlashAttention headers keep their BSD-3-Clause
notices (`licenses/flash-attention-BSD-3-Clause.txt`, `licenses/flash-attention-AUTHORS`). CUTLASS and FlashInfer are
build dependencies and are not relicensed.
