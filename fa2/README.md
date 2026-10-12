# fa2/: fp8-KV FlashAttention-2 for sm86

Source of the fp8 e4m3 KV-cache attention path for `FLASH_ATTN` (Q stays bf16, the fp8 KV becomes bf16 in
shared memory, head dims 128 and 256, split-KV and GQA packing for decode and verify). It is carried here
and checked in CI against the flash-attention commit vLLM pins. Tracked in syv-ai/HyperQwen#153, where the
maintainer decided against adopting the prebuilt plugin binary and asked for a source port that CI checks
against pinned upstream, like `patches/`.

This directory holds the sources and the check. Building and installing them is not part of it yet.

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

## Licences

`LICENSE` (Apache-2.0) and `NOTICE` are the plugin's. The vendored FlashAttention headers keep their BSD-3-Clause
notices (`licenses/flash-attention-BSD-3-Clause.txt`, `licenses/flash-attention-AUTHORS`). CUTLASS and FlashInfer are
build dependencies and are not relicensed.
