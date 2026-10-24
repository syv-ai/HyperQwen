# The vLLM 0.32.0 pin

What the move from 0.31.0 to 0.32.0 changed in this repo, what was re-measured, and what it costs.

[← back to the main README](../README.md)

## Dependencies

`vllm==0.32.0`. torch 2.13.0 and FlashInfer 0.7.0.post1 are unchanged from 0.31.0. vLLM 0.32.0 needs transformers
>=5.16.1,<5.20.0 and tokenizers >=0.23.2, so `docker/requirements.txt` moves to transformers 5.19.0 and tokenizers
0.23.3. It pins `xgrammar==0.2.8`; the image builds xgrammar from source instead (below). `flashinfer-cubin` is no
longer in vLLM's `requirements/cuda.txt` (0.32.0 installs FlashInfer's kernels with `vllm download-kernels`, #58765), so
`scripts/pin-bump.py` reads the cubin pin from the `flashinfer-python` pin, which the cubin package has to equal
anyway (#372). `fa2/PIN` moves to 0.32.0: vLLM 0.32.0 builds flash-attention at 9cd61de, the same commit as 0.31.0, so
`fa2/` is checked against unchanged sources.

## Patch series

54 files (50 in the series, 4 KVarN), re-exported from a vLLM branch that carries the series as one commit per topic on
v0.32.0, each re-cut against 0.32.0 (`Cut-against`) with a note where upstream moved code under it. Applied to
pristine v0.32.0, the series plus the KVarN files reproduce that branch's `vllm/` tree with 0 differing files. The rebase stopped 16 times; each stop's upstream cause and resolution:

Retired, because upstream now does it:

- `serve-404-served-names`: our vllm #59889 is in 0.32.0 (with the wording `Valid aliases:`). The patch, its series
  line and its PATCHES.md row are gone; `verify.sh` checks the upstream wording.
- Half of `mamba-chunked-prefill-align`: its `chunk_o.py` hunk, because #60502 loads `g` with `mask=m_t, other=0.0`,
  zeroing the lanes the topic zeroed. Its `mamba_utils.py` half stays.
- Half of `kvarn-v2-runner-0.32.0`: its `mamba_hybrid.py` hunk, because #55601 seeds the hybrid Mamba state index with
  `cache_config.mamba_block_size`, the same fix.

Re-cut where upstream moved code beside ours:

- `spec-decode-attn` and `prefill-attn-int8`: #61126 removed cascade attention, so both branches move out one
  indentation level in `flash_attn.py`; the added lines are the topics' own, 0 removed.
- `hybrid-sw-block-promote`: #57169 deleted the hunk's anchor; `_promote_indivisible_block_sizes` keeps its place before
  `unify_kv_cache_spec_page_size`.
- `qwen3_5-mtp-draft-vocab`: #60152's `is_unused_checkpoint_weight` check runs first, then the draft vocab head's.
- `kvarn-0.32.0`: #57057 (UltraQuant 4-bit KV) took `KVQuantMode` value 11, which was KVarN's. An `IntEnum` duplicate is a
  silent alias, so KVarN would have read as UltraQuant; KVarN is now 100 (no code reads the number).
- `engine-stall-sentinel`, `cudagraph-memory-from-allocator`, `pinned-kv-empty-cache`, `qwen3_5-embed-uva`: re-anchored
  beside #59781, #59160, 0.32's `set_current_vllm_config` wrap and #60623 (Transformers now ships the Qwen 3.5 configs).
- `vllm-pr40371-prompt-progress`: upstream added an `error` parameter to `make_request_output` and
  `_new_request_output` where the topic adds `num_computed_tokens`. Two call sites passed the count positionally, which
  after the merge would bind it to `error` and drop progress without an error. Both now pass it by keyword.

Merged, with a cost on one card family:

- `spec-decode-scratch-*`: #58482 sizes the Triton backend's 3D scratch for 64 segments on SM120 (RTX 50). This series'
  pool owns that scratch at 16 segments, so the 64-segment mode is off in this backend and every card runs 16, as on
  0.31.0. RTX 50 cards using Triton attention lose #58482's occupancy gain; results are unchanged. A pool that carries
  segments would be its own change. No RTX 50 card was available to measure it.

Checks with no card: `scripts/port-removed-names.py` finds 1 removed name in our added lines, a docstring word, passed
with `--accept`; `scripts/port-changed-signatures.py` finds 0 calls that no longer bind over the 197 it can resolve.
Three breaks those checks cannot see were found by boots and an override scan, and are fixed:
`DFlash2Speculator.propose()` must take #60754's `dp_sync_state` and `num_speculative_tokens` (SPEC=dflash2 was dead at
engine start); `next_num_draft_tokens()` can return `None`, which `min()` refused in warm-up; and 0.32.0 passes a
`kv_cache_spec` to `get_supported_kernel_block_sizes`, which KVarN's override did not take (CTX=huge dead at start). Two
more were outside the series: `fa2/`'s build-time test built `FlashAttentionMetadata` with the cascade fields #61126 removed,
and `verify.sh`'s `has_flashinfer()` check needs the venv's `bin/` on `PATH` since #58360 (it looks for ninja), as the
launchers already run it. The vendored `pthfa/` imports no name 0.32.0 removed, and its overrides match their bases.

## xgrammar

vLLM 0.32.0 pins xgrammar 0.2.8, on which a tool schema's `maxLength` drops decode to ~7 tok/s under
`TOOL_STRICT=parameter`. The image builds xgrammar from source with the bounded-string fix (mlc-ai/xgrammar#989) and
the Qwen tool-call framing fix (`xgrammar/`, `bash xgrammar/install.sh`). On this image (an RTX 4090, the same request,
one run per row): 181.2, 180.9 and 138.2 tok/s at `maxLength` 200000, 20000 and 2000, against 5.9, 6.9 and 7.4 on 0.2.8,
with the same tool-call arguments on both builds.

## Acceptance

0.31 arm: 0.31.0 with this repo's series as of main before the port (and, for the last night, syv-ai main ce9615c).
0.32 arm: this branch. Two cards: a native RTX 3090 and a native RTX 4090. One boot per cell unless stated.

- **Every profile boots at its shipped defaults on the 3090**, and the KV pools are equal or larger on 0.32.0 (batch
  221,907 tokens against 217,268; `alternative.sh` 307,058 against 300,675).
- **Tokens per step at default temperature, seeded, two boots per image:** `alternative.sh` +0.2%, DFlash2 k7 -0.9% on
  the 3090; single-user +0.6%, `alternative.sh` -0.2%, DFlash2 k7 +0.2% on the 4090. Each is inside the gap between the
  0.31 arm's own two boots. Greedy, the copy probe's verbatim-copy task produced the same output hash on both pins on
  every profile compared (MTP and DFlash2 `CTX=long`, DFlash2 k15); its other two tasks differ in places, one of three
  on MTP `CTX=long`. Greedy decoding with speculation is not guaranteed bit-exact, so matching hashes are an
  observation, not a property.
- **Prefix-cache correctness** (`bench/residue_sweep.py`, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1`): 0 broken of 128 on
  both pins on the 3090 and on the 4090, 0 broken cold, every 4090 answer one hash; the server's own log shows the
  cache hit (97.7-98.3%). The 4090 ran a smaller pool (203,378 tokens), because the shipped pin does not fit the
  4090 it ran on.
- **First-request JIT:** 0 compiles during inference on both pins.
- **`CTX=huge`'s KVarN pool** at the shipped 3090 defaults is 268,169 tokens on both pins.
- **Two long conversations' reuse:** `alternative.sh` 93.5% for both on both pins; with an 8 GB CPU tier, 57,664 tokens
  (96.3%) for both on both pins.
- **Vision:** 512 tokens per 1024x512 image, the ninth image refused, eight 2048x1024 images beside 60k of text served
  on both pins.
- **Batch:** the forced 64-prompt step and the 64-way burst of 128 serve on both pins; margin after the step 177 MiB on
  the 3090 on both pins (0.31.0's port measured 174).
- **pth-flash-attn's 8.6 default** (`SPEC=mtp CTX=long`, `SPEC=dflash2 CTX=long`): boots with `PTH_FA=1` chosen by the
  launcher on both pins, and the three copy-probe answers have the same hashes on both pins on the DFlash2 profile
  (tokens per step 7.701 on both). On the 4090 the launcher does not choose it (compute capability 8.9).

## FlashInfer outside the launchers

A process that starts vLLM 0.32.0 without the launchers (a bare `docker run` with its own command, say) prints
"FlashInfer kernels are disabled", which follows #58765 (FlashInfer's kernels now come from `vllm download-kernels`).
0 of 40 boots through the launchers printed it. Why the launchers' environment avoids it was not isolated, so a custom
entry point should start vLLM the way the launchers do, or check its log for that line.

## What is still unproven

- One boot per cell, except the seeded speed checks, so small gaps are not claims.
- The RTX 50 cost above is arithmetic, not a measurement.
- Batch margins are nvidia-smi peaks sampled at 50 ms, so upper bounds.

## Decisions made in this port, one line each (reject any by name)

- xgrammar built from source over the 0.2.8 pin, retiring when a release carries mlc-ai/xgrammar#989.
- The Triton backend's 64-segment SM120 mode stays off while the scratch pool is 16-segment.
- `KVQuantMode.KVARN` is 100.
- `pin-bump.py` takes the cubin pin from `flashinfer-python` when `cuda.txt` has no `flashinfer-cubin` line (#372's form).
