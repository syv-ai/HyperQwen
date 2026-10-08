# KVarN KV cache, ported to vLLM 0.30.0

[KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL, Apache-2.0) is a
KV-cache compression scheme — Hadamard rotation, iterative variance
normalization, 4-bit keys / 2-bit values per 128-token tile — shipped as a
native vLLM attention backend inside a fork of vLLM 0.23.0. This directory is
that backend ported onto the vLLM 0.30.0 this repo runs, dense (non-MLA) path
only, and tuned for the Qwen3.8-27B / RTX 3090 setup here.

What's in it:

- `files/vllm/...` — the KVarN modules (backend, Triton kernels, config,
  Sinkhorn reference), copied from KVarN and adapted to the 0.28.0 backend API
  (the original adaptation markers are retained in the source files).
- `kvarn-0.30.0.patch` — the small hunks upstream vLLM needs to know the
  new `kvarn_*` cache dtypes (cache dtype literals, dtype map, backend registry
  + priority, a `KVQuantMode.KVARN`, the KV-cache spec branch in the attention
  layer, and the hybrid-model page alignment branch).
- `kvarn-v2-runner-0.30.0.patch` — the V2 runner, sliding-cache, and DFlash2
  correctness fixes layered on top of the base port.
- `kvarn-recycled-pages-0.30.0.patch` — both runners hand KVarN each step's
  block ids (`note_scheduled_blocks` in `kvarn_attn.py`), and KVarN releases
  without flushing whatever it still holds for a page another KV-cache group
  has taken. Without it a finished request's last block, or an evicted retired
  sink, could be flushed to int4 into a page that was already another
  request's mamba state, which reads back as NaN: the request then prints `!`
  (token 0) forever (#208).
- `kvarn-fp16-dequant-0.30.0.patch` — registers `KVARN_FP16_DEQUANT` in
  `envs.py`, so the modules read it through `vllm.envs` and it is part of the
  torch.compile cache key (see "Environment knobs" below).
- `install.sh` — copies the modules into the venv's `site-packages/vllm` (found by asking the venv's python, so any Python version)
  and applies the four patches at `--fuzz 0` (safe to re-run; a rejected hunk stops it).
  Like the `patches/` files, each one is the source and is written by `scripts/export-patch.sh` from a commit that
  sits after the whole `patches/` series, so they are never edited by hand.

Port notes, for whoever bumps vLLM next:

- 0.30.0 (from 0.29.0): `KVQuantMode.KVARN` is value 11, because upstream inserted `NVFP4_DS_MLA` at 10 (every
  use is by name). #54713 threads `replay_boundaries` through `cache_blocks`, and the v2-runner override forwards
  it. #53007 rewrote `_largest_kernel_block_within`; the rule that keeps the DFlash2 drafter's padded
  sliding-window block a divisor of the primary block (128 against 2176) is carried into it as `divisor_of`.
  Without it, `CTX=huge SPEC=dflash2 PREFIX_CACHE=1` is refused at boot by `prefix_match_unit` (#179), and the
  pool moves from 268,169 to 298,067. #55353 removed `CommonAttentionMetadata._seq_lens_cpu`; the backend's
  `getattr` falls back to an exact `seq_lens.tolist()`.

- 0.29.0 (from 0.28.0): vLLM no longer asks a backend for its KV cache shape or stride
  order; every layer's physical layout comes from its spec as `[blocks, heads, states,
  content]` bytes under one process-wide layout. The backend therefore declares
  `supported_kv_cache_layouts = (LBNHC,)` and folds the runner's 4D view back into one
  tile per (block, head) at its two entry points (`_as_tile_view`, a `view`, so a wrong
  layout fails instead of copying). The old `attn_utils.py` strided-view hunk and the four
  `kv_cache_utils.py` hunks are retired (their reasons are in the patch preambles). The
  older ports (`kvarn-0.27.1.patch`, `kvarn-v2-runner.patch`) are removed from the tree;
  git history keeps them.

- 0.28.0's attention spec uses `cache_dtype_str="auto"` for specs whose
  `kv_quant_mode` is `NONE`; KVarN's shape depends on the preset, so the port
  adds `KVQuantMode.KVARN` and passes the packed slot size through
  `FullAttentionSpec(state_content_bytes=...)`. Without that the engine dies
  at KV-cache init.
- The impl→builder wiring uses `get_layers_from_vllm_config` instead of
  KVarN's `attention.py` `impl.layer_name` hunk, and a small owner registry so
  the MTP draft layer isn't flushed by two builders.
- Pools are materialized during `profile_run` (forward with
  `attn_metadata=None`) so vLLM's memory profiler charges them correctly —
  no `gpu_worker.py` hunk needed.
- Per-token slot padding to a power of two (KVarN did it for Gemma-4's mixed
  head dims) is off by default here (`KVARN_POW2_SLOT=1` restores it): with
  head_dim 256 that is 840 B/token/layer instead of 1024 (fp8: 2048).
- The hybrid alignment makes the attention block 2048 tokens (page must match
  the 1.63 MB Gated-DeltaNet page); vLLM splits it into 128-token kernel
  tiles, KVarN's invariant `tile == kernel block` holds.
- Small robustness fixes: NaN guards in the online-softmax kernels for
  fully-masked chunks / all-empty split-K rows, no per-context recompiles of
  the packed-KV kernel, verify-plan padding zeroed for CUDA-graph replays.
- Not ported: the MLA path, `TQSlidingWindowSpec` (no sliding-window layers
  here), the Gemma-4 config hunk.

Environment knobs (KVarN):

- `KVARN_FP16_DEQUANT=1` — run the three fused decode kernels
  (`_kvarn_fused_decode_kernel`, `_kvarn_fused_decode_stage1`,
  `_kvarn_fused_verify_stage1`) in fp16: the dequant math (q codes, per-channel
  and per-row scales, the `p` tile) stays fp16, the dots still accumulate in
  fp32. Halving the working tiles' width halves their registers, which is what
  lifts the long-context split-K decode. **Default off**, and with it off the
  three kernels compile to the same TTGIR/PTX (and n_regs/n_spills) as the port
  without the knob. Registered in `vllm/envs.py` by
  `kvarn-fp16-dequant-0.30.0.patch` and read through `vllm.envs`, so it is part
  of vLLM's torch.compile cache key: the two settings never share a compile
  directory (switching it costs one cold compile). With the knob on, the split-K
  stage1 and verify kernels autotune only over BLOCK_N 16/32, num_warps=4 and no
  maxnreg, which spill 8 registers or fewer in every measured fp16 shape. The
  configs that spill heavily (72 registers or more) are 2.5-24x slower at long
  context, and the warmup-shape autotune could pick one of them. The rule also
  drops BLOCK_N=64 w4, which spills only at verify QLEN=8: at verify QLEN 2 and in
  stage1 the kept list is 8.4% and 3% slower than it. With the knob off the autotune list is unchanged.
- `KVARN_SHARED_VERIFY=1` — the shared-dequant uniform verify kernel. Still off
  by default: serving with it corrupts the MTP drafter's proposals through a
  mechanism that is not isolated yet (see the comment at the driver's guard).
- the rest (`KVARN_NUM_KV_SPLITS`, `KVARN_SPLIT_K`, `KVARN_FUSED_DECODE`,
  `KVARN_POOL_MEM_FRAC`, `KVARN_FA_SCRATCH_CAP`, `KVARN_SPEC_DEBUG`, …) are in
  `vllm/envs.py` once `install.sh` has run.

Measured on the 3090 (details in [docs/long-context.md](../docs/long-context.md)): 262k context fits
(420k-token pool at 4 slots vs ~200k with fp8), needle-in-a-haystack correct
at 4k…240k, perplexity +0.16%, decode ~20% slower than fp8 at 100k context,
MTP works, short-request throughput lower (2048-token blocks make each
request cost as much as fp8's 800-token block, and prefill flushes cost time).
