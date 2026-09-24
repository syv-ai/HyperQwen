# KVarN KV cache, ported to vLLM 0.29.0

[KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL, Apache-2.0) is a
KV-cache compression scheme — Hadamard rotation, iterative variance
normalization, 4-bit keys / 2-bit values per 128-token tile — shipped as a
native vLLM attention backend inside a fork of vLLM 0.23.0. This directory is
that backend ported onto the vLLM 0.29.0 this repo runs, dense (non-MLA) path
only, and tuned for the Qwen3.8-27B / RTX 3090 setup here.

What's in it:

- `files/vllm/...` — the KVarN modules (backend, Triton kernels, config,
  Sinkhorn reference), copied from KVarN and adapted to the 0.28.0 backend API
  (the original adaptation markers are retained in the source files).
- `kvarn-0.29.0.patch` — the small hunks upstream vLLM needs to know the
  new `kvarn_*` cache dtypes (cache dtype literals, dtype map, backend registry
  + priority, a `KVQuantMode.KVARN`, the KV-cache spec branch in the attention
  layer, and the hybrid-model page alignment branch).
- `kvarn-v2-runner-0.29.0.patch` — the V2 runner, sliding-cache, and DFlash2
  correctness fixes layered on top of the base port.
- `install.sh` — copies the modules into the venv's `site-packages/vllm` (found by asking the venv's python, so any Python version)
  and applies both patches at `--fuzz 0` (safe to re-run; a rejected hunk stops it).
  Both patch files are exported from their commits on the fork branch (`cpuchip/vllm`
  `qwen38/0.29`), which sit after the whole `patches/` series, so they are never edited by hand.

Port notes, for whoever bumps vLLM next:

- 0.29.0 (from 0.28.0): vLLM no longer asks a backend for its KV cache shape or stride
  order; every layer's physical layout comes from its spec as `[blocks, heads, states,
  content]` bytes under one process-wide layout. The backend therefore declares
  `supported_kv_cache_layouts = (LBNHC,)` and folds the runner's 4D view back into one
  tile per (block, head) at its two entry points (`_as_tile_view`, a `view`, so a wrong
  layout fails instead of copying). The old `attn_utils.py` strided-view hunk and the four
  `kv_cache_utils.py` hunks are retired (their reasons are in the patch preambles); the
  `kvarn-0.27.1.patch` and `kvarn-v2-runner.patch` files are the older ports, kept for
  the diff history.

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

Measured on the 3090 (details in [docs/long-context.md](../docs/long-context.md)): 262k context fits
(420k-token pool at 4 slots vs ~200k with fp8), needle-in-a-haystack correct
at 4k…240k, perplexity +0.16%, decode ~20% slower than fp8 at 100k context,
MTP works, short-request throughput lower (2048-token blocks make each
request cost as much as fp8's 800-token block, and prefill flushes cost time).
