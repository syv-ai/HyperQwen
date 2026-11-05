# The vLLM 0.33.0 pin

What the move from 0.32.0 to 0.33.0 changed in this repo, what was re-measured, and what it costs.

[← back to the main README](../README.md)

## Dependencies

`vllm==0.33.0`, which moves torch from 2.13.0 to 2.14.0, torchvision to 0.29, `flashinfer-python` from 0.7.0.post1 to
0.7.2.post1 and Triton to 3.8.0. 0.33.0 accepts transformers below 5.21, so `docker/requirements.txt` keeps 5.19.0 and
tokenizers 0.23.3. `fa2/PIN` moves to flash-attention b273a026, the commit 0.33.0 builds; it is four commits past
0.32's 9cd61de, all under `flash_attn/cute`, and `fa2/check_upstream.py` passes on it (18 mapped files byte for
byte). torch 2.14's ATen headers need C++20, so `fa2/build.py` now compiles with `-std=c++20`. `pthfa/` builds
unchanged, because it uses the stable C ABI rather than ATen.

## Patch series

61 files (57 in the series, 4 KVarN), re-exported from a vLLM branch that carries the series as one commit per topic on
v0.33.0, each re-cut against 0.33.0 (`Cut-against`) with a note where upstream moved code under it. Applied to
pristine v0.33.0, the series plus the KVarN files reproduce that branch's `vllm/` tree with 0 differing files. The
rebase stopped 21 times; each stop's upstream cause and resolution:

Retired, or smaller, because upstream now does it:

- `vllm-pr61050-lru-running-prefix`: #61050 is in 0.33.0 (e113018f50). The patch, its series line and its PATCHES.md
  row are gone. Upstream's merged form may differ from the head this series carried, so the CPU-tier reuse check
  below is what shows it still covers our case (it does).
- `bench-probe-errors`: #59888 carries the 404, connection and generic hints. The patch keeps the 401 and timeout
  hints, the probe's Authorization header and the key on the `/metrics` scrapes.
- `bench-sse-keepalive`: #51172 passes chunks unstripped, which was this patch's main fix, and upstream's
  `if not chunk_bytes:` is kept everywhere (ours could drop a blank-line separator that arrived as its own chunk).
  What is left is the audio function's skip of SSE comments.
- `sampler-warmup-cuda`: #59260 registers the same top-k/top-p warm-ups on CUDA, but only with speculative decoding
  on. The patch stays, because it also covers `SPEC=off`.

Re-cut where upstream moved code beside ours:

- `hybrid-kv-groups-v2-cudagraph`, `hybrid-sw-block-promote`, `async-mamba-kv-sizing`: #61183 moved the KV cache
  planning code from `v1/core/kv_cache_utils.py` to the new `v1/core/kv_cache_planning.py`, and the hunks move with
  it unchanged. In `async-mamba-kv-sizing` the group helpers now take a `KVCacheConfigBuilder` first, so the check
  gets one from `get_kv_cache_config_builder(vllm_config)` the way `get_kv_cache_configs` does; the two-argument
  calls would have been a TypeError the first time the check ran.
- `spec-decode-scratch-token-units`, `spec-decode-scratch-within-budget`, `spec-decode-int8-kv`: #56148 gives
  upstream its own multi-query 3D Triton path for speculative verification (query length up to 8, not INT4), and it
  reads the same split-K scratch this series' pool owns. The pool's plan sizes that scratch for both readers, because
  no batch carries more tokens than the scheduler places; upstream falls back to 2D on a short scratch, so an
  undersized buffer would cost speed, not memory safety. The 64-segment SM120 mode stays off, as on 0.32.0.
- `kvarn-0.33.0`: #59774 (ROCm Octave KV) and #52670 (pluggable KV dtypes) add `KVQuantMode` values beside KVarN's
  100, which stays. In v0.33.0 itself `CUSTOM` and `OCTAVE_K3V4` are both 12, so `CUSTOM` is an alias of
  `OCTAVE_K3V4`; that is an upstream issue on ROCm modes this repo does not run.
- `dflash2-lookup-drafting`, `dflash2-z-adaptive-emitted`, `dflash2-prewarm`: re-anchored beside #64458's draft
  broadcast and #58984's dummy-run clamp; every warm-up launch passes #59166's new `is_padding` argument.
- `dflash2-ngram-chains`: #61161 draws a fresh random number for each verification round, which the chains path
  picks up through the shared sampler. 0.33.0 also changed `prepare_context_anchor` to take `(num_reqs,
  query_start_loc, num_rejected)`, and the chains caller now passes those (below). Separately, #57169 (in 0.32.0)
  removed `BlockTables.kernel_block_sizes` and moved upstream's DFlash caller to `block_sizes[gid]`, but the chains
  copy of that caller kept the old attribute. So since our 0.32.0 pin, `VLLM_DFLASH2_CHAIN=1` has stopped the engine
  on the first step that engages a chain. The caller now reads `block_sizes[gid]`, as upstream's does.
- `vllm-pr40371-prompt-progress`: 0.33.0 split the chat stream into a generator that yields chunk objects and a
  wrapper that frames them as SSE. The patch still yielded an already framed string from the inner generator, so the
  wrapper raised on the first progress chunk and a `return_progress` stream ended after one error chunk. It now
  yields the chunk.
- `sampler-small-topk-fast-softmax`: #61996's `has_topp_only_rows` keeps its 4th place and `k_max` follows it;
  `softmax_fp32` sits where #53987 moved the rejection sampler's softmax.
- `engine-completion-log`, `engine-stall-sentinel`, `kvarn-recycled-pages-0.33.0`, `kvarn-fp16-dequant-0.33.0`,
  `vllm-pr40371-prompt-progress` (whose fix is above): re-anchored beside #58778's `continuing_reqs`, new `envs.py` entries, the runner's
  encoder-input update and #55226's moved pooling return.

Added:

- `vllm-pr64137-logits-stats-no-specialize`: on 0.33.0, #61451 made two rejection-sampler Triton kernels take
  `num_logits`, which changes with the batch, so the first concurrent requests compiled them again during serving.
  #64137 (merged after 0.33.0) stops Triton specializing on it; applied unchanged.

Checks with no card: `scripts/port-changed-signatures.py` found one call that no longer binds before the fix (the
chains caller of `prepare_context_anchor`, which applied cleanly and would have been a TypeError on the first
`VLLM_DFLASH2_CHAIN=1` request) and 0 of 202 after it. `scripts/port-removed-names.py` finds 2 removed names in our
added lines, both words in comments. The override scan finds no series override that misses a base parameter, and
the out-of-tree plugins (`pthfa/`, `fa2/`, `lsa/`, `bench/`) import no name 0.33.0 removed. Neither script found the
two bugs above: `kernel_block_sizes` is an attribute read whose name still exists on another class, and the prompt
progress string was a change in what a generator yields. The RTX 4090 run's probes found both. Four post-install tests
built vLLM objects that 0.33.0 changed (the offload group and spec configs, `get_kv_cache_configs`' new argument, a
block pool attribute, the split chat stream generator) and are updated; the stream test now builds its parser the way the
server does, with `ParserManager` from the launchers' flags.

## Acceptance

0.32 arm: this repo's main with vLLM 0.32.0, built from the fork at the same series. 0.33 arm: this branch. One native
RTX 3090, the two images alternating in one session, one boot per cell unless stated.

- **Every profile boots at its shipped defaults**, and the 64-way burst of 128 requests serves 128/128 on both. The
  batch profile's forced 64-prompt step serves on both, with a margin of 207 MiB after it on 0.33.0 (171 on 0.32.0).
  KV pools: batch 215,482 tokens against 217,766 (-1.05%), single-user 85,992 against 85,234, `alternative.sh`
  300,643 against 299,247. Burst throughput is within 1.1% on every profile.
- **Tokens per step, two boots per image:** at default temperature, seeded (5 prompts x 8 seeds), no profile moves
  beyond the gap between the 0.32 arm's own boots, except DFlash2 k7, which is higher on 0.33.0 by about that gap,
  the direction #61161 would move it. Greedy speculation is not bit-exact across boots on either image (the 0.32
  arm's two boots match on 4 of 5 greedy texts), so greedy is compared against each image's own spread: inside it on
  4 of 5 profiles. DFlash2 k15 read 0.64% lower on two boots per image, and at four boots per image the greedy
  means are 0.11% apart with overlapping ranges, so greedy is inside the spread on all five. At k15 the seeded
  reading is 1.2% lower on 0.33.0, inside the boot-to-boot range, but with all four 0.33.0 boots below all four 0.32.0
  boots, so it is probably a real k15 shift. Its cause is unmeasured, and at k7 the sign is the other way. The
  unseeded reading (-5.4%) is not evidence of a rate change: with no seed, three of its five prompts replay the same
  sampled text on every boot of an image, so it compares two fixed sampling paths per prompt. The copy
  probe reads 7.701 at k7 and 14.472 at k15 on both, and every boot uses its shipped draft count.
- **Knobs:** the 25 the launchers export reach the engine on both images, and the invalid values refused on 0.32.0
  are refused on 0.33.0.
- **First-request JIT:** 0 compiles during inference on both images with one request at a time. Under concurrent
  traffic (8 streams, 3 requests each, two waves, cold) 0.33.0 compiled one more kernel than 0.32.0,
  `_compute_local_logits_stats_kernel`, with a 0.5 s stall mid-stream. `vllm-pr64137-logits-stats-no-specialize` fixes
  it: on that load the patched image compiles only what 0.32.0 also compiles (`_gather_draft_sampled_kernel` on the
  default profile, `_attn_packed` on `alternative.sh`'s first request), and an image without the patch, booted in
  the same run, compiled it again.
- **Prefix-cache correctness** (`bench/residue_sweep.py`, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1`): 0 broken of 128 on
  both images, 0 broken cold, every answer one hash (the same as 0.31 and 0.32). The KV dtype reads KVarN, and the `CTX=huge`
  pool is 262,633 tokens on both.
- **Two long conversations' reuse:** `alternative.sh` 96.3% for both on 0.33.0 (93.5% on 0.32.0, one block lower);
  with an 8 GB CPU tier, 57,664 and 59,360 tokens on 0.33.0 against 57,664 for both on 0.32.0.
- **MTP prefix caching now hits on this model.** With the default MTP profile and `PREFIX_CACHE=1`, turn 2 reuses
  59,616 of 59,885 tokens on 0.33.0, where 0.32.0 reused none. In the single-user pool (85,992 tokens) that hit lets
  the scheduler start the second conversation's turn beside the first, which then preempts, so total time is
  unchanged (258.1 s against 257.5 s) and turn 2's wait moves from the first token to a stall mid-stream. So
  `usage.cached_tokens` overstates the saving when the pool cannot hold both. The cause is #61589's lookahead block
  hashes, which stop the trailing block being dropped for MTP: with `"disable_lookahead_block_hashes": true` in
  `--speculative-config` the same boot reuses nothing, runs one request at a time and decodes turn 2 at normal
  speed, as 0.32.0 does.
- **Vision:** turning vision on costs no KV on either image, both read the printed word, an image is the same number
  of tokens on both, and the ninth image is refused with the same error.
- **LSA** (`LSA=1 SPEC=mtp CTX=long`, `lsa/tests/ruler_mk.py` at 50000 and 100000): boots on both, 8/8 and 6/6 on
  both, with identical trials.
- **Streamed tool calls** on `single-user/start_qwen.sh` and `alternative.sh`, thinking on and off: 3 `tool_calls`
  deltas with the expected arguments and `finish_reason` `tool_calls` on both images, 8 of 8.

### RTX 4090

One native RTX 4090 (Ubuntu 26.04), the same two images, one boot per image per cell. The 0.33 image is the one
before the two fixes above, which were then checked on the 3090 (the last item).

- **Every profile boots at its shipped defaults**, and the 64-way burst of 128 requests serves 128/128 on every
  profile on both. Batch's forced 64-prompt step serves on both, with 262 MiB to spare on 0.33.0 (280 on 0.32.0). KV
  pools are equal on batch (207,106 tokens), single-user (71,218), `CTX=huge` (198,485) and DFlash2 k15 (45,070);
  `alternative.sh` is 302,735 against 296,457. Burst throughput is within 3.0% on every profile.
- **Tokens per step:** every boot uses its shipped draft count, and the copy probe reads the same on both (4.867 at
  MTP k4, 7.701 at DFlash2 k7, 14.472 at k15). Greedy is within 2.3% on every profile. Seeded default temperature
  reads 0.7-1.4% lower on 0.33.0 on all three profiles that ran it; with one boot per image that is not a claim, and
  on the 3090 those profiles were inside the boot-to-boot spread.
- **First-request JIT, whole boot including the 64-way burst:** the same kernels on both images on every profile
  (`_attn_packed` on `alternative.sh`, `_gather_draft_sampled_kernel` on single-user, none on the others), so
  `vllm-pr64137-logits-stats-no-specialize` holds on this card under concurrent traffic.
- **Prefix-cache correctness** (`SPEC=dflash2 CTX=huge PREFIX_CACHE=1`): 0 broken of 128 on both, 0 broken cold,
  every answer the same hash as on the 3090.
- **Two long conversations' reuse:** `alternative.sh` 96.3% on 0.33.0 against 93.5%, the same as the 3090. On a
  production-shaped `alternative.sh` line (`MAX_SEQS=4`, `MAX_LEN=160000`, vision on, an 8 GB CPU tier) turn 2
  reuses 99.1% for both conversations on 0.33.0 and waits 0.8 s for its first token, against 85.0% and 82.1% and
  10-12 s on 0.32.0.
- **Tools and vision:** strict tool choice, streamed tool calls (both thinking settings) and the eight-image vision
  probe pass on both.
- **Found here:** `return_progress` failed on 0.33.0 only, and `VLLM_DFLASH2_CHAIN=1` stopped the engine on both
  images. The fixes are in the series above. On the 3090, the 0.33 image with the two fixed files mounted over it
  serves both: chains 5 of 5 requests with 22 engaged chain steps, where the unfixed image fails all 5, and a
  29,845-token prompt streams 17 progress chunks where the unfixed image sends one error chunk. 0.32.0 sends 16 on
  that prompt, because 0.33.0 ends prefill at a block boundary and runs the last 37 tokens as their own step.

## Native install

The README's install path, run as extracted from `docs/install.md` on a host with Python 3.14, system nvcc 12.4 and
Ubuntu 26.04 (glibc 2.43): pip resolves vLLM 0.33.0, torch 2.14.0, FlashInfer 0.7.2.post1 and Triton 3.8.0, the
patches apply, and `verify.sh --no-server` passes 91 of 91. `pthfa/install.sh` fails on that host, on 0.32.0 as well:
glibc 2.43 declares `rsqrt` in a way CUDA 13.0's `math_functions.h` conflicts with. The image builds on Ubuntu 24.04
and is not affected. The doc pins no transformers, so the native install got 5.20.2 where the image has 5.19.0.

## What is still unproven

- The 4090 ran the 0.33 image from before the two fixes. The fixed image has not been built; the fixes were checked
  by mounting the two files over that image on the 3090.
- Why 0.33.0 runs a separate prefill step for the tail past the last block boundary is not read.
- Two boots per image for tokens per step on the 3090 and one on the 4090, so gaps under the boot-to-boot spread are
  not claims.
- Batch margins are nvidia-smi peaks sampled at 50 ms, so upper bounds.

## Decisions made in this port, one line each (reject any by name)

- `vllm-pr61050-lru-running-prefix` retired at 0.33.0, on the CPU-tier reuse check.
- `vllm-pr64137-logits-stats-no-specialize` carried until the pin that has #64137.
- `fa2/` compiles at C++20.
- `KVQuantMode.KVARN` stays 100.
