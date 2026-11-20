# The vLLM 0.34.0 pin

What the move from 0.33.0 to 0.34.0 changed in this repo, what was re-measured, and what it costs.

[← back to the main README](../README.md)

## What it costs, first

- **Batch mode serves W4A16.** vLLM 0.34.0 removed Marlin's 8-bit activations, the kernels included (vllm #65542),
  and setting `VLLM_MARLIN_INPUT_DTYPE` is now a startup error. Batch mode defaulted to int8 activations on its MLP
  GEMMs, which batch/README.md's table puts at +24% end-to-end (876 against 707 tok/s at 128/512) and +27% steady
  decode (~1,050 against ~830) over W4A16, at 64 concurrent requests on vLLM 0.27.1. That gain is gone on this pin.
  Upstream's route to W4A8 is the Humming kernels (`--quantization humming` with
  `VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8"}'`; humming-kernels 0.1.19 ships with 0.34.0). They are not
  wired into these launchers yet, and whether they load this checkpoint correctly is untested.
- **`INT8_ACT` is refused.** Every launcher now exits with the reason (exit 2, before the engine starts) on a
  non-empty `INT8_ACT`, instead of exporting a variable the engine rejects. Single-user and `alternative.sh` already
  defaulted to empty; batch's default moves from `int8` to empty. `PREFILL_ATTN=int8` (our own attention kernel) is
  unaffected.

## Dependencies

`vllm==0.34.0`. torch stays at 2.14.0 and torchvision at 0.29.0. `flashinfer-python` moves from 0.7.2.post1 to 0.7.3
(docs/install.md's cubin line follows). 0.34.0 accepts transformers below 5.23, so `docker/requirements.txt` keeps
5.19.0 and tokenizers 0.23.3. It requires huggingface_hub 2.0 or newer, so the pin moves from 1.32.0 to 2.4.0, the
version vLLM 0.34.0's own CI lock uses (transformers 5.19.0 declares huggingface-hub <3.0). `fa2/PIN` moves to
flash-attention d8dca8f0, the commit 0.34.0 builds: two commits past 0.33's b273a026, neither under
`csrc/flash_attn/src`, the cutlass submodule unchanged, and `fa2/check_upstream.py` passes on it (18 mapped files
byte for byte).

## The extensible KV cache

vLLM 0.34 turns on an extensible KV cache by default on CUDA. It reserves the KV pool's address range, commits memory
only after warmup and CUDA-graph capture, and sizes the pool from what it measures free then, less a margin for
activation spikes warmup didn't see (the larger of 1 GiB and 75% of the measured peak, plus 10% of what warmup
leaves; `vllm/v1/worker/extensible_kv_cache.py`). Upstream says to leave `gpu_memory_utilization` unset with it on,
and every launcher here sets `GPU_UTIL`, so with it on we paid our cap and its margin both.

**Every launcher passes `--disable-extensible-kv-cache`**, which keeps 0.33's sizing. Batch's `GPU_UTIL` moves from
0.94 to 0.93 with it. To turn the cache on, put `--enable-extensible-kv-cache` in `EXTRA_ARGS` (the launchers then
leave the flag to you). One boot per setting:

| card, profile | off | on, `GPU_UTIL` kept | on, `GPU_UTIL=1.0` |
|---|---|---|---|
| native 3090, `alternative.sh`-style at 160k, 4 seqs, vision, an 8 GB CPU tier (`GPU_UTIL=0.90`) | 339 blocks | 281 (-17%) | 353 (+4%) |
| native 3090, batch | 287 blocks at 0.94, 277 at 0.93, 64 seats | 252 at 0.94, 63 seats | 255, 63 seats |
| 4090 under WSL2, the same 160k profile | 368 blocks | 292 (-21%) | 292 (-21%) |
| 4090 under WSL2, `alternative.sh` at its shipped 256k | 431 blocks | refuses: 5.08 GiB needed, 4.29 available | refuses |

With it on, batch stays a seat short at any `GPU_UTIL`: the 1 GiB margin floor binds, not the cap. Under WSL2 the
cap doesn't bind at all; the free memory WSL reports after warmup, less the margin, does. Every arm that booted
passed the same stress load (four concurrent requests with two ~58k-token prompts and an 8-image prompt, a second
turn, then a 16-request burst), and greedy acceptance read the same on and off, so the setting moves capacity.

Turning it on pays, as far as measured, on a native-Linux card nothing else uses, with `GPU_UTIL=1.0`: a few
percent more KV than off on the single-user profile, plus upstream's margin against activation spikes. Not for batch,
not under WSL2, and not with `LSA=1`, which turns it off itself: with it on the plugin's index tables follow the whole
reserved range, and a 50k/100k ruler read 0/8 and 0/6 with no error (8/8 and 6/6 off), so the plugin refuses it.
KVarN runs with it on (`kvarn-extensible-kv-commit`: the deferred int4 flush commits the blocks it writes before
writing them), with the same texts as off and a 6.9-13.5% smaller pool. No native 4090 was measured.

## Patch series

66 files (62 in the series, 4 KVarN), re-exported from a vLLM branch that carries the series as one commit per topic
on v0.34.0. Applied to pristine v0.34.0, the series plus the KVarN files reproduce that branch's `vllm/` tree with 0
differing files; a one-byte change to one patch shows as 1 differing file. The rebase stopped 21 times.

Retired, because upstream now does it or removed what it patched:

- `vllm-pr61612-flashinfer-async-barrier`, replaced by `vllm-pr64165-flashinfer-plan-event`. vLLM #64165 (merged
  11-11, after the 0.34.0 branch) fixes the same FlashInfer plan-buffer race behind #328's PP=2 illegal memory
  access: the builder records a CUDA event after each `plan()` and waits on it before the next, so the host blocks
  only when it is more than one plan ahead of the GPU. The port it replaces took the blocking `seq_lens.cpu()`
  again at every step under async scheduling.
- `marlin-int8-layer-select`, `marlin-int8-negative-scales` and `marlin-int8-asym-zp`: #65542 removed the
  int8-activation Marlin path they patched.
- `sampler-small-topk-fast-softmax`: #63373 removed Model Runner V1, and every caller of its fast path was a V1
  file. On 0.33 the launchers already ran V2, where the patch was inert.
- `quant-scale-view-preserve-pages`: its whole change is in 0.34.0 through #56492.
- `vllm-pr64137-logits-stats-no-specialize`: #64137 is in 0.34.0.

Re-cut where upstream moved code under a topic (each stop's cause and resolution is in the topic's `Cut-against`
note and in the fork's history):

- `dflash-context-capacity`: draft-config setup moved into `SpeculatorConfig.configure()` (#63205); the check now
  runs there, before the drafter's max_model_len override.
- `kvarn-v2-runner`: 0.34.0 added its own `divisor_of` to `_largest_kernel_block_within`, used when there is no
  shared page. KVarN's padded drafter page keeps its rule (a divisor of the primary block, a multiple of 128 first):
  upstream's rule alone picks 272 on a 2176-token primary at `DFLASH_TOKENS=7`, which `prefix_match_unit` (128)
  refuses (#179).
- `gdn-multi-prefill-checkpoints`: the Triton prefill branch now gathers its initial state with
  `gather_initial_states`; the checkpoint hook runs after it, Triton backend only, as before.
- `gdn-ba-pad-small-m`: upstream now runs the small `in_proj_ba` GEMM on an aux stream under CUDA graph capture; the
  padded op is used on both of upstream's paths.
- `dflash2-prewarm`: the prewarm launches take #65488's grid, `num_reqs` and `NUM_PAD_PROGRAMS`, so they compile the
  variant the real launch uses.
- `qwen3_5-embed-quant`, `qwen3_5-mtp-draft-vocab` and `qwen3_5-embed-uva`: on CUDA, 0.34.0 runs Qwen3.5 through
  `vllm/models/qwen3_5/nvidia` (#51012), so each patches those files as well as the old ones, which still serve other
  platforms.
- `dflash2-lookup-drafting` goes back to 0.34.0's own quantized-weight fallback for the drafter's context KV (#64357)
  in place of its dense unpack.
- `spec-decode-scratch-within-budget` sizes the int4 scratch from the caller's geometry, which is what 0.34.0's
  metadata builder reads (#56996).

New in this port:

- `qwen3_5-nvidia-gated-norm-warmup`: the new CUDA path's fused GDN norm (`_fused_gdn_gated_rms_norm_kernel`)
  compiled during the first requests; it is warmed at boot.
- `vision-tower-cpu-offload-meta`: 0.34.0 builds the vision tower on the meta device on every pipeline rank after the
  first, and the vision offload copied to host at construction, so `VISION=1` with `VLLM_VISION_CPU_OFFLOAD_GB` did
  not boot at PP=2. The offload is skipped where the tower is on meta.
- `kvarn-extensible-kv-commit`: KVarN's deferred int4 flush commits the extensible cache's blocks it writes before
  writing them (above, "The extensible KV cache").

Breaks no conflict showed, found by reading upstream's changes and by `scripts/port-changed-signatures.py` /
`port-removed-names.py` (item 0), all fixed in their topics:

- `dflash2-lookup-drafting` and `dflash2-ngram-chains`: 0.34.0 made the draft-logits cache a `DraftLogitsCache`
  (#64356), and both point-mass launches still passed it as a tensor. They now pass `.values`, its row mapping and
  the values' strides.
- `marlin-repack-staged-sm80`: its repack call still passed `is_a_8bit`, which #65542 removed from the function and
  the file. A NameError at weight load on every card the staged repack serves.
- `marlin-tune-table`: the tuned route passed `a_scales`, which 0.34.0's `marlin_gemm` no longer takes; the
  standalone build keeps the slot and gets None.
- `spec-sampler-prewarm`: its warmup ran only when the worker reported `use_v2_model_runner`, a flag 0.34.0 removed
  with V1, so it would have skipped itself on every boot.

## Acceptance

0.33 arm: 0.33.0 with the 0.33 series, built from the vLLM fork. 0.34 arm: this series on 0.34.0, built the same way.
A native RTX 3090 with the two images alternating in one session, one boot per cell unless stated. The items ran on an
image from before the extensible-cache default and the `DFLASH_TOKENS=15` length below, with the cache on wherever no
KV pin was set; each fix was then checked on a later image.

- Static: `scripts/port-changed-signatures.py` binds 227 calls on our added lines against 0.34.0 with 0 breaks (2
  before the fixes above); `port-removed-names.py` finds 4 uses, all our own parameter name or words in comments.
- Every launcher setting (408 cases) through 0.34.0's `create_engine_config()` on CPU in the image: 320 build, 24 are
  the launchers' own `INT8_ACT` refusal, and 64 are `SPEC=none` with `REQ_METRICS=1` (the alias misses #444's fix,
  syv-ai #464). Nothing new RAISEs on 0.34. 0.34.0 deprecates `cudagraph_capture_sizes` and
  `max_cudagraph_capture_size`, which the launchers pass, and says `--mamba-cache-mode` will be removed.
- Every profile at its shipped defaults with a 64-way burst of 128 requests: 128/128 on both. With the cache off at
  `GPU_UTIL=0.93`, batch's forced 64-prompt step serves 64 of 64 with 373 MiB to spare (0.33: 171 MiB at int8, 231 at
  W4A16), and the burst runs at 875.5 tok/s, against 866.6 for 0.33 at W4A16 and 1,201.5 at int8.
- Tokens per step at greedy, two boots per image: every boot uses its shipped draft count, and the copy probe reads
  7.701 at k7 and 14.472 at k15 on both. Greedy is inside the 0.33 boots' range on MTP long, DFlash2 k7 and
  `alternative.sh`. MTP fast k4 reads 3.714 on both 0.34 boots, above both 0.33 boots (3.674, 3.681) and every MTP
  fast greedy reading on this card from 0.31 to 0.33 (3.674-3.701, 9 boots), with the same five texts as one 0.33
  boot; the cause is not measured. Seeded default temperature on each profile is no farther from the 0.33 mean than
  the two 0.33 boots are from each other.
- Every knob the launchers export reaches the engine on both, and the invalid values refused on 0.33 are refused on
  0.34, apart from the int8-activation knobs, which are gone.
- First requests on a cold volume (8 streams of 3 requests, two waves): no kernel compiled during inference on 0.34
  at the single-user defaults or `alternative.sh`. On 0.33, `alternative.sh` compiles `_attn_packed` in both waves.
- `bench/residue_sweep.py` at `SPEC=dflash2 CTX=huge PREFIX_CACHE=1`: 0 broken of 128 on both, 0 broken cold, and
  all 256 answers per image one hash, the same on both.
- Two ~59K conversations at once: `alternative.sh` 96.3% reuse on 0.33 against 98.1% on 0.34; with an 8 GB CPU
  tier, 57,664 and 59,360 tokens on 0.33 against 58,752 for both. On the MTP profile with `PREFIX_CACHE=1`, 0.33's
  99.6% was #61589's admission hit, which preempted the other conversation and saved no time (turn-2 decode 4.0 and
  4.2 tok/s, 258.7 s in all). 0.34 runs the two one after the other, 0% hit, turn-2 decode 107.8 and 96.3 tok/s,
  256.4 s, with the cache on or off; one conversation alone reuses 99.5%.
- The launcher defaults the READMEs document, 13 combinations of SPEC, CTX and DFLASH_TOKENS: 9 booted on the first
  image, with the cache on. The other 4 boot with the fixes in this port: three KVarN profiles with no KV pin, which
  died in CUDA-graph capture with the cache on, and `SPEC=dflash2 CTX=fast DFLASH_TOKENS=15`, whose page went from
  480 to 512 tokens (0.34.0 rounds the hybrid attention page up to every backend's smallest kernel block), so its
  57,344 default no longer fit the 5.2 GiB pin; it defaults to 55,296.
- Vision (0, 1 and 8 images read, the ninth refused with the documented 400), LSA (`lsa/tests/ruler_mk.py` 8/8 and
  6/6 on both, once `LSA=1` turns the cache off) and streamed tool calls (3 `tool_calls` deltas, `finish_reason`
  `tool_calls`, on both thinking settings) are unchanged.
- The native install from `docs/install.md`, run as extracted on a Python 3.14 host with a system nvcc 12.4: every
  step exits 0 and `verify.sh` passes, except pthfa's optional build, which fails on that host's glibc 2.43 headers
  with CUDA 13.0 as it did on 0.33. A patch run with one corrupted patch exits naming it and leaves the venv unchanged
  (a stat manifest of 155,737 entries and the vllm package's hashes, before and after).
- #328's load at PP=2 on a rented 2x3090, 60 minutes: no crash with #64165 and without the old port (551 requests
  served; the 47 refused are the load's own over-limit image requests); the same build with only #64165 reversed hit
  the illegal memory access 5 s in. One machine, one boot each.
- On a production RTX 4090 (native Linux, `alternative.sh` with vision, an 8 GB CPU tier and 4 seats at 160k), against
  0.33 on the same probes: auth, tool calls, vision, `return_progress` and a 16-request burst read the same. Greedy
  decode on one 600-token prompt is 104.5 tok/s against 112.9 (-7.4%), and with two ~59K conversations at once both
  turn-2 prompts hit 50,112 tokens, all from the CPU tier, against 59,360 (turn-2 first token 9.9 s against
  0.7-0.8 s). One prompt sent twice does hit the GPU prefix cache. The decode cause is not measured yet. The
  reuse cost was `alternative.sh`'s dense retention with a KV tier, which now defaults to `0`
  ([gotchas.md](gotchas.md) 60): on an RTX 3090 with the same profile, both turn 2s hit 58,752 tokens on the GPU
  at `0`, and the tier still serves a forced miss.

## What is still unproven

- Humming W4A8 on this checkpoint (correctness with its negative group scales, the per-layer selection, speed).
- Turing (sm75, #448) and GA100 (#451) were not run on this pin.
- SM120 keeps upstream's 64-segment decode mode off, as on 0.33, because the 3D scratch comes from the budgeted pool
  (`spec-decode-scratch-within-budget`), which is sized for 16 segments.
- Whether `gdn-ba-pad-small-m` still pays once upstream hides the ba GEMM on an aux stream.
- Why the production 4090 decodes 7.4% slower on 0.34 (Acceptance, last item). An RTX 3090 shows no 0.34 cost per
  decode step on the same profile (26.10 against 26.12 ms).
- The seat-budget log line (`kv_cache_utils.py`, "KV cache seat budget") counts 9 fixed blocks a request on batch
  where 4 are held, so it implies about 25 resident requests where 63-64 run. The same on 0.33; not this port's.
- KVarN's `build_for_drafting` override had no caller on the V2 runner on 0.33 either (only V1 proposers called it);
  0.34.0 removes the last caller. Pre-existing, unchanged by this port.

## Decisions made in this port, one line each (reject any by name)

- Batch serves W4A16 and refuses `INT8_ACT`, rather than wiring Humming in the same change.
- huggingface_hub 2.4.0 (upstream CI's), with transformers kept at 5.19.0 rather than moved to CI's 5.22.0.
- #64165 carried without its test file, as with the other backports.
- The launchers turn the extensible KV cache off, rather than keep upstream's default and document the cost.
- `alternative.sh` keeps prefix retention at `0` with a KV tier too, rather than dense as since 0.29.
- No constraints file for vLLM's unpinned dependencies, although `openai` (3.55.1 to 3.56.0) and `anthropic` (2.10.0
  to 2.11.0) moved between two `--no-cache` builds of this branch four hours apart.
