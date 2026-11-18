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

## Patch series

63 files (59 in the series, 4 KVarN), re-exported from a vLLM branch that carries the series as one commit per topic
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

Pending: this section is filled from the runs, against 0.33.0 (main-033-663bdd7) on the same box.

## What is still unproven

- Humming W4A8 on this checkpoint (correctness with its negative group scales, the per-layer selection, speed).
- Turing (sm75, #448) and GA100 (#451) were not run on this pin.
- SM120 keeps upstream's 64-segment decode mode off, as on 0.33, because the 3D scratch comes from the budgeted pool
  (`spec-decode-scratch-within-budget`), which is sized for 16 segments.
- Whether `gdn-ba-pad-small-m` still pays once upstream hides the ba GEMM on an aux stream.
- KVarN's `build_for_drafting` override had no caller on the V2 runner on 0.33 either (only V1 proposers called it);
  0.34.0 removes the last caller. Pre-existing, unchanged by this port.

## Decisions made in this port, one line each (reject any by name)

- Batch serves W4A16 and refuses `INT8_ACT`, rather than wiring Humming in the same change.
- huggingface_hub 2.4.0 (upstream CI's), with transformers kept at 5.19.0 rather than moved to CI's 5.22.0.
- #64165 carried without its test file, as with the other backports.
