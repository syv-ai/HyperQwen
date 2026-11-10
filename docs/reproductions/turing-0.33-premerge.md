# Archived initial Turing 0.33 measurements and investigation

This preserves the earlier patch-set measurements and diagnostic history.
For the current merged build, see [the GPU remeasurement](turing-0.33.md).


Measured November 9, 2026 on two modified 22 GB RTX 2080 Ti cards with NVLink
and the host `turing-unlock` module active. Native Python 3.14.8 venv;
vLLM 0.33.0, torch 2.14.0, Triton 3.8.0, FlashInfer 0.7.2.post1, and the six
Turing patches from this branch. `bash verify.sh --install` passed. The server
used `Qwen3.8-27B-W4A16-AutoRound`, and DFlash2 runs used
`Qwen3.8-27B-DFlash2-W4A16`. See [Turing setup](../turing.md).

**Patch-set provenance:** the standard 0.33 measurements used the Turing port
from repository commit `611eb30`; diagnostics used that base with the temporary
overrides described below. The 0.32 comparisons used the preserved older venv.
These runs predate upstream
`9268d8d` (PR #434) added periodic GDN prefill checkpoints. The later merge
registers that default-off feature and rejects its unsupported combination
with SM75 CUDA prefill. The merged series has source/CPU integration checks,
but has not been installed into the measured server or rebenchmarked here.
These tables are measurements of the recorded build, not post-merge results.

The local venv used `pyarrow==26.0.1` in place of the repository's
`pyarrow==20` pin because version 20 has no Python 3.14 wheel. No model or
benchmark code was changed for this substitution. The measurements are native,
not Docker measurements. Full-precision benchmark outputs, diagnostic timings
and counters, synthetic prompts, and fixed-output token IDs are archived in
[turing-0.33-results.json](data/turing-0.33-results.json). Tables below round
these values for display. Tensor fixtures, generated GPU code, and launch logs
remain in local ignored `logs/`; they are not part of the portable data archive.

## Method

Each configuration was started separately and allowed to finish loading and
graph capture before requests. `scripts/bench_turing.py` sent one request at a
time, using greedy decoding, `ignore_eos`, and a synthetic document with a fixed
structure and fresh identifier. Each length had three runs: first cold, next
two with the same prompt and prefix caching. Cold TTFT is time to the first
nonempty text/reasoning delta; cached TTFT is the median of the next two runs.
The client decode-rate estimate is `(usage.completion_tokens - 1)` divided by
the interval between the first and last such deltas. It uses
token counts, not chunk counts, but does not timestamp every individual token
or measure isolated GPU kernel time. A warmup request preceded each configuration's benchmark.
The prompts differ slightly in length and text between configurations, and
speculative acceptance depends on generated text. These numbers describe these
completions, not general throughput or an isolated patch speedup.
The three repetitions share a prompt and usually the same completion; they
are not three independent prompt samples. No confidence intervals or
statistical significance estimates were calculated.

Default launch: `bash start-turing.sh` (TP=2, DFlash2, fp16 KV, native
FlashInfer graphs, CUDA GDN prefill, `GPU_UTIL=0.90`, `MAX_SEQS=2`). The
benchmark command was:

```bash
venv/bin/python scripts/bench_turing.py --lengths 1000 4000 7000 24000 --gen 512 --runs 3 --json logs/bench-033-tp2.json
```

| Configuration | Prompt tokens | Cold TTFT | Cached TTFT | Decode runs (tok/s) | Median |
| --- | ---: | ---: | ---: | --- | ---: |
| TP2 DFlash2 | 1,067 | 0.67 s | 0.17 s | 210.1 / 210.4 / 209.4 | 210.1 |
| | 4,068 | 2.34 s | 0.17 s | 217.2 / 217.0 / 216.7 | 217.0 |
| | 7,074 | 3.96 s | 0.26 s | 177.0 / 177.8 / 178.7 | 177.8 |
| | 24,062 | 14.57 s | 0.31 s | 183.4 / 183.3 / 184.6 | 183.4 |
| TP1 DFlash2 | 1,069 | 1.07 s | 0.24 s | 96.8 / 96.7 / 96.3 | 96.7 |
| | 4,066 | 3.97 s | 0.20 s | 132.0 / 132.1 / 131.4 | 132.0 |
| | 7,052 | 6.88 s | 0.44 s | 115.1 / 115.2 / 114.9 | 115.1 |
| TP1 no speculation | 1,067 | 1.04 s | 0.30 s | 32.9 / 32.8 / 32.7 | 32.8 |
| | 4,068 | 3.95 s | 0.14 s | 32.1 / 32.1 / 32.0 | 32.1 |
| | 7,052 | 6.94 s | 0.31 s | 31.7 / 31.7 / 31.6 | 31.7 |
| | 24,063 | 25.71 s | 0.20 s | 29.8 / 29.7 / 29.8 | 29.8 |

TP1 DFlash2 used
`CUDA_VISIBLE_DEVICES=0 TP=1 KV_MEM=3000000000 DFLASH_MAX_LEN=8192 bash start-turing.sh`;
its benchmark omitted `24000` from `--lengths`. TP1 no speculation used
`CUDA_VISIBLE_DEVICES=0 TP=1 SPEC=off bash start-turing.sh` and all four
lengths. Both used `--gen 512 --runs 3`. Boot log KV pools: TP2 default
317,842 tokens; TP1 DFlash2 14,716; TP1 no speculation 64,815.
The 8,192-token TP1 DFlash2 context limit is the tested configuration, not a
demonstrated maximum capacity. No 24k TP1 DFlash2 result was measured.

## Standard configuration comparison with 0.32

The 0.32 measurements were collected November 8 and the 0.33 measurements
November 9 on the same machine, with the Turing patches for each version and
the launch profiles above. No diagnostic reversion, compiler override, or
output constraint is included in this table. Changes are
`100 × (median_033 / median_032 - 1)`, calculated before rounding.
This compares the full stacks: torch 2.13.0 → 2.14.0, Triton 3.7.1 → 3.8.0,
and FlashInfer 0.7.0.post1 → 0.7.2.post1 also changed. It does not isolate the
six Turing patches from the dependency upgrade.

| Configuration | Approximate prompt | 0.32 decode | 0.33 decode | Change |
| --- | ---: | ---: | ---: | ---: |
| TP2 DFlash2 | 1k | 163.6 tok/s | 210.1 tok/s | +28.4% |
| | 4k | 201.1 tok/s | 217.0 tok/s | +7.9% |
| | 7k | 171.7 tok/s | 177.8 tok/s | +3.6% |
| | 24k | 187.4 tok/s | 183.4 tok/s | -2.2% |
| TP1 DFlash2 | 1k | 126.3 tok/s | 96.7 tok/s | -23.5% |
| | 4k | 128.7 tok/s | 132.0 tok/s | +2.6% |
| | 7k | 114.3 tok/s | 115.1 tok/s | +0.7% |
| TP1 no speculation | 1k | 32.7 tok/s | 32.8 tok/s | +0.1% |
| | 4k | 32.1 tok/s | 32.1 tok/s | +0.1% |
| | 7k | 31.7 tok/s | 31.7 tok/s | +0.0% |
| | 24k | 29.7 tok/s | 29.8 tok/s | +0.1% |

These are observed single-request rates on separately generated prompts and
completions. They combine execution cost with draft acceptance and do not
isolate a version's kernel efficiency. The unconstrained TP1 1k decrease
remains the recorded result; the diagnostic controls below do not replace it
with a faster constrained-output number.

## 24k prompt, 800 generated tokens

Each row used `venv/bin/python scripts/bench_turing.py --lengths 24000 --gen 800 --runs 3 --json logs/bench-033-NAME.json`, with the named server launch option.

| Launch option / JSON name | Prompt tokens | Cold TTFT | Cached TTFT | Decode runs (tok/s) | Median |
| --- | ---: | ---: | ---: | --- | ---: |
| Default / `graphs-on` | 24,063 | 14.20 s | 0.31 s | 205.3 / 205.8 / 206.6 | 205.8 |
| `VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=0` / `graphs-off` | 24,063 | 14.32 s | 0.33 s | 104.6 / 103.8 / 104.6 | 104.6 |
| `KV_DTYPE=fp8` / `fp8` | 24,064 | 14.57 s | 2.16 s | 209.7 / 191.3 / 191.2 | 191.3 |
| `VLLM_GDN_SM75=0` / `gdn-off` | 24,061 | 15.79 s | 0.33 s | 210.2 / 209.8 / 209.3 | 209.8 |

All four used TP2 DFlash2; other launch options remained at default. The FP8
KV pool held 536,108 tokens, and the Triton GDN pool held 280,424. Native spec
graphs on versus off measured a 1.97× decode-rate ratio. The CUDA GDN run had
1.59 s (10.1%) shorter cold TTFT than the Triton GDN run. Each configuration
had one cold request; these separate boots used fresh identifiers and may
generate different completions. These ratios are not isolated kernel speedups.

## Functional checks

`venv/bin/python scripts/check_sm75_attention_graphs.py` passed causal and
noncausal graph replay against eager attention at lengths 8, 31, 32, 33, 1024,
23700, 65, and 8, with zero maximum error in this check.

`venv/bin/python scripts/check_greedy_repeatability.py --rounds 3 --max-tokens 256 --save logs/repeat-033-default.json`
passed: five of five prompts gave byte-identical outputs across three rounds
within the default server boot. This does not test cross-restart repeatability.

## Controlled investigation of the 1k single-GPU drop

**Current finding:** the saved fixed-prompt results support a substantial
completion-dependent draftability component. The [fixed-output control below](#holding-the-completion-fixed)
made both versions emit identical token IDs: both ran the saved old completion
at about 124 tok/s and the saved new completion at about 102 tok/s. The
source/compiler changes below affect computation boundaries and numeric
results. The grammar constraint also changes the request path, so this is
supporting evidence, not a quantitative attribution of the unrestricted gap.
The evidence does not establish a uniform 23.5% execution-speed regression or
attribute the original observation to one cache-policy expression.

The original TP1 1k result fell from 126.3 to 96.7 tok/s (23.5% from unrounded rates). Those
benchmark runs used different fresh document identifiers and did not save their
completion text or per-request acceptance counters. Their exact textual
divergence cannot be reconstructed from the saved JSON.

To investigate, both preserved venvs were run with the same current launcher,
TP1, `KV_MEM=3000000000`, `DFLASH_MAX_LEN=8192`, and default DFlash2 settings.
Three prompts were built once with `bench_turing.make_prompt`, target length
1000 and identifiers `fixed-comparison-0000000000000000`,
`fixed-comparison-0000000000000001`, and `fixed-comparison-0000000000000002`.
Each was sent unchanged to both versions, three times, with 512 generated
tokens, temperature 0, and `ignore_eos`. Generated text and Prometheus counter
deltas were saved in local `logs/fixed-1k-{032,033}.json`; the exact prompts
are in `logs/fixed-1k-prompts.json`.

| Prompt | 0.32 median tok/s | 0.33 median tok/s | Change | 0.32 tokens/step | 0.33 tokens/step | 0.32 / 0.33 ms per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 90.28 | 84.41 | -6.5% | 3.66 | 3.45 | 40.43 / 40.36 |
| 1 | 123.55 | 102.09 | -17.4% | 5.05 | 4.17 | 40.55 / 40.69 |
| 2 | 87.67 | 94.86 | +8.2% | 3.62 | 3.87 | 40.76 / 40.81 |

Here and in the diagnostic tables below, "tokens/step" denotes the
counter-based mean accepted length, `1 + accepted_tokens / num_drafts`, not
an exact emitted-token count: the final accepted block can be clipped by
`max_tokens`. "ms per step" is
client-observed decode time divided by the number of drafts, so it includes
request boundary effects and is not a kernel profiler measurement. Each prompt
had identical acceptance counts across its three rounds within a version.
Completion text differed between versions for all three prompts. The almost
unchanged client-side step-time estimate and changed accepted lengths are
consistent with completion-dependent throughput; this is not a measurement
of a uniform 23.5% execution slowdown on 0.33.

Additional probes supplied both versions with identical input token IDs at the
common prefixes before textual divergence, then requested 16 tokens with top-5
log probabilities. This recomputes the prefix as prefill and therefore does not
exactly replay its original incremental decode. It nevertheless confirms
different token scores and choices across stacks:

| Fixed-prefix probe | 0.32 log probabilities | 0.33 log probabilities |
| --- | --- | --- |
| Prompt 2, next word | `model`: -1.288966; `patterns`: -1.288966; selected `model` | `patterns`: -1.246017; `model`: -1.308517; selected `patterns` |
| Prompt 1, punctuation after `essay` | period: -0.701109; comma: -0.716734; selected period | comma and period: -0.709009; selected comma |

These observations support the mechanism: stack-dependent numeric scores change
near-tied greedy choices, the completion changes, and DFlash2 accepts a different
number of draft tokens. The document identifier also affects the completion:
changing only that identifier produced 84–102 tok/s on the same 0.33 server.
Those initial tests did not identify the source change. The subsequent
isolations below identify a changed DFlash cache policy and compiler-dependent
GDN arithmetic; the original unsaved completion still cannot be reconstructed.
Raw score probes are in local `logs/divergence-{032,033}.json`.

### Upstream DFlash trailing-block drop changes the prefill checkpoint

In the installed 0.32 source, `SpeculativeConfig.use_eagle_block_drop()` returns
`use_eagle() and not disable_eagle_block_drop`, including for DFlash. Pristine
upstream 0.33.0 adds `needs_eagle_block_drop()`, which explicitly exempts
DFlash/DSpark: their context KV reads target hidden states alone, whereas
EAGLE-like drafters also read the next token. This change is present in the
upstream tag's [`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/speculative.py),
not introduced by the Turing port.

With 448-token cache blocks and these roughly 1,072-token prompts, dropping the
trailing block places the reusable Mamba checkpoint at token 448. Exempting
DFlash moves it to token 896. The scheduler splits prefill at that checkpoint;
both the cold split and the cached-resend boundary therefore change. That
changes the shapes and boundaries at which the fp16 model computes its prompt
and recurrent state, and can change near-tied token choices.

To isolate this policy, the **same 0.32 venv** was restarted with the existing
speculative configuration option `"disable_eagle_block_drop": true`. No model
or kernel source was edited, and the boot logs show **the same AOT cache paths**
for the backbone, draft head, and candidate selector. The compiled binaries
themselves were not independently checksummed for this comparison.
The saved fixed prompts and request settings remained identical.

| Prompt | 0.32 default tok/s | 0.32 no block drop tok/s | Change | Tokens/step, default → no drop | ms/step, default → no drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 90.28 | 84.17 | -6.8% | 3.66 → 3.41 | 40.43 → 40.47 |
| 1 | 123.55 | 91.81 | -25.7% | 5.05 → 3.74 | 40.55 → 40.63 |
| 2 | 87.67 | 73.01 | -16.7% | 3.62 → 2.99 | 40.76 → 40.69 |

All three completions changed. Each prompt repeated the same acceptance counts
across its three rounds. The logs confirmed the new 896-token boundary and
cached TTFT fell from approximately 0.64–0.68 s to 0.24 s. Thus, changing this
policy alone reproduces changed completions, lower acceptance, and a slowdown
as large as the originally observed drop, while per-step cost stays nearly
unchanged. It does not reproduce the exact 0.33 completions or rates, because
the rest of that stack also differs. Raw results are in local
`logs/fixed-1k-032-no-block-drop.json`.

Disabling lookahead hashes alone on unmodified 0.33 did not alter this behavior:
`"disable_lookahead_block_hashes": true` retained the 896-token boundary
and the same acceptance counts as default 0.33. DFlash is independently exempt
from the trailing-block drop. This control is saved in
`logs/fixed-1k-033-legacy-hashes.json`.

### Isolating compiler-dependent GDN arithmetic

A subsequent operation-level comparison used identical saved fp16 inputs on the
same GPU. The recurrent GDN kernel function
`fused_recurrent_gated_delta_rule_fwd_kernel` has an identical Python AST in
the installed 0.32 and 0.33 sources (SHA-256
`98de695d7facd9c682687dd1630b9abfe7cac753811755f9c75237fb5810fa88`).
Its output nevertheless changed between the two stacks. To separate the
compiler from vLLM and torch, standalone diagnostic processes imported the
other venv's Triton through an isolated `PYTHONPATH`; no package installation
or serving source file was changed.

| vLLM / torch | Triton | Recurrent GDN result |
| --- | --- | --- |
| 0.32 / 2.13 | 3.7.1 | Reference A |
| 0.33 / 2.14 | 3.8.0 | Result B |
| 0.33 / 2.14 | 3.7.1 | Exactly A, including state |
| 0.32 / 2.13 | 3.8.0 | Exactly B, including state |

The fixture used 8 query tokens, 16 key heads, 48 value heads, head dimension
128, fp16 initial state, and in-kernel Q/K normalization. Between A and B,
27 of 49,152 output elements differed (maximum absolute difference
0.00006103515625), as did 443 of 7,077,888 saved state elements (maximum
0.00048828125). The saved state includes nine allocated slots. Generated PTX
also differed: this specialization contained 358 `fma.rn.f32` instructions
under 3.7.1 versus 260 under 3.8.0, with different separate add/multiply counts.
These are static instruction counts, not execution-time measurements.

On the other saved fixtures, Marlin W4A16 GEMM (both fp32-reduction settings),
FlashInfer FA2 attention, gated RMS normalization, and eager RMS normalization
matched exactly. Compiled RMS normalization also showed small differences
across compiler stacks. These bounded checks do not establish equivalence for
all inputs or rule out additional differences elsewhere in the model.

This identifies the Triton 3.7.1 to 3.8.0 upgrade as a concrete source of
changed recurrent arithmetic, even with unchanged GDN kernel source. It does
not by itself establish that this is the sole cause of every changed completion
or quantify its contribution to the original 23.5% observation. Local artifacts
are `logs/ops-{032,033,213-38,214-37}.pt`, `logs/ops-inputs.pt`, and
`logs/gdn-{37,38}.{ptx,ttir,ttgir,llir}`.

A full-server diagnostic also ran 0.33/torch 2.14 with Triton 3.7.1 imported
through the temporary path and a separate temporary vLLM cache. Its medians
were 82.86, 101.39, and 101.27 tok/s, versus default 0.33's 84.41, 102.09,
and 94.86. Tokens/step were 3.38, 4.17, and 4.16. Reverting the compiler alone
therefore did not recover the old 0.32 results. This was a diagnostic override,
not a supported dependency pin or a serving-code fix. Results are in
`logs/fixed-1k-033-triton37.json`.

These controls identify **changed computation boundaries from upstream DFlash
block-drop policy, plus changed compiler arithmetic**, with speed following
the resulting completion and acceptance.
Neither observation establishes that the original 23.5% is a uniform execution
regression or assigns an exact share of that unsaved run to each change. The
default 0.33 dependency stack and launcher settings were restored afterward;
only documentation was changed in the repository.

### Reverting the block-drop method on 0.33: reversible follow-up

The reverse experiment tested whether restoring the old method on 0.33 would
restore the old performance. The installed `vllm/config/speculative.py` was
backed up, then this single return expression was changed temporarily:

```diff
- return self.needs_eagle_block_drop() and not self.disable_eagle_block_drop
+ return self.use_eagle() and not self.disable_eagle_block_drop
```

Three separate TP1 server boots used the same saved prompts and settings as
above: a fresh unmodified 0.33 baseline, the one-line change alone, and that
change with `"disable_lookahead_block_hashes": true` in the speculative config.
Each prompt ran three times with 512 output tokens. All three boot logs show
the same backbone, draft-head, and candidate-selector AOT cache paths.

| Prompt | Earlier 0.32 reference | Fresh 0.33 baseline | Old method only | Old method + legacy hashes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 90.3 tok/s | 84.2 tok/s | 84.1 tok/s | 121.9 tok/s |
| 1 | 123.5 tok/s | 101.6 tok/s | 101.5 tok/s | 91.3 tok/s |
| 2 | 87.7 tok/s | 94.5 tok/s | 94.5 tok/s | 91.8 tok/s |

These are medians of three runs. The old-method-only variant produced exactly
the same completions and acceptance counts as the fresh 0.33 baseline on every
prompt. Its reusable boundary remained at 896 tokens. Restoring that expression
alone therefore **did not recover the old speeds**.

The reason is a second 0.33 control path: changing the method to return true
also enables `VllmConfig.num_hash_lookahead_tokens`. In `Scheduler.__init__`,
a nonzero value explicitly sets `self.use_eagle_block_drop = False`. To
actually recover the legacy 448-token boundary, the diagnostic also had to
disable lookahead block hashes. The logs and usage counters confirmed 448
cached tokens in that combined variant, versus 896 in the other two.

The combined variant changed acceptance and produced mixed performance, not a
general recovery of 0.32 behavior. Prompt 0 improved, while prompts 1 and 2
slowed relative to the fresh 0.33 baseline. Its cold and cached completions
differed on prompts 0 and 1; prompt 2's text matched across rounds, but its
acceptance counts differed. None of the three first-run completions matched
the saved 0.32 reference. By contrast, the baseline and method-only variants
were byte-identical across rounds. These observations further limit any claim
that reverting this one method would undo the full-stack upgrade's effects.

**Rollback was part of the experiment:** the controller used a `finally` block
to stop the test server, restore the saved source byte-for-byte, clear that
module's bytecode cache, verify its SHA-256, and restart the default TP2 0.33
server without either diagnostic override. The restored source SHA-256 is
`9243ffcbba50e25c8214386710015db99603bdbabdaeca88a1d56748dc63cccc`.
No source modification from this experiment is retained. Local backups,
manifest, rollback instructions, per-run results, and logs are in
`logs/block-drop-revert-test/`; `summary.json` records output comparisons and
the AOT artifact paths. The report records this diagnostic result, not a
proposed serving-code fix.

### Additional source changes: context projection and normalization

Upstream 0.33 also moved part of DFlash context preparation inside the full
CUDA graph. In 0.32, `DFlashSpeculator.propose()` combines all auxiliary hidden
states in one projection before replay. In 0.33, `_prepare_graph_replay()` and
`_store_hidden_states()` stage the graph's first context rows separately, and
`_prepare_graph_forward()` projects those staged rows during replay. At TP1
with seven speculative tokens, this separates the first eight rows from the
rest of a prefill chunk. The drafter's `combine_hidden_states()` implementation
is unchanged, but the sizes of its Marlin GEMM calls change.

A frozen-input probe used the actual drafter `fc.weight_packed` and
`fc.weight_scale`, input width 25,600, output width 5,120, and the normal fp32
Marlin reduction. Splitting a projection into eight rows plus the remainder
changed 116,237 of 901,120 output values at 176 rows (relative L2 0.001646),
and 164,376 of 3,194,880 at 624 rows (relative L2 0.001283). Same-shape repeats
were exact. The full and split outputs each matched exactly across the two
venvs: this is sensitivity to the changed call shape, rather than evidence
that the Marlin kernel became slower or universally changed its result.
Inputs were saved synthetic auxiliary states, not captured request states;
the weights were the real checkpoint weights. The probe and artifacts are in
local `logs/context-projection-diagnostic/`.

A separate 0.33 server diagnostic restored the old context-preparation order
through a temporary import overlay. It kept default 0.33 cache policy,
dependencies, and the same three AOT cache paths as the fresh baseline; installed
files were not edited. The saved 1k prompts gave:

| Prompt | Fresh 0.33 tok/s | Old context order tok/s | Old context tokens/step |
| --- | ---: | ---: | ---: |
| 0 | 84.21 | 86.28 | 3.514 |
| 1 | 101.58 | 101.21 | 4.137 |
| 2 | 94.47 | 93.97 | 3.880 |

All three repeated identically within the diagnostic boot. Prompts 0 and 2
changed text relative to baseline; prompt 1's text stayed identical. This
confirms another source change can affect the draft computation, but it did
**not** recover the large remaining gap for prompt 1. The overlay was removed
by restarting the default server, and installed source checksums matched
before and after. Local records are `logs/context-graph-ablation/` and
`logs/fixed-1k-033-old-context-order.json`.

Several apparent source differences were ruled out or bounded:

- `qwen3_dflash2.py`, including the fp16 residual-scaling implementation, and
  `dflash2/lookup.py` are identical between the installed builds.
- Draft seed mixing changed, but Gumbel noise is bypassed at temperature zero;
  that seed change does not explain these greedy draws. FLy verification was
  not enabled in the tested configuration.
- Draft quantization mapping was refactored, but the new specialized mapping
  concerns ModelOpt configurations. This checkpoint uses compressed-tensors;
  the fallback retains the prior layer-alias handling. Both venvs have
  compressed-tensors 0.17.0 and cuBLAS 13.1.1.3 and read the same model files.
- The new CUDA `fla_rms_norm_gated` wrapper is not selected by this launcher's
  `custom_ops=["+rms_norm", "+silu_and_mul", "none"]`: gated normalization
  dispatches to the native implementation. Its actual `forward_static`
  calculation, both eager and compiled, matched across versions on a frozen
  8 × 48 × 128 fixture. This bounded result does not rule out all compiler
  differences elsewhere; the recurrent GDN and RMS probes above did show them.

### Holding the completion fixed

To separate execution cost from the completion's draftability, both preserved
versions ran the **same input token IDs** for fixed prompt 1 and were constrained
to reproduce each of its two saved completions. A literal regular expression
was supplied through `/v1/completions` `structured_outputs`, with
`--structured-outputs-config '{"enable_in_reasoning":true}'` on the diagnostic
servers so the constraint also applied to reasoning text. These are diagnostic
requests, not replacements for the unconstrained benchmark above.

Both boots used TP1, DFlash2, the default fp16 KV and graph settings,
`KV_MEM=3000000000`, `DFLASH_MAX_LEN=8192`, temperature zero, and the original
code and dependency stack for that version. The prompt was warmed first.
Each completion ran three times. `return_token_ids=true` captured the actual
generated IDs; all six runs for a given completion emitted exactly the same
513 IDs, including the final stop token. The request limit was 1,024 tokens
to allow the entire saved 512-token completion and its stop token.

| Fixed completion | 0.32 median decode | 0.33 median decode | 0.32 tokens/step | 0.33 tokens/step |
| --- | ---: | ---: | ---: | ---: |
| Saved 0.32 text | 123.93 tok/s | 124.07 tok/s | 5.020 | 5.069 |
| Saved 0.33 text | 102.40 tok/s | 101.58 tok/s | 4.163 | 4.163 |

The old text took 102 drafting steps on 0.32 and 101 on 0.33; the new text took
123 on both, with the same counts across each case's three rounds. Changing
the completion changes the rate substantially **on either software version**.
With the completion held fixed, the median version difference was within 1%
in these two cases. The constrained 0.33 result on the old text also closely
matches the earlier unconstrained 0.32 result of 123.55 tok/s; its new-text
result matches the fresh unconstrained 0.33 result of 101.58 tok/s.

Structured decoding can prune grammar-invalid draft tokens before verification,
so `accepted/drafted` fractions from this control must not be compared directly
to ordinary requests. The constraint can change verification lengths and other
execution details, so this is not an isolated test of the unrestricted request
path. The table uses drafting-step counts and identical emitted token
sequences, and its rates are reported as a diagnostic comparison only.
It does not establish a universal throughput or recover the original unsaved
1k completion. It supports a substantial completion/draftability effect on
this saved prompt, but does not measure the share of the unrestricted gap
caused by text alone or rule out request-path-specific regressions. The
original 23.5% observation cannot be assigned exact causal shares from its
saved JSON alone.

No serving source was edited for this control. The venv selection and diagnostic
structured-output setting were restored to the default 0.33 TP2 service.
Input IDs, frozen completions, generated token IDs, counters, and summaries are
in local `logs/fixed-output-diagnostic/`.

## Follow-up: unconstrained TP1 short-prompt performance

Improving short-prompt decoding on one GPU is follow-up work; no production
optimization is included in these findings. The standard 1k measurement remains
96.7 tok/s. The three saved 1k prompts provide an additional unconstrained 0.33
baseline of 84.21, 101.58, and 94.47 tok/s from the fresh-baseline boot.
Future work should use those same saved prompts, record generated token IDs
and accepted lengths, and report cold and cached TTFT separately. Improvements
must be demonstrated on unconstrained requests across prompts; the fixed-output
control is not a production workaround or a recovered headline speed.
