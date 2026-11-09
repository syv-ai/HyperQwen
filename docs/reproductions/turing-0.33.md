# Turing GPU remeasurement after the upstream and moderator merges

Measured November 9, 2026 on two modified 22 GB RTX 2080 Ti cards with NVLink,
250 W limits and the host Turing unlock active. All runs are native Python
3.14.8, not container measurements. This report supplies the current numbers
for [the Turing guide](../turing.md).

## Build and controls

The candidate combines upstream `9268d8d` (periodic GDN checkpoints), our
`83d80e2` integration, and moderator branch `cpuchip:turing-0.33-cuts` at
`f85f5ad`. Five Turing cuts are taken directly from that branch; the SM75 CUDA
prefill cut retains the later unsupported-checkpoint-combination guard.
`VLLM_GDN_PREFILL_CHECKPOINTS=0` throughout this campaign.

A separate `venv-033-remeasured` was installed from the vLLM 0.33.0 wheel,
then all 64 series patches and four KVarN patches were applied. Serving
dependencies were copied into that independent environment; 205 distribution
versions match the preserved environment, and all 13 vLLM native wheel
extensions match byte-for-byte. Installation verification passed with zero
failures, patch reapplication was a no-op, and all 3,245 package Python files
byte-compiled. The patched source checksum stamp and individual patch hashes
are recorded with the measurement data.

The runtime is torch 2.14.0 (CUDA 13.0 build), Triton 3.8.0, FlashInfer Python
and cubin 0.7.2.post1, compressed-tensors 0.17.0 and pyarrow 26.0.1. Native
extension builds selected the host toolchain at `CUDA_HOME=/opt/cuda`, nvcc
13.4.92. Triton's compiler and cached/prebuilt FlashInfer kernels remain part
of their recorded package versions. That explicit CUDA-path setting is identical on the restarted pre-merge
and candidate controls; it is not a claim about Docker's pinned compiler.
Compiler caches are used after warmup. Cache rebuilds and environment paths
can change numeric results; this is a build comparison, not an attribution of
every output difference to a particular source edit.
Native reproduction on this host requires `export CUDA_HOME=/opt/cuda` before
launching a profile; a different path setting can select a different compile
cache even if it resolves to the same compiler.

## Method

One request at a time, temperature 0, top_p 1, `ignore_eos=true`, 512 generated
tokens for the main tables and 800 for the variant table. No output constraint,
diagnostic source reversion, or dependency downgrade is used. Prompts were
built once and frozen; every boot rechecked the chat-template token IDs. Each
length has one uncached request and two repeats. A separate frozen prompt is
used for the 800-token variants so the earlier 512-token suite does not warm it.

Cold TTFT is time to the first nonempty text/reasoning delta; cached TTFT is
the median of the next two runs. The client decode proxy is
`(completion_tokens - 1) / (last_nonempty_delta_time - first_nonempty_delta_time)`.
It is not isolated GPU kernel time. Output token IDs, usage, and acceptance
counters are saved. Generation-counter deltas match each request's output
count, and the initial gate required sustained idle request/GPU metrics and
no unrelated GPU processes before replacement.
The recorded counter-based mean accepted length is
`1 + accepted_tokens / num_drafts`; final-block clipping means it is not an
exact emitted-token count per iteration.

Three repetitions of one prompt do not establish general throughput or
statistical significance. Greedy output and draft acceptance can change with
configuration, tensor parallelism, compiler artifacts, or cached state. The
synthetic document and `ignore_eos` can also favor repetitive output.

Full-precision results, the exact client, synthetic prompt fixtures, unique
output token sequences, hardware snapshots, dependency inventory and check
logs are in [the measurement archive](data/turing-0.33-merged-results.json).
The earlier version comparison and source-isolation investigation are
[archived separately](turing-0.33-premerge.md); those are not current speed quotes.

For an exact-cohort rerun, extract the archive's `fixtures` under their recorded
filenames and `benchmark_client_source` as `benchmark.py` into a measurement
directory. Adjust the client's `ROOT` to your checkout and keep `BASE` pointing
to that directory. Start one fresh profile at a time with the recorded settings,
then use `--label NAME --lengths 1000 4000 7000 24000 --gen 512`; omit 24000 for
the 8k TP1 DFlash2 profile. The variant suite uses `--lengths 24000 --gen 800`.
Do not regenerate prompt identifiers when comparing builds; the client checks
the frozen template token IDs before timing requests.

## Current merged-build measurements: 512 output tokens

| Configuration | Prompt tokens | Cold TTFT | Prompt tokens / cold TTFT | Cached TTFT | Decode median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 GPUs, TP=2, DFlash2 | 1,070 | 0.64 s | 1,680 tok/s | 0.15 s | 249.0 tok/s |
|  | 4,069 | 2.35 s | 1,729 tok/s | 0.16 s | 167.2 tok/s |
|  | 7,053 | 3.99 s | 1,766 tok/s | 0.25 s | 191.0 tok/s |
|  | 24,066 | 14.64 s | 1,643 tok/s | 0.33 s | 209.7 tok/s |
| 1 GPU, TP=1, DFlash2 | 1,070 | 1.07 s | 1,002 tok/s | 0.24 s | 167.7 tok/s |
|  | 4,069 | 3.95 s | 1,031 tok/s | 0.19 s | 107.0 tok/s |
|  | 7,053 | 6.87 s | 1,026 tok/s | 0.44 s | 133.9 tok/s |
| 1 GPU, TP=1, no speculation | 1,070 | 1.03 s | 1,042 tok/s | 0.30 s | 32.7 tok/s |
|  | 4,069 | 3.95 s | 1,029 tok/s | 0.14 s | 32.1 tok/s |
|  | 7,053 | 6.95 s | 1,014 tok/s | 0.31 s | 31.7 tok/s |
|  | 24,066 | 25.77 s | 934 tok/s | 0.21 s | 29.7 tok/s |

| Configuration | Prompt tokens | Decode runs, cold / cached / cached (tok/s) | Median | Identical token IDs across runs |
| --- | ---: | --- | ---: | --- |
| 2 GPUs, TP=2, DFlash2 | 1,070 | 242.8 / 250.2 / 249.0 | 249.0 | True |
|  | 4,069 | 166.3 / 167.2 / 167.2 | 167.2 | True |
|  | 7,053 | 191.0 / 190.9 / 191.8 | 191.0 | True |
|  | 24,066 | 210.9 / 209.5 / 209.7 | 209.7 | True |
| 1 GPU, TP=1, DFlash2 | 1,070 | 167.7 / 167.7 / 167.1 | 167.7 | True |
|  | 4,069 | 107.1 / 107.0 / 106.9 | 107.0 | True |
|  | 7,053 | 133.9 / 134.0 / 133.9 | 133.9 | True |
| 1 GPU, TP=1, no speculation | 1,070 | 32.9 / 32.7 / 32.5 | 32.7 | True |
|  | 4,069 | 32.1 / 32.1 / 32.1 | 32.1 | True |
|  | 7,053 | 31.7 / 31.7 / 31.7 | 31.7 | True |
|  | 24,066 | 29.7 / 29.7 / 29.7 | 29.7 | True |

Configured limits: TP2 DFlash2 262,136; TP1 DFlash2 8,192 with
`KV_MEM=3000000000 DFLASH_MAX_LEN=8192`; TP1 plain decode 32,768 with `SPEC=off`.
These are tested profiles, not measured maximum single-card capacities. No
24k single-card DFlash2 or full-window 262k performance claim is made.
Reported KV pools are 317,842 tokens for default TP2, 14,716 for
TP1 DFlash2, and 64,815 for TP1 plain decoding.

## Paired pre-merge versus merged controls

Both builds received the same frozen inputs, dependency versions, compiler
setting and inference profile. The paired TP2 reference is a restarted,
matched-toolchain run, not the initial live-server warm-cache sample retained
in the archive. These controls compare the two installed builds, including
their compiled artifacts; changed outputs limit isolated speedup claims.

| Profile | Target prompt length | Pre-merge tok/s | Merged tok/s | Change | Identical output IDs in all paired rounds |
| --- | ---: | ---: | ---: | ---: | --- |
| tp2 | 1000 | 253.67 | 248.97 | -1.85% | True |
| tp2 | 4000 | 154.66 | 167.16 | +8.08% | False |
| tp2 | 7000 | 190.51 | 190.99 | +0.25% | True |
| tp2 | 24000 | 211.21 | 209.72 | -0.71% | True |
| tp1-dflash | 1000 | 167.47 | 167.71 | +0.14% | True |
| tp1-dflash | 4000 | 109.11 | 106.98 | -1.96% | True |
| tp1-dflash | 7000 | 124.70 | 133.89 | +7.38% | False |
| tp1-nospec | 1000 | 32.67 | 32.74 | +0.21% | True |
| tp1-nospec | 4000 | 32.03 | 32.08 | +0.15% | True |
| tp1-nospec | 7000 | 31.66 | 31.66 | -0.01% | True |
| tp1-nospec | 24000 | 29.75 | 29.73 | -0.09% | True |

Identical output IDs do not imply identical draft proposals or accepted-length
histories. For example, vLLM reported 76 draft generations for the TP2 1k case
before the merge and 77 afterward, while emitting the same 512 tokens. The archive records
those counters separately; the rate differences are not isolated kernel costs.

## Additional fixed 1k prompts on one GPU

These are the three saved `fixed-comparison-*` document prompts from the earlier
investigation, now rerun on both current environments. They remain unconstrained;
no fast completion was selected or forced. The spread is evidence that a single
short-prompt number is not a universal rate.

| Prompt | Pre-merge tok/s | Merged tok/s | Change | Identical output IDs in all paired rounds |
| --- | ---: | ---: | ---: | --- |
| 0 | 120.07 | 117.40 | -2.22% | False |
| 1 | 99.74 | 101.21 | +1.47% | True |
| 2 | 73.74 | 126.70 | +71.82% | False |

## Two-GPU variants: 24k prompt, 800 output tokens

All rows use the same 800-token-suite prompt and DFlash2. Only the named
backend/KV setting changes; each is a fresh boot with a cold first request.

| Configuration | Decode median | Cold TTFT | Prompt tokens / cold TTFT | Cached TTFT |
| --- | ---: | ---: | ---: | ---: |
| Full graphs, fp16 KV (default) | 202.2 tok/s | 14.76 s | 1,631 tok/s | 0.33 s |
| Native spec graphs off, fp16 KV | 123.7 tok/s | 14.45 s | 1,666 tok/s | 0.34 s |
| Full graphs, FP8 KV | 170.5 tok/s | 14.52 s | 1,658 tok/s | 2.13 s |
| Full graphs, fp16 KV, Triton GDN prefill | 200.9 tok/s | 16.06 s | 1,499 tok/s | 0.36 s |

| Configuration | Decode runs (tok/s) | Median | Identical token IDs across runs |
| --- | --- | ---: | --- |
| Full graphs, fp16 KV (default) | 202.2 / 200.8 / 202.5 | 202.2 | True |
| Native spec graphs off, fp16 KV | 123.1 / 123.7 / 124.1 | 123.7 | True |
| Full graphs, FP8 KV | 210.3 / 170.5 / 170.0 | 170.5 | False |
| Full graphs, fp16 KV, Triton GDN prefill | 201.2 / 200.0 / 200.9 | 200.9 | True |

The native-graph on/off median ratio is 1.63× for this fixture. Ratios
across configurations include output/acceptance differences where present and
must not be reported as isolated kernel gains. FP8 increases reported KV
capacity from 317,842 to 536,108 tokens (+68.7%). Its cold and cached
outputs differ in this trial; this is not an accuracy evaluation or a decode
speed recommendation. FP16 KV remains the standard profile.

## GPU validation and final state

The SM75 attention checker passed its causal and noncausal graph/eager
comparisons at lengths 8, 31, 32, 33, 1024, 23700, 65, and 8 on its fixed
fixture. The default merged server passed all five greedy-repeatability
prompts across three rounds, and the post-reference restoration comparison
matched that saved text after a restart with the same compile cache. These
checks do not establish bf16/fp32 reference accuracy, all-model-shape coverage,
concurrent-request invariance, or fresh-cache repeatability.

The independent merged environment remains selected and its default TP2
server is running. The pre-merge environment is retained for rollback. No
production speed optimization was applied in this campaign. The next task is
to improve unconstrained one-GPU short-prompt performance across the saved
suite, with output and acceptance evidence; these measurements are its baseline.
