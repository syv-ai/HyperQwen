# Turing: RTX 2080 Ti 22 GB on vLLM 0.33.0

For two cards, launch `bash start-turing.sh` for fp16 W4A16 target + DFlash2, TP=2,
FlashInfer attention, and a 262136-token maximum request length. In Docker use
`docker compose --profile turing up -d` ([docker.md](docker.md#rtx-20-series-turing-sm75)).

The speed tables below use the standard 0.33 configuration with this branch's
Turing patches. No diagnostic code reversion, compiler version downgrade, or forced-output
constraint is included. Version comparisons and diagnostic results are in the
[reproduction report](reproductions/turing-0.33.md).

## Turing unlock (faster FP16 tensor cores)

GeForce TU102 cards run FP16 tensor-core matrix multiplies with FP32 accumulation at reduced speed.
[turing-unlock](https://github.com/nlzy/turing-unlock) (a Turing adaptation of
[gpu-unlock](https://github.com/Zagitta/gpu-unlock)) is a Linux kernel module that lifts that limit
on the TU102, so FP16 tensor math runs at full rate. It is installed on the host, not in this repo
or in the Docker image, and it loads at boot. Its README warns that full-speed tensor cores draw a
lot of power and can damage hardware; use it at your own risk.

On this machine both cards report it active (`turing_unlock` 2.0.0, `targets=all`, kernel log
`TU102 FEAT UNLOCKED` for `0000:01:00.0` and `0000:02:00.0`; check with
`journalctl -k -b | grep turing_unlock` and `lsmod | grep turing_unlock`). Its own README measures
cuBLAS FP16 matrix multiplies with FP32 accumulate at 250 W: 52.8 to 77.9 TFLOPS on one 22 GB card
and 63.6 to 106.5 TFLOPS on another (M=N=K=8192), locked versus unlocked.

**The speed numbers in this document were measured on a machine with the unlock active** (loaded
since the November 7, 2026 boot); no run was made with it off. There is no locked-versus-unlocked
comparison of the model itself, so how much of the decode and prefill speed comes from the unlock is
not known. The patches and launcher settings described here do not depend on it.

## How the model runs: W4A16 in fp16, DFlash2, batching

Everything below is what the server logs at boot (`logs/turing.log`) on two RTX 2080 Ti with the
`turing-unlock` module loaded (see above). The cards are the modified 22 GB variety: `nvidia-smi`
reports 22,528 MiB and a 250 W limit per card, with two NVLink links between them. The 22 GB
modification and the unlock are independent: the first is about memory capacity, the second about
tensor-core speed.

**Weights: W4A16 through Marlin.** The checkpoint is `Qwen3.8-27B-W4A16-AutoRound`, a
`compressed-tensors` `pack-quantized` model: 4-bit signed integer weights in groups of 128 with
per-group scales, and an 8-bit `lm_head`. Activations are never quantized. vLLM selects
`MarlinLinearKernel` for `CompressedTensorsWNA16`, which keeps the weights packed in VRAM,
unpacks them inside the GEMM and multiplies against fp16 activations on the tensor cores. Turing
has no native bf16 tensor-core support, so the launcher forces `--dtype float16`
end to end:
- the checkpoint's bf16 tensors (norms, the conv weights) are converted to fp16 on load, which vLLM
  logs as `Attempted to load weight ... with dtype torch.bfloat16 into parameter with dtype
  torch.float16`;
- the Mamba/GDN state cache is fp16 (`--mamba-ssm-cache-dtype float16`, overriding the model's fp32
  setting);
- the KV cache is fp16 by default (`KV_DTYPE=fp8` is the optional alternative below).

Two things go wrong in fp16 without this repo's patches: the DFlash2 drafter's residual stream
exceeds fp16's 65,504 and every draft is NaN (`patches/dflash2-fp16-residual-scale.patch`), and
the split-KV verify kernel only accepted bf16 (`patches/spec-attn-fp16.patch`; the launcher
sets `FP16_SPEC_OK=1`, which lifts the old "fp16 needs SPEC=off" refusal).

**Speculative decoding: DFlash2 with lookup drafting.** `SPEC=dflash2` (the default here) uses a
separate 1.2 GB W4A16 drafter, `Qwen3.8-27B-DFlash2-W4A16`, configured as `method=dflash` with
`num_speculative_tokens=7`, so each step the target verifies an 8-token block. The drafter predicts
the block in one pass with noncausal sliding-window attention (window 2048), and
`VLLM_DFLASH2_LOOKUP=1` adds n-gram lookup drafting from the context, which helps when the model
repeats text it has already seen. Drafts are verified by the target, so the output is what the target itself would
choose (up to the numeric caveats in the repeatability section). Acceptance varies
with the completion; `SPEC=off` runs plain decoding.

**Measured stack:** vLLM 0.33.0, torch 2.14.0, Triton 3.8.0, FlashInfer
0.7.2.post1, Python 3.14.8, and the six Turing patches cut against 0.33.0.
The November 9 runs and rounded per-run rates are in
[the reproduction record](reproductions/turing-0.33.md).
These tables were remeasured after upstream `9268d8d` and moderator cut
`f85f5ad` were integrated. Periodic prefill checkpoints remain off. The native
build used `CUDA_HOME=/opt/cuda` (nvcc 13.4.92), torch's CUDA 13.0 runtime, and
the dependency versions above; this is not a Docker performance measurement.
For native reproduction on this host, export `CUDA_HOME=/opt/cuda` before
launching; compiler-path settings and rebuilt caches can change the result.

| Part | Used for | Notes |
| --- | --- | --- |
| Marlin (`MarlinLinearKernel`) | W4A16 linear layers | fp16 activations, from vLLM |
| FlashInfer 0.7.2.post1 (FA2 paged prefill) | attention for target and drafter | opted in with `VLLM_FLASHINFER_SM75=1`; full CUDA graphs with `VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=1` |
| CUDA GDN kernel | recurrent prefill of the linear-attention layers | `VLLM_GDN_SM75=1`, compiled once with nvcc |
| Triton GDN | speculative decode of those layers | vLLM's fused CUDA decode kernel needs bf16, so fp16 falls back to Triton (logged at boot) |
| NCCL / custom all-reduce over NVLink | tensor parallel, TP=2 | the heads (24 query, 4 KV) allow TP 1, 2 or 4 |
| Full CUDA graphs | target verification and the drafter | captured at boot (`Capturing dflash2 CUDA graphs (FULL)`) |

**Batching.** This is vLLM's continuous batching, and it is on. The launcher sets `MAX_SEQS=2`, so up
to two requests share each step, with a 2048-token step budget (`max_num_batched_tokens=2048`; vLLM
logs that it caps this at 2048 because of the speculative settings). Prompts are read in chunks
inside that budget, which is what the per-request "reading the prompt" progress shows,
scheduling is asynchronous, and prefix caching is on (Mamba cache mode `align`). At the maximum 262,136-token context the KV pool holds 317,842
tokens, 1.21 times one full-length request. Raise `MAX_SEQS` for more concurrent requests; this has
not been benchmarked here. The throughput numbers in this document are single
stream, and, as noted below, concurrent requests can change greedy output relative to running
them one at a time.

## Full CUDA graphs

`patches/flashinfer-sm75-spec-graphs.patch` enables an opt-in native FA2
paged-prefill graph path. The launcher sets
`VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=1`; use `=0` for the original behavior.
Both target verification and noncausal DFlash query attention capture full
graphs. Planning still occurs eagerly before replay; this does not claim
in-place draft metadata updates.

Measured November 9, 2026 on vLLM 0.33.0, single stream, 24,070 prompt
tokens and 800 generated tokens. Decode is the median of three client streaming
measurements, using `(output_tokens - 1) / (last_chunk_time - first_chunk_time)`.
Here the chunk timestamps are for nonempty text/reasoning deltas.
Cold TTFT is the first run's time to first streamed token; cached TTFT is the
median of the next two runs. All rows use DFlash2,
TP=2 and the CUDA GDN prefill kernel except the last row:

| Configuration | Decode | Cold TTFT | Prompt tokens / cold TTFT | Cached TTFT |
| --- | ---: | ---: | ---: | ---: |
| Full graphs, fp16 KV (default) | 202.2 tok/s | 14.76 s | 1,631 tok/s | 0.33 s |
| Native spec graphs off, fp16 KV | 123.7 tok/s | 14.45 s | 1,666 tok/s | 0.34 s |
| Full graphs, FP8 KV | 170.5 tok/s | 14.52 s | 1,658 tok/s | 2.13 s |
| Full graphs, fp16 KV, Triton GDN prefill | 200.9 tok/s | 16.06 s | 1,499 tok/s | 0.36 s |

Prompt tokens divided by cold TTFT is an effective prefill-rate proxy, including
the first generation step and request overhead. Cached TTFT includes the
uncached suffix and first generation step; it is not a pure prefix-cache timing.

The default run measured 202.2 versus 123.7 tok/s with native spec graphs off
(1.63×) on the frozen 800-token-suite prompt. Configuration changes can
change generated text and acceptance, so this ratio does not isolate graph
overhead alone. Attention graph replay matched eager output at lengths 8, 31, 32, 33, 1024,
23700, 65, then 8, for causal full attention and noncausal window-2048
attention on the 0.33 stack. Reproduce with `venv/bin/python scripts/check_sm75_attention_graphs.py`.

Concurrent requests can change greedy completions relative to running them
individually, including with graphs disabled. This setup does not promise
batch-invariant arithmetic.

## One GPU versus two

These tables use frozen synthetic prompts built with `scripts/bench_turing.py`'s
document generator and an archived measurement client. Each length has one
uncached request and two repeats, using the streaming-rate formula above.
Settings: greedy, `ignore_eos`, 512 generated tokens, one request at a time.
Output token IDs and acceptance counters are saved. Measured November 9, 2026
on vLLM 0.33.0, on the same unlocked machine for all configurations. Repetitive
synthetic output can favor speculation, so these are fixture-specific rates.
The exact client, prompts and per-run results are in
[the reproduction record](reproductions/turing-0.33.md).

| Configuration | Prompt | Cold TTFT | Prompt tokens / cold TTFT | Cached TTFT | Decode (median of 3) |
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

How each was launched, and what the card holds:

| | Launch | Configured context limit | Weights | KV pool |
| --- | --- | ---: | ---: | ---: |
| 2 GPUs | `bash start-turing.sh` | 262,136 | split over 2 | 317,842 tokens |
| 1 GPU, DFlash2 | `CUDA_VISIBLE_DEVICES=0 TP=1 KV_MEM=3000000000 DFLASH_MAX_LEN=8192 bash start-turing.sh` | 8,192 | 15.79 GiB with the drafter | 14,716 tokens |
| 1 GPU, no speculation | `CUDA_VISIBLE_DEVICES=0 TP=1 SPEC=off bash start-turing.sh` | 32,768 | 14.41 GiB | 64,815 tokens |

What this shows:

- In this frozen-prompt suite, one-card DFlash2 decoded at
  107.0–167.7 tok/s,
  versus about 32.7 tok/s for plain decoding at 1k.
- Three additional fixed 1k prompts measured 117.4, 101.2 and 126.7 tok/s.
  The main table's 167.7 tok/s is specific to its completion, not a general
  short-prompt expectation or a claimed optimization over the earlier build.
- The single-card DFlash2 profile uses the reduced KV pool and 8,192-token
  context limit shown above. This is a tested profile, not a demonstrated
  maximum capacity; larger single-card DFlash2 contexts were not benchmarked.
- The two-card profile is configured for 262,136 tokens; performance here was
  measured only up to 24k. Neither this limit nor the KV-pool count is a
  full-window performance or concurrency test.
- Plain one-card decoding measured 32.7 tok/s at 1k and
  29.7 at 24k. Its 24k cold TTFT was 25.77 s.
- The same inputs can generate different text across configurations, changing
  draft acceptance and speed. The additional three fixed 1k prompts and paired
  build controls are in the reproduction report. No diagnostic forced-output
  speed is substituted for these ordinary-request results.

## Optional FP8 KV

`KV_DTYPE=fp8 bash start-turing.sh` keeps fp16 queries and full graphs while
storing attention KV in FP8 E4M3. The reported pool rose from 317,842 to
536,108 tokens (+68.7%). The 24k/800-token trial measured
170.5 tok/s median decode, 14.52 s cold TTFT, and
2.13 s cached TTFT. Cold and cached completions differed;
this is not an accuracy evaluation. FP16 KV stays the default.

## Optional CUDA GDN prefill

`patches/gdn-sm75-cuda-prefill.patch` adapts the recurrent CUDA kernel from
[weicj's Turing fork](https://github.com/weicj/vLLM-2080Ti-Definitive), commit
`93c9013e375b1ecbf2da88179d2a31175a0d5c99`. The Turing launcher enables it with `VLLM_GDN_SM75=1`.
It requires nvcc and a C++ compiler, and compiles once before graph capture.
The original Triton path remains available with `VLLM_GDN_SM75=0`.
The optional `VLLM_GDN_PREFILL_CHECKPOINTS=1` feature requires that Triton/FLA
path. Enabling it with SM75 CUDA prefill raises an error before checkpoint
allocation; keep it at its default of 0 with the standard Turing profile.

The 0.33.0 24k/800-token runs measured 14.76 s cold TTFT with
CUDA GDN versus 16.06 s with Triton GDN. Cached TTFT was
0.33 versus 0.36 s, and median decode
was 202.2 versus 200.9 tok/s. Each configuration had one cold
request; outputs may differ, so this is not an isolated kernel benchmark or
a demonstrated decode benefit. Startup compilation and graph warmup are
outside these request measurements.

## Verified interconnect

A live topology check showed `NV2` between GPU0 and GPU1,
with two active links per GPU (`nvidia-smi topo -m` and
`nvidia-smi nvlink --status`). The server selects CUSTOM/PYNCCL collectives.
The FlashInfer all-reduce
warning only disables that particular implementation for world_size=2.

The guest exposes 22 Xeon E5-2699 v4 CPU cores, one NUMA node, under KVM.
CPU affinity/host scheduling tuning is a potential benchmark, not a measured
speedup. CPU model offload is not enabled; the model already fits on the GPUs.

## Watch prefill and decode speed

The Turing launcher enables per-request progress (modeled on Project Maya, see Credits) in its own terminal
and appends the server output to `logs/turing.log`. Set
`VLLM_TERMINAL_PROGRESS=0` to disable progress or `HYPERQWEN_LOG_FILE` to choose
another log path. It prints reading-the-prompt progress, a prefill timing/rate
with cached tokens separated, answering progress once per second, and a done
or cancellation summary. Reading progress excludes scheduled but unfinished
async chunks. Request IDs distinguish concurrent requests; prompt and generated
text are never logged by this feature. Prefill rates exclude cached tokens;
decode rates exclude the first generated token and queue/prefill time.

Run `python3 scripts/watch_turing.py` in another terminal to follow just these
lines. Use `--lines 0` to show only future events, or `--log /path/to/server.log`
for a custom server log. Ctrl+C stops the watcher, not the server.

`tail -f logs/turing.log` shows all server output, including errors and vLLM's
usual aggregate throughput lines. Future launches of `bash start-turing.sh`
print progress directly without needing the watcher.

The earlier Prometheus view is still available with
`python3 scripts/watch_turing.py --metrics`. `--once` shows completed timing
totals since server startup. Those are aggregate summaries and do not offer
Maya's request-specific progress. Several completions between polls are grouped.

## Terminal dashboard

`python3 scripts/tui_turing.py` draws [Project Maya](https://github.com/mw00/project-maya)'s boxed
dashboard (big decode tok/s, prefill, context, per-GPU and hardware tiles, recent requests) from
vLLM's `/metrics`, `nvidia-smi`, `/proc` and the `[hyperqwen]` request log. It never sends requests.
Maya's code (MIT) is not vendored: the script imports it from your own Project Maya checkout, found
through `MAYA_DIR`, `../project-maya` or `~/project-maya`, and runs under the checkout's `.venv` when
the current Python has no Textual. Without a checkout it falls back to a plain curses display
(`scripts/tui_turing_stdlib.py`). `HYPERQWEN_URL`, `VLLM_API_KEY` and `HYPERQWEN_LOG_FILE` select the
server and log; `q` leaves and the server keeps running. `scripts/turing_metrics.py` is the adapter
that turns vLLM's data into the JSON Maya's dashboard reads.

## Greedy repeatability

`python3 scripts/check_greedy_repeatability.py` sends five fixed prompts (short, code,
reasoning, story, and a long-context question over a ~630-line document) at temperature 0, three rounds
each, one request at a time, and fails if any prompt's output differs between rounds.
The long-context case exercises cold and cached prefill; short prompts may not
fill a cache block. `--save a.json` records the outputs and `--compare a.json` checks a later run against them.
The check requests up to 256 tokens per completion; a response can stop earlier.

On November 9, 2026, the 0.33.0 default server returned byte-identical
completions for all five prompts across three rounds, including cold and cached
prefill within the merged-build boot. A comparison after one restart with the
same compile cache also matched all five saved texts. The logs and output
records are in `logs/merged-remeasurement/`.

This check establishes repeatability for the tested boot and prompts, not
accuracy against a bf16/fp32 reference or invariance to rebuilt compiler caches,
changed settings, or concurrent requests. Fresh-cache repeatability was not
established for this merged stack.

To compare settings, run with `--save a.json`, restart with the other setting,
then use `--compare a.json`. Record the stack, compiler-cache state, generated
text, and acceptance counters with the timing results; an arithmetic change
can alter a near-tied greedy choice and the completion's draftability. The
[reproduction report](reproductions/turing-0.33.md) contains the version and
diagnostic comparisons.

## Credits

- The per-request progress output (reading the prompt, prefill rate with cached tokens
  separated, answering, done) is modeled on the terminal output of Project Maya (https://github.com/mw00/project-maya, MIT, Peasant Smith and the Project Maya contributors).
- The CUDA GDN prefill kernel is adapted from weicj's
  [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) (Apache-2.0), see above.
- `scripts/tui_turing.py` imports the dashboard modules of the same project at run time (see above).
- The tensor-core unlock used for the measurements is [turing-unlock](https://github.com/nlzy/turing-unlock),
  itself an adaptation of Zagitta's [gpu-unlock](https://github.com/Zagitta/gpu-unlock); it is installed on
  the host and is not part of this repo.
