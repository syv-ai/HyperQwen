# Turing: RTX 2080 Ti 22 GB on vLLM 0.33.0

For two cards, launch `bash start-turing.sh` for fp16 W4A16 target + DFlash2, TP=2,
FlashInfer attention, and a 262136-token maximum request length. In Docker use
`docker compose --profile turing up -d` ([docker.md](docker.md#rtx-20-series-turing-sm75)).

The speed tables below use the standard 0.33 configuration with this branch's
Turing patches. No diagnostic code reversion, compiler override, or forced-output
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
These measurements predate the later upstream periodic-checkpoint patch
([gdn-checkpoints.md](gdn-checkpoints.md)); that feature was not enabled in the
measured build, and the merged patch series has not been rebenchmarked here.

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

Measured November 9, 2026 on vLLM 0.33.0, single stream, about 24,063 prompt
tokens and 800 generated tokens. Decode is the median of three client streaming
measurements, using `(output_tokens - 1) / (last_chunk_time - first_chunk_time)`.
Here the chunk timestamps are for nonempty text/reasoning deltas.
Cold TTFT is the first run's time to first streamed token; cached TTFT is the
median of the next two runs. All rows use DFlash2,
TP=2 and the CUDA GDN prefill kernel except the last row:

| Configuration | Decode | Cold TTFT | Prompt tokens / cold TTFT | Cached TTFT |
| --- | ---: | ---: | ---: | ---: |
| Full graphs, fp16 KV (default) | 205.8 tok/s | 14.20 s | 1,694 tok/s | 0.31 s |
| Native spec graphs off, fp16 KV | 104.6 tok/s | 14.32 s | 1,680 tok/s | 0.33 s |
| Full graphs, FP8 KV | 191.3 tok/s | 14.57 s | 1,652 tok/s | 2.16 s |
| Full graphs, fp16 KV, Triton GDN prefill | 209.8 tok/s | 15.79 s | 1,524 tok/s | 0.33 s |

Prompt tokens divided by cold TTFT is an effective prefill-rate proxy, including
the first generation step and request overhead. Cached TTFT includes the
uncached suffix and first generation step; it is not a pure prefix-cache timing.

The default run measured 205.8 versus 104.6 tok/s with native spec graphs off
(1.97×). These separate runs used fresh prompt identifiers and may generate
different text, so the ratio does not isolate graph overhead alone. Attention
graph replay matched eager output at lengths 8, 31, 32, 33, 1024,
23700, 65, then 8, for causal full attention and noncausal window-2048
attention on the 0.33 stack. Reproduce with `venv/bin/python scripts/check_sm75_attention_graphs.py`.

Concurrent requests can change greedy completions relative to running them
individually, including with graphs disabled. This setup does not promise
batch-invariant arithmetic.

## One GPU versus two

`python3 scripts/bench_turing.py` measures single-stream speed against a running server: for each prompt
length it sends a never-seen prompt (cold prefill), then the same prompt twice more (cached prefill), and
reports decode tokens per second after the first token for each of the three runs. Settings: greedy,
`ignore_eos`, 512 generated tokens, prompts built from a repetitive synthetic document, so read decode
rates as "this kind of text" rather than a universal figure. Measured November 9, 2026
on vLLM 0.33.0. All three configurations ran on the same unlocked machine (see above),
one request at a time. Rounded per-run rates are in [the reproduction record](reproductions/turing-0.33.md).

| Configuration | Prompt | Cold TTFT | Prompt tokens / cold TTFT | Cached TTFT | Decode (median of 3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 GPUs, TP=2, DFlash2 | 1,067 | 0.67 s | 1,599 tok/s | 0.17 s | 210.1 tok/s |
| | 4,068 | 2.34 s | 1,737 tok/s | 0.17 s | 217.0 tok/s |
| | 7,074 | 3.96 s | 1,786 tok/s | 0.26 s | 177.8 tok/s |
| | 24,062 | 14.57 s | 1,651 tok/s | 0.31 s | 183.4 tok/s |
| 1 GPU, TP=1, DFlash2 | 1,069 | 1.07 s | 1,001 tok/s | 0.24 s | 96.7 tok/s |
| | 4,066 | 3.97 s | 1,025 tok/s | 0.20 s | 132.0 tok/s |
| | 7,052 | 6.88 s | 1,026 tok/s | 0.44 s | 115.1 tok/s |
| 1 GPU, TP=1, no speculation | 1,067 | 1.04 s | 1,027 tok/s | 0.30 s | 32.8 tok/s |
| | 4,068 | 3.95 s | 1,029 tok/s | 0.14 s | 32.1 tok/s |
| | 7,052 | 6.94 s | 1,016 tok/s | 0.31 s | 31.7 tok/s |
| | 24,063 | 25.71 s | 936 tok/s | 0.20 s | 29.8 tok/s |

How each was launched, and what the card holds:

| | Launch | Configured context limit | Weights | KV pool |
| --- | --- | ---: | ---: | ---: |
| 2 GPUs | `bash start-turing.sh` | 262,136 | split over 2 | 317,842 tokens |
| 1 GPU, DFlash2 | `CUDA_VISIBLE_DEVICES=0 TP=1 KV_MEM=3000000000 DFLASH_MAX_LEN=8192 bash start-turing.sh` | 8,192 | 15.79 GiB with the drafter | 14,716 tokens |
| 1 GPU, no speculation | `CUDA_VISIBLE_DEVICES=0 TP=1 SPEC=off bash start-turing.sh` | 32,768 | 14.41 GiB | 64,815 tokens |

What this shows:

- **Speculation is the big single-GPU win:** DFlash2 decoded at 97 to 132 tok/s against 32 tok/s
  without it (about 2.9x at 1k and 4.1x at 4k) in these runs. The single-card
  DFlash2 profile was tested with an 8,192-token context limit and a reduced KV
  pool. This is a tested profile, not a measured maximum capacity; larger
  single-card DFlash2 contexts were not benchmarked here.
- **Two cards over one, with the same speculation:** 1.5x to 2.2x the decode rate (210 vs 97, 217 vs
  132, 178 vs 115 tok/s at about 1k, 4k and 7k tokens) and about 1.6x to 1.8x the prefill rate, and the
  configured context limit grows from 8,192 to 262,136 tokens. The performance
  measurements above cover prompts up to 24k, not a full-window stress test.
- **Without speculation one card slows with context:** 32.8 tok/s at 1k tokens, 29.8 at 24k, and a
  24k-token prompt has a cold TTFT of 25.71 s.
- Cached time to first token includes the first decode step, so the DFlash2 and no-speculation values
  do not isolate prefix-cache speed. Different generated text and draft
  acceptance also affect the one- versus two-card ratios. Improving
  unconstrained single-GPU short-prompt decoding remains follow-up work; no
  diagnostic result is substituted for the 96.7 tok/s measurement above.

## Optional FP8 KV

`KV_DTYPE=fp8 bash start-turing.sh` keeps fp16 queries and full graphs while
storing attention KV in FP8 E4M3. On 0.33.0, the KV pool rose from 317,842 to
536,108 tokens (68.7%). The 24k/800-token trial measured 191.3 tok/s median
decode, 14.57 s cold TTFT, and 2.16 s cached TTFT. FP16 KV
stays the default. Quantized KV can change outputs; this trial is not an accuracy eval.

## Optional CUDA GDN prefill

`patches/gdn-sm75-cuda-prefill.patch` adapts the recurrent CUDA kernel from
[weicj's Turing fork](https://github.com/weicj/vLLM-2080Ti-Definitive), commit
`93c9013e375b1ecbf2da88179d2a31175a0d5c99`. The Turing launcher enables it with `VLLM_GDN_SM75=1`.
It requires nvcc and a C++ compiler, and compiles once before graph capture.
The original Triton path remains available with `VLLM_GDN_SM75=0`.
The optional `VLLM_GDN_PREFILL_CHECKPOINTS=1` feature requires that Triton/FLA
path. Enabling it with SM75 CUDA prefill raises an error before checkpoint
allocation; keep it at its default of 0 with the standard Turing profile.

The 0.33.0 24k/800-token runs measured 14.20 s cold TTFT
with CUDA GDN versus 15.79 s with Triton GDN, a 10.1% reduction in this trial. Cached time
to first token was 0.31 versus 0.33 s. Median decode was 205.8 versus 209.8
tok/s. Each configuration had one cold run with a fresh identifier; this is
not an isolated kernel benchmark and does not establish a decode benefit.

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
Each completion is 256 tokens (86 for the short prompt).

On November 9, 2026, the 0.33.0 default server returned byte-identical
completions for all five prompts across three rounds, including cold and cached
prefill within one boot (`logs/repeat-033-default.json`). Cross-restart
repeatability has not been checked on 0.33.0.

This check establishes repeatability for the tested boot and prompts, not
accuracy against a bf16/fp32 reference or invariance to rebuilt compiler caches,
changed settings, or concurrent requests. Cross-restart and fresh-cache checks
were not repeated for this stack.

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
