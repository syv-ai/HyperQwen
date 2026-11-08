# Turing: two RTX 2080 Ti 22 GB cards

Launch `bash start-turing.sh` for fp16 W4A16 target + DFlash2, TP=2,
FlashInfer attention, and a 262136-token maximum request length. In Docker use
`docker compose --profile turing up -d` ([docker.md](docker.md#rtx-20-series-turing-sm75)).

## Full CUDA graphs

`patches/flashinfer-sm75-spec-graphs.patch` enables an opt-in native FA2
paged-prefill graph path. The launcher sets
`VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=1`; use `=0` for the original behavior.
Both target verification and noncausal DFlash query attention capture full
graphs. Planning still occurs eagerly before replay; this does not claim
in-place draft metadata updates.

Measured November 8, 2026, single stream, 23,674 prompt tokens, 800 generated
tokens, prefill excluded from decode timing:

| Configuration | Decode | Cold prefill | Cold prefill rate | Cached prefill |
| --- | ---: | ---: | ---: | ---: |
| FlashInfer, graphs disabled | 89.7–92.2 tok/s | 15.6 s | 1,518 tok/s | 0.8 s |
| FlashInfer, full graphs | 201.1–203.8 tok/s | 15.7 s | 1,508 tok/s | 0.8 s |
| FlashInfer, full graphs, FP8 KV | 194.9–198.7 tok/s | 15.7 s | 1,508 tok/s | 7.0 s |
| Full graphs, fp16 KV, CUDA GDN | 185.3–199.1 tok/s | 14.2 s | 1,667 tok/s | 0.7 s |

Prefill rate is the 23,674-token prompt divided by its cold prefill time. Cached prefill only
computes the tokens after the cached prefix, so a tok/s figure for it would not be comparable.

Two greedy 512-token completions matched exactly before and after enabling
graphs. Their end-to-end rates rose from 83.3/61.0 to 175.9/129.1 tok/s.
Attention graph replay matched eager output at lengths 8, 31, 32, 33, 1024,
23700, 65, then 8, for causal full attention and noncausal window-2048
attention. Reproduce with `venv/bin/python scripts/check_sm75_attention_graphs.py`.

Concurrent requests can change greedy completions relative to running them
individually, including with graphs disabled. This setup does not promise
batch-invariant arithmetic.

## Rank balance and draft attention

With full graphs, both GPUs reported 76–77% utilization. A two-rank torch
profile recorded 201626 kernels on each rank, totaling 4.00 and 3.98 seconds
of GPU kernel time. Disabling graphs reproduced an approximately 90%/44%
utilization split. Host launch/synchronization overhead is the leading
explanation; the original imbalance is not evidence that only rank 0 runs
the drafter.

The drafter selects FlashInfer, with noncausal sliding-window attention.
Its full query graph is captured separately from the target verification graph.

## Optional FP8 KV

`KV_DTYPE=fp8 bash start-turing.sh` keeps fp16 queries and full graphs while
storing attention KV in FP8 E4M3. The measured pool rose from 317632 to 474838
tokens (49.5%). Decode measured 194.9/198.7 tok/s. Cold prefill stayed 15.7 s;
cached prefill regressed to 7.0 s because the mixed 880/64 block groups require
3520-token prefix alignment and disable fine-grained prefix hits. FP16 KV stays
the default. Quantized KV can change outputs; this trial is not an accuracy eval.

## Optional CUDA GDN prefill

`patches/gdn-sm75-cuda-prefill.patch` adapts the recurrent CUDA kernel from
[weicj's Turing fork](https://github.com/weicj/vLLM-2080Ti-Definitive), commit
`93c9013e375b1ecbf2da88179d2a31175a0d5c99`. The Turing launcher enables it with `VLLM_GDN_SM75=1`.
It requires nvcc and a C++ compiler, and compiles once before graph capture.
The original Triton path remains available with `VLLM_GDN_SM75=0`.

At the target TP=2 geometry (8 key heads, 24 value heads, dimension 128),
a 2048-token microbenchmark measured 11.45 ms for Triton versus 2.73 ms
for CUDA. The bounded comparison used the first GDN-state Triton autotune
configuration. Output/state relative L2 errors were 0.00051/0.00037. A separate
direct recurrence check passed. The integration copies initial state before
the kernel's in-place update and supports packed variable-length requests.

The end-to-end gain is smaller than the isolated kernel gain: cold 24k prefill
measured 14.2 s, down from 15.7 s. Cached prefill was 0.7 s. Decode measured
199.1 and 185.3 tok/s; the slower run had mean acceptance length 5.27 versus
6.02 on the faster run. Both original deterministic 512-token completions still
matched exactly. Packed two-request output/state checks also passed, including
an exact check that the input state was not modified.

A repeat cold run with a fresh prefix (23681 tokens) again measured 14.2 s
prefill (1,668 tok/s over 23,681 tokens) and 207.8 tok/s decode. This confirms the prefill reduction; decode
varies with the completion and draft acceptance.

## Verified interconnect

A live topology check after optimization showed `NV2` between GPU0 and GPU1,
with two active links per GPU (`nvidia-smi topo -m` and
`nvidia-smi nvlink --status`). The earlier handoff's PCIe-only description was
incorrect. The server selects CUSTOM/PYNCCL collectives, and the recorded
profile includes custom peer all-reduce kernels. The FlashInfer all-reduce
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
Round 1 is a cold prefill and later rounds hit the prefix cache, so cold-vs-cached drift
is covered. On the default setup (full graphs, FlashInfer, CUDA GDN, DFlash2, fp16 KV)
all five prompts were byte-identical across three rounds (Nov 8, 2026).

To compare two configurations, run with `--save a.json` on one, restart with the other
setting (for example `VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=0` or `VLLM_GDN_SM75=0`), and run
with `--compare a.json`. Concurrent requests are not exercised, and as noted above they can
change greedy output on any vLLM setup.

## Credits

- The per-request progress output (reading the prompt, prefill rate with cached tokens
  separated, answering, done) is modeled on the terminal output of Project Maya (https://github.com/mw00/project-maya, MIT, Peasant Smith and the Project Maya contributors).
- The CUDA GDN prefill kernel is adapted from weicj's
  [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) (Apache-2.0), see above.
- `scripts/tui_turing.py` imports the dashboard modules of the same project at run time (see above).
