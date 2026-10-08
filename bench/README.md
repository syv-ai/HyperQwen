# bench/

Tests, benchmarks, reproducers and tools for this stack. Each script's docstring or header
says how to run it. This page says which scripts are tests, what each one needs, and what
its exit code means. CI runs the ones that `.github/workflows/patch-integrity.yml` and the
`Dockerfile` name.

## Exit codes

| exit | meaning |
|---|---|
| 0 | The check ran and passed, or the script is a measurement. |
| 1 | The check ran and the verdict is FAIL. |
| 2 | The run was invalid, so there is no verdict: zero eligible cases, an overflowed tier, no request that succeeded. The script refuses to pass. |
| 3 | The test's own negative control failed, so no verdict from it can be trusted. |

- A measurement prints numbers and exits 0 whatever they are. A measurement that cannot run does not exit 0.
- A script that does not detect an invalid run crashes on it, for example with no server. A Python traceback exits 1, so exit 1 is not always a verdict.
- A designed skip (a GPU without the feature, a kernel without the path under test) prints `SKIP` or `skip:` and exits 0.
- A new script follows this table and gets a row below.

## Files

Needs: **CPU** is Python 3 and its standard library. **torch** is a CPU torch wheel. **image**
is the patched vLLM venv (the Docker image or `venv/`), with no GPU. **GPU** is the patched
venv on a CUDA card. **server** is a running server on `PORT` (default 18020), with the key
from `VLLM_API_KEY` or `api_key.txt`. **model** is the checkpoint under `models/`.

| file | kind | needs | exit | what it does |
|---|---|---|---|---|
| `act_calib.py` | measurement | GPU, model | 0 | Error that int8 activations add to each linear layer. Picks `INT8_LAYERS`. |
| `api_smoke.py` | smoke test, run by hand | server | 0 if all 12 features pass, 2 if no request reaches a server, else 1 | Request-level API features: logprobs, n, stop, seeds, structured output, penalties, streaming, thinking. |
| `bugb_sweep.py` | reproducer | server, model | 0, or 1 if a length is broken | The prompt lengths you give, against the broken-length defect (gotcha 37). |
| `conc_ladder.py` | measurement | server | 0 | Decode tok/s per stream at N=1..8, with passes, preemptions and KV occupancy. |
| `concurrent_collapse.py` | reproducer | server | 0, or 1 if a trial collapsed or a request errored | The "!!!!" collapse from #208. |
| `demo/` | demo | Node | n/a | The canvas renderer for the README video. See `demo/README.md`. |
| `demo_capture.py` | demo | server | 0 | Records one lane of the demo video: each token and the time it arrived. |
| `interleave_dose.py` | measurement | server | 0 | Prefix reuse against a dose of interleaved traffic, with the single-conversation control. |
| `labd_accept.py` | measurement | server | 0, or 1 if the server ignores `return_tokens_as_token_ids` | Teacher-forced tokens per step for lookup-augmented drafting. |
| `labd_bench.py` | measurement | server | 0 | Six long-context greedy tasks: decode speed and tokens per step. |
| `labd_soak.py` | test | server | 0 or 1 | The long verify block at batch > 1. |
| `make_long_corpus.py` | data tool | `~/bench/labd_corpus.txt`, vLLM source | 0, or 1 if an input is missing | Builds the long-context LABD corpus. |
| `mq3d_capacity_property.py` | test | CPU | 0, 1, 2 if no batch was eligible, 3 if `--mutate` did not break the property | The int4 3D scratch capacity property. `--mutate seq-rows` is its negative control and exits 0 when it breaks the property. |
| `mq3d_layer2_oracle.py` | test | GPU | 0 or 1 | Independent Layer-2 oracle for the int4 3D scratch patch. Writes `mq3d_layer2_verdicts.jsonl` beside itself unless `ORACLE_OUT` is set. |
| `mq3d_layer2_verdicts.jsonl` | data | | | The oracle's records from the 4090 run. |
| `mq3d_layer2_verdicts-3090.jsonl` | data | | | The oracle's records from the 3090 rerun, with the shipped patch's sha. |
| `mq3d_scratch_pool_test.py` | test | image | 0 or 1 | The int4 3D scratch pool, on CPU tensors. |
| `needle_reuse.py` | test | server | 0 or 1 | The answer lives inside the reused prefix. |
| `needle_test.py` | probe | server | 0 if RETRIEVED, 1 if MISSED | A passcode at a depth in a long cold prompt. |
| `prefill_ab.sh` | measurement | GPU, model | 0, or 1 if it cannot start its server | Boots a server with an env set and measures the prefill rows. |
| `prefix_alternation.py` | reproducer | server | 0, 1 if the arm reproduces the defect, or 2 if it checked no turn | Prefix reuse under two alternating conversations. |
| `prompts_real.jsonl` | data | | | The realistic prompts that `run_benchmarks.sh`, `real_rep.sh`, `prefill_ab.sh` and `act_calib.py` read. |
| `quality_battery.py` | quality measurement | server, `bench/quality-data/` | 0 | Perplexity on three corpora and GSM8K accuracy. |
| `real_rep.sh` | measurement | server, model | the last repeat's exit | The C1 row of `run_benchmarks.sh`, repeated. |
| `replay_offload_serve.py` | reproducer | server with an offload tier | 0 if SERVED, 1 if NOT-SERVED, 2 if INVALID-TIER-OVERFLOW | Whether the CPU offload tier serves a stored request back. |
| `residue_sweep.py` | reproducer | server, model | 0, or 1 if a residue is broken | One prompt for each length residue mod 128. |
| `run_benchmarks.sh` | measurement | server, model | 0, or 1 if there is no server | The README numbers. |
| `seat_ttft.py` | measurement | server | 0, or 2 if every request failed | Cold and warm TTFT at N=1. |
| `spec_attn_ctx_scan.py` | measurement | GPU | 0 | Verify attention cost against context length. |
| `test_bench_sse_keepalive.py` | test | image | 0 or 1 | `vllm bench serve` against a server that sends an SSE keep-alive. |
| `test_kvarn_recycled_pages.py` | test | image | 0 or 1 | KVarN's recycled-page drop (#208). |
| `test_lookup_kernels.py` | kernel test | GPU | 0 or 1 | The LABD kernels against a Python reference. |
| `test_marlin_int8_asym.py` | kernel test | GPU | 0 or 1 | Marlin W4A8-INT8 with zero points against W4A16. |
| `test_model_verification.py` | test | CPU | 0 or 1 | Model verification and selection. Cuts a heredoc out of `verify.sh`. |
| `test_no_key_bind.sh` | test | CPU, bash, tar | 0 or 1. Skips under `/.dockerenv`. | The `--host` each launcher binds, with and without a key (#204). Cuts a block out of `verify.sh` and evaluates it. |
| `test_prefill_attn_bigpool.py` | kernel test | GPU | 0 or 1 | int32 overflow in the int8 prefill kernel on a large pool (#86). |
| `test_prepare_crash.py` | test | torch and the prepare stack | 0 or 1 | Crash injection for the model preparation scripts (#195). Cuts a heredoc out of `docker/prepare.sh`. |
| `test_prepare_state.py` | test | CPU | 0 or 1 | `state()` in `docker/prepare.sh` on torn files (#195). Cuts a heredoc out of `docker/prepare.sh`. |
| `test_spec_decode_attn.py` | kernel test | GPU | 0 or 1 | The split-KV spec-decode attention against a reference. The timing rows are informational. |
| `test_spec_decode_bigpool.py` | kernel test | GPU | 0 or 1. The fp8 part skips below sm89 or without the fp8 path. | int32 overflow in `_spec_attn_partial` on a large pool (#86). |
| `test_spec_decode_fp8.py` | kernel test | GPU, sm89+ | 0 or 1. Skips below sm89 or without the fp8 path. | The fp8 path of the split-KV kernel. The timing rows are informational. |
| `tune_gdn.py` | measurement | GPU | 0 | The Gated DeltaNet decode kernel over block sizes and warp counts. |
| `verbatim.py` | library and test | CPU | 0 or 1 | How much of an answer copies its source. The two sweeps use it. Run alone, it tests itself. |
| `warmup.sh` | lifecycle | server | 0 or 1 | Warms the serving path after boot. `single-user/qwen-server.sh` runs it when `WARMUP=1`. |

One test outside `bench/` gates CI too:

| file | kind | needs | exit | what it does |
|---|---|---|---|---|
| `kvarn/tests/test_kvarn_fp16_dequant_torch.py` | test | torch | 0 or 1 | Numerics of `KVARN_FP16_DEQUANT`. |
