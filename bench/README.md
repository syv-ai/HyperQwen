# bench/

Tests, benchmarks, reproducers and tools for this stack. Each script's docstring or header
says how to run it. This page says which scripts are tests, what each one needs, and what
its exit code means. CI runs the ones that `.github/workflows/patch-integrity.yml` and the
`Dockerfile` name.

Under Docker Desktop (WSL2) the VM's clock can run fast against the host's, by 8.3% on one
RTX 3090, and can change rate partway through a run. Durations timed there read long by up to
that much, so compare A/B arms that ran interleaved in one sitting rather than absolute numbers.

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
venv on a CUDA card. **server** is a running server at `VLLM_API` (a trailing `/v1` is fine), else on
`PORT` (default 18020), with the key from `OPENAI_API_KEY`, `VLLM_API_KEY` or `api_key.txt`, in that
order (`bench/harness.py`). The model is `VLLM_MODEL` if set, else the server's own: the Python scripts name
none, and the `.sh` scripts ask `/v1/models`. The `.sh` measurements take `PORT` and ignore `VLLM_API`;
`run_benchmarks.sh` and `warmup.sh` also take `HOST`. **model** is the checkpoint under `models/`.

| file | kind | needs | exit | what it does |
|---|---|---|---|---|
| `act_calib.py` | measurement | GPU, model | 0 | Error that int8 activations add to each linear layer. Picks `INT8_LAYERS`. |
| `test_dflash_context_capacity.py` | test | image, no GPU | 0 or 1 | DFlash constructor rejects oversized target contexts and reserves its fixed lookahead without rewriting draft configuration. |
| `test_doctor.py` | test | CPU, bash | 0 or 1 | Doctor health failures, configuration-pipeline failures and redacted output. |
| `api_smoke.py` | smoke test, run by hand | server | 0 if all 12 features pass, 2 if no request reaches a server, else 1 | Request-level API features: logprobs, n, stop, seeds, structured output, penalties, streaming, thinking. |
| `bugb_sweep.py` | reproducer | server, model | 0, or 1 if a length is broken | The prompt lengths you give, against the broken-length defect (gotcha 37). |
| `code_corpus.py` | library | CPU | n/a | Source selection for quality, calibration and long LABD corpora; see `CODE_CORPUS_DIR` below. |
| `conc_ladder.py` | measurement | server | 0 | Decode tok/s per stream at N=1..8, with passes, preemptions and KV occupancy. `--salt S` fixes the prompts across boots for an A/B. |
| `concurrent_collapse.py` | reproducer | server | 0, or 1 if a trial collapsed or a request errored | The "!!!!" collapse from #208. |
| `demo/` | demo | Node | n/a | The canvas renderer for the README video. See `demo/README.md`. |
| `demo_capture.py` | demo | server, frozen LABD corpus | 0, 2 if the corpus is missing, blank or too short | Records one lane of the demo video: each token and the time it arrived. |
| `test_demo_corpus.py` | test | CPU | 0 or 1 | The demo rejects invalid corpora before requests and sends the frozen document to the copy task. |
| `harness.py` | library | CPU | n/a | How the scripts reach the server: the key, the URL, the model, the request, the stream and `/metrics`. For the bash scripts, `python3 bench/harness.py spec` prints the drafts and accepted-token counters and `model` the served name. |
| `interleave_dose.py` | measurement | server | 0 | Prefix reuse against a dose of interleaved traffic, with the single-conversation control. |
| `labd_accept.py` | measurement | server | 0, or 1 if the server ignores `return_tokens_as_token_ids` | Teacher-forced tokens per step for lookup-augmented drafting. |
| `labd_bench.py` | measurement | server | 0 | Six long-context greedy tasks: decode speed and tokens per step. |
| `labd_corpus.py` | library | CPU | n/a | Reuses a nonblank frozen corpus without rewriting it, or builds this checkout's docs on first run. A missing `labd_corpus_long.txt` must be built with `make_long_corpus.py`. A blank frozen head is refused even when a long corpus has a nonblank source tail. |
| `labd_soak.py` | test | server | 0 or 1 | The long verify block at batch > 1. |
| `make_long_corpus.py` | data tool | `~/bench/labd_corpus.txt`, vLLM source | 0, or 1 if an input is missing or blank | Builds the long-context LABD corpus without extending a blank frozen head. |
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
| `quality_battery.py` | quality measurement | server, `bench/quality-data/`, code source for PPL | 0, 1 if an input is missing, 2 for invalid arguments or an empty GSM8K dataset | Perplexity on three corpora and GSM8K accuracy; reports the actual number of evaluated questions. PPL requires a code source selected by `CODE_CORPUS_DIR` / checkout venv / active-install fallback; `--gsm-only` does not need vLLM locally. |
| `real_rep.sh` | measurement | server, model | the last repeat's exit | The C1 row of `run_benchmarks.sh`, repeated. |
| `replay_offload_serve.py` | reproducer | server with an offload tier | 0 if SERVED, 1 if NOT-SERVED, 2 if invalid (missing corpus/metrics, counter reset, tier overflow, request failure or X not evicted) | Whether a fast replay loads bytes from the CPU offload tier. Thinking is off; streams without content have no TTFT verdict. A fast replay without loaded bytes is `INVALID-NOT-EVICTED`: adjust the pool/evictors before drawing a conclusion. |
| `residue_sweep.py` | reproducer | server, model | 0, or 1 if a residue is broken | One prompt for each length residue mod 128. |
| `run_benchmarks.sh` | measurement | server, model | 0, or 1 if there is no server | The README numbers. |
| `seat_ttft.py` | measurement | server | 0, or 2 if every request failed | Cold and warm TTFT at N=1. |
| `spec_attn_ctx_scan.py` | measurement | GPU | 0 | Verify attention cost against context length. |
| `t4_battery.py` | driver, run by hand | GPU, server (it boots its own), model, `bench/quality-data/`, both LABD corpora | 0 if every bar passes (also with one arm, where the cross-arm bars are n/a), 1 if a bar fails, 2 if a bar had no data | The T4 battery (#153): `SPEC=dflash2 CTX=long` int8 + Triton against `FA2_FP8KV=1` (`--arms int8,fp8fa2`, the default) and the `PTH_FA=1` int8 kernel arm (`--arms int8,pth`: the same bars against int8, soak on pth), with quality, residue (cold boot), decode at ~60k and ~90k (opt-in ~149k) and a soak, on each boot's server identity (any PID or EngineCore change, or a death line in the log, fails the cell). `--help` lists the bars; `--dry-run` prints the plan and needs no GPU. |
| `test_bench_sse_keepalive.py` | test | image | 0 or 1 | `vllm bench serve` against a server that sends an SSE keep-alive. |
| `test_conc_ladder_salt.py` | test | CPU | 0 or 1 | `conc_ladder.py --salt` against a stub server: the same prompts on a second run, and the prefix-cache hit reported. |
| `test_chat_hardening.py` | test | CPU | 0 or 1 | Explicit/default template scope, path deduplication and help without writes. |
| `test_gdn_ba_lora.py` | test | image | 0 or 1 | CPU/source check of the GDN projection gate and LoRA dispatch, plain padding numerics and fake output shapes. |
| `test_fa2_slab_plan.py` | test | CPU | 0 or 1 | Bounded K/V slab coverage, causal alignment and fixed workspace size. |
| `test_fa2_slab_dispatch.py` | test | CPU | 0 or 1 | Opt-in dispatch, workspace sharing and fallback for decode, capture and unsupported inputs. |
| `test_fa2_slab_gpu.py` | kernel test | sm86 GPU, installed PTH/vLLM/FA2 | 0 or 1 | Explicit `--allow-gpu`: slab attention against full FA2 and FP32, including paged layouts and stable LSE merging; `--benchmark` adds timings. See [the adapter notes](../pthfa/LOCAL_CHANGES.md). |
| `test_harness.py` | test | CPU, bash | 0 or 1 | `harness.py` against a stub server, including refusal of SSE streams closed before `[DONE]`. Its key chain is checked against `resolve_api_key.sh`. |
| `test_kv_cache_seat_budget.py` | test | image | 0 or 1 | The KV cache seat budget line (`kv-cache-seat-budget.patch`) against five measured boots, with vLLM's own spec classes. |
| `test_kvarn_recycled_pages.py` | test | image | 0 or 1 | KVarN's recycled-page drop (#208). |
| `test_labd_bench_clock.py` | test | CPU | 0 or 1 | `labd_bench.py` against a stub server, with the wall clock stepped back: TTFT and decode rate as the stub served them. |
| `test_labd_corpus.py` | test | CPU | 0 or 1 | LABD corpus discovery/reuse, blank frozen heads, and rejection in benchmarks, soak and sweeps before requests or tokenizer loading. |
| `test_long_corpus.py` | test | CPU | 0 or 1 | Explicit/checkout/fallback source selection without importing vLLM, missing-source/cohort refusals and byte-preserved frozen heads. |
| `test_lookup_kernels.py` | kernel test | GPU | 0 or 1 | The LABD kernels against a Python reference. |
| `test_marlin_int8_asym.py` | kernel test | GPU | 0 or 1 | Marlin W4A8-INT8 with zero points against W4A16. |
| `test_model_verification.py` | test | CPU | 0 or 1 | Model verification and selection. Cuts a heredoc out of `verify.sh`. |
| `test_alt_wsl_gpu_util.sh` | test | CPU, bash, tar | 0 or 1 | `single-user/alternative.sh` defaults `GPU_UTIL` to 0.95 natively and 0.93 under WSL2 (vLLM sees 22.75 GiB of a 3090 there, #401); an explicit value wins in both. Faked with `WSL_DISTRO_NAME`, stub `vllm`, `PRINT_ARGV=1`. |
| `test_no_key_bind.sh` | test | CPU, bash, tar | 0 or 1. Skips under `/.dockerenv`. | The `--host` each launcher binds, with and without a key (#204). Cuts a block out of `verify.sh` and evaluates it. |
| `test_offload_mamba_swa_boundaries.py` | test | image, CPU only | 0 or 1 | Actual native store masks and CPU-manager lookup retain SWA history at Mamba replay boundaries under sparse retention. |
| `test_pin_bump.py` | test | CPU, git | 0 or 1 | Missing-pattern/co-pin rejection without partial edits; every target updated on valid bumps, future repo pins, Python-only FlashInfer pin and dry runs. |
| `test_prefill_attn_bigpool.py` | kernel test | GPU | 0 or 1 | int32 overflow in the int8 prefill kernel on a large pool (#86). |
| `test_prepare_crash.py` | test | torch and the prepare stack | 0 or 1 | Crash injection for the model preparation scripts (#195). Cuts a heredoc out of `docker/prepare.sh`. |
| `test_prepare_state.py` | test | CPU | 0 or 1 | `state()` in `docker/prepare.sh` on torn files (#195). Cuts a heredoc out of `docker/prepare.sh`. |
| `test_port_paths.py` | test | CPU, bash, git | 0 or 1 | Export, port drift and triage against a real checkout with spaces in its path. |
| `test_quant_schema.py` | test | torch and the prepare stack | 0 or 1 | What `prepare/quant_schema.py` writes for every in-place quant script: the packed tensors and their scale dtype, the killed-run check, the index entries, the config group. |
| `test_quality_battery.py` | test | CPU | 0 or 1 | Quality CLI help/validation without optional dependencies, GSM8K sample accounting and empty-data refusal. |
| `test_quant_scale_views.py` | test | image (CPU), GPU optional | 0 or 1 | Scale views preserve shared hybrid pages and existing scales for INT4/INT8/FP8, HND/NHD and storage offsets; PTH_FA inheritance; `--device cuda:0` also checks the actual INT8 writer. |
| `test_replay_offload.py` | test | CPU | 0 or 1 | Replay verdicts on CPU stubs: eviction, TTFT and byte deltas, fresh/legacy counters, exact metric families and interrupted requests. |
| `test_generation_state.py` | test | CPU | 0 or 1 | Resume after torn JSON/UTF-8 tails, record-shape refusal, process exclusion before engine startup, and lock release on exceptions/process exit. |
| `test_spec_decode_attn.py` | kernel test | GPU | 0 or 1 | The split-KV spec-decode attention against a reference. The timing rows are informational. |
| `test_spec_decode_bigpool.py` | kernel test | GPU | 0 or 1. The fp8 part skips below sm89 or without the fp8 path. | int32 overflow in `_spec_attn_partial` on a large pool (#86). |
| `test_spec_decode_fp8.py` | kernel test | GPU, sm89+ | 0 or 1. Skips below sm89 or without the fp8 path. | The fp8 path of the split-KV kernel. The timing rows are informational. |
| `test_t4_battery.py` | test | CPU, bash | 0 or 1 | `t4_battery.py`'s dry-run plan, parsers, restart gate (on a fake `/proc`), verdict, and one run against a stub launcher. |
| `test_vision_knobs.py` | test | CPU, bash | 0 or 1 | All three launchers' Vision limits, CPU tower-offload budget and boolean controls against a stub vLLM. |
| `test_vision_smoke.py` | test | CPU, bash | 0 or 1 | Pixel-dependent Vision verification against stub HTTP servers, including a server that only answers text. |
| `tune_gdn.py` | measurement | GPU | 0 | The Gated DeltaNet decode kernel over block sizes and warp counts. |
| `verbatim.py` | library and test | CPU | 0 or 1 | How much of an answer copies its source. The two sweeps use it. Run alone, it tests itself. |
| `vision_smoke.py` | smoke test | server with `VISION=1` | 0 if both color orders are read, 1 if a response fails, 2 if no server answers | Two different orders of three large color panels, the same question; expected color names/order appear only in the pixels. Run by `VISION=1 bash verify.sh`; a setting in the server's environment alone does not enable this check. |
| `warmup.sh` | lifecycle | server | 0 or 1 | Warms the serving path after boot. `single-user/qwen-server.sh` runs it when `WARMUP=1`. |

For comparable code scores, set `CODE_CORPUS_DIR` to the same fixed **vllm package
directory** for every arm, for example `CODE_CORPUS_DIR=/path/to/fixed/vllm`.
`quality_battery.py`, `make_long_corpus.py` and `act_calib.py` use that tree. Without
an explicit selection, they prefer `<repo>/venv/lib/python3.12/site-packages/vllm`
to preserve existing inputs, and discover the active Python's vLLM only when that
directory is absent. The chosen directory and file/window counts are printed.
An invalid explicit directory never falls back; missing source or eligible PPL
windows is an error with instructions to set `CODE_CORPUS_DIR`. Use one immutable
source tree for comparisons across vLLM versions. `--gsm-only` skips this selection.

One test outside `bench/` gates CI too:

| file | kind | needs | exit | what it does |
|---|---|---|---|---|
| `kvarn/tests/test_kvarn_fp16_dequant_torch.py` | test | torch | 0 or 1 | Numerics of `KVARN_FP16_DEQUANT`. |

## Async Mamba KV sizing
## Continuous usage while tool parsing buffers output

After installing the pinned vLLM and applying `patches/series`, run:

```bash
CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_async_mamba_kv_sizing.py
CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_kv_cache_seat_budget.py
```

These CPU checks cover async state reservations, speculative/checkpoint blocks,
grouped admission, null-block capacity, budget overrides and automatic fitting.
A replay fixture checks all 99 allocator snapshots from a 163K-token prefill,
including its 219-block peak. The residency-log fixtures retain the older boot
metrics while checking the corrected conservative async estimates. No model
or GPU is needed. The Docker build runs both suites after applying the patches.
CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_continuous_stream_usage.py
This uses Qwen's parser and the chat streaming generator with synthetic engine
outputs. It checks continuous counts during a buffered array argument, hidden
reasoning, final-only usage and no-usage streams. No model or GPU is needed.
The Docker build runs the same regression after applying the patches.
