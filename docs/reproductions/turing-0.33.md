# Turing reproduction: vLLM 0.33.0

Measured November 9, 2026 on two modified 22 GB RTX 2080 Ti cards with NVLink
and the host `turing-unlock` module active. Native Python 3.14.8 venv;
vLLM 0.33.0, torch 2.14.0, Triton 3.8.0, FlashInfer 0.7.2.post1, and the six
Turing patches from this branch. `bash verify.sh --install` passed. The server
used `Qwen3.8-27B-W4A16-AutoRound`, and DFlash2 runs used
`Qwen3.8-27B-DFlash2-W4A16`. See [Turing setup](../turing.md).

The local venv used `pyarrow==26.0.1` in place of the repository's
`pyarrow==20` pin because version 20 has no Python 3.14 wheel. No model or
benchmark code was changed for this substitution. The measurements are native,
not Docker measurements. The full JSON output is in local ignored `logs/` files;
the per-run decode rates below preserve the raw measurements in the repository.

## Method

Each configuration was started separately and allowed to finish loading and
graph capture before requests. `scripts/bench_turing.py` sent one request at a
time, using greedy decoding, `ignore_eos`, and a deterministic synthetic
document. Each length had three runs: first cold, next two with the same prompt
and prefix caching. Cold and cached figures are time to first streamed token;
decode excludes the first token. Decode rates count `usage.completion_tokens`,
not stream chunks. A warmup request preceded each configuration's benchmark.
The prompts differ slightly in length and text between configurations, and
speculative acceptance depends on generated text. These numbers describe these
completions, not general throughput or an isolated patch speedup.

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

## 24k prompt, 800 generated tokens

Each row used `venv/bin/python scripts/bench_turing.py --lengths 24000 --gen 800 --runs 3 --json logs/bench-033-NAME.json`, with the named server launch option.

| Launch option / JSON name | Prompt tokens | Cold TTFT | Cached TTFT | Decode runs (tok/s) | Median |
| --- | ---: | ---: | ---: | --- | ---: |
| Default / `graphs-on` | 24,063 | 14.20 s | 0.31 s | 205.3 / 205.8 / 206.6 | 205.8 |
| `VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=0` / `graphs-off` | 24,063 | 14.32 s | 0.33 s | 104.6 / 103.8 / 104.6 | 104.6 |
| `KV_DTYPE=fp8` / `fp8` | 24,064 | 14.57 s | 2.16 s | 209.7 / 191.3 / 191.2 | 191.3 |
| `VLLM_GDN_SM75=0` / `gdn-off` | 24,061 | 15.79 s | 0.33 s | 210.2 / 209.8 / 209.3 | 209.8 |

All four used TP2 DFlash2; other launch options remained at default. The FP8
KV pool held 536,108 tokens, and the Triton GDN pool held 280,424. Graph-on
versus graph-off median decode was 1.97×. CUDA GDN shortened cold TTFT by
1.59 s (10.1%) relative to Triton GDN in this trial. These are comparisons of
separate server boots and their completions.

## Functional checks

`venv/bin/python scripts/check_sm75_attention_graphs.py` passed causal and
noncausal graph replay against eager attention at lengths 8, 31, 32, 33, 1024,
23700, 65, and 8, with zero maximum error in this check.

`venv/bin/python scripts/check_greedy_repeatability.py --rounds 3 --max-tokens 256 --save logs/repeat-033-default.json`
passed: five of five prompts gave byte-identical outputs across three rounds
within the default server boot. This does not test cross-restart repeatability.
