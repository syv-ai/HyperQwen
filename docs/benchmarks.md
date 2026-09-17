# Benchmarks

[← back to the main README](../README.md)

Full tables per mode in [batch/README.md](../batch/README.md) and
[single-user/README.md](../single-user/README.md); quality in
[quality.md](quality.md). Reproduce any of it with
`bash bench/run_benchmarks.sh batch|single` against your own server.

## vs. ninfer-3090

[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) is a standalone C++/CUDA engine
that publishes cohort benchmarks for this model on this card. Theirs are 1,024-token
answers from 29-34-token prompts, greedy, MTP3, int8 KV, prefix reuse off, an
8,192-token context window, and **thinking on** at `reasoning_effort=medium`, so their
1,024 tokens include reasoning. Ours are 8 realistic chat prompts (English, Danish,
code), 1,024-token answers, model-default sampling, thinking off:

| Cohort | ninfer-3090 (MTP3) | this repo, batch | single-user, MTP | single-user, DFlash2 |
|---|---|---|---|---|
| C1 | 71.00 tok/s | 45.5 | 111.1 | **121.8** |
| C2 | 90.66 tok/s | 86.3 | 191.8 | **195.5** |
| C4 | 100.28 tok/s | 168.3 | 268.5 | **278.9** |
| C8 | 165.33 tok/s | 324.9 | **407.3** | 389.9† |
| C64 (128 in / 512 out) | not supported | **~1,035** | — | — |

† measured before `PREFIX_CACHE=1` became the single-user default. With
prefix caching on, cached prefixes (and, under `--mamba-cache-mode align`,
their recurrent-state pages) stay resident through the cohort ladder, so the
DFlash2 residency ceiling bites earlier and this cell reads ~324 tok/s with a
2-3 s TTFT on the current stack — independently measured at 321.8 in
[#40](https://github.com/syv-ai/HyperQwen/issues/40), which is what
prompted the re-measurement. C1–C4 read the same or slightly better than the
table. One card, many concurrent users: `SPEC=mtp` remains the right mode.

Decode rate, C × 1000 / mean TPOT. All four of our columns were re-measured together
on the current stack with `bench/run_benchmarks.sh`, keeping the second run after each
restart as the script advises; greedy instead of default sampling reads
131.2 / 214.6 / 285.7 / 405.5 for DFlash2. Run-to-run spread on the same server is
5-8%, so treat one-decimal differences between the three right-hand columns as noise —
C1 and C8 are where the modes genuinely separate.

Theirs is the **decode** column of their table; their end-to-end column reads
70.19 / 89.43 / 97.89 / 161.28, and an earlier version of this table quoted *those*
against our decode rate, which was not like-for-like. What still is not like-for-like,
in their favour and ours: their C1 is a single prompt in a single run with no error
bars, thinking is on for them and off for us, and they publish no power limit or driver
version — ours is an RTX 3090 pinned at 250 W. Peak VRAM is comparable (23.0 vs
22.1 GiB at C8). The gap is mostly vLLM's continuous batching plus the memory this
repo's requantization frees up.

## Quality

The whole stack is quantized, so the honest question is what it costs. Short
version: **IFBench 78.3** prompt-level strict vs 79.5 for the unquantized model
(one point), **perplexity 8.09** on 33k held-out tokens, **GSM8K 96.5%** (200
questions, greedy). Speculative decoding — MTP, DFlash2 and the lookup drafter —
is exact by construction and changes none of it; the int8-activation steps in
batch mode are the only knobs that trade accuracy for speed, and they cost
0.9-3.7% perplexity depending on how far you push them. Per-configuration
tables: [quality.md](quality.md).

## Results from other hardware

Moved to [reproductions/](reproductions/README.md).
