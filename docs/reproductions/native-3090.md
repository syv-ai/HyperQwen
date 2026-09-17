# Reproduction: RTX 3090, native Python 3.14, Ubuntu 26.04

*August 2026. Native venv, no container. `verify.sh --no-server` clean, all fifteen patches
applied. Every run below discards a warmup pass, per the README's own instruction.*

[← back to Python 3.14](../python-314.md) · [← main README](../../README.md)

## Hardware and stack

| | |
|---|---|
| GPU | RTX 3090, 24 GiB, driver 610.43.02, 250 W |
| OS / Python | Ubuntu 26.04, Python **3.14.4** (system), `python3.14-dev` installed |
| vLLM | 0.27.1 + all fifteen `patches/`, torch 2.13.0+cu130, Triton 3.7.1 |
| model | `dbirks/Qwen3.8-27B-W4A16-AutoRound` + repo requantization + fast variant + DFlash2 drafter |
| KVarN | installed (`kvarn/install.sh`) |

## `bench/run_benchmarks.sh single`, greedy, vs the README

| cohort | ours e2e | ours decode | README |
|---|---:|---:|---:|
| C1 | **122.46** | 125.9 | 121.8 |
| C2 | 203.13 | 231.7 | 195.5 |
| C4 | 270.91 | 320.3 | 278.9 |
| C8 | 372.62 | 466.5 | 389.9 |

Within a few percent throughout. **The README's table reproduces on Python 3.14.**

## Context profiles as measured

| profile | KV dtype | max_model_len | pool | perplexity¹ |
|---|---|---:|---:|---:|
| `CTX=fast` | bf16 | 65,536 | 68,605-72,992² | 3.1174 |
| `CTX=long` | int8 per-token-head | 131,072 | 136,429 | 3.1100 |
| `CTX=huge` | KVarN 4/2-bit | 245,760 | 268,169 | 3.1107 |

¹ Teacher-forced, identical 12,000-character passage, echo+logprobs. Only the KV dtype
varies. **All three within 0.24%.**
² Varies with `SPEC`/`DFLASH_TOKENS`; both values observed.

## Aggregate tok/s, context × concurrency

`SPEC=dflash2 DFLASH_TOKENS=7 PREFIX_CACHE=1`, 256-token answers, prefix warmed before each
cohort so no row is measuring cache warmth instead of concurrency.

| profile | depth | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| fast | shallow | 137.3 | 239.5 | **321.3** | 300.6 |
| fast | deep (44k) | 66.8 | 99.7 | 19.2 | 33.4 |
| long | shallow | 121.1 | 230.0 | 304.8 | 311.0 |
| long | deep (44k) | 47.2 | 63.7 | 69.5 | 70.8 |
| huge | shallow | 134.1 | 220.9 | 157.6 | 232.2 |
| huge | deep (44k) | 43.6 | 50.1 | 53.3 | 52.8 |

**Scope note: every row above is speculation-ON, and at depth speculation is worth roughly
2×.** fermion's first reading of this had the opposite sign and was retracted: their
deep-decode probe counted SSE chunks, and that stack emits **one chunk per speculative step**,
so every spec-on number was low by the emitted/step factor (~3×). Token-true, at 72k: int4
with the wide-verify kernel **70-75 tok/s**, `long` **64-68**, against spec-off **35.5**.
Acceptance at depth is .286/.290 across the two KV dtypes: identical, so the decay is
model-and-depth rather than dtype. **These deep rows are therefore speculator and profile
jointly**, and there is no spec-off column here to separate them.

*The counting hazard is worth stating for anyone reproducing this:* under speculation, a
chunk count is a **step** count. Spec-off is immune at one token per step, which is exactly
why the wrong world looks self-consistent. Everything in this document takes its token counts
from `usage.completion_tokens` or the server's `generation_tokens_total` counter, never from
delta counting; `bench/conc_ladder.py` uses SSE deltas only for first/last timestamps.

`CTX=huge` costs roughly 10% shallow and 8% deep against `long`, for 1.9× the window. At
depth the larger pools hold up where `fast` does not.

## Prefix cache

Same 96,370-token document, three consecutive turns, `PREFIX_CACHE=1`:

```
turn 1:  232.9 s   ← pays the prefill
turn 2:    5.9 s
turn 3:    6.0 s
```

The README predicts *"5.9 s afterwards"* for a 112k document. Reproduced exactly.

## KV quality at 4/2 bits

Perplexity is a weak instrument here: it scores the model's fit to text it is *looking at*,
over a short window, which is not where a quantised cache fails. Needle-in-a-haystack on
`CTX=huge` (KVarN 4/2-bit), one distinctive fact planted at varying depth in a long
document, exact-match recall:

| prompt tokens | needle at | recall |
|---:|---:|---|
| 29,653 | 25% | HIT |
| 96,368 | 50% | HIT |
| 168,542 | 75% | HIT |
| **218,085** | 50% | **HIT** |

**Caveat, stated because the table would otherwise imply more than it shows:** the bf16
profile cannot exceed 65,536 tokens, so only the first rung has a like-for-like control.
This demonstrates that KVarN recalls correctly at 218k, not that it equals bf16 there.

## Prompt shape dominates, and it explains a reported number

A community report of ~208 tok/s did not match our first prose measurement (107.2), but the
author later published the prompt and the distinction: **200 was a peak, the average on that
prompt was ~160, and it was "very simple, code heavy output": "Please write a Tetris clone
that can run in a browser."** Prose, he noted, runs 100-120.

Running that exact prompt here, same configuration, three consecutive runs within 0.2%:

| prompt | ours | reported |
|---|---:|---:|
| *"Please write a Tetris clone that can run in a browser."* | **188.3** | ~160 avg, 200 peak |
| reflective prose essay | **113.1** | 100-120 |

Both land inside or above the reported band. **The discrepancy was never hardware or
configuration: it was that code generation and prose are different workloads.** `LOOKUP`
drafting predicts repetitive structure well (braces, indentation, boilerplate), so long runs
clear per verify step; prose branches more and accepts less.

**A tok/s figure for this stack is meaningless without the prompt that produced it.**

## Earlier, before the prompt was known

A community report of **~208 accepted tok/s** on a 3090 at `CTX=long` with a basic prompt
did not reproduce here: **107.2 tok/s median**, same posted configuration
(`SPEC=dflash2 DFLASH_TOKENS=7 PREFIX_CACHE=1 CTX=long`), four consecutive runs within 0.2%
of each other. Reproduction-shaped prompts do climb, consistent with `LOOKUP` drafting:

| task | tok/s |
|---|---:|
| free generation | 102.7 |
| quote a passage back verbatim | 125.5 |
| repeat a passage with a substitution | 156.6 |

The original report says "peaked at around 208", but a peak is not a median; the prompt was
not published. Recorded as unreproduced rather than disputed.

---

*Added August 2026, second pass. Everything below shares the stack above; where it does not,
the row says so.*

## Concurrency ladder: the 3090/4090 gap widens with load

`bench/conc_ladder.py --n 1,2,4,8 --out 256 --reps 2`, `CTX=long`, `MAX_SEQS=8`, shipped
`KV_MEM` pin. The 4090 column is fermion's run of the identical command under WSL2
([wsl2-4090.md](../wsl2-4090.md)).

| N | 3090 /stream | 4090 /stream | ratio | 3090 agg | 4090 agg | kv% 3090 | kv% 4090 | preempt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 129.1 | 149.7 | 1.16× | 128 | — | 19.7 | — | 0 / 0 |
| 2 | 91.8 | 119.8 | 1.31× | 236 | 278 | 38.3 | — | 0 / 0 |
| 4 | 46.6 | 79.0 | 1.70× | **316** | **517** | 73.5 | 71.1 | 0 / 0 |
| 8 | 30.3 | 57.2 | 1.89× | 230\* | 351\* | 98.2 | 98.2 | 2 / 2 |

\* Tail measurement: no instant at which all 8 streams were decoding.

Two things fall out, and the second is the one worth carrying away:

- **The concurrency ceiling is pool geometry, not the card.** N=8 goes tail-mode on *both*
  boxes at **kv 98.2% with 2 preemptions each**: same occupancy to the decimal, same failure
  mode, same count. A 4090 does not buy N=8 in long mode; it buys the same wall, reached
  faster. And **N=8 is strictly dominated by N=4 in aggregate on both cards** (230 vs 316
  here, 351 vs 517 there): not a tradeoff, a loss.
- **The 4090's per-stream advantage widens with N: +16% at N=1 → +89% at N=8.** This is
  consistent with, and extends, the README's own explanation of its +1.9% 4090 row
  ([#32](https://github.com/syv-ai/HyperQwen/issues/32)): *batch-1 decode is
  bandwidth-bound, the extra compute has nothing to bite on.* Raise N and the forward pass
  goes compute-bound, so sm_89 pulls away. **Sizing a card from a single-stream benchmark
  under-buys for concurrency by roughly 5× the error you think you are making.**

**What this ladder cannot see, and it is the binding constraint on mixed load.** Every stream
here carries a uniform ~4k prompt, so every prefill is small and comparable: the ladder
measures *decode* concurrency and is structurally blind to **prefill head-of-line blocking**.
fermion measured the case it misses: one 72.6k "whale" against 4k "minnows", where the deep
prefill monopolises the engine for its full ~65 s, admitted minnows ration to ~1 tok/s, and
the rest queue behind the entire prefill (TTFT 63-82 s). Cached whale, same scenario: 72.2 s
→ 13.5 s wall. **So "N=4 is the sweet spot, N=8 is strictly dominated" is scoped to uniform
short-prompt load**: under mixed depth the constraint is prefill admission, not decode
concurrency, so the ladder's advice does not transfer.

*Instrument caveats, both found the hard way:* `kv%` is documented as **peak** occupancy
sampled every 250 ms, and our N=4 run lasts 1.70× longer than fermion's on the same output
budget, so it gets 1.70× the draws at catching the same transient. **A peak-of-poll compared
across runs of different duration is structurally biased upward on the slower machine.**
Compare `preempt` and tail-mode onset instead: discrete, duration-invariant, and they matched
exactly. Separately, `conc_ladder.py:245` puts `int(time.time())` inside every prompt salt, so
token counts differ per box, per run, per rep by design.

## `VLLM_DFLASH2_LOOKUP`, and how to measure an env toggle at all

Every env-toggle A/B on this stack is a **cross-boot** comparison (the toggle needs a
restart), so it is only readable against a measured boot-to-boot floor. Ours, on this build
(`fa11c73`, pre-[#38](https://github.com/syv-ai/HyperQwen/issues/38)), warmed,
median of 2 passes per boot, env read back from `/proc/<pid>/environ` rather than trusted from
the shell:

| | boots | prose | code |
|---|---:|---|---|
| `LOOKUP=1` (default) | 3 | 142.85 · 142.89 · 142.78 → **142.84** | 185.53 · 185.41 · 185.50 → **185.48** |
| `LOOKUP=0` | 2 | 157.64 · 157.54 → **157.59** | 185.63 · 185.57 → **185.60** |
| boot floor (worst spread) | | **0.070%** | **0.065%** |
| effect | | **+10.33% (147× the floor), stands** | +0.06% (1× the floor), **unresolved** |

*Audited after the fact, because the harness timed whole requests including prefill:* re-ran
streamed with TTFT split out, and the effect is the same three ways: whole-window unstreamed
**+10.33%**, whole-window streamed **+9.92%**, **decode-only +9.96%**. The arms were clean for
the dull reason: ~200-token prompts against a working prefix cache leave no prefill worth
contaminating.

**On novel generation the lookup lane is net overhead on prose here, and invisible on code.**
Its value is context reproduction, exactly as the `DFLASH_TOKENS=15` documentation says.
Code is *inside our floor*, so we report it as unresolved rather than as zero: fermion
measures a real +7.7% there, and our silence is not evidence against it.

**The boot floor is a property of what state the boot inherited, not of the stack.** Ours is
0.07% across plain restarts that reuse the venv and `torch.compile` cache. Fermion measured
**8.7%** across a boot that followed an *image rebuild* with cold JIT caches; a
"regression" found that way was retracted once warm boots were compared. The README's
"first run reads low" warning operates at **boot** granularity; one warmup generation does not
warm a boot.

*Falsifier for everything in this section:* single RTX 3090 at 250 W, 453 MiB driver-reserved,
nothing else resident, native Linux with no container, `llama-chip` stopped for the duration
and verified healthy afterward. Prompts are two fixed strings (one prose, one code), not the
harness cohorts: comparable to each other, loosely comparable to anything else.


## int4 KV (#42): the 256k window serves on a 3090, and recall holds at 218k

`single-user/alternative.sh`, `MAX_LEN=256000`, `SPEC=off`, nothing else changed.

| | |
|---|---|
| GPU KV cache size | **266,520 tokens** |
| max_model_len served | 256,000 |
| VRAM | 22,612 MiB |
| attention block size the engine chose | **1696** |

Needle planted at depth and asked back (one distinctive literal, four positions):

| prompt tokens | depth | recall | wall |
|---:|---:|:---:|---:|
| 29,653 | 25% | **HIT** | 36 s |
| 96,368 | 50% | **HIT** | 215 s |
| 168,542 | 75% | **HIT** | 563 s |
| **218,085** | 50% | **HIT** | 893 s |

218,085 is the same depth at which KVarN 4/2-bit recalls exactly on this box, so **int4 is
not worse than the alternative already trusted here.**

**Read this narrowly.** A needle is a *recall* test, not a *quality* test: a distinctive
literal is close to the easiest thing a degraded KV can still retrieve. Four depths, n=1
each, one needle string, spec off. It licenses *"int4 does not lose the plot at 218k"* and
**not** *"int4 is lossless at 218k"*; the second wants a teacher-forced comparison against
bf16 at depth, which nobody has run.

*`SPEC=off` is not incidental:* fermion measured spec-on 25.2 against spec-off 35.5 tok/s at
72k depth on this profile: **speculation is a 1.4× slowdown there**, because acceptance
around .30 across 7 draft tokens cannot beat plain q=1 decode.

### The prefix-match unit is per-(model, kv-dtype, draft-length), not per-card

This 3090 reports `Setting attention block size to 1696 tokens` under int4, **identical to
fermion's 4090.** The card enters only through how much pool fits. The axis that does move it
is the draft length: at `DFLASH_TOKENS=3` the same stack reports 1616. So **a user changing
`DFLASH_TOKENS` silently changes the only valid `--prefix-match-unit` underneath themselves**,
and any value has to be read from the boot log rather than inherited from another machine or
another profile.
