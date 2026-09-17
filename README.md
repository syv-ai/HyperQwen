<div align="center">

<h1>HyperQwen</h1>

<p><b>Large Qwen models, served fast on the GPUs people actually own.</b></p>

<p>
<a href="https://github.com/syv-ai/HyperQwen/actions/workflows/docker-image.yml"><img alt="docker image" src="https://github.com/syv-ai/HyperQwen/actions/workflows/docker-image.yml/badge.svg"></a>
<a href="https://github.com/syv-ai/HyperQwen/actions/workflows/patch-integrity.yml"><img alt="patch integrity" src="https://github.com/syv-ai/HyperQwen/actions/workflows/patch-integrity.yml/badge.svg"></a>
<a href="https://github.com/syv-ai/HyperQwen/pkgs/container/hyperqwen"><img alt="ghcr.io" src="https://img.shields.io/badge/ghcr.io-syv--ai%2Fhyperqwen-2496ED?logo=docker&logoColor=white"></a>
<a href="https://github.com/vllm-project/vllm"><img alt="vLLM 0.28.0" src="https://img.shields.io/badge/vLLM-0.28.0-5C3EE8"></a>
<a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/github/license/syv-ai/HyperQwen"></a>
<a href="https://github.com/syv-ai/HyperQwen/stargazers"><img alt="stars" src="https://img.shields.io/github/stars/syv-ai/HyperQwen?style=flat"></a>
</p>

![Stock vLLM against this repo, same card, same prompts](docs/media/demo.gif)

</div>

**Today:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a single 24 GB
consumer GPU with vLLM — 150k token context and an OpenAI-compatible API with key
auth, in two ready-made modes. On an RTX 3090 at 250 W: 127 tok/s single-stream
in single-user mode, ~1,035 tok/s aggregate decode at 64 concurrent in batch mode,
and 381 tok/s when the answer quotes its own prompt.

**Where it is going:** more Qwen models, more classes of card. What this repo
really is, under the serving setup, is a patch series against a pinned vLLM plus
a model-preparation pipeline — requantized heads, a calibrated draft vocabulary,
speculation that drafts out of the prompt. Most of that is not specific to one
checkpoint or one card; it is just that one checkpoint and one card are what has
been measured to death. See [Roadmap](#roadmap-more-qwen-models-more-cards) for
what is and is not portable yet, and
[Help us test more hardware](#help-us-test-more-hardware) if you have credits to
spare on a GPU we have never touched.

## Quick start

```bash
git clone https://github.com/syv-ai/HyperQwen && cd HyperQwen
cp .env.example .env                      # PowerShell: Copy-Item .env.example .env

docker compose --profile single up -d     # one or a few people chatting
docker compose --profile batch  up -d     # API backend, many concurrent requests
```

First start pulls the image (9.5 GB) and requantizes the model (~20 GB, once,
into `./models`), then serves on `:18020`. One GPU runs one mode at a time.

- **Before exposing it** — the server binds `0.0.0.0` with no auth:
  `echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env`
- **Docker Desktop on WSL2** — keep `VLLM_WSL2_ENABLE_PIN_MEMORY=1` in `.env`, or
  the V2 runner aborts with `RuntimeError: UVA is not available`
- **No compose, or no Docker at all** —
  [docs/docker.md](docs/docker.md#plain-docker-no-compose) ·
  [docs/install.md](docs/install.md)

## Which setup do I want?

Three questions, and the answers are the whole of your `.env`.

```mermaid
flowchart TD
  Q1{"Who is sending<br/>the requests?"}
  Q1 -->|"an API backend, a pipeline,<br/>8+ at once"| A["<b>A — batch</b>"]
  Q1 -->|"one or a few<br/>people chatting"| Q2{"Longest prompt<br/>you will send?"}
  Q2 -->|"under 64k tokens"| Q3{"Do the answers quote<br/>the prompt back?"}
  Q2 -->|"up to 150k"| D["<b>D — long context</b>"]
  Q2 -->|"200k and beyond"| E["<b>E — huge context</b>"]
  Q3 -->|"no — normal chat"| B["<b>B — single, default</b>"]
  Q3 -->|"yes — code edits, RAG,<br/>rewrites, translation"| C["<b>C — reproduction</b>"]
```

Starting from the `.env` you copied in Quick start, which already ships **B**:

| | change in `.env` | start with | what you get |
|---|---|---|---|
| **A** | nothing | `--profile batch` | ~1,035 tok/s aggregate at 64 concurrent |
| **B** | nothing | `--profile single` | 127 tok/s single stream, 64k context |
| **C** | add `DFLASH_TOKENS=15` | `--profile single` | 381 tok/s while quoting, at 4 slots and 56k |
| **D** | `SPEC=mtp` and `CTX=long` | `--profile single` | 150k context, ~95-100 tok/s |
| **E** | add `CTX=huge` | `--profile single` | 240k context, 67 tok/s mixed and 164 while quoting |

Torn between B and C: B is the safe default, and C only pays off when the output
really does repeat the input — it trades request slots and context for it. D drops
back to MTP speculation on purpose, because DFlash2 past 64k is worth it only for
reproduction and loses to `SPEC=mtp CTX=long` about 2:1 on everything else; at E
the KVarN cache buys the context instead, so DFlash2 stays on. A needs no edit at
all — batch ignores `SPEC` and says so on startup. Every knob, and what each costs:
[single-user/README.md](single-user/README.md#if-you-are-the-only-user-do-this) ·
[batch/README.md](batch/README.md).

### The numbers behind that

| | **batch** → [batch/](batch/) | **single** → [single-user/](single-user/) |
|---|---|---|
| **best for** | API backends, pipelines, many concurrent requests | one or a few people chatting |
| **64 concurrent** (128 in / 512 out) | **~1,035 tok/s** decode, 948 e2e — ~1,222 with every layer int8 | n/a — 8 slots |
| **single stream** (C1) | 46 tok/s | **127 tok/s** — 121 on the older MTP path |
| **quoting its own prompt** | 46 tok/s | **381 tok/s** at 25k context |
| **prefill**, 1k in | ~1,810 tok/s | ~1,440 — **~1,850-1,940** with `INT8_ACT=int8` |
| **how** | 16-bit recurrent state, int8 tensor-core GEMMs | 7 drafts proposed per pass, 15 tokens verified per step off the context |

Speculation wins below ~8 concurrent users, plain batching above — earlier on long
sessions, where a speculating request reserves recurrent-state pages the pool has
few of ([measurement](docs/long-context.md)). Both modes share one install.
[Full prefill matrix](batch/README.md#prefill) ·
[how each number was won](docs/optimizations.md).

## Roadmap: more Qwen models, more cards

The name changed because the scope did. This started as "Qwen3.8-27B on one RTX
3090" and the README still reflects that, because that is the pair which has been
measured properly. The plan is to widen both axes. Being specific about what
transfers, since "supports N models" is easy to claim and expensive to be wrong
about:

**Portable already — nothing model- or card-specific in it.** The vLLM patch
series (`patches/`, 38 files, one line each in [PATCHES.md](PATCHES.md)), the
KVarN long-context backend, int8 Marlin GEMM layer selection, the SSE keep-alive,
the engine stall sentinel. These are vLLM fixes that happen to have been written
here.

**Model-specific, and this is the real work.** The int8-QK prefill attention
kernel is gated on this checkpoint's exact geometry — `num_heads == 24`,
`num_kv_heads == 4`, `head_size == 256` — and falls back to FA2 for anything
else, so a new model gets correctness for free and none of the prefill win until
the gate is generalized. The 40k-token draft vocabulary is calibrated on this
model's own output distribution. The DFlash2 drafter is a per-model checkpoint.
`prepare/` assumes a head layout that only just learned to handle heads in
different shards ([#120](https://github.com/syv-ai/HyperQwen/issues/120)).

**Card-specific.** The Marlin tune tables are measured on sm86. sm120 (RTX 5090)
reproduces ([#35](https://github.com/syv-ai/HyperQwen/issues/35)); sm80 runs but
has an open speculation fault at any k
([#98](https://github.com/syv-ai/HyperQwen/issues/98),
[#72](https://github.com/syv-ai/HyperQwen/issues/72)). Multi-GPU works at TP=2
and TP=4; TP=3 and PP=3 are invalid for this model's head count, which is a
property of the checkpoint rather than a missing feature.

**What we want next.** Smaller Qwen checkpoints so 16 GB and 12 GB cards get the
same treatment rather than a shrug; a larger one for 48 GB; the EXL3 route in
[#103](https://github.com/syv-ai/HyperQwen/issues/103) as a second quantization
path. If you want a specific model or card prioritized, say so in an issue — the
order is mostly driven by who turns up with a reproduction.

## Help us test more hardware

**We are looking for Runpod or Vast.ai credits.** One RTX 3090 at 250 W is the
entire hardware budget behind every number in this README, and it is shared with
training jobs. That constraint shows: the tables above lean on community
reproductions for every card that is not a 3090, an sm80 speculation bug has sat
open because nobody here owns an sm80 card to debug it on, and each architecture
sweep is scheduled around whatever else needs the GPU that week.

Rented compute converts directly into things this repo does not currently have:

- **An architecture matrix that is measured rather than collected.** sm80, sm89,
  sm90, sm120 and multi-GPU under one harness, one power protocol, one set of
  prompts — instead of a table where every row came from a different person's
  client and cannot honestly be compared to the row above it.
- **Fixing the bugs we cannot reproduce.**
  [#98](https://github.com/syv-ai/HyperQwen/issues/98) and
  [#72](https://github.com/syv-ai/HyperQwen/issues/72) are sm80 faults diagnosed
  entirely from other people's logs. A few hours on a rented A100 probably closes
  both.
- **Porting faster.** The vLLM 0.29.0 port
  ([#106](https://github.com/syv-ai/HyperQwen/issues/106)) is blocked on
  packaging details that need a machine to bisect on, while the one card here is
  the card serving production.
- **More models.** Every new checkpoint needs its draft vocabulary calibrated and
  its drafter trained — GPU-hours, not cleverness.

What you get: results published here as reproductions with the raw harness
output, credit in this section and on the runs themselves, and — if you would
rather have it than the credit — an honest writeup of what your hardware does
badly, which is usually the more useful half.

If you can help, open an issue titled "compute offer" and we will take it from
there. Small amounts are genuinely useful: a single day on one unfamiliar card
has historically been worth more to this project than a month on a familiar one.


## Documentation

Everything that used to be in this README, in the folder it belongs to.

**Running it**

| | |
|---|---|
| [batch/](batch/) · [single-user/](single-user/) | The two serving modes: full benchmark tables, every env knob, systemd units. |
| [docs/install.md](docs/install.md) | The bare-metal venv install, for when you are not using the container. |
| [docs/docker.md](docs/docker.md) | The container image, and an independent WSL2 reproduction. |
| [docs/clients.md](docs/clients.md) | Pointing coding agents and OpenAI-protocol clients at the server. |
| [docs/multi-gpu.md](docs/multi-gpu.md) | What transfers to a second card, and which tensor-parallel shapes this model allows. |
| [docs/third-party-checkpoints.md](docs/third-party-checkpoints.md) | Serving a checkpoint other than the base model, and what has to be re-prepared. |

**How it works, and how fast**

| | |
|---|---|
| [docs/optimizations.md](docs/optimizations.md) | Every optimization in full: why it was needed, what it measured, which patch implements it — including what each step buys in tokens per second, the two speculative-decoding modes, and the lookup drafter. |
| [docs/benchmarks.md](docs/benchmarks.md) | The comparison against ninfer-3090, and quality per configuration. |
| [docs/quality.md](docs/quality.md) | IFBench, perplexity and GSM8K per configuration. |
| [docs/long-context.md](docs/long-context.md) | 262k context with the KVarN 4/2-bit KV cache, `CTX=huge` with DFlash2 at 240k, vLLM's own per-token-head KV modes, and the experimental int4 route. |
| [docs/reproductions/](docs/reproductions/README.md) | Other people's hardware running this stack, reports and full write-ups. |
| [docs/gotchas.md](docs/gotchas.md) | Things that each cost us hours — read before debugging something that looks like a vLLM bug. |

**Building it**

| | |
|---|---|
| [PATCHES.md](PATCHES.md) | One line per patch in `patches/`: what it does, its upstream reference, and what retires it. |
| [docs/spec-decode-scratch-token-units.md](docs/spec-decode-scratch-token-units.md) | Why the speculative-decode scratch buffer is sized in tokens, and the int4 MQ-3D verify kernel — the design notes behind two of the patches. |
| [prepare/](prepare/) | The one-time model-preparation scripts (also `docker compose run --rm prepare`). |
| [drafter/](drafter/) | How the draft vocabulary, the int4 drafters and the DFlash2 requantization were built — including what did not work. |
| [kvarn/](kvarn/) | The KVarN 4/2-bit KV cache port. |

## License

Apache-2.0, same as the model.
