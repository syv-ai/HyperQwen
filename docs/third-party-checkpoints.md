# Third-party checkpoints

Serving a checkpoint other than the base model — uncensored builds and other
Qwen3.8-family derivatives — and what has to be re-prepared for each one.

[← back to the main README](../README.md)

`MODEL=` points the launchers at any Qwen3.8-27B checkpoint in the same
`compressed-tensors` shape. Two routes, easiest first.

**Ready-made:**
[leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound](https://huggingface.co/leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound)
([#45](https://github.com/syv-ai/HyperQwen/issues/45)) is an
abliterated Qwen3.8-27B already quantized with this repo's own recipe —
AutoRound W4A16 body plus the `prepare/` head requant — so it serves without
any preparation. Its author measured ~100 tok/s warm at `SPEC=dflash2
CTX=huge` on a 3090 with coherent output and a 45k-context needle retrieved,
and a second tester confirmed `SPEC=mtp` works. Community-built and
community-verified; not benchmarked on this repo's reference box.

### Ready-made, Swift + uncensored

[ultimaterex/Swift-Qwen3.8-27B-Uncensored-W4A16-AutoRound](https://huggingface.co/ultimaterex/Swift-Qwen3.8-27B-Uncensored-W4A16-AutoRound)

**Why Swift matters here.** Every extra token a model spends "thinking" before
it answers costs latency and money, and that cost compounds hard in an
agentic loop that re-reasons on every tool call. UkisAI's
[Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) adapter
targets exactly that: a reasoning-efficiency LoRA trained to reach the same
answer with a shorter chain of thought, with a claimed 58.3% reduction in
thinking tokens. Pairing it with an uncensored base compounds the value for
automation use cases specifically — fewer stalled tool calls from
unnecessary refusals, on top of the speed gain.

**What we did.** Took
[d0xin/Swift-Qwen3.8-27B-Uncensored-BF16](https://huggingface.co/d0xin/Swift-Qwen3.8-27B-Uncensored-BF16)
(Swift adapter merged, then rank-1 residual-stream ablation on top) and ran
it through this repo's own quantization pipeline end to end:

1. **AutoRound W4A16**, reproducing this repo's published recipe against the
   Swift+uncensored base instead of the official checkpoint — `in_proj_a`/
   `in_proj_b`, the vision tower, and the MTP draft head kept BF16, everything
   else quantized to W4A16 g128 symmetric.
2. **This repo's `prepare/` scripts** on top — GPTQ-recalibrated int4 requant
   of `lm_head` and the MTP module (the first upload shipped both at int8),
   with `embed_tokens` kept int8, which is what the pipeline does:
   `prepare/quant_embed.py` has `BITS = 8` and `quant_heads_stream.py` has
   `HEAD_BITS = 8`, and the artifact's `config.json` agrees (`group_2`,
   `num_bits: 8`, a `[248320, 1280]` I32 pack — int4 would be 640 columns).
   The `mtp_draft_vocab_ids.pt` it ships is this repo's own 40,960-id list
   (`prepare/draft_vocab_ids.json`), not one counted over this checkpoint's
   own outputs. Rebuilding it from the finetune's own outputs is the stronger
   option and is measurable — a Swift build in
   [#139](https://github.com/syv-ai/HyperQwen/pull/139) found Swift emits only
   ~25.9k distinct tokens against base Qwen's ~54k, so its draft head shrinks
   to 25,879 rows at 99.8% coverage and the acceptance deficit disappears.

**Validation.** Boots and serves correctly at `CTX=fast`, `CTX=long` and
`CTX=huge`, with `SPEC=dflash2` and with `SPEC=mtp`. The 148 tok/s
(`CTX=fast`-equivalent) and 96 tok/s (`CTX=long`/KVarN-equivalent) figures
in the first revision of this entry were not `bench/run_benchmarks.sh`
output -- they came from vLLM's per-request `metrics.tokens_per_second` on
a single 50-token boot-verification request, which is dominated by
per-request overhead at that completion length and reproduces anywhere from
~50 to ~175 tok/s depending on the try. Removed rather than replaced with
an unverifiable number; happy to add real `run_benchmarks.sh` C1 figures if
useful.

Quality battery (`bench/quality_battery.py`, `SPEC=dflash2`,
`PREFIX_CACHE=1` -- unaffected by gotcha 46, which is specific to
`SPEC=mtp`) on `CTX=huge`: perplexity en 11.22 / da 11.49 / code 3.50,
GSM8K 200q 95.5% acc, 362 mean tokens. In line with this repo's own numbers
([docs/benchmarks.md](benchmarks.md), GSM8K 96.5%;
[docs/long-context.md](long-context.md), 95.0-97.0%); no regression from the
MTP tensor fix, which does not touch the tensors these numbers exercise.

Both the tensor fix and the boot were checked independently in the thread, on
the live files rather than on a local copy: 2,022 index entries against 2,022
keys with nothing duplicated or orphaned, and `bash verify.sh --no-server
MODEL=<dir>` passing 7/7 model checks — including the cross-shard duplicate
check that exists for exactly this failure. Since the entry's claim is that it
serves with no preparation, run that command against your download before you
serve it.

**`SPEC=mtp`.** An earlier upload of this checkpoint left the superseded int8
`mtp.*` tensors in `model-00004-of-00004.safetensors` next to the int4
recalibration in `model_extra_tensors.safetensors`, with the index routing
only three keys to the extras file. vLLM loads every key in a file it opens,
so `mtp.fc.weight_packed` appeared at two widths and `SPEC=mtp` failed at
load with a shape mismatch (`SPEC=dflash2` never loads those tensors, so it
was unaffected). Fixed in
[ultimaterex/Swift-Qwen3.8-27B-Uncensored-W4A16-AutoRound@29c95a9](https://huggingface.co/ultimaterex/Swift-Qwen3.8-27B-Uncensored-W4A16-AutoRound/commit/29c95a9c5a0b36246aae41b45c2e6732d9db48e0):
the 27 duplicated tensors were removed from shard 4 and the index now routes
all 27 to the extras file. No key appears in more than one file, and vLLM
0.28.0 boots and serves with `SPEC=mtp` on the fixed files.

A 6-task correctness battery against the base checkpoint (arithmetic, code
generation, factual recall, a constraint-logic puzzle, a security-training
explanation, strict output-format compliance) came back **6/6 on both**,
with a measured **31.2% reduction in reasoning tokens** on that same
battery — real, but well short of the adapter's own 58.3% headline claim,
and not uniform across tasks (one logic puzzle in the battery actually used
*more* reasoning tokens than the baseline). Treat the efficiency gain as a
per-task estimate, not a guarantee.

### Ready-made, asymmetric AWQ with the vision tower

[Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM-HyperQwen](https://huggingface.co/Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM-HyperQwen)
and
[Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM-HyperQwen](https://huggingface.co/Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM-HyperQwen)
are the llm-compressor AWQ exports of the uncensored finetune and of the base model --
int4 asymmetric g128 *with zero points*, the vision tower, the MTP head and the SSM gate
projections in bf16 -- after `prepare/quant_heads_stream.py` and `build_draft_vocab.py`,
so they serve without preparation; the `-fast` siblings carry the int4-GPTQ `lm_head`
from `drafter/gptq_lm_head.py` (calibrated on 600k teacher-forced UltraChat tokens:
KL 0.0070 RTN -> 0.0024 GPTQ; building one from a single-shard export like these takes
the `drafter/` fixes in #181). Asymmetric bodies need
`patches/marlin-int8-asym-zp.patch` for `INT8_ACT=int8` (batch mode, the production
line): without it vLLM refuses the zero-point weights on the int8 path at load.
`verify.sh --no-server` passes all eight model checks on the four of them. Measured on an
RTX 3090 at 350 W on vLLM 0.29.0 (the #148 port with this patch on top) with `VISION=1`
(tower offloaded to host RAM), C1 at the model's default sampling, second harness run
kept:

| | C1 | tok/step | pool |
|---|---|---|---|
| `SPEC=mtp CTX=fast` | 107.4 tok/s | 2.71 | 70,933 |
| `SPEC=dflash2 CTX=fast KV_MEM=4300000000 DFLASH_MAX_LEN=49152` | 123.0 | 3.15 | 49,662 |
| `SPEC=mtp CTX=long MAX_LEN=100000` | 84.4 | 2.58 | 164,705 |
| the production line (`DFLASH_TOKENS=15 INT8_ACT=int8 PREFILL_ATTN=int8`, `DFLASH_MAX_LEN=36864`) | 117.5 (TTFT 96 ms) | 3.12 | 37,834 |
| `-fast` sibling, `SPEC=dflash2 CTX=fast KV_MEM=4600000000 DFLASH_MAX_LEN=49152` | **135.9** | 3.29 | 53,233 |
| batch mode, `GPU_UTIL=0.94 MAX_LEN=100000` (the shipped 0.972 / 150k ran out of memory at warmup with the tower on, #182) | 1,169 tok/s decode at 64 concurrent | | 215,267 |

DFlash2 needs its pool pinned lower than the fast variant's default for the same reason
the philbert440 export does (15.7 GiB of weights after requantization against 14.71), and
a 15-token verify block needs `DFLASH_MAX_LEN=36864` on top. The full table with the int8
and fast-variant rows, and the container that ships all four checkpoints:
[Ar4ikov/vllm-hyprfastQwen](https://github.com/Ar4ikov/vllm-hyprfastQwen).

### Ready-made, ThinkingCap (reasoning fine-tune)

[dewamade/ThinkingCap-Qwen3.8-27B-AutoRound-W4A16-fast](https://huggingface.co/dewamade/ThinkingCap-Qwen3.8-27B-AutoRound-W4A16-fast)
is BottleCap AI's reasoning fine-tune
[ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B)
in this repo's fast-variant layout, so it serves without preparation
([#211](https://github.com/syv-ai/HyperQwen/issues/211)). It starts from an
AutoRound W4A16 export in AutoRound's GPTQ layout, which this stack's Marlin
path does not load, and repacks the same int4 g128 symmetric weights into
`compressed-tensors`. The int4 GPTQ `lm_head` and MTP module (all `mtp.*` keys
in `model_extra_tensors.safetensors`), the int8 `embed_tokens` and this repo's
stock `mtp_draft_vocab_ids.pt` match the shipped fast variant (config groups
and the ids file's hash checked against it). A `-emb4` companion,
[dewamade/ThinkingCap-Qwen3.8-27B-AutoRound-W4A16-fast-emb4](https://huggingface.co/dewamade/ThinkingCap-Qwen3.8-27B-AutoRound-W4A16-fast-emb4),
swaps in an int4 embedding table for ~0.6 GB more pool. The author's numbers,
RTX 3090 at 250 W, vLLM 0.29.0 at 73fd65d, `SPEC=mtp CTX=long VISION=1
PREFIX_CACHE=1`:

| | pool | PPL en / da / code | GSM8K (200) |
|---|---|---|---|
| fast | 201,275 | 10.77 / 10.91 / 3.39 | 0.970 |
| fast-emb4 | 218,112 | 10.78 / 10.92 / 3.39 | 0.970 |

Community-built; not benchmarked on the reference box, and no decode rate was
reported. The base model declares its own license on the Hub (`other`); read
it before redistributing.

**Any other export**, including single-shard ones too big for the base model's
three `quant_*.py` scripts to read into RAM, goes through the streaming
requant (contributed in
[#37](https://github.com/syv-ai/HyperQwen/pull/37)). The worked
example is
[philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ)
— an abliterated (de-refused) Qwen3.8-27B, W4A16 AWQ, with the vision tower and
the grafted MTP head both preserved. Prepare it once, then serve it:

```bash
venv/bin/python prepare/fetch_thirdparty.py          # ~18.6 GB; or: fetch_thirdparty.py <hf-repo>
venv/bin/python prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16
venv/bin/python prepare/build_draft_vocab.py  models/Qwen3.8-27B-Uncensored-W4A16 \
  --ids prepare/draft_vocab_ids.json

MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 SPEC=mtp CTX=long PREFIX_CACHE=1 \
  MAX_LEN=100000 bash single-user/start_qwen.sh
```

It needs `prepare/quant_heads_stream.py` rather than the three `quant_*.py` steps
the base model uses, because of a property of the checkpoint and not of the model:
it ships as **one 18.6 GB shard**, which the three scripts read into RAM whole before
rewriting. Its **asymmetric AWQ** body is not a reason: all four scripts declare the
groups they add symmetric with no zero point
([#212](https://github.com/syv-ai/HyperQwen/pull/212)). The streaming script produces
the same tensors otherwise; `bash verify.sh --no-server` with `MODEL=` set checks the
result exactly as it checks the base model.

**`SPEC=dflash2` needs its pool resized for this checkpoint.** After requantization
it is 15.68 GiB of weights against the fast variant's 14.71, and the DFlash2 branch
pins the KV pool *in bytes* (`KV_MEM`) rather than sizing it from
`--gpu-memory-utilization`, so the pool does not give that gigabyte back. The server
loads, captures graphs, and then dies on the split-KV verify buffer:

```
Model loading took 15.71 GiB
reserved 5.2 GiB memory for KV Cache as specified by kv_cache_memory_bytes config
torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 926.44 MiB is free
```

Hand that gigabyte back and it comes up. `CTX=long` with int8 KV (`FA2_FP8KV=0`; the default is now fp8 on the fa2/ plugin, which
this run did not measure) is the one to spend it
on, because it buys roughly twice the context per byte of pool that `CTX=fast` does:

```bash
MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 SPEC=dflash2 CTX=long FA2_FP8KV=0 PREFIX_CACHE=1 \
  KV_MEM=4456028569 DFLASH_MAX_LEN=98304 bash single-user/start_qwen.sh
```

Measured here, RTX 3090 at 250 W: a 4.15 GiB pool holding **103,033 tokens** at
98,304 `max-model-len` (4.6% margin) and **85.7 tok/s** greedy on a 400-token
answer. The checkpoint keeps its vision tower, and that run had `VISION=1` — images
came back described correctly — so the numbers are an upper bound on what the
default `VISION=0` needs, which drops the tower's weights entirely.

`SPEC=mtp` needs no `KV_MEM` of its own: its pool is profiled from `GPU_UTIL` rather
than pinned, so it absorbs the extra gigabyte by shrinking the pool for you. It is the
mode to reach for first on this checkpoint. The pool it lands on will not hold
`CTX=long`'s stock 150k, though, which is what the `MAX_LEN=100000` above is — the
figure this checkpoint has been run at.

**A fully-built fast variant of a finetune:**
[liamwh/Swift-Qwen3.8-27B-W4A16-syv-fast](https://huggingface.co/liamwh/Swift-Qwen3.8-27B-W4A16-syv-fast)
is [ukisai/Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b)
(the "reduced reasoning" finetune) in the single-user fast-variant layout —
so it serves with no preparation at all, and it is also a worked example of
building that layout for a checkpoint the prebuilt fast variant does not
cover. The AWQ body is
[TheUnderscore's](https://huggingface.co/TheUnderscore/Swift-Qwen3.8-27b-W4A16-AWQ)
carried through unmodified; the lm_head and MTP are int4-GPTQ calibrated on
the finetune's OWN hidden states (lm_head KL 0.00234 against the shipped
fast variant's 0.0029), and the draft vocab is counted over 4.23M tokens of
its own outputs on a coding-agent-weighted corpus. That last step found
something worth knowing for any finetune: Swift emits only ~25.9k distinct
tokens (base Qwen: ~54k), so its draft head is 25,879 rows rather than
40,960 — smaller and slightly faster, at 99.8% held-out coverage (the base
model's id list covers 96.7% of Swift's output).

Measured on a 3090 at 350 W, `SPEC=mtp CTX=long MAX_LEN=114688`: 98.4 tok/s
at 0.660 MTP acceptance — parity with the shipped fast variant (98.2,
0.634) on a body one AWQ-zero-point heavier. With `SPEC=dflash2` and the
generic drafter the acceptance deficit people worry about with finetunes
did not materialise once the heads and vocab were rebuilt: 0.385 against
the base's 0.387.

Two things the build taught, for anyone rebuilding this layout for another
export. The upstream AWQ ships lm_head and embed_tokens in *different*
shards — the case the streaming requant was just generalised to handle —
and, if you assemble the fast dir with hardlinks the way `drafter/`'s
scripts do, strip the superseded int8 MTP tensors from any shard you keep:
vLLM loads every key in an opened file, so a stale `mtp.fc.weight_packed`
sitting next to the bf16 norms collides with the int4 one in
`model_extra_tensors.safetensors` and the load dies on a shape assert. The
model card documents the full recipe and both licences the Swift Open
License requires a derivative to carry.
