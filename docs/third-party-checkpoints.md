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
2. **This repo's `prepare/` scripts** on top — int8 shrink of `lm_head` and
   `embed_tokens`, int8 requant of the MTP module, and a 40,960-token draft
   vocabulary build for speculative decoding.

**Validation.** Boots and serves correctly at both a `CTX=fast`-equivalent
tier and a `CTX=long`/KVarN-equivalent tier on a 3090:

| | Decode throughput |
|---|---|
| `CTX=fast`-equivalent | 148 tok/s |
| `CTX=long`/KVarN-equivalent | 96 tok/s |

A 6-task correctness battery against the base checkpoint (arithmetic, code
generation, factual recall, a constraint-logic puzzle, a security-training
explanation, strict output-format compliance) came back **6/6 on both**,
with a measured **31.2% reduction in reasoning tokens** on that same
battery — real, but well short of the adapter's own 58.3% headline claim,
and not uniform across tasks (one logic puzzle in the battery actually used
*more* reasoning tokens than the baseline). Treat the efficiency gain as a
per-task estimate, not a guarantee.

**Any other export**, including single-shard and asymmetric-AWQ ones the base
model's three `quant_*.py` scripts cannot open, goes through the streaming
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
the base model uses, for two reasons that are properties of the checkpoint and not
of the model: it ships as **one 18.6 GB shard**, which the three scripts read into
RAM whole before rewriting, and its body is **asymmetric AWQ**, which those scripts
would copy onto the symmetric tensors they write — vLLM then looks for a
`weight_zero_point` that was never written. The streaming script handles both and
produces the same tensors otherwise; `bash verify.sh --no-server` with `MODEL=` set
checks the result exactly as it checks the base model.

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

Hand that gigabyte back and it comes up. `CTX=long` (int8 KV) is the one to spend it
on, because it buys roughly twice the context per byte of pool that `CTX=fast` does:

```bash
MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 SPEC=dflash2 CTX=long PREFIX_CACHE=1 \
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
