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
