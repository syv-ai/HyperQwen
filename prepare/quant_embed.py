"""Requantize the token embedding table to int8 (group-128, symmetric),
in place. Companion to quant_lm_head.py — run that one first.

Qwen3.8-27B has untied embeddings, so embed_tokens is a second 2.5 GB bf16
matrix on top of lm_head. vLLM ships a dequant-on-gather path for int-quantized
embeddings (CompressedTensorsEmbeddingWNA16Int) but the qwen3_5 model code
never passes quant_config to VocabParallelEmbedding, so you also need the
two-line patch in patches/qwen3_5-embed-quant.patch.

Usage: python prepare/quant_embed.py /path/to/Qwen3.8-27B-W4A16-AutoRound

Rewrites the shard holding embed_tokens, config.json and the safetensors index, in
that order, through prepare/atomic_publish.py (a temp file and a rename). The index
goes last: docker/prepare.sh's state() reads only the index, so a killed run leaves the
step pending, and the next run completes it, reusing a shard that already holds the
packed embeddings (#195).

Measured on an RTX 3090: another ~1.3 GB freed, round-trip error 0.56%.

The config.json schema is checked before the first shard is read: a checkpoint
these scripts cannot extend exits with one line and an untouched directory
instead of a rewritten shard (#241).
"""

import json
import sys

import torch
from safetensors import safe_open

from atomic_publish import backup_once, save_tensors, write_json
from quant_schema import clone_group, index_packed, load_config, pack, packed_already

BITS = 8

d = sys.argv[1].rstrip("/") + "/"
# Before anything is read or written: a checkpoint these scripts cannot extend used to
# fail on qc["ignore"] after the shard had already been replaced (#241).
c, qc = load_config(d)

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
key = next(k for k in wm if k.endswith("embed_tokens.weight"))
base = key[:-len(".weight")]
shard = wm[key]
print(f"{key} lives in {shard}")

tensors = {}
with safe_open(d + shard, framework="pt") as f:
    meta = f.metadata()
    names = set(f.keys())
    if key in names:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

if not packed_already(names, base, shard, ".bak_embed"):
    # the embedding path creates scales in params_dtype (bf16), unlike the linears
    packed, err = pack(base, tensors.pop(key), BITS, torch.bfloat16)
    print(f"round-trip relative error: {err:.4f}")
    assert err < 0.01, "quantization error too high, aborting"
    tensors.update(packed)

    # ".bak_embed", not ".bak": quant_lm_head.py writes ".bak" for its own shard, and a
    # checkpoint that puts embed_tokens and lm_head in one shard would have the second
    # script overwrite the first one's pristine backup. drafter/train_mtp.py reads both.
    backup_once(d + shard, ".bak_embed")
    save_tensors(tensors, d + shard, meta or {"format": "pt"})
    del tensors

# group_0, as the other scripts clone: quant_lm_head.py's group_1 may not exist yet,
# and the shard is already replaced at this point.
clone_group(qc, "embed", BITS)
write_json(d + "config.json", c)

# The index is the commit point, so it goes last.
index_packed(wm, base, shard)
write_json(d + "model.safetensors.index.json", idx)
print("done")
