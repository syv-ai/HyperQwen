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
"""

import copy
import json
import sys

import torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

from atomic_publish import backup_once, save_tensors, write_json

GROUP = 128
BITS = 8
QMAX = 127

d = sys.argv[1].rstrip("/") + "/"

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
key = next(k for k in wm if k.endswith("embed_tokens.weight"))
packed = [key.replace(".weight", "." + s) for s in ("weight_packed", "weight_scale", "weight_shape")]
shard = wm[key]
print(f"{key} lives in {shard}")

tensors = {}
with safe_open(d + shard, framework="pt") as f:
    meta = f.metadata()
    names = set(f.keys())
    if key in names:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

if key not in names:
    # A killed run published the shard but not the index: finish that run.
    if not all(k in names for k in packed):
        sys.exit(f"{shard} holds neither {key} nor its packed form; restore it from {shard}.bak_embed")
    print(f"{shard} already holds the packed embeddings (completing an interrupted run)")
else:
    w = tensors.pop(key).to(torch.float32)
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)

    deq = (q.reshape(out_f, -1, GROUP).to(torch.float32) * scale).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    print(f"round-trip relative error: {err:.4f}")
    assert err < 0.01, "quantization error too high, aborting"

    tensors[packed[0]] = pack_to_int32(q, BITS, packed_dim=1).contiguous()
    # the embedding path creates scales in params_dtype (bf16), unlike the linears
    tensors[packed[1]] = scale.squeeze(-1).to(torch.bfloat16).contiguous()
    tensors[packed[2]] = torch.tensor([out_f, in_f], dtype=torch.int64)

    # ".bak_embed", not ".bak": quant_lm_head.py writes ".bak" for its own shard, and a
    # checkpoint that puts embed_tokens and lm_head in one shard would have the second
    # script overwrite the first one's pristine backup. drafter/train_mtp.py reads both.
    backup_once(d + shard, ".bak_embed")
    save_tensors(tensors, d + shard, meta or {"format": "pt"})
    del tensors

c = json.load(open(d + "config.json"))
qc = c["quantization_config"]
g2 = copy.deepcopy(qc["config_groups"]["group_1"])
g2["targets"] = ["re:.*embed_tokens$"]
g2["weights"]["num_bits"] = BITS
qc["config_groups"]["group_2"] = g2
write_json(d + "config.json", c)

# The index is the commit point, so it goes last.
del wm[key]
for k in packed:
    wm[k] = shard
write_json(d + "model.safetensors.index.json", idx)
print("done")
