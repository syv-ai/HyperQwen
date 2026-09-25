"""Requantize lm_head to int8 (group-128, symmetric) in compressed-tensors
pack-quantized format, in place.

The published W4A16 quants of Qwen3.8-27B leave lm_head in bf16 — that's a
2.5 GB matrix (248k vocab) read every decode step. int8 halves the read and
frees ~1.3 GB of VRAM for the KV/state pool. Measured +12% aggregate
throughput on an RTX 3090, round-trip error 0.64% (Frobenius).

Usage: python prepare/quant_lm_head.py /path/to/Qwen3.8-27B-W4A16-AutoRound

Rewrites the shard containing lm_head.weight, config.json and the safetensors
index, in that order. The first pre-quant copy of each is kept next to it
(.bak for the shard, .bak-quant for config and index) and never overwritten.

Each file is written through prepare/atomic_publish.py (a temp file and a
rename), and the index goes last: docker/prepare.sh's state() reads only the
index, so a killed run leaves the step pending, and the next run completes it,
reusing a shard that already holds the packed lm_head (#195).
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
KEY = "lm_head.weight"
PACKED = [f"lm_head.{s}" for s in ("weight_packed", "weight_scale", "weight_shape")]

d = sys.argv[1].rstrip("/") + "/"

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
shard = wm[KEY]
print(f"{KEY} lives in {shard}")

tensors = {}
with safe_open(d + shard, framework="pt") as f:
    meta = f.metadata()
    names = set(f.keys())
    if KEY in names:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

if KEY not in names:
    # A killed run published the shard but not the index: finish that run.
    if not all(k in names for k in PACKED):
        sys.exit(f"{shard} holds neither {KEY} nor the packed lm_head; restore it from {shard}.bak")
    print(f"{shard} already holds the packed lm_head (completing an interrupted run)")
else:
    w = tensors.pop(KEY).to(torch.float32)
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)

    deq = (q.reshape(out_f, -1, GROUP).to(torch.float32) * scale).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    print(f"round-trip relative error: {err:.4f}")
    assert err < 0.01, "quantization error too high, aborting"

    tensors["lm_head.weight_packed"] = pack_to_int32(q, BITS, packed_dim=1).contiguous()
    # linear layers use fp16 scales in this checkpoint
    tensors["lm_head.weight_scale"] = scale.squeeze(-1).to(torch.float16).contiguous()
    tensors["lm_head.weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)

    backup_once(d + shard, ".bak")
    save_tensors(tensors, d + shard, meta or {"format": "pt"})
    del tensors

c = json.load(open(d + "config.json"))
backup_once(d + "config.json", ".bak-quant")
qc = c["quantization_config"]
qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head"]
# The MTP draft head is stored in bf16 but missing from the ignore list, which
# breaks loading when speculative decoding is enabled (single-user mode).
for m in (
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
):
    if m not in qc["ignore"]:
        qc["ignore"].append(m)
g1 = copy.deepcopy(qc["config_groups"]["group_0"])
g1["targets"] = ["re:.*lm_head$"]
g1["weights"]["num_bits"] = BITS
qc["config_groups"]["group_1"] = g1
write_json(d + "config.json", c)

# The index is the commit point, so it goes last.
backup_once(d + "model.safetensors.index.json", ".bak-quant")
del wm[KEY]
for k in PACKED:
    wm[k] = shard
write_json(d + "model.safetensors.index.json", idx)
print("done")
