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

The config.json schema is checked before the first shard is read: a checkpoint
these scripts cannot extend exits with one line and an untouched directory
instead of a rewritten shard (#241).
"""

import json
import sys

import torch
from safetensors import safe_open

from atomic_publish import backup_once, save_tensors, write_json
from quant_schema import MTP_LINEARS, clone_group, index_packed, load_config, pack, packed_already

BITS = 8
KEY = "lm_head.weight"

d = sys.argv[1].rstrip("/") + "/"
# Before anything is read or written: a checkpoint these scripts cannot extend used to
# fail on qc["ignore"] after the shard had already been replaced (#241).
c, qc = load_config(d)

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

if not packed_already(names, "lm_head", shard, ".bak"):
    # linear layers use fp16 scales in this checkpoint
    packed, err = pack("lm_head", tensors.pop(KEY), BITS, torch.float16)
    print(f"round-trip relative error: {err:.4f}")
    assert err < 0.01, "quantization error too high, aborting"
    tensors.update(packed)

    backup_once(d + shard, ".bak")
    save_tensors(tensors, d + shard, meta or {"format": "pt"})
    del tensors

backup_once(d + "config.json", ".bak-quant")
qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head"]
# The MTP draft head is stored in bf16 but missing from the ignore list, which
# breaks loading when speculative decoding is enabled (single-user mode).
qc["ignore"] += [m for m in MTP_LINEARS if m not in qc["ignore"]]
clone_group(qc, "lm_head", BITS)
write_json(d + "config.json", c)

# The index is the commit point, so it goes last.
backup_once(d + "model.safetensors.index.json", ".bak-quant")
index_packed(wm, "lm_head", shard)
write_json(d + "model.safetensors.index.json", idx)
print("done")
