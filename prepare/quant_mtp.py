"""Requantize the MTP (multi-token-prediction) draft module to int8 or int4
(group-128, symmetric) in compressed-tensors pack-quantized format, in place.

The published W4A16 quant leaves the whole `mtp.*` module in bf16 (~850 MB:
mtp.fc plus one full decoder layer). In single-user mode that module runs once
per draft token, so at 4 drafts/step it is read four times per step; int8
halves that traffic, int4 quarters it. The draft head only steers speculation
(acceptance rate) — the sampled distribution stays exact either way — so this
is a pure speed knob. Measured acceptance change: int8 none.

Usage: python prepare/quant_mtp.py /path/to/Qwen3.8-27B-W4A16-AutoRound [--bits 8|4] [--keep-fc]
--keep-fc leaves mtp.fc (the 10240->5120 input projection, 105 MB) in bf16.

Rewrites model_extra_tensors.safetensors, config.json and the safetensors index, in
that order; the first pre-quant copy of each is kept as <file>.bak-mtp and never
overwritten. Each file is written through prepare/atomic_publish.py (a temp file and a
rename), and the index goes last: docker/prepare.sh's state() reads only the index, so
a killed run leaves the step pending, and the next run completes it, reusing a shard
that already holds the packed linears (#195).
"""

import copy
import json
import sys

import torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

from atomic_publish import backup_once, save_tensors, write_json

SUFFIXES = ("weight_packed", "weight_scale", "weight_shape")

GROUP = 128
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
QMAX = 2 ** (BITS - 1) - 1
KEEP_FC = "--keep-fc" in sys.argv
MTP_LINEARS = ([] if KEEP_FC else ["mtp.fc"]) + [
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
]

d = sys.argv[1].rstrip("/") + "/"
idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
shards = {wm[m + ".weight"] for m in MTP_LINEARS}
assert len(shards) == 1, f"mtp weights span several shards: {shards}"
shard = shards.pop()
print(f"mtp linears live in {shard}, quantizing to int{BITS} g{GROUP}")

tensors = {}
with safe_open(d + shard, framework="pt") as f:
    meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

changed = False
for m in MTP_LINEARS:
    if m + ".weight" not in tensors:
        # A killed run published the shard but not the index: finish that run.
        if not all(f"{m}.{s}" in tensors for s in SUFFIXES):
            sys.exit(f"{shard} holds neither {m}.weight nor its packed form; restore it from {shard}.bak-mtp")
        print(f"  {m}: already packed in {shard} (completing an interrupted run)")
        continue
    w = tensors.pop(m + ".weight").to(torch.float32)
    out_f, in_f = w.shape
    assert in_f % GROUP == 0, (m, w.shape)
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)
    deq = (q.reshape(out_f, -1, GROUP).to(torch.float32) * scale).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    print(f"  {m}: {tuple(w.shape)} round-trip rel error {err:.4f}")
    tensors[m + ".weight_packed"] = pack_to_int32(q, BITS, packed_dim=1).contiguous()
    tensors[m + ".weight_scale"] = scale.squeeze(-1).to(torch.float16).contiguous()
    tensors[m + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    changed = True

if changed:
    backup_once(d + shard, ".bak-mtp")
    save_tensors(tensors, d + shard, meta or {"format": "pt"})
del tensors

c = json.load(open(d + "config.json"))
backup_once(d + "config.json", ".bak-mtp")
qc = c["quantization_config"]
qc["ignore"] = [i for i in qc["ignore"] if i not in MTP_LINEARS]
g = copy.deepcopy(qc["config_groups"]["group_0"])
g["targets"] = ["re:^mtp\\.layers\\..*"] if KEEP_FC else ["re:^mtp\\..*"]
g["weights"]["num_bits"] = BITS
qc["config_groups"]["group_3"] = g
write_json(d + "config.json", c)

# The index is the commit point, so it goes last.
backup_once(d + "model.safetensors.index.json", ".bak-mtp")
for m in MTP_LINEARS:
    del wm[m + ".weight"]
    for s in SUFFIXES:
        wm[f"{m}.{s}"] = shard
write_json(d + "model.safetensors.index.json", idx)
print("done")
