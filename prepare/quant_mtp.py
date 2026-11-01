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

The config.json schema is checked before the first shard is read: a checkpoint
these scripts cannot extend exits with one line and an untouched directory
instead of a rewritten shard (#241).
"""

import json
import sys

import torch
from safetensors import safe_open

from atomic_publish import backup_once, save_tensors, write_json
from quant_schema import GROUP, MTP_LINEARS, clone_group, index_packed, load_config, pack, packed_already

BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
KEEP_FC = "--keep-fc" in sys.argv
LINEARS = [m for m in MTP_LINEARS if not (KEEP_FC and m == "mtp.fc")]

d = sys.argv[1].rstrip("/") + "/"
# Before anything is read or written: a checkpoint these scripts cannot extend used to
# fail on qc["ignore"] after the shard had already been replaced (#241).
c, qc = load_config(d)

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
shards = {wm[m + ".weight"] for m in LINEARS}
assert len(shards) == 1, f"mtp weights span several shards: {shards}"
shard = shards.pop()
print(f"mtp linears live in {shard}, quantizing to int{BITS} g{GROUP}")

tensors = {}
with safe_open(d + shard, framework="pt") as f:
    meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

changed = False
for m in LINEARS:
    if packed_already(tensors, m, shard, ".bak-mtp"):
        continue
    w = tensors.pop(m + ".weight")
    packed, err = pack(m, w, BITS, torch.float16)
    print(f"  {m}: {tuple(w.shape)} round-trip rel error {err:.4f}")
    tensors.update(packed)
    changed = True

if changed:
    backup_once(d + shard, ".bak-mtp")
    save_tensors(tensors, d + shard, meta or {"format": "pt"})
del tensors

backup_once(d + "config.json", ".bak-mtp")
qc["ignore"] = [i for i in qc["ignore"] if i not in LINEARS]
clone_group(qc, "mtp-keep-fc" if KEEP_FC else "mtp", BITS)
write_json(d + "config.json", c)

# The index is the commit point, so it goes last.
backup_once(d + "model.safetensors.index.json", ".bak-mtp")
for m in LINEARS:
    index_packed(wm, m, shard)
write_json(d + "model.safetensors.index.json", idx)
print("done")
