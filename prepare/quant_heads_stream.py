"""Requantize lm_head, embed_tokens and the MTP module to int8/int4
(group-128, symmetric) in compressed-tensors pack-quantized format, in place.

Same math and output as quant_lm_head.py / quant_embed.py / quant_mtp.py (all four
quantize and declare through prepare/quant_schema.py), but for single-shard checkpoints,
which those three cannot handle. They read a whole shard into a dict before rewriting it;
philbert440/Qwen3.8-27B-Uncensored-* ships one 18.6 GB model.safetensors (2384 tensors),
which does not fit in RAM here. This streams the shard tensor-by-tensor instead, copying
untouched tensors as raw bytes, so peak RSS is a few GB regardless of shard size.

Usage: venv/bin/python prepare/quant_heads_stream.py /path/to/model [--mtp-bits 8|4] [--keep-fc]

This is what the uncensored checkpoint needs (prepare/fetch_thirdparty.py); the base
model is 7 shards, so quant_lm_head/quant_embed/quant_mtp serve it fine.

The rewritten shards replace the originals; the first pre-quant copy of each stays
next to it as <shard>.bak-orig (a hardlink to the original file, as the old rename kept
it: no copy of an 18.6 GB shard, never overwritten, so it keeps holding the bf16 lm_head
that drafter/gptq_lm_head.py reads), and config.json and the safetensors index are
backed up as .bak-quant.

Every file goes through prepare/atomic_publish.py (a temp file and a rename), and the
index is written last: docker/prepare.sh's state() reads only the index, so a killed run
leaves the step pending, and the next run completes it, reusing a shard that already
holds the packed tensors (#195).

The config.json schema is checked before the first shard is read: a checkpoint
these scripts cannot extend exits with one line and an untouched directory
instead of a rewritten shard (#241).
"""

import json
import os
import struct
import sys

import torch
from safetensors import safe_open

from atomic_publish import backup_once, publish, save_tensors, write_json
from quant_schema import GROUP, MTP_LINEARS, clone_group, index_packed, load_config, pack, packed_already

HEAD_BITS = 8
MTP_BITS = int(sys.argv[sys.argv.index("--mtp-bits") + 1]) if "--mtp-bits" in sys.argv else 8
KEEP_FC = "--keep-fc" in sys.argv
LINEARS = [m for m in MTP_LINEARS if not (KEEP_FC and m == "mtp.fc")]

d = sys.argv[1].rstrip("/") + "/"
# Before the shards are streamed and rewritten: a checkpoint these scripts cannot extend
# used to fail on qc["ignore"] after the shard had already been replaced (#241).
c, qc = load_config(d)

DTYPE_STR = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32",
    torch.int8: "I8", torch.int32: "I32", torch.int64: "I64", torch.uint8: "U8",
}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def stream_rewrite(src, dst, drop, add):
    """Copy src to dst, omitting tensor names in `drop` and appending `add`
    (name -> tensor). Untouched tensors are copied as raw bytes."""
    hdr, data_start = read_header(src)
    meta = hdr.pop("__metadata__", None)
    keep = [(k, v) for k, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
            if k not in drop]

    new_hdr, off = {}, 0
    if meta is not None:
        new_hdr["__metadata__"] = meta
    plan = []
    for k, v in keep:
        b0, b1 = v["data_offsets"]
        size = b1 - b0
        new_hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + size]}
        plan.append(("copy", data_start + b0, size))
        off += size
    for k, t in add.items():
        t = t.contiguous()
        size = t.numel() * t.element_size()
        new_hdr[k] = {"dtype": DTYPE_STR[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + size]}
        plan.append(("write", t, size))
        off += size

    blob = json.dumps(new_hdr, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for kind, a, size in plan:
            if kind == "copy":
                fi.seek(a)
                left = size
                while left:
                    chunk = fi.read(min(left, 64 << 20))
                    if not chunk:
                        raise IOError(f"short read in {src}")
                    fo.write(chunk)
                    left -= len(chunk)
            else:
                # .numpy() has no bfloat16; reinterpret as bytes instead
                fo.write(memoryview(a.contiguous().view(torch.uint8).numpy()))
    return off


def keep_original(shard):
    """Keep the pre-quant shard as <shard>.bak-orig, once. A hardlink, not a copy: the
    shard can be 18.6 GB, and publish() gives the live path a new inode, so the link
    keeps the original bytes. The first link is never replaced."""
    if not os.path.exists(shard + ".bak-orig"):
        os.link(shard, shard + ".bak-orig")


idx_path = d + "model.safetensors.index.json"
idx = json.load(open(idx_path))
wm = idx["weight_map"]


def weight_name(entry):
    """The `<base>.weight` spelling of an index entry: the index is written last, so a
    run that got as far as the index leaves only the packed entries behind."""
    return entry[: -len("weight_packed")] + "weight" if entry.endswith("weight_packed") else entry


# Where each weight lives, read before writing anything, so an interrupted run still
# resolves the shard of a key whose packed entry is already in the index.
shards = {weight_name(k): v for k, v in wm.items()}

emb_base = next(k for k in shards if k.endswith("embed_tokens.weight"))[:-len(".weight")]

# ---- lm_head + embed_tokens: one streaming pass per shard that holds them.
# Single-shard exports (the original case) land in one group; multi-shard
# exports with the two heads in different shards get one pass per shard ----
groups = {}
# linears take fp16 scales; the embedding path creates them in params_dtype
for base, scale_dtype in (("lm_head", torch.float16), (emb_base, torch.bfloat16)):
    groups.setdefault(shards[base + ".weight"], []).append((base, scale_dtype))

for big, bases in groups.items():
    hdr, _ = read_header(d + big)
    todo = [(base, dt) for base, dt in bases if not packed_already(hdr, base, big, ".bak-orig")]

    if todo:
        add = {}
        with safe_open(d + big, framework="pt") as f:
            for base, scale_dtype in todo:
                w = f.get_tensor(base + ".weight")
                packed, err = pack(base, w, HEAD_BITS, scale_dtype)
                print(f"  {base}.weight: {tuple(w.shape)} int{HEAD_BITS} g{GROUP}, round-trip rel error {err:.4f}")
                assert err < 0.01, f"quantization error too high for {base}.weight, aborting"
                add.update(packed)
                del w, packed

        print(f"rewriting {big} (streaming)")
        keep_original(d + big)
        tmp = d + big + ".tmp"
        stream_rewrite(d + big, tmp, drop={base + ".weight" for base, _ in bases}, add=add)
        publish(tmp, d + big)
        del add

    for base, _ in bases:
        index_packed(wm, base, big)

# ---- MTP module (small shard, fits in RAM) ----
mtp_shards = {shards[m + ".weight"] for m in LINEARS}
assert len(mtp_shards) == 1, f"mtp weights span several shards: {mtp_shards}"
mtp_shard = mtp_shards.pop()
print(f"mtp linears live in {mtp_shard}, quantizing to int{MTP_BITS} g{GROUP}")

tensors = {}
with safe_open(d + mtp_shard, framework="pt") as f:
    mtp_meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
changed = False
for m in LINEARS:
    if packed_already(tensors, m, mtp_shard, ".bak-orig"):
        continue
    w = tensors.pop(m + ".weight")
    packed, err = pack(m, w, MTP_BITS, torch.float16)
    print(f"  {m}: {tuple(w.shape)} round-trip rel error {err:.4f}")
    tensors.update(packed)
    del w, packed
    changed = True

if changed:
    # The head pass may have kept this shard already (single-shard exports): the first
    # .bak-orig is the pristine one and stays that way.
    keep_original(d + mtp_shard)
    save_tensors(tensors, d + mtp_shard, mtp_meta or {"format": "pt"})
del tensors

for m in LINEARS:
    index_packed(wm, m, mtp_shard)

# ---- config.json ----
cfg_path = d + "config.json"
backup_once(cfg_path, ".bak-quant")
qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head" and i not in LINEARS]
clone_group(qc, "lm_head", HEAD_BITS)
clone_group(qc, "embed", HEAD_BITS)
clone_group(qc, "mtp-keep-fc" if KEEP_FC else "mtp", MTP_BITS)
write_json(cfg_path, c)

# The index is the commit point, so it goes last.
backup_once(idx_path, ".bak-quant")
write_json(idx_path, idx)
print("done")
