"""Requantize lm_head, embed_tokens and the MTP module to int8/int4
(group-128, symmetric) in compressed-tensors pack-quantized format, in place.

Same math and output as quant_lm_head.py / quant_embed.py / quant_mtp.py, but
for checkpoints those three cannot handle:

  - single-shard checkpoints. They read a whole shard into a dict before
    rewriting it; philbert440/Qwen3.8-27B-Uncensored-* ships one 18.6 GB
    model.safetensors (2384 tensors), which does not fit in RAM here. This
    streams the shard tensor-by-tensor instead, copying untouched tensors as
    raw bytes, so peak RSS is a few GB regardless of shard size.

  - asymmetric bodies. The three scripts deepcopy config_groups.group_0 and
    override only num_bits/targets. AWQ exports have symmetric=false and
    zp_dtype=torch.int8, so the cloned group would declare asymmetric quant
    for the symmetric tensors written here and vLLM would look for a
    weight_zero_point that does not exist. The groups written below always say
    symmetric=true / zp_dtype=null.

Usage: venv/bin/python prepare/quant_heads_stream.py /path/to/model [--mtp-bits 8|4] [--keep-fc]

This is what the uncensored checkpoint needs (prepare/fetch_uncensored.py); the base
model is 7 shards and symmetric, so quant_lm_head/quant_embed/quant_mtp serve it fine.

The rewritten shards replace the originals; the first pre-quant copy of each stays
next to it as <shard>.bak-orig (a hardlink to the original file, as the old rename kept
it: no copy of an 18.6 GB shard, never overwritten, so it keeps holding the bf16 lm_head
that drafter/gptq_lm_head.py reads), and config.json and the safetensors index are
backed up as .bak-quant.

Every file goes through prepare/atomic_publish.py (a temp file and a rename), and the
index is written last: docker/prepare.sh's state() reads only the index, so a killed run
leaves the step pending, and the next run completes it, reusing a shard that already
holds the packed tensors (#195).
"""

import copy
import json
import os
import struct
import sys

import torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

from atomic_publish import backup_once, publish, save_tensors, write_json

GROUP = 128
HEAD_BITS = 8
MTP_BITS = int(sys.argv[sys.argv.index("--mtp-bits") + 1]) if "--mtp-bits" in sys.argv else 8
KEEP_FC = "--keep-fc" in sys.argv
ROWS = 16384  # quantize this many rows at a time, to bound peak RSS
SUFFIXES = ("weight_packed", "weight_scale", "weight_shape")

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

DTYPE_STR = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32",
    torch.int8: "I8", torch.int32: "I32", torch.int64: "I64", torch.uint8: "U8",
}


def quantize(w, bits):
    """int-N group-wise symmetric quant, row-chunked. Returns packed/scale/err."""
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = w.shape
    assert in_f % GROUP == 0, w.shape
    packed_parts, scale_parts = [], []
    num, den = 0.0, 0.0
    for lo in range(0, out_f, ROWS):
        chunk = w[lo:lo + ROWS].to(torch.float32)
        g = chunk.reshape(chunk.shape[0], in_f // GROUP, GROUP)
        s = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-10)
        q = torch.clamp(torch.round(g / s), -qmax - 1, qmax).to(torch.int8)
        deq = (q.to(torch.float32) * s).reshape(chunk.shape[0], in_f)
        num += (deq - chunk).pow(2).sum().item()
        den += chunk.pow(2).sum().item()
        packed_parts.append(pack_to_int32(q.reshape(chunk.shape[0], in_f), bits, packed_dim=1).contiguous())
        scale_parts.append(s.squeeze(-1).contiguous())
        del chunk, g, q, deq
    return torch.cat(packed_parts), torch.cat(scale_parts), (num / den) ** 0.5


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

lm_key = "lm_head.weight"
emb_key = next(k for k in shards if k.endswith("embed_tokens.weight"))

# ---- lm_head + embed_tokens: one streaming pass per shard that holds them.
# Single-shard exports (the original case) land in one group; multi-shard
# exports with the two heads in different shards get one pass per shard ----
groups = {}
for key, scale_dtype in ((lm_key, torch.float16), (emb_key, torch.bfloat16)):
    groups.setdefault(shards[key], []).append((key, scale_dtype))

for big, keys in groups.items():
    hdr, _ = read_header(d + big)
    todo = []
    for key, scale_dtype in keys:
        base = key[:-len(".weight")]
        packed_names = [f"{base}.{s}" for s in SUFFIXES]
        if key not in hdr:
            # A killed run published the shard but not the index: finish that run.
            if not all(k in hdr for k in packed_names):
                sys.exit(f"{big} holds neither {key} nor its packed form; "
                         f"restore it from {big}.bak-orig")
            print(f"  {key}: already packed in {big} (completing an interrupted run)")
            continue
        todo.append((key, scale_dtype, base))

    if todo:
        add = {}
        with safe_open(d + big, framework="pt") as f:
            for key, scale_dtype, base in todo:
                w = f.get_tensor(key)
                out_f, in_f = w.shape
                packed, scale, err = quantize(w, HEAD_BITS)
                print(f"  {key}: {(out_f, in_f)} int{HEAD_BITS} g{GROUP}, round-trip rel error {err:.4f}")
                assert err < 0.01, f"quantization error too high for {key}, aborting"
                add[base + ".weight_packed"] = packed
                # linears take fp16 scales; the embedding path creates them in params_dtype
                add[base + ".weight_scale"] = scale.to(scale_dtype)
                add[base + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
                del w, packed, scale

        print(f"rewriting {big} (streaming)")
        keep_original(d + big)
        tmp = d + big + ".tmp"
        stream_rewrite(d + big, tmp, drop={k for k, _ in keys}, add=add)
        publish(tmp, d + big)
        del add

    for key, _ in keys:
        base = key[:-len(".weight")]
        wm.pop(key, None)  # already gone when the index of a finished run is re-read
        for s in SUFFIXES:
            wm[f"{base}.{s}"] = big

# ---- MTP module (small shard, fits in RAM) ----
mtp_shards = {shards[m + ".weight"] for m in MTP_LINEARS}
assert len(mtp_shards) == 1, f"mtp weights span several shards: {mtp_shards}"
mtp_shard = mtp_shards.pop()
print(f"mtp linears live in {mtp_shard}, quantizing to int{MTP_BITS} g{GROUP}")

tensors = {}
with safe_open(d + mtp_shard, framework="pt") as f:
    mtp_meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
changed = False
for m in MTP_LINEARS:
    packed_names = [f"{m}.{s}" for s in SUFFIXES]
    if m + ".weight" not in tensors:
        # A killed run published the shard but not the index: finish that run.
        if not all(k in tensors for k in packed_names):
            sys.exit(f"{mtp_shard} holds neither {m}.weight nor its packed form; "
                     f"restore it from {mtp_shard}.bak-orig")
        print(f"  {m}: already packed in {mtp_shard} (completing an interrupted run)")
        continue
    w = tensors.pop(m + ".weight")
    out_f, in_f = w.shape
    packed, scale, err = quantize(w, MTP_BITS)
    print(f"  {m}: {(out_f, in_f)} round-trip rel error {err:.4f}")
    tensors[m + ".weight_packed"] = packed
    tensors[m + ".weight_scale"] = scale.to(torch.float16)
    tensors[m + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    del w, packed, scale
    changed = True

if changed:
    # The head pass may have kept this shard already (single-shard exports): the first
    # .bak-orig is the pristine one and stays that way.
    keep_original(d + mtp_shard)
    save_tensors(tensors, d + mtp_shard, mtp_meta or {"format": "pt"})
del tensors

for m in MTP_LINEARS:
    wm.pop(m + ".weight", None)  # already gone when the index of a finished run is re-read
    for s in SUFFIXES:
        wm[f"{m}.{s}"] = mtp_shard

# ---- config.json ----
cfg_path = d + "config.json"
c = json.load(open(cfg_path))
backup_once(cfg_path, ".bak-quant")
qc = c["quantization_config"]


def group(bits, targets):
    g = copy.deepcopy(qc["config_groups"]["group_0"])
    g["targets"] = targets
    w = g["weights"]
    w["num_bits"] = bits
    # tensors written here are symmetric with no zero point, regardless of
    # what the body group uses (AWQ bodies are asymmetric).
    w["symmetric"] = True
    w["zp_dtype"] = None
    w["group_size"] = GROUP
    w["strategy"] = "group"
    w["type"] = "int"
    return g


qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head" and i not in MTP_LINEARS]
qc["config_groups"]["group_1"] = group(HEAD_BITS, ["re:.*lm_head$"])
qc["config_groups"]["group_2"] = group(HEAD_BITS, ["re:.*embed_tokens$"])
qc["config_groups"]["group_3"] = group(
    MTP_BITS, ["re:^mtp\\.layers\\..*"] if KEEP_FC else ["re:^mtp\\..*"]
)
write_json(cfg_path, c)

# The index is the commit point, so it goes last.
backup_once(idx_path, ".bak-quant")
write_json(idx_path, idx)
print("done")
