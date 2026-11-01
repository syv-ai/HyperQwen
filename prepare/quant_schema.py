"""What prepare/'s in-place quant scripts read and write, in one place.

load_config() refuses a checkpoint these scripts cannot extend, before they write anything
(#241). pack(), packed_already(), index_packed() and clone_group() are the tensors, the
killed-run check, the index entries and the config group that every script writes the
same way. The file I/O is atomic_publish.py's.

prepare/quant_lm_head.py, quant_embed.py, quant_mtp.py and quant_heads_stream.py do not
build a quantization config. They read the one already in config.json, clone
`config_groups.group_0` and append a group of their own. A checkpoint whose config.json is
not already in compressed-tensors pack-quantized form has neither group_0 nor an `ignore`
list, so that read raises KeyError -- and because the shard is written before config.json
and the index (that order is deliberate, see atomic_publish.py), the error arrives after
the shard has already been replaced.

A native AutoRound export is the usual cause: its `quantization_config` carries
bits/group_size/sym/data_type and `quant_method: "auto-round"`, which is not a shape these
scripts can extend -- it has to be converted, not modified. The directory name is not a
guide either way: the base model they were written for is a compressed-tensors re-export
that keeps "-AutoRound" in its name.

load_config() runs before the first shard is opened, so an unsupported checkpoint gets one
line and an untouched directory instead of a half-rewritten one.
"""

import copy
import json
import sys

GROUP = 128
SUFFIXES = ("weight_packed", "weight_scale", "weight_shape")
# The base checkpoint's ignore list names the MTP module's linears in this order.
MTP_LINEARS = ("mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj",
               "mtp.layers.0.mlp.up_proj", "mtp.layers.0.self_attn.q_proj",
               "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
               "mtp.layers.0.self_attn.o_proj")
# The groups the scripts add, each a clone of group_0: what it covers -> (name, targets).
GROUPS = {"lm_head": ("group_1", ["re:.*lm_head$"]),
          "embed": ("group_2", ["re:.*embed_tokens$"]),
          "mtp": ("group_3", ["re:^mtp\\..*"]),
          "mtp-keep-fc": ("group_3", ["re:^mtp\\.layers\\..*"])}

SEE = ('prepare/README.md, "Which checkpoints these scripts modify" -- they extend an '
       "existing\n  compressed-tensors config, they do not convert one.")


def _reject(path, reason, detail, qc):
    sys.exit(f"{path}: {reason}\n  {detail}\n"
             f"  quantization_config has: {', '.join(sorted(qc)) or '(nothing)'}\n"
             f"  {SEE}\n  Nothing was written.")


def load_config(d):
    """Read <d>/config.json and return (config, quantization_config), after checking that
    it is the compressed-tensors schema the in-place scripts extend. Exits non-zero, before
    any write, when it is not."""
    path = d + "config.json"
    try:
        with open(path) as f:
            c = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{path} does not exist: {d} is not a model directory")
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON ({e}); an interrupted run may have left it half-written")

    qc = c.get("quantization_config")
    if not isinstance(qc, dict):
        sys.exit(f"{path} has no quantization_config, so there is no group_0 to extend.\n"
                 f"  {SEE}\n  Nothing was written.")

    method = qc.get("quant_method")
    if method is not None and method != "compressed-tensors":
        _reject(path, f"quant_method is {method!r}, not 'compressed-tensors'",
                "this looks like a native quantizer export (an AutoRound one carries bits/\n"
                "  group_size/sym/data_type). It has to be converted to compressed-tensors\n"
                "  pack-quantized first; these scripts cannot do that. A checkpoint already\n"
                "  prepared for HyperQwen needs none of prepare/ at all.", qc)

    if not isinstance(qc.get("ignore"), list):
        _reject(path, "quantization_config has no 'ignore' list",
                "the scripts rewrite it (they drop lm_head and the mtp.* linears from it).", qc)

    groups = qc.get("config_groups")
    if not isinstance(groups, dict) or "group_0" not in groups:
        _reject(path, "quantization_config has no config_groups.group_0",
                "every group these scripts add is a clone of group_0.", qc)

    print(f"config.json: compressed-tensors, extending config_groups.group_0 "
          f"({len(groups)} existing group(s), {len(qc['ignore'])} ignored modules)")
    return c, qc


def pack(base, w, bits, scale_dtype, rows=16384):
    """Quantize w ([out, in], `in` a multiple of GROUP) to int-`bits` by symmetric
    round-to-nearest over GROUP-wide groups. Returns the compressed-tensors entries
    <base>.weight_packed/.weight_scale/.weight_shape and the round-trip relative error.

    scale_dtype has no default: the linears take fp16 scales, and the embedding path
    creates its scales in params_dtype (bf16). Rows are quantized `rows` at a time to
    bound peak RSS; rows are independent, so the bytes do not depend on `rows`."""
    import torch
    from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = w.shape
    assert in_f % GROUP == 0, (base, tuple(w.shape))
    packed, scales, num, den = [], [], 0.0, 0.0
    for lo in range(0, out_f, rows):
        x = w[lo:lo + rows].to(torch.float32)
        g = x.reshape(len(x), in_f // GROUP, GROUP)
        s = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-10)
        q = torch.clamp(torch.round(g / s), -qmax - 1, qmax).to(torch.int8)
        num += ((q.to(torch.float32) * s).reshape(x.shape) - x).pow(2).sum().item()
        den += x.pow(2).sum().item()
        packed.append(pack_to_int32(q.reshape(x.shape), bits, packed_dim=1))
        scales.append(s.squeeze(-1))
    return ({f"{base}.weight_packed": torch.cat(packed).contiguous(),
             f"{base}.weight_scale": torch.cat(scales).to(scale_dtype).contiguous(),
             f"{base}.weight_shape": torch.tensor([out_f, in_f], dtype=torch.int64)},
            (num / den) ** 0.5 if den else float("nan"))


def packed_already(names, base, shard, backup):
    """Whether <shard>, holding the tensors `names`, has <base> packed and not
    <base>.weight: a killed run published the shard but not the index, and this run
    completes it (#195). Exits when the shard holds neither form."""
    if base + ".weight" in names:
        return False
    if not all(f"{base}.{s}" in names for s in SUFFIXES):
        sys.exit(f"{shard} holds neither {base}.weight nor its packed form; restore it from {shard}{backup}")
    print(f"  {base}: already packed in {shard} (completing an interrupted run)")
    return True


def index_packed(wm, base, shard):
    """Point the index's weight_map at <base>'s packed entries in <shard>."""
    wm.pop(base + ".weight", None)
    for s in SUFFIXES:
        wm[f"{base}.{s}"] = shard


def clone_group(qc, part, bits):
    """Add the config group for `part` (a GROUPS key): group_0 with its targets, declaring
    what pack() writes. That is symmetric with no zero point whatever the body group says
    (an AWQ body is asymmetric, #197)."""
    name, targets = GROUPS[part]
    g = copy.deepcopy(qc["config_groups"]["group_0"])
    g["targets"] = list(targets)
    g["weights"].update(num_bits=bits, symmetric=True, zp_dtype=None, group_size=GROUP,
                        strategy="group", type="int")
    qc["config_groups"][name] = g
