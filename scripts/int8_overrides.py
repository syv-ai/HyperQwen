#!/usr/bin/env python3
"""Print the --hf-overrides JSON that gives INT8_LAYERS int8 activations on a compressed-tensors int4 checkpoint.

vLLM 0.34 serves int8 activations on int weights through compressed-tensors' CompressedTensorsWNA8Int scheme
(Humming kernels), chosen per config group when that group's input_activations are int8, symmetric, dynamic and
per token. This copies the checkpoint's own quantization_config and adds one group after the existing ones: the
weight spec of the group that targets "Linear", with targets ["re:.*(?:<INT8_LAYERS>).*"] and int8 per-token
activations. compressed-tensors takes the first target that matches a layer name, so layers the checkpoint already
puts in a group of their own (an int8 lm_head, the embeddings, the MTP module) keep it, and the ignore list still
wins over every group. INT8_LAYERS is a Python regex searched in the layer name, as on 0.33 (mlp, gate_up, "." for
every linear).

  python3 scripts/int8_overrides.py <model dir> <INT8_LAYERS>

Exits 2 with the reason when the checkpoint is not one this applies to.
"""
import copy
import json
import os
import re
import sys

ACT = {"num_bits": 8, "type": "int", "symmetric": True, "dynamic": True, "strategy": "token"}


def fail(msg):
    print(f"[start_qwen] INT8_ACT=int8: {msg}", file=sys.stderr)
    sys.exit(2)


def search_target(layers):
    try:
        flags = re.compile(layers).flags
        leading = r"\(\?[aiLmsux]+\)|\(\?#(?:\\[\s\S]|[^\\)])*\)"
        newline = ""
        if flags & re.VERBOSE:
            leading += r"|[ \t\n\r\v\f]+|#[^\n]*(?:\n|$)"
            newline = "\n"
        prefix = re.match(f"(?:{leading})*", layers)[0]
        pattern = f"{prefix}{newline}.*(?:{layers[len(prefix):]}{newline}).*"
        re.compile(pattern)
    except (re.error, ValueError) as e:
        fail(f"INT8_LAYERS={layers!r} is not a regex: {e}")
    return f"re:{pattern}"


def overrides(config, layers):
    q = config.get("quantization_config")
    if not isinstance(q, dict) or q.get("quant_method") != "compressed-tensors":
        fail("needs a compressed-tensors checkpoint (quantization_config.quant_method), "
             f"this one is {None if not isinstance(q, dict) else q.get('quant_method')!r}")
    if q.get("format") != "pack-quantized":
        fail(f"needs the pack-quantized format, this checkpoint is {q.get('format')!r}")
    groups = q.get("config_groups") or {}
    base = next((g for g in groups.values() if "Linear" in (g.get("targets") or [])), None)
    if base is None:
        fail("no config group targets Linear")
    w = base.get("weights") or {}
    if w.get("type") != "int" or w.get("dynamic"):
        fail(f"the Linear group's weights are not static int ({w.get('type')!r})")
    if base.get("input_activations") is not None:
        fail("the Linear group already declares input activations; nothing to add")
    target = search_target(layers)
    q = copy.deepcopy(q)
    name = "group_int8_act"
    while name in q["config_groups"]:
        name += "_"
    q["config_groups"][name] = dict(copy.deepcopy(base), targets=[target], input_activations=ACT)
    return {"quantization_config": q}


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    model, layers = sys.argv[1], sys.argv[2]
    if not layers:
        fail("INT8_LAYERS is empty; set a regex (mlp, gate_up, or . for every linear)")
    path = os.path.join(model, "config.json")
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except OSError as e:
        fail(f"cannot read {path} ({e.strerror}); MODEL must be a local checkpoint directory")
    print(json.dumps(overrides(config, layers), separators=(",", ":")))


if __name__ == "__main__":
    main()
