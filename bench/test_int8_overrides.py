#!/usr/bin/env python3
"""scripts/int8_overrides.py: the group it adds, what it leaves alone, and what it refuses.

CPU only. With vLLM importable (the image), the override also goes through compressed-tensors' own config and
target matching, so the layers INT8_LAYERS names get int8 per-token activations and nothing else changes group.
  python3 bench/test_int8_overrides.py
"""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "int8_overrides.py")
W4 = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128, "dynamic": False,
      "actorder": None, "block_structure": None, "observer": "memoryless_minmax", "observer_kwargs": {}}
W8 = dict(W4, num_bits=8)
CONFIG = {"quantization_config": {
    "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
    "config_groups": {
        "group_0": {"format": "pack-quantized", "targets": ["Linear"], "weights": W4, "input_activations": None},
        "group_1": {"format": "pack-quantized", "targets": ["re:.*lm_head$"], "weights": W8, "input_activations": None},
        "group_3": {"format": "pack-quantized", "targets": ["re:^mtp\\..*"], "weights": W8, "input_activations": None},
    },
    "ignore": ["model.visual.blocks.0.mlp.linear_fc1", "model.language_model.layers.0.linear_attn.in_proj_a"],
}}
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    fails += not ok


def run(config, layers):
    with tempfile.TemporaryDirectory() as d:
        if config is not None:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f)
        p = subprocess.run([sys.executable, SCRIPT, d, layers], capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr


rc, out, err = run(CONFIG, "mlp")
check("mlp: exits 0", rc == 0, err)
ov = json.loads(out)["quantization_config"] if rc == 0 else {}
g = ov.get("config_groups", {})
check("mlp: the original groups first, in order, untouched",
      list(g)[:3] == ["group_0", "group_1", "group_3"]
      and all(g[k] == CONFIG["quantization_config"]["config_groups"][k] for k in ("group_0", "group_1", "group_3")))
add = g.get("group_int8_act", {})
check("mlp: one added group, last", list(g)[3:] == ["group_int8_act"])
check("mlp: targets the regex as a layer-name search", add.get("targets") == ["re:.*(?:mlp).*"], add.get("targets"))
check("mlp: group_0's weights", add.get("weights") == W4)
check("mlp: int8 symmetric dynamic per-token activations",
      add.get("input_activations") == {"num_bits": 8, "type": "int", "symmetric": True, "dynamic": True,
                                       "strategy": "token"}, add.get("input_activations"))
check("mlp: ignore list and top-level keys unchanged",
      ov.get("ignore") == CONFIG["quantization_config"]["ignore"]
      and {k: v for k, v in ov.items() if k != "config_groups"}
      == {k: v for k, v in CONFIG["quantization_config"].items() if k != "config_groups"})

regex_cases = ("(?i)MLP", "(?i)^MODEL", "(?x) mlp # target the MLP", "(?x) # comment\n (?i) MLP # trailing",
               "(?# start)(?i)MLP", r"(?i)(mlp)\.\1$")
layer_names = ("model.language_model.layers.0.mlp.gate_up_proj", "model.layers.0.self_attn.qkv_proj",
               "model.layers.0.mlp.MLP", "lm_head")
for layers in regex_cases:
    rc, out, err = run(CONFIG, layers)
    target = json.loads(out)["quantization_config"]["config_groups"]["group_int8_act"]["targets"][0] if rc == 0 else ""
    try:
        same_matches = all(bool(re.match(target[3:], name)) == bool(re.search(layers, name)) for name in layer_names)
    except re.error:
        same_matches = False
    check(f"{layers!r}: generated target preserves layer-name search", rc == 0 and same_matches, target or err)

refusals = [
    ("no config.json", None, "mlp"),
    ("not compressed-tensors", {"quantization_config": {"quant_method": "gptq"}}, "mlp"),
    ("no Linear group", {"quantization_config": dict(CONFIG["quantization_config"], config_groups={})}, "mlp"),
    ("input activations already set",
     {"quantization_config": dict(copy.deepcopy(CONFIG["quantization_config"]), config_groups={
         "group_0": dict(CONFIG["quantization_config"]["config_groups"]["group_0"],
                         input_activations={"num_bits": 8, "type": "int"})})}, "mlp"),
    ("an invalid regex", CONFIG, "("),
    ("incompatible regex flags", CONFIG, "(?a)(?u)mlp"),
    ("empty INT8_LAYERS", CONFIG, ""),
]
for name, config, layers in refusals:
    rc, out, err = run(config, layers)
    check(f"refuses {name} (exit 2, a reason, no JSON)", rc == 2 and "INT8_ACT=int8" in err and not out.strip(),
          f"rc {rc} err {err.strip()[:120]!r}")

try:
    import torch
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        find_matched_target,
        should_ignore_layer,
    )
except ImportError as e:
    print(f"skip vLLM matching ({e})")
else:
    mlp_layers = {"model.language_model.layers.0.mlp.gate_up_proj", "model.language_model.layers.0.mlp.down_proj"}
    for layers, want_int8 in (("mlp", mlp_layers), ("(?i)MLP", mlp_layers), ("(?x) mlp # MLP", mlp_layers),
                              (".", {"model.language_model.layers.0.mlp.gate_up_proj",
                                     "model.language_model.layers.0.mlp.down_proj",
                                     "model.language_model.layers.3.self_attn.qkv_proj",
                                     "model.language_model.layers.0.linear_attn.in_proj_qkvz"})):
        q = json.loads(run(CONFIG, layers)[1])["quantization_config"]
        c = CompressedTensorsConfig.from_config(q)
        for n in ("model.language_model.layers.0.mlp.gate_up_proj", "model.language_model.layers.0.mlp.down_proj",
                  "model.language_model.layers.3.self_attn.qkv_proj",
                  "model.language_model.layers.0.linear_attn.in_proj_qkvz", "lm_head", "mtp.layers.0.mlp.down_proj"):
            t = find_matched_target(layer_name=n, module=torch.nn.Linear(1, 1), targets=c.target_scheme_map.keys(),
                                    fused_mapping=c.packed_modules_mapping)
            act = c.target_scheme_map[t].get("input_activations")
            got = act is not None and act.num_bits == 8
            check(f"vLLM {layers!r}: {n} int8={n in want_int8}", got == (n in want_int8), f"target {t}")
        check(f"vLLM {layers!r}: an ignored layer stays ignored",
              should_ignore_layer("model.visual.blocks.0.mlp.linear_fc1", ignore=c.ignore,
                                  fused_mapping=c.packed_modules_mapping))

print(f"{'FAIL' if fails else 'PASS'}: {fails} failed")
sys.exit(1 if fails else 0)
