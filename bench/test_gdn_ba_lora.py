#!/usr/bin/env python3
"""CPU/source regression for the GDN ba fast path and LoRA dispatch.

Needs CPU torch and patched vLLM source, not a GPU or a model. VLLM_SOURCE
selects a package directory; otherwise use the installed package. Execute
the actual gate, projection branch and op functions with small layer/config
stand-ins. The wrapper uses vLLM's actual LoRA attribute delegation method.
"""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


class Unquantized:
    pass


class Plain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(6, 3))
        self.bias = None
        self.quant_method = Unquantized()

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight), None


class GdnBaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = os.environ.get("VLLM_SOURCE")
        if selected:
            root = Path(selected)
        else:
            spec = importlib.util.find_spec("vllm")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError("set VLLM_SOURCE to the patched vllm package directory")
            root = Path(next(iter(spec.submodule_search_locations)))
        path = root / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenGatedDeltaNetAttention")
        constructor = next(n for n in layer.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assignment = next(n for n in ast.walk(constructor) if isinstance(n, ast.Assign)
                          and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == "ba_proj_pad_small_m")
        cls.gate_code = compile(ast.Expression(assignment.value), str(path), "eval")
        forward = next(n for n in layer.body if isinstance(n, ast.FunctionDef) and n.name == "forward_cuda")
        branch = next(n for n in forward.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Attribute) and n.test.attr == "ba_proj_pad_small_m")
        project = ast.FunctionDef(name="project", args=ast.arguments(posonlyargs=[],
                                  args=[ast.arg(arg="self"), ast.arg(arg="hidden_states")],
                                  kwonlyargs=[], kw_defaults=[], defaults=[]),
                                  body=[branch, ast.Return(ast.Name(id="ba", ctx=ast.Load()))], decorator_list=[])
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ("gdn_ba_proj", "gdn_ba_proj_fake")]
        getter_tree = ast.parse((root / "lora/layers/base.py").read_text(encoding="utf-8"))
        base = next(n for n in getter_tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseLayerWithLoRA")
        getter = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__getattr__")
        delegated = ast.ClassDef(name="Delegated", bases=[ast.Name(id="Module", ctx=ast.Load())],
                                 keywords=[], body=[getter], decorator_list=[])
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 *functions, project, delegated], type_ignores=[])
        cls.scope = dict(torch=torch, Module=torch.nn.Module)
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), cls.scope)

    def select(self, base, *, lora=None, knob=True, cuda=True):
        return eval(self.gate_code, dict(self=NS(in_proj_ba=base),
                    envs=NS(VLLM_GDN_BA_PAD_SMALL_M=knob), vllm_config=NS(lora_config=lora),
                    current_platform=NS(is_cuda=lambda: cuda), UnquantizedLinearMethod=Unquantized))

    def project(self, base, enabled, value):
        calls = []
        def op(*args):
            calls.append(True)
            return self.scope["gdn_ba_proj"](*args)
        function = self.scope["project"]
        original = function.__globals__["torch"]
        function.__globals__["torch"] = NS(nn=torch.nn, ops=NS(vllm=NS(gdn_ba_proj=op)))
        try:
            result = function(NS(in_proj_ba=base, ba_proj_pad_small_m=enabled), value)
        finally:
            function.__globals__["torch"] = original
        return result, calls

    def test_lora_enabled_before_wrapper_replacement_keeps_adapter_contribution(self):
        plain = Plain()
        enabled = self.select(plain, lora=object())
        class Adapter(self.scope["Delegated"]):
            def __init__(self):
                super().__init__()
                self.base_layer = plain

            def forward(self, value):
                base, bias = self.base_layer(value)
                low_rank = (value @ torch.ones(3, 2)) @ torch.ones(2, 6)
                return base + low_rank, bias
        wrapped = Adapter()
        self.assertIs(wrapped.weight, plain.weight)
        value = torch.ones(4, 3)
        result, calls = self.project(wrapped, enabled, value)
        torch.testing.assert_close(result, torch.full((4, 6), 6.0))
        self.assertFalse(enabled)
        self.assertEqual(calls, [])

    def test_plain_projection_preserves_values_and_shape_across_padding_window(self):
        plain = Plain()
        generator = torch.Generator().manual_seed(17)
        plain.weight.data.copy_(torch.randn(6, 3, generator=generator))
        self.assertTrue(self.select(plain))
        for rows in (0, 1, 2, 3, 4, 5, 8):
            with self.subTest(rows=rows):
                value = torch.randn(rows, 3, generator=generator)
                result, calls = self.project(plain, True, value)
                torch.testing.assert_close(result, torch.nn.functional.linear(value, plain.weight))
                self.assertEqual(calls, [True])
                self.assertEqual(self.scope["gdn_ba_proj_fake"](value, plain.weight).shape, result.shape)

    def test_existing_opt_out_platform_quantization_and_bias_guards_remain(self):
        plain = Plain()
        self.assertFalse(self.select(plain, knob=False))
        self.assertFalse(self.select(plain, cuda=False))
        plain.quant_method = object()
        self.assertFalse(self.select(plain))
        plain.quant_method = Unquantized()
        plain.bias = torch.nn.Parameter(torch.ones(6))
        self.assertFalse(self.select(plain))


if __name__ == "__main__":
    unittest.main()
