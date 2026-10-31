#!/usr/bin/env python3
"""CPU/source regression for GDN checkpoint opt-in. No torch import or GPU required.

VLLM_SOURCE names a patched vllm package directory; otherwise use the installed
package's source without importing its runtime. Execute actual source methods,
with only the unrelated base-layer/config objects replaced by small stand-ins.
"""
import ast
from dataclasses import dataclass, replace
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


def function(path, name, owner=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


@dataclass(frozen=True)
class Spec:
    num_prefill_checkpoint_blocks: int = 0
    prefill_checkpoint_alignment: int | None = None


class Base:
    def get_kv_cache_spec(self, config):
        return self.original


class CheckpointSafetyTests(unittest.TestCase):
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
        getter = function(root / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
                          "get_kv_cache_spec", "QwenGatedDeltaNetAttention")
        wrapper = ast.ClassDef(name="CheckedLayer", bases=[ast.Name(id="Base", ctx=ast.Load())],
                               keywords=[], body=[getter], decorator_list=[])
        interface = root / "v1/kv_cache_interface.py"
        nodes = [function(interface, name) for name in
                 ("get_mamba_prefill_checkpoint_position", "is_mamba_prefill_checkpoint_valid")]
        nodes += [function(root / "model_executor/layers/mamba/checkpoint.py", "compute_mamba_prefill_checkpoints"),
                  function(root / "v1/core/single_type_kv_cache_manager.py", "_cache_partial_tail_block", "MambaManager")]
        cls.envs = NS(VLLM_GDN_PREFILL_CHECKPOINTS=False)
        cls.scope = dict(Base=Base, MambaSpec=Spec, replace=replace, envs=cls.envs,
                         current_platform=NS(is_cuda=lambda: True),
                         _resolve_gdn_prefill_backend=lambda config: ("triton", "triton"),
                         cdiv=lambda x, y: (x + y - 1) // y)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 wrapper, *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(root / "checkpoint_safety_source"), "exec"), cls.scope)

    def layer(self, spec):
        layer = self.scope["CheckedLayer"]()
        layer.original = spec
        return layer

    def test_default_off_preserves_the_original_spec(self):
        self.envs.VLLM_GDN_PREFILL_CHECKPOINTS = False
        spec = Spec()
        self.assertIs(self.layer(spec).get_kv_cache_spec(object()), spec)

    def test_non_mamba_spec_is_unchanged(self):
        self.envs.VLLM_GDN_PREFILL_CHECKPOINTS = True
        spec = object()
        self.assertIs(self.layer(spec).get_kv_cache_spec(object()), spec)

    def test_enabled_export_reserves_exactly_one_interior_block(self):
        self.envs.VLLM_GDN_PREFILL_CHECKPOINTS = True
        spec = Spec()
        config = NS(cache_config=NS(mamba_cache_mode="align"))
        selected = self.layer(spec).get_kv_cache_spec(config)
        self.assertEqual(selected.num_prefill_checkpoint_blocks, 1)
        self.assertEqual(selected.prefill_checkpoint_alignment, 1)
        self.assertEqual(spec, Spec())

    def test_unsupported_backend_and_cache_mode_refuse_before_allocation(self):
        self.envs.VLLM_GDN_PREFILL_CHECKPOINTS = True
        config = NS(cache_config=NS(mamba_cache_mode="all"))
        with self.assertRaisesRegex(RuntimeError, "mamba_cache_mode=align"):
            self.layer(Spec()).get_kv_cache_spec(config)
        config.cache_config.mamba_cache_mode = "align"
        self.scope["current_platform"] = NS(is_cuda=lambda: False)
        try:
            with self.assertRaisesRegex(RuntimeError, "CUDA Triton/FLA"):
                self.layer(Spec()).get_kv_cache_spec(config)
        finally:
            self.scope["current_platform"] = NS(is_cuda=lambda: True)
        self.scope["_resolve_gdn_prefill_backend"] = lambda config: ("flashinfer", "flashinfer")
        try:
            with self.assertRaisesRegex(RuntimeError, "CUDA Triton/FLA"):
                self.layer(Spec()).get_kv_cache_spec(config)
        finally:
            self.scope["_resolve_gdn_prefill_backend"] = lambda config: ("triton", "triton")

    def test_allocator_valid_positions_cannot_be_row_ends(self):
        compute = self.scope["compute_mamba_prefill_checkpoints"]
        eligible = 0
        for drop in (False, True):
            for start in (0, 2048, 4096):
                for length in range(128, 16385, 128):
                    offsets, _ = compute([start + length], [length], 2048, 832, 1, drop, 2048)
                    if offsets[0]:
                        eligible += 1
                        self.assertLess(offsets[0], length)
        self.assertGreater(eligible, 0)
        offsets, _ = compute([8192], [8192], 2048, 2048, 1, False, 2048)
        self.assertEqual(offsets, [6144])

    def test_manager_assumes_the_backend_wrote_the_reserved_checkpoint(self):
        # This is why silently declining in the writer is unsafe: the manager has
        # no backend acknowledgement and publishes the reserved physical block.
        calls = []
        pool = NS(hash_block_size=2048,
                  cache_partial_block=lambda **kwargs: calls.append(kwargs) or "published")
        manager = NS(block_pool=pool, _checkpoints={"r": (6144, 2)},
                     req_to_blocks={"r": [None] * 2 + [NS(block_hash_num_tokens=None, block_id=8)]},
                     kv_cache_group_id=0, block_size=2048, drop_eagle_checkpoint_block=False,
                     cache_hit_alignment_tokens=2048,
                     kv_cache_spec=Spec(),
                     reachable_block_mask=lambda *args, **kwargs: [True])
        request = NS(request_id="r", num_prompt_tokens=8192)
        result = self.scope["_cache_partial_tail_block"](manager, request, 8192, None)
        self.assertEqual(result, "published")
        self.assertEqual(calls[0]["num_tokens"], 6144)
        self.assertEqual(calls[0]["block"].block_id, 8)


if __name__ == "__main__":
    unittest.main()
