#!/usr/bin/env python3
"""DFlash startup context-capacity checks against the installed patched vLLM.

Needs the image/venv, no GPU or model weights. Execute the actual validation
and its constructor dispatch with small config stand-ins.
"""
import ast
import inspect
import textwrap
from types import SimpleNamespace as NS
import unittest

from vllm.config.speculative import SpeculativeConfig


class DFlashCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(textwrap.dedent(inspect.getsource(SpeculativeConfig.__post_init__)))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Compare)
                    and any(isinstance(c, ast.Constant) and c.value == 'dflash' for c in n.test.comparators)
                    and '_validate_dflash_context_capacity' in ast.unparse(n))
        cls.dispatch = compile(ast.fix_missing_locations(ast.Module(body=[call], type_ignores=[])),
                               '<actual-speculative-constructor>', 'exec')

    def select(self, target, capacity=262144, drafts=7, method='dflash'):
        obj = NS(method=method, target_model_config=NS(max_model_len=target),
                 draft_model_config=NS(max_model_len=capacity), num_speculative_tokens=drafts)
        exec(self.dispatch, dict(self=obj, SpeculativeConfig=SpeculativeConfig))
        return obj

    def test_shipped_profiles_fit_without_changing_config(self):
        for context in (8192, 65536, 131072, 221184, 245760):
            for drafts in (7, 15):
                with self.subTest(context=context, drafts=drafts):
                    obj = self.select(context, drafts=drafts)
                    self.assertEqual(obj.draft_model_config.max_model_len, 262144)

    def test_long_target_is_rejected_before_any_model_execution(self):
        for context in (262145, 264550, 1048576):
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, 'DFlash.*context capacity 262144'):
                    self.select(context)

    def test_fixed_query_block_has_room_at_the_boundary(self):
        self.select(262136)
        for context in (262137, 262143, 262144):
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, '8 lookahead positions'):
                    self.select(context)

    def test_explicit_larger_capacity_is_respected(self):
        self.select(1000000, capacity=1048576)
        self.select(1048568, capacity=1048576)

    def test_other_speculative_methods_keep_their_existing_policy(self):
        for method in ('mtp', 'eagle', 'eagle3', 'draft_model', 'dspark'):
            with self.subTest(method=method):
                self.select(1048576, method=method)


if __name__ == '__main__':
    unittest.main()
