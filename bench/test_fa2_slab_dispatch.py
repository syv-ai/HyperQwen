"""Exercise the actual adapter dispatch with CPU stand-ins, without torch/vLLM."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pthfa/python'))


class Workspace:
    def __init__(self, device):
        self.device = device
        self.owned = True
        self.route_logged = True
        self.run = Mock()

    def claim_stream(self):
        return self.owned


class Tensor:
    def __init__(self, size, dtype=None):
        self.size = size
        self.dtype = dtype

    def numel(self):
        return self.size

    def __getitem__(self, key):
        return self

    def item(self):
        raise AssertionError('dispatch must not read GPU scalars')


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.bf16 = object()
        self.cuda = SimpleNamespace(
            get_device_capability=Mock(return_value=(8, 6)),
            current_device=Mock(return_value=0),
            is_current_stream_capturing=Mock(return_value=False))
        self.fa_supported = Mock(return_value=True)
        fake_torch = SimpleNamespace(bfloat16=self.bf16, cuda=self.cuda,
                                    device=lambda *args: SimpleNamespace(index=0))
        modules = {
            'torch': fake_torch,
            'vllm.triton_utils': SimpleNamespace(
                triton=SimpleNamespace(jit=lambda *a, **k: a[0] if a else lambda f: f),
                tl=SimpleNamespace(constexpr=object)),
            'vllm.v1.attention.backends.fa_utils': SimpleNamespace(
                is_fa_version_supported=self.fa_supported),
            'vllm.logger': SimpleNamespace(init_logger=lambda *args: Mock()),
        }
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                '_fa2_slab_dispatch_test', ROOT/'pthfa/python/pth_flash_attn/fa2_slab.py')
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.modules = modules
        self.module.SlabWorkspace = Workspace
        self.config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            model_config=SimpleNamespace(dtype=self.bf16))
        self.impl = SimpleNamespace(num_heads=24, num_kv_heads=4,
                                    head_size=256, window=0, scale=.0625)
        self.metadata = SimpleNamespace(
            prefill=SimpleNamespace(token_start=0, token_end=2048), decode=None,
            num_actual_tokens=2048, causal=True, use_cascade=False,
            max_query_len=2048, seq_lens=Tensor(1), query_start_loc=Tensor(2),
            max_seq_len=150528, block_table=object())
        self.query = Tensor(2048, self.bf16)
        self.output = Tensor(2048, self.bf16)
        self.caches = object()
        self.workspace = Workspace('cuda')

    def prepare(self, flag='1', config=True):
        with patch.dict(os.environ, {'PTH_FA2_SLAB_PREFILL': flag}), \
                patch.dict(sys.modules, self.modules):
            return self.module.prepare_workspace(self.impl, self.config if config else None)

    def dispatch(self):
        return self.module.try_prefill(
            self.impl, self.workspace, self.query, self.output, self.caches, self.metadata)

    def test_disabled_and_missing_config_allocate_nothing(self):
        self.assertIsNone(self.prepare('0'))
        self.assertIsNone(self.prepare(config=False))
        self.assertEqual(self.module._WORKSPACES, {})
        self.cuda.get_device_capability.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            self.prepare('true')

    def test_workspace_is_shared_only_within_one_model_and_device(self):
        first = self.prepare()
        self.assertIs(first, self.prepare())
        self.config.model_config = SimpleNamespace(dtype=self.bf16)
        self.assertIsNot(first, self.prepare())

    def test_unsupported_shape_batch_dtype_device_or_fa2_falls_back(self):
        for name, value in [('num_heads', 12), ('num_kv_heads', 8),
                            ('head_size', 128), ('window', 1024)]:
            with self.subTest(name=name), patch.object(self.impl, name, value):
                self.assertIsNone(self.prepare())
        self.config.scheduler_config.max_num_seqs = 2
        self.assertIsNone(self.prepare())
        self.config.scheduler_config.max_num_seqs = 1
        self.config.model_config.dtype = object()
        self.assertIsNone(self.prepare())
        self.config.model_config.dtype = self.bf16
        self.cuda.get_device_capability.return_value = (8, 9)
        self.assertIsNone(self.prepare())
        self.cuda.get_device_capability.return_value = (8, 6)
        self.fa_supported.return_value = False
        self.assertIsNone(self.prepare())
        self.assertEqual(self.module._WORKSPACES, {})

    def test_eligible_prefill_uses_host_metadata_without_reading_gpu_lengths(self):
        self.assertTrue(self.dispatch())
        self.workspace.run.assert_called_once_with(
            self.query, self.caches, self.metadata.block_table,
            self.metadata.query_start_loc, 150528, .0625, self.output)

    def test_decode_mixed_batches_cascade_noncausal_and_bad_queries_fall_back(self):
        cases = [('prefill', None), ('decode', object()), ('causal', False),
                 ('use_cascade', True), ('num_actual_tokens', 5),
                 ('num_actual_tokens', 2049), ('max_query_len', 17),
                 ('seq_lens', Tensor(2)), ('query_start_loc', Tensor(3))]
        for name, value in cases:
            with self.subTest(name=name, value=value), \
                    patch.object(self.metadata, name, value):
                self.assertFalse(self.dispatch())
        for name, value in [('token_start', 1), ('token_end', 2047)]:
            with self.subTest(name=name), patch.object(self.metadata.prefill, name, value):
                self.assertFalse(self.dispatch())
        self.query.dtype = object()
        self.assertFalse(self.dispatch())
        self.workspace.run.assert_not_called()

    def test_capture_foreign_stream_and_missing_workspace_fall_back(self):
        self.cuda.is_current_stream_capturing.return_value = True
        self.assertFalse(self.dispatch())
        self.cuda.is_current_stream_capturing.return_value = False
        self.workspace.owned = False
        self.assertFalse(self.dispatch())
        self.workspace.run.assert_not_called()
        self.workspace = None
        self.assertFalse(self.dispatch())


if __name__ == '__main__':
    unittest.main()
