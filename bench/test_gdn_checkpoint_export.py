#!/usr/bin/env python3
"""Exact GDN checkpoint planning/history, plus optional real CUDA recurrence.

Use the patched image: python bench/test_gdn_checkpoint_export.py
Add --device cuda:0 for production-head-size FLA prefix/resume numerics.
CPU checks use actual planner/export methods; only H2D transfer is replaced
with CPU tensor creation. GPU checks call the real FLA kernels.
"""
import argparse
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointMetadata
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as layer
from vllm.v1.attention.backends import gdn_attn as backend
from vllm.third_party.flash_linear_attention.ops import index as fla_index

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', default='cpu')
args, remaining = parser.parse_known_args()


def builder():
    value = object.__new__(backend.GDNAttentionMetadataBuilder)
    value.gdn_prefill_backend = 'triton'
    return value


class PlannerAndHistoryTests(unittest.TestCase):
    def test_mixed_decode_prefill_plan_and_ragged_offsets(self):
        meta = MambaPrefillCheckpointMetadata(
            torch.tensor([0, 3, 0, 5], dtype=torch.int32),
            torch.tensor([0, 10, 0, 12], dtype=torch.int32), [0, 3, 0, 5])
        def transfer(values, dtype=None, device=None):
            return torch.tensor(values, dtype=dtype, device=device)
        with patch.object(backend, 'async_tensor_h2d', transfer), \
             patch.object(fla_index, 'async_tensor_h2d', transfer):
            plan = builder()._build_checkpoint_plan(
                meta, 1, 1, torch.tensor([0, 1, 8, 9, 16]), torch.device('cpu'))
        self.assertEqual(plan.source_slices, [(0, 3), (8, 5)])
        self.assertEqual(plan.conv_source_starts, [1, 9])
        self.assertEqual(plan.prefill_rows.tolist(), [0, 2])
        self.assertEqual(plan.non_spec_rows.tolist(), [1, 3])
        self.assertEqual(plan.state_indices.tolist(), [10, 12])
        self.assertEqual(plan.state_indices.dtype, torch.int64)
        self.assertEqual(plan.query_start_loc.tolist(), [0, 3, 8])

    def test_ineligible_rows_do_not_create_a_plan(self):
        meta = MambaPrefillCheckpointMetadata(torch.zeros(2, dtype=torch.int32),
                                              torch.zeros(2, dtype=torch.int32), [0, 0])
        self.assertIsNone(builder()._build_checkpoint_plan(
            meta, 0, 0, torch.tensor([0, 5, 8]), torch.device('cpu')))

    def test_interior_geometry_without_a_reserved_page_is_not_exported(self):
        # The allocator intentionally reserves no checkpoint for an intermediate
        # prefill chunk. Its geometrically valid position maps to the null page.
        meta = MambaPrefillCheckpointMetadata(
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([backend.NULL_BLOCK_ID], dtype=torch.int32), [3])
        self.assertIsNone(builder()._build_checkpoint_plan(
            meta, 0, 0, torch.tensor([0, 8]), torch.device('cpu')))

    def test_convolution_history_includes_initial_prefix_and_spec_scratch(self):
        for dtype in (torch.float16, torch.bfloat16):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    generator = torch.Generator().manual_seed(19)
                    raw = torch.randn(20, 5, generator=generator).to(torch.bfloat16)
                    states = torch.randn(16, 5, 10, generator=generator).to(dtype)
                    if strided:
                        states = states.transpose(1, 2).contiguous().transpose(1, 2)
                    before = states.clone()
                    dst = torch.tensor([10, 11, 12], dtype=torch.int64)
                    rows = torch.tensor([1, 2, 3], dtype=torch.int64)
                    plan = NS(source_slices=[(0, 1), (3, 2), (7, 6)],
                              conv_source_starts=[1, 4, 8],
                              state_indices=dst, non_spec_rows=rows)
                    meta = NS(checkpoint_plan=plan,
                              non_spec_state_indices_tensor=torch.tensor([1, 2, 3, 4]),
                              has_initial_state=torch.tensor([False, True, False, True]))
                    layer.QwenGatedDeltaNetAttention._store_gdn_checkpoint_conv(
                        NS(conv_kernel_size=4), meta, raw, states)
                    for i, (start, length) in enumerate(zip([1, 4, 8], [1, 2, 6])):
                        history = raw[start:start+length].T.to(dtype)
                        initial = before[i+2, :, :3] if i != 1 else torch.zeros(5, 3, dtype=dtype)
                        expected = torch.cat((initial, history), dim=-1)[:, -3:]
                        torch.testing.assert_close(states[10+i, :, :3], expected, rtol=0, atol=0)
                        self.assertEqual(states[10+i, :, 3:].count_nonzero().item(), 0)
                    torch.testing.assert_close(states[:10], before[:10], rtol=0, atol=0)
                    torch.testing.assert_close(states[13:], before[13:], rtol=0, atol=0)

    def test_no_checkpoint_preserves_the_cache(self):
        cache = torch.randn(3, 5, 3)
        before = cache.clone()
        layer.QwenGatedDeltaNetAttention._store_gdn_checkpoint_conv(
            NS(conv_kernel_size=4), NS(checkpoint_plan=None), torch.randn(8, 5), cache)
        torch.testing.assert_close(cache, before, rtol=0, atol=0)


@unittest.skipUnless(args.device.startswith('cuda'), 'CUDA recurrence check requested with --device')
class CUDARecurrentExportTests(unittest.TestCase):
    def test_ragged_prefix_states_and_resumed_suffixes(self):
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        rng = torch.Generator(device=device).manual_seed(1729)
        lengths = [80, 100, 73]
        starts = [0, 80, 180]
        total = sum(lengths)
        shape = (1, total, 2, 128)
        q = torch.nn.functional.normalize(torch.randn(shape, generator=rng, device=device), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(torch.randn(shape, generator=rng, device=device), dim=-1).to(torch.bfloat16)
        v = (torch.randn(shape, generator=rng, device=device) * .05).to(torch.bfloat16)
        g = (-torch.rand((1, total, 2), generator=rng, device=device) * .1).to(torch.float32)
        beta = torch.rand((1, total, 2), generator=rng, device=device).to(torch.bfloat16)
        initial = torch.randn((3, 2, 128, 128), generator=rng, device=device) * .01
        fn = layer.fla_chunk_gated_delta_rule
        for offsets in ([1, 7, 2], [3, 7, 63], [64, 7, 65]):
            cp = MambaPrefillCheckpointMetadata(
                torch.tensor(offsets, device=device, dtype=torch.int32),
                torch.tensor([5, backend.NULL_BLOCK_ID, 9], device=device, dtype=torch.int32), list(offsets))
            plan = builder()._build_checkpoint_plan(
                cp, 0, 0, torch.tensor([0, 80, 180, 253]), device)
            meta = NS(checkpoint_plan=plan)
            for saved_dtype in (torch.float32, torch.float16):
                with self.subTest(offsets=offsets, dtype=saved_dtype):
                    paged = torch.full((12, 2, 128, 128), -7., dtype=saved_dtype, device=device)
                    layer.QwenGatedDeltaNetAttention._store_gdn_prefill_checkpoint(
                        NS(chunk_gated_delta_rule=fn), meta, initial, q, k, v, g, beta, paged)
                    for row, block in ((0, 5), (2, 9)):
                        start, offset, length = starts[row], offsets[row], lengths[row]
                        cropped = [a[:, start:start+offset].contiguous() for a in (q, k, v, g, beta)]
                        _, reference = fn(q=cropped[0], k=cropped[1], v=cropped[2], g=cropped[3], beta=cropped[4],
                                          initial_state=initial[row:row+1], output_final_state=True,
                                          cu_seqlens=torch.tensor([0, offset], device=device, dtype=torch.int32),
                                          use_qk_l2norm_in_kernel=False)
                        torch.testing.assert_close(paged[block], reference[0].to(saved_dtype), rtol=.002, atol=.0002)
                        full = [a[:, start:start+length].contiguous() for a in (q, k, v, g, beta)]
                        full_out, _ = fn(q=full[0], k=full[1], v=full[2], g=full[3], beta=full[4],
                                         initial_state=initial[row:row+1], output_final_state=True,
                                         cu_seqlens=torch.tensor([0, length], device=device, dtype=torch.int32),
                                         use_qk_l2norm_in_kernel=False)
                        suffix = [a[:, offset:].contiguous() for a in full]
                        resumed, _ = fn(q=suffix[0], k=suffix[1], v=suffix[2], g=suffix[3], beta=suffix[4],
                                        initial_state=paged[block:block+1].float(), output_final_state=True,
                                        cu_seqlens=torch.tensor([0, length-offset], device=device, dtype=torch.int32),
                                        use_qk_l2norm_in_kernel=False)
                        torch.testing.assert_close(resumed, full_out[:, offset:], rtol=.01, atol=.001)
                    untouched = [i for i in range(12) if i not in (5, 9)]
                    self.assertTrue(torch.all(paged[untouched] == -7).item())


if __name__ == '__main__':
    unittest.main(argv=[__file__, *remaining])
