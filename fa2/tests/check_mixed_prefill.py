"""Mixed prefill regression; CPU routing oracle or real CUDA operators."""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args, remaining = parser.parse_known_args()
if args.device == "cpu":
    # Import the actual vLLM metadata classes, but never require a CUDA binary
    # for the CPU routing checks. Native prefill must be explicitly OOM-mocked.
    from vllm.v1.attention.backends import flash_attn
    with patch.dict(sys.modules, {"vllm.vllm_flash_attn": SimpleNamespace(
            flash_attn_varlen_func=Mock(side_effect=AssertionError("Unexpected native prefill")))}):
        from fa2_fp8kv_vllm import backend
else:
    from fa2_fp8kv_vllm import backend


@dataclass(frozen=True)
class Scales:
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor


def reference(query, keys, values, table, lengths, starts, scales, scale):
    results = []
    page_size = keys.shape[1]
    for row, (start, end) in enumerate(zip(starts, starts[1:])):
        length = int(lengths[row])
        positions = torch.arange(length, device=query.device)
        pages = table[row, positions // page_size].long()
        k = (keys.float()[pages, positions % page_size] * scales._k_scale).bfloat16().float()
        v = (values.float()[pages, positions % page_size] * scales._v_scale).bfloat16().float()
        k = k.repeat_interleave(query.shape[1] // keys.shape[2], dim=1)
        v = v.repeat_interleave(query.shape[1] // keys.shape[2], dim=1)
        for first in range(start, end, 128):
            last = min(first + 128, end)
            scores = torch.einsum("qhd,khd->hqk", query[first:last].float(), k) * scale
            visible = torch.arange(first - start, last - start, device=query.device) + length - (end - start)
            scores.masked_fill_(positions[None, :] > visible[:, None], -torch.inf)
            results.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
    return torch.cat(results).bfloat16()


class MixedPrefill(unittest.TestCase):
    device = "cpu"

    def run_batch(self, queries, *, capture=False, oom=False, upper_bound=True, contexts=()):
        torch.manual_seed(1358)
        device = self.device
        dim, heads, kv_heads, page_size = 128, 2, 1, 16
        starts = [0]
        for count in queries:
            starts.append(starts[-1] + count)
        lengths = torch.tensor(contexts or [q + 17 + i * 7 for i, q in enumerate(queries)],
                               dtype=torch.int32, device=device)
        blocks = (int(lengths.max()) + page_size - 1) // page_size
        # Separate shuffled pages expose cross-request table/slice mistakes.
        table = torch.randperm(blocks * len(queries), device=device).reshape(len(queries), blocks).int()
        cache = torch.randn(blocks * len(queries), page_size, kv_heads, 2 * dim,
                            device=device).to(torch.float8_e4m3fn).transpose(1, 2)
        keys, values = cache.transpose(1, 2).split(dim, dim=-1)
        query = torch.randn(starts[-1] + 3, heads, dim, device=device, dtype=torch.bfloat16)
        guarded = torch.full((starts[-1] + 5, 2, heads, dim), 11.0,
                             device=device, dtype=torch.bfloat16)
        output = guarded[1:-1, 0]
        scales = Scales(torch.tensor(0.5, device=device), torch.tensor(0.25, device=device))
        common = SimpleNamespace(
            causal=True, max_query_len=max(queries), num_reqs=len(queries),
            query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
            # Async speculative decode may overestimate a decode row's length.
            seq_lens_cpu_upper_bound=lengths.cpu() + torch.tensor([9 if q <= 64 else 0 for q in queries])
                if upper_bound else None,
        )
        metadata = backend.native.FlashAttentionMetadata(
            num_actual_tokens=starts[-1], max_query_len=max(queries),
            query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
            max_seq_len=int(lengths.max()), seq_lens=lengths, block_table=table,
            slot_mapping=torch.zeros(starts[-1], dtype=torch.int64, device=device),
            use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
            prefix_kv_lens=None, suffix_kv_lens=None, causal=True,
        )
        builder = object.__new__(backend.Fp8MetadataBuilder)
        builder.dcp_world_size = 1
        builder.model_config = SimpleNamespace(max_model_len=blocks * page_size)
        with patch.object(backend.native.FlashAttentionMetadataBuilder, "build", return_value=metadata):
            metadata = builder.build(0, common)
        # Distributed rank initialization belongs to vLLM, not this unit check.
        impl = object.__new__(backend.Fp8Attention)
        impl.__init__(heads, dim, dim ** -0.5, kv_heads, kv_cache_dtype="fp8_e4m3")
        impl.dcp_world_size = impl.pcp_world_size = 1
        calls = []

        def bounded_operation(q, k, v, out, cu_q, seq_k, pages, ks, vs,
                              max_q, max_k, causal, left, right, scale, splits, grouped):
            self.assertLessEqual(max_q, 2048, "kernel envelope exceeded")
            self.assertTrue(causal)
            self.assertEqual(left, -1)
            self.assertLessEqual(int(seq_k.max()), max_k)
            calls.append((q.shape[0], seq_k.tolist()))
            out.copy_(reference(q, k, v, pages, seq_k, cu_q.tolist(), Scales(ks, vs), scale))
            return out, torch.empty(0)

        with ExitStack() as stack:
            if device == "cpu":
                stack.enter_context(patch.object(backend, "extension_selected", return_value=True))
                stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=capture))
                stack.enter_context(patch.object(torch.ops.fa2_fp8kv, "forward", bounded_operation, create=True))
            if oom:
                stack.enter_context(patch.object(backend.Fa2Prefill, "forward", side_effect=torch.OutOfMemoryError))
            impl.forward(scales, query, query, query, cache, metadata, output)
            if capture and device == "cuda":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        impl.forward(scales, query, query, query, cache, metadata, output)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    impl.forward(scales, query, query, query, cache, metadata, output)
                lengths.sub_(1)
                graph.replay()
        expected = reference(query[:starts[-1]], keys, values, table, lengths, starts, scales, impl.scale)
        torch.testing.assert_close(output[:starts[-1]], expected, atol=0.025, rtol=0.025)
        self.assertTrue(bool((guarded[[0, -1]] == 11).all()))
        self.assertTrue(bool((guarded[:, 1] == 11).all()))
        self.assertTrue(bool((output[starts[-1]:] == 11).all()))
        return calls

    def test_decode_then_long_prefill(self):
        self.run_batch((1, 4097))

    def test_long_prefill_then_decode(self):
        self.run_batch((2049, 8))

    def test_two_prefills_with_empty_row(self):
        self.run_batch((2049, 0, 2051))

    def test_missing_cpu_context_bound(self):
        self.run_batch((1, 2049), upper_bound=False)

    def test_single_prefill_without_cpu_context_bound(self):
        self.run_batch((2049,), upper_bound=False)

    def test_single_prefill_oom_fallback(self):
        self.run_batch((2049,), oom=True)

    def test_kernel_boundary(self):
        calls = self.run_batch((1, 2048))
        if self.device == "cpu":
            self.assertEqual(len(calls), 1)

    def test_long_context_cuda(self):
        if self.device != "cuda":
            self.skipTest("Long-context kernel validation requires CUDA")
        self.run_batch((1, 2049), contexts=(131077, 262143))

    def test_decode_capture_keeps_batched_operator(self):
        calls = self.run_batch((1, 8), capture=True)
        if self.device == "cpu":
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    MixedPrefill.device = args.device
    torch.set_num_threads(2)
    with torch.inference_mode():
        unittest.main(argv=[__file__, *remaining])
