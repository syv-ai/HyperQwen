# Reused from models/qwen3.8-27b/vllm/patches/fa2-fp8kv/check.py.
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
import argparse
import json
import math

import torch
from fa2_fp8kv_vllm import paths


@dataclass(frozen=True)
class Fp8Alphabet:
    scale: float

    def check(self) -> dict:
        values = torch.arange(256, device='cuda', dtype=torch.int32).to(torch.uint8)
        values[127] = 0
        values[255] = 0
        v = torch.zeros((1, 16, 2, 128), device='cuda', dtype=torch.float8_e4m3fn)
        v[0, 0] = values.view(torch.float8_e4m3fn).reshape(2, 128)
        k = torch.zeros_like(v)
        q = torch.zeros((1, 2, 128), device='cuda', dtype=torch.bfloat16)
        out = torch.empty_like(q)
        scale = torch.tensor([self.scale], device='cuda', dtype=torch.float32)
        result, _ = torch.ops.fa2_fp8kv.forward(
            q, k, v, out, torch.tensor([0, 1], device='cuda', dtype=torch.int32),
            torch.tensor([1], device='cuda', dtype=torch.int32),
            torch.tensor([[0]], device='cuda', dtype=torch.int32), scale, scale,
            1, 1, False, -1, -1, 1 / math.sqrt(128), 1)
        expected = (v[0, :1].float() * scale).bfloat16()
        assert torch.equal(result, expected), 'FP8 finite alphabet conversion mismatch'
        return {'case': 'fp8-alphabet', 'scale': self.scale, 'passed': True}


@dataclass(frozen=True)
class PagedCase:
    name: str
    dim: int
    heads: int
    kv_heads: int
    queries: tuple[int, ...]
    keys: tuple[int, ...]
    page: int
    causal: bool = False
    left: int = -1
    right: int = -1
    splits: int = 1
    key_scale: float = 1.0
    value_scale: float = 1.0
    strided: bool = False

    def check(self) -> dict:
        torch.manual_seed(42)
        device = 'cuda'
        dtype = torch.float8_e4m3fn
        blocks = [math.ceil(length / self.page) for length in self.keys]
        physical = sum(blocks) + 2
        padding = 16 if self.strided else 0
        shape = (physical, self.page, self.kv_heads, self.dim + padding)
        k_storage = (torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.7).to(dtype)
        v_storage = torch.randn(shape, device=device, dtype=torch.bfloat16).to(dtype)
        k, v = k_storage[..., :self.dim], v_storage[..., :self.dim]
        if padding:
            k_storage[..., self.dim:] = float('nan')
            v_storage[..., self.dim:] = float('nan')
        order = torch.randperm(physical).tolist()
        table_storage = torch.zeros((len(self.keys), max(blocks) + 3), device=device, dtype=torch.int32)
        table = table_storage[:, :max(blocks)]
        cursor = 0
        sequences = []
        for batch, (length, count) in enumerate(zip(self.keys, blocks)):
            pages = order[cursor:cursor + count]
            cursor += count
            sequences.append(pages)
            if count:
                table[batch, :count] = torch.tensor(pages, device=device, dtype=torch.int32)
                if length % self.page:
                    k[pages[-1], length % self.page:] = float('nan')
                    v[pages[-1], length % self.page:] = float('nan')
        for unused in order[cursor:]:
            k[unused] = float('nan')
            v[unused] = float('nan')

        total = sum(self.queries)
        q_storage = torch.randn((total, self.heads, self.dim + padding), device=device, dtype=torch.bfloat16)
        out_storage = torch.full_like(q_storage, float('nan'))
        q, out = q_storage[..., :self.dim], out_storage[..., :self.dim]
        cu_q = torch.tensor([0, *accumulate(self.queries)], device=device, dtype=torch.int32)
        seq_k = torch.tensor(self.keys, device=device, dtype=torch.int32)
        ks = torch.tensor([self.key_scale], device=device, dtype=torch.float32)
        vs = torch.tensor([self.value_scale], device=device, dtype=torch.float32)
        result, lse = torch.ops.fa2_fp8kv.forward(
            q, k, v, out, cu_q, seq_k, table, ks, vs,
            max(self.queries), max(self.keys), self.causal, self.left, self.right,
            1 / math.sqrt(self.dim), self.splits,
        )
        torch.cuda.synchronize()

        expected = torch.empty_like(q)
        expected_lse = torch.empty((self.heads, total), device=device, dtype=torch.float32)
        start = 0
        group = self.heads // self.kv_heads
        for count_q, count_k, pages in zip(self.queries, self.keys, sequences):
            if not count_q:
                continue
            selected = torch.tensor(pages, device=device, dtype=torch.long)
            q_positions = torch.arange(count_q, device=device) + count_k - count_q
            k_positions = torch.arange(count_k, device=device)
            mask = torch.ones((count_q, count_k), device=device, dtype=torch.bool)
            if self.causal:
                mask &= k_positions[None, :] <= q_positions[:, None]
            if self.left >= 0:
                mask &= k_positions[None, :] >= q_positions[:, None] - self.left
            if self.right >= 0:
                mask &= k_positions[None, :] <= q_positions[:, None] + self.right
            valid = mask.any(dim=-1)
            for head in range(self.kv_heads):
                # Process one KV head at a time to bound reference-test memory.
                key = (k[selected, :, head, :].reshape(-1, self.dim)[:count_k].float() * ks).bfloat16().float()
                value = (v[selected, :, head, :].reshape(-1, self.dim)[:count_k].float() * vs).bfloat16().float()
                query = q[start:start + count_q, head * group:(head + 1) * group].float().transpose(0, 1)
                scores = torch.matmul(query, key.T) / math.sqrt(self.dim)
                scores.masked_fill_(~mask, -float('inf'))
                weights = scores.softmax(dim=-1)
                weights = torch.where(valid[None, :, None], weights, 0)
                expected[start:start + count_q, head * group:(head + 1) * group] = torch.matmul(weights, value).transpose(0, 1).bfloat16()
                expected_lse[head * group:(head + 1) * group, start:start + count_q] = torch.where(
                    valid[None, :], scores.logsumexp(-1), float('inf'))
            start += count_q
        difference = result.float() - expected.float()
        relative = (difference.norm() / expected.float().norm().clamp_min(1e-8)).item()
        maximum = difference.abs().max().item()
        assert result.isfinite().all(), self.name + ': non-finite output'
        assert relative < 0.012 and maximum < 0.04, (self.name, relative, maximum)
        assert torch.equal(lse.isfinite(), expected_lse.isfinite()), self.name + ': LSE mask mismatch'
        finite = expected_lse.isfinite()
        assert torch.allclose(lse[finite], expected_lse[finite], atol=0.03, rtol=0.005), self.name + ': LSE mismatch'
        if padding:
            assert out_storage[..., self.dim:].isnan().all(), self.name + ': output padding overwritten'
        return {'case': self.name, 'max_abs': maximum, 'relative_l2': relative,
                'peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1), 'passed': True}


parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true')
parser.add_argument('--library', default=str(paths.library()))
args = parser.parse_args()
torch.ops.load_library(args.library)
torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(0.20)
cases = [
    Fp8Alphabet(1.0),
    Fp8Alphabet(0.37),
    PagedCase('draft-short', 128, 16, 4, (7,), (53,), 16),
    PagedCase('target-causal', 256, 12, 2, (8,), (151,), 16, True),
    PagedCase('ragged-split', 128, 16, 4, (1, 7), (11, 70), 16, True, splits=4),
    PagedCase('strided-scales', 256, 12, 2, (3, 8), (77, 193), 32,
              splits=4, key_scale=0.37, value_scale=1.7, strided=True),
    PagedCase('sliding-window', 128, 16, 4, (7,), (317,), 16, left=63, right=0, splits=4),
    PagedCase('bidirectional-window', 128, 16, 4, (7,), (317,), 16, left=63, right=63, splits=4),
    PagedCase('hybrid-page', 256, 12, 2, (8,), (3333,), 1648, True, splits=4),
    # HyperQwen's shapes (not in the plugin): target D256 with 24 q / 4 kv heads, drafter D128 with 8 kv
    # heads (its q-head count is a guess) and the noncausal 2048 window.
    PagedCase('hq-target-verify', 256, 24, 4, (8,), (3333,), 1648, True, splits=4),
    PagedCase('hq-target-decode', 256, 24, 4, (1,), (4097,), 16, True, splits=128),
    PagedCase('hq-draft-window', 128, 32, 8, (8,), (2055,), 16, left=2047, splits=32),
]
if args.full:
    cases += [
        PagedCase('prefill-2048', 256, 12, 2, (2048,), (4097,), 256, True),
        PagedCase('draft-262k', 128, 16, 4, (7,), (262144,), 256, left=2047, right=0),
        PagedCase('draft-262k-bidirectional', 128, 16, 4, (7,), (262144,), 256, left=2047, right=2047, splits=0),
        PagedCase('target-262k', 256, 12, 2, (8,), (262144,), 256, True, splits=128),
        PagedCase('ragged-empty-splits', 256, 12, 2, (1, 7), (1, 73), 16, True, splits=32),
        PagedCase('causal-leading-mask', 128, 16, 4, (7,), (3,), 16, True, splits=4),
    ]
with torch.inference_mode():
    for case in cases:
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps(case.check()), flush=True)
print(json.dumps({'passed_cases': len(cases)}), flush=True)
