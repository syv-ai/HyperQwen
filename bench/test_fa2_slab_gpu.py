"""FA2 slab correctness against full-buffer FA2 and native PTH; explicit opt-in."""
import argparse
import json
import math
import unittest

import torch
from pth_flash_attn import ops
from pth_flash_attn.fa2_slab import SlabWorkspace, _merge
from pth_flash_attn.fa2_slab_plan import slabs, workspace_bytes
from pth_flash_attn.planner import plan_launch
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


def cache(seq, layout='hnd', block=816):
    pages = math.ceil(seq/block)+1
    shape = (pages, 4, block, 520) if layout == 'hnd' else (pages, block, 4, 520)
    raw = torch.empty(shape, dtype=torch.int8, device='cuda')
    view = raw.transpose(1, 2) if layout == 'hnd' else raw
    kc, vc = view.split(260, dim=-1)
    k, v = kc[..., :256], vc[..., :256]
    ks = kc[..., 256:].view(torch.float32).squeeze(-1)
    vs = vc[..., 256:].view(torch.float32).squeeze(-1)
    for t in (k, v):
        t.copy_(torch.randint(-128, 128, t.shape, dtype=torch.int8, device='cuda'))
    for t in (ks, vs):
        t.copy_(torch.rand_like(t)*.015+.002)
    order = torch.randperm(pages, dtype=torch.int32, device='cuda')
    storage = torch.full((2, pages+3), -1, dtype=torch.int32, device='cuda')
    table = storage[:1, :pages]
    table[0].copy_(order)
    if seq % block:
        page = int(order[(seq-1)//block])
        ks[page, seq % block:] = float('nan')
        vs[page, seq % block:] = float('nan')
    return (k, v, ks, vs), table


def full_kv(caches, table, seq):
    k, v, ks, vs = caches
    pages = table[0, :math.ceil(seq/k.shape[1])].long()
    # Multiply after flattening, independently of the slab gather's addressing.
    result = []
    for data, scales in ((k, ks), (v, vs)):
        x = data[pages].reshape(-1, 4, 256)[:seq].float()
        s = scales[pages].reshape(-1, 4)[:seq].float()
        result.append((x*s[..., None]).bfloat16())
    return result


def full_fa2(q, kv, cu_q, cu_k, out):
    return flash_attn_varlen_func(
        q=q, k=kv[0], v=kv[1], cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=len(q), max_seqlen_k=len(kv[0]), softmax_scale=.0625,
        causal=True, fa_version=2, num_splits=1, out=out)


def native(q, caches, table, seq, out):
    plan = plan_launch(num_tokens=len(q), num_reqs=1, max_query_len=len(q),
                       G=6, Hkv=4, D=256, num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
                       is_prefill=True, ws_token_splits=0)
    lengths = torch.tensor([seq], dtype=torch.int32, device='cuda')
    cu = torch.tensor([0, len(q)], dtype=torch.int32, device='cuda')
    empty = torch.empty(0, dtype=torch.float32, device='cuda')
    def run():
        ops.pth_flash_attn_fwd(out, q, *caches, table, lengths, cu,
                             len(q), plan.wm, plan.num_splits, .0625, empty, empty)
    return run


def normalized_rms(a, b):
    return ((a.float()-b.float()).square().mean()/b.float().square().mean()).sqrt().item()


class SlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = SlabWorkspace(torch.device('cuda', torch.cuda.current_device()))

    def test_gather_and_attention_match_full_fa2_across_boundaries(self):
        torch.manual_seed(83)
        for seq, qlen, layout in [(17, 17, 'hnd'), (2048, 2048, 'nhd'),
                (16384, 256, 'hnd'), (16385, 256, 'nhd'),
                (32769, 512, 'hnd'), (50003, 2048, 'nhd'), (153088, 256, 'hnd')]:
            with self.subTest(seq=seq, qlen=qlen, layout=layout):
                caches, table = cache(seq, layout)
                q = torch.randn((qlen, 24, 256), dtype=torch.bfloat16, device='cuda')
                cu_q = torch.tensor([0, qlen], dtype=torch.int32, device='cuda')
                cu_k = torch.tensor([0, seq], dtype=torch.int32, device='cuda')
                kv = full_kv(caches, table, seq)
                reference = full_fa2(q, kv, cu_q, cu_k, torch.empty_like(q))
                # Padded output row strides exercise the final merge stores.
                out = torch.empty((qlen, 25, 256), dtype=q.dtype, device='cuda')[:, :24]
                self.ws.run(q, caches, table, cu_q, seq, .0625, out)
                torch.cuda.synchronize()
                self.assertTrue(torch.isfinite(out).all().item())
                last = slabs(seq, qlen)[-1]
                for actual, expected in zip((self.ws.k, self.ws.v), kv):
                    torch.testing.assert_close(actual[:last.end-last.start],
                        expected[last.start:last.end], rtol=0, atol=0)
                rms = normalized_rms(out, reference)
                self.assertLess(rms, .004)
                # An independent FP32 attention check on the first/last query
                # rows guards both the FA2 LSE convention and causal offsets.
                idx = torch.tensor([0, qlen-1], device='cuda')
                kk = kv[0].float().repeat_interleave(6, dim=1).transpose(0, 1)
                vv = kv[1].float().repeat_interleave(6, dim=1).transpose(0, 1)
                score = q[idx].float().transpose(0, 1) @ kk.transpose(1, 2) * .0625
                positions = seq-qlen+idx
                score.masked_fill_(torch.arange(seq, device='cuda')[None, None] > positions[None, :, None], -float('inf'))
                oracle = (score.softmax(-1) @ vv).transpose(0, 1)
                oracle_rms = normalized_rms(out[idx], oracle)
                self.assertLess(oracle_rms, .005)
                print(json.dumps(dict(case='attention', seq=seq, query=qlen, layout=layout,
                    slabs=len(slabs(seq, qlen)), full_fa2_normalized_rms=rms,
                    fp32_normalized_rms=oracle_rms, gather_bit_exact=True)), flush=True)

    def test_merge_is_stable_for_extreme_lse_and_empty_partitions(self):
        qlen = 3
        part = torch.randn((qlen, 24, 256), dtype=torch.bfloat16, device='cuda')
        other = torch.randn_like(part)
        a = torch.tensor([-1000., 1000., -float('inf')], device='cuda').expand(24, -1).contiguous()
        b = torch.tensor([1000., -1000., -float('inf')], device='cuda').expand(24, -1).contiguous()
        out = torch.empty_like(part)
        _merge[(qlen, 24)](part, a, self.ws.acc, self.ws.lse, out,
            *a.stride(), *out.stride(), FIRST=True, FINAL=False, num_warps=4)
        _merge[(qlen, 24)](other, b, self.ws.acc, self.ws.lse, out,
            *b.stride(), *out.stride(), FIRST=False, FINAL=True, num_warps=4)
        expected = torch.stack((other[0], part[1], torch.zeros_like(part[2])))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_native_comparison_and_fixed_allocation(self):
        torch.manual_seed(97)
        caches, table = cache(32769)
        q = torch.randn((2048, 24, 256), dtype=torch.bfloat16, device='cuda')
        cu = torch.tensor([0, len(q)], dtype=torch.int32, device='cuda')
        old, new = torch.empty_like(q), torch.empty_like(q)
        run = native(q, caches, table, 32769, old)
        run(); self.ws.run(q, caches, table, cu, 32769, .0625, new)
        error = normalized_rms(new, old)
        self.assertLess(error, .008)
        actual = sum(t.numel()*t.element_size() for t in
                     (self.ws.k, self.ws.v, self.ws.partial, self.ws.acc, self.ws.lse, self.ws.cu_k))
        self.assertEqual(actual, sum(workspace_bytes().values()))
        print(json.dumps(dict(case='memory_and_pth', persistent_bytes=actual,
            kv_bytes=workspace_bytes()['kv'], pth_normalized_rms=error)), flush=True)


def benchmark():
    ws = SlabWorkspace(torch.device('cuda', torch.cuda.current_device()))
    torch.manual_seed(101)
    for seq in (8192, 65536, 150528):
        caches, table = cache(seq)
        q = torch.randn((2048, 24, 256), dtype=torch.bfloat16, device='cuda')
        cu = torch.tensor([0, len(q)], dtype=torch.int32, device='cuda')
        out = torch.empty_like(q)
        runs = {'pth': native(q, caches, table, seq, out),
                'fa2-slab': lambda: ws.run(q, caches, table, cu, seq, .0625, out)}
        for name, run in runs.items():
            for _ in range(3):run()
            torch.cuda.synchronize()
            samples = []
            for _ in range(15):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); run(); end.record(); end.synchronize()
                samples.append(start.elapsed_time(end))
            print(json.dumps(dict(case='kernel_benchmark', path=name, sequence=seq,
                query=len(q), median_ms=sorted(samples)[len(samples)//2],
                min_ms=min(samples), samples=len(samples))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-gpu', action='store_true', required=True)
    parser.add_argument('--benchmark', action='store_true')
    args, rest = parser.parse_known_args()
    assert torch.cuda.get_device_capability() == (8, 6)
    suite = unittest.main(argv=['test_gpu.py', *rest], verbosity=2, exit=False)
    if not suite.result.wasSuccessful():
        raise SystemExit(1)
    # Run the performance sample only when the caller explicitly requests it.
    if args.benchmark:benchmark()
