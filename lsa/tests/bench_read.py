# SPDX-License-Identifier: Apache-2.0
"""Time one layer's LSA work for a cold prefill and for decode, on synthetic data, without vLLM.

    python lsa/tests/bench_read.py [context] [chunk]

Prints milliseconds a layer for: index keys, Ward over the closed blocks, scoring+expansion, the
sparse read; then a decode step (4 query tokens, split-KV). Multiply by 16 for Qwen3.8-27B.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from hyperqwen_lsa import core, kernels, read  # noqa: E402

dev = torch.device("cuda")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
HQ, HKV, D, ROT, BS = 24, 4, 256, 64, 816
torch.manual_seed(0)
nblocks = (T + BS - 1) // BS + 2
block_table = torch.randperm(nblocks, device=dev)[None, : nblocks - 2].int().contiguous()
slots_all = core.slots_of(block_table, torch.zeros(1, dtype=torch.long, device=dev), torch.arange(T, device=dev)[None], BS)[0]
kc = torch.randint(-127, 127, (nblocks, BS, HKV, D + 4), dtype=torch.int8, device=dev)
vc = torch.randint(-127, 127, (nblocks, BS, HKV, D + 4), dtype=torch.int8, device=dev)
ks = torch.rand(nblocks, BS, HKV, device=dev) / 100
vs = torch.rand(nblocks, BS, HKV, device=dev) / 100
w_q = torch.randn(HKV * (D - ROT), 64, device=dev)
w_k = torch.randn(HKV * (D - ROT), 64, device=dev)
tables = core.LayerTables.allocate(nblocks * BS, dev, BS)
design = core.Design()
nb_max = 150000 // 8 + 1


def timed(fn, reps=1):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / reps


# compile every kernel once, outside the timed loop
_n = 256
_q = torch.randn(_n, HQ, D, device=dev, dtype=torch.bfloat16)
read.lsa_attention(_q, w_q, HKV, ROT, tables, kc[..., :D], vc[..., :D], ks, vs, torch.empty_like(_q),
                   torch.tensor([0, _n], dtype=torch.int32, device=dev), block_table,
                   torch.zeros(_n, dtype=torch.long, device=dev), torch.arange(5000, 5000 + _n, device=dev),
                   torch.full((_n,), 4608, device=dev), 1, _n, nb_max, D ** -0.5, design, chunk=256, num_splits=4)
core.cluster_blocks(tables, block_table, torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev),
                    torch.ones(1, dtype=torch.bool, device=dev), BS)
torch.cuda.synchronize()
totals = dict(keys=0.0, ward=0.0, select_attn=0.0)
for start in range(0, T, CHUNK):
    end = min(T, start + CHUNK)
    n = end - start
    key = torch.randn(n, HKV, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(n, HQ, D, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    totals["keys"] += timed(lambda: core.write_index_keys(tables, core.index_keys(key, w_k, ROT), slots_all[start:end]))
    first, last = start // 512, end // 512
    rows = torch.zeros(last - first, dtype=torch.long, device=dev)
    starts = torch.arange(first, last, device=dev) * 512
    valid = torch.ones_like(rows, dtype=torch.bool)
    totals["ward"] += timed(lambda: core.cluster_blocks(tables, block_table, rows, starts, valid, BS))
    if end <= design.sink + design.top_k + design.window:
        continue
    qsl = torch.tensor([0, n], dtype=torch.int32, device=dev)
    treq = torch.zeros(n, dtype=torch.long, device=dev)
    tpos = torch.arange(start, end, device=dev)
    ce = torch.full((n,), end // 512 * 512, device=dev)
    totals["select_attn"] += timed(lambda: read.lsa_attention(
        q, w_q, HKV, ROT, tables, kc[..., :D], vc[..., :D], ks, vs, out, qsl, block_table, treq, tpos, ce, 1, n,
        nb_max, D ** -0.5, design, chunk=int(os.environ.get("LSA_PREFILL_CHUNK", 256)), num_splits=4))
print(f"prefill {T} tokens, one layer: " + ", ".join(f"{k} {v:.0f} ms" for k, v in totals.items()),
      f"-> x16 layers {16 * sum(totals.values()) / 1000:.1f} s")

# split the selection from the read
n = 2048
q = torch.randn(n, HQ, D, device=dev, dtype=torch.bfloat16)
qsl = torch.tensor([0, n], dtype=torch.int32, device=dev)
treq = torch.zeros(n, dtype=torch.long, device=dev)
tpos = torch.arange(T - n, T, device=dev)
ce = torch.full((n,), T // 512 * 512, device=dev)
qidx = core.index_queries(q, w_q, HKV, ROT)
tails = read.tail_starts(tpos, ce, design.window)
vis = read.visible_buckets(tpos, ce)
sc = kernels.score_buckets(qidx, qsl, vis, block_table, tables.centroids, 0, 256, nb_max, BS, 1, n)
t_score = timed(lambda: kernels.score_buckets(qidx, qsl, vis, block_table, tables.centroids, 0, 256, nb_max, BS, 1, n), 5)
best, picked = kernels.top_p(sc, vis, 0, design.top_p, parts=32)
t_top = timed(lambda: kernels.top_p(sc, vis, 0, design.top_p, parts=32), 5)
width = design.sink + design.top_k + design.window + 512
t_exp = timed(lambda: kernels.expand_slots(picked, best, tables.members, block_table, treq[:256], tpos[:256], tails[:256],
                                           design.top_k, design.sink, BS, width, cb=64), 5)
slots, lens = kernels.expand_slots(picked, best, tables.members, block_table, treq[:256], tpos[:256], tails[:256],
                                   design.top_k, design.sink, BS, width, cb=64)
out = torch.empty_like(q)
t_attn = timed(lambda: kernels.slot_attention(q[:256], kc[..., :D], vc[..., :D], ks, vs, out[:256], slots, lens,
                                              D ** -0.5, num_splits=4, block_n=16, num_warps=2), 5)
print(f"256 queries at {T}: score {t_score:.2f} ms, top_p {t_top:.2f} ms, expand {t_exp:.2f} ms, attention {t_attn:.2f} ms")

# decode: 4 tokens (MTP verify), split-KV
n = 4
q = torch.randn(n, HQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
qsl = torch.tensor([0, n], dtype=torch.int32, device=dev)
treq = torch.zeros(n, dtype=torch.long, device=dev)
tpos = torch.arange(T - n, T, device=dev)
ce = torch.full((n,), (T - n) // 512 * 512, device=dev)
for splits in (8, 16, 32):
    read.lsa_attention(q, w_q, HKV, ROT, tables, kc[..., :D], vc[..., :D], ks, vs, out, qsl, block_table, treq, tpos,
                       ce, 1, n, nb_max, D ** -0.5, design, chunk=n, num_splits=splits)
    t = timed(lambda: read.lsa_attention(q, w_q, HKV, ROT, tables, kc[..., :D], vc[..., :D], ks, vs, out, qsl, block_table, treq, tpos,
                                         ce, 1, n, nb_max, D ** -0.5, design, chunk=n, num_splits=splits), 20)
    print(f"decode step (4 tokens) at {T}, splits {splits}: {t:.3f} ms a layer (eager, incl. launch overhead)")
