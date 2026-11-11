# SPDX-License-Identifier: Apache-2.0
"""Hold the port to the upstream torch reference, on a GPU, without vLLM.

    LSA_REF=<checkout of Prefrontal-AI/latent-sparse-attention>/src python lsa/tests/check_core.py

1. Ward: the partition of every cluster block equals lsa.core.ward.ward_rounds'.
2. Selection: on a paged cache with a shuffled block table, each query's key set (sinks, selected,
   tail) equals lsa.core.reference.read's.
3. Attention: the expansion to physical slots equals core.expand's lists; the read equals
   lsa.core.attention.topk_attention over them on a bf16 cache and on an int8_per_token_head cache
   (against the dequantized keys), split-KV or not; and at a budget covering the context the whole read
   path (read.lsa_attention) equals dense causal attention.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ["LSA_REF"])
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from lsa.core import reference, ward as ref_ward            # noqa: E402
from lsa.core.attention import topk_attention               # noqa: E402

from hyperqwen_lsa import core, read                         # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        failures.append(name)


def clustered_unit_rows(t, d, centers=96, noise=0.35):
    """Keys with structure: noisy copies of a few directions, as real index keys are."""
    c = F.normalize(torch.randn(centers, d), dim=-1)
    rows = c[torch.randint(0, centers, (t,))] + noise * torch.randn(t, d) / d ** 0.5
    return F.normalize(rows, dim=-1)


# 1. Ward -------------------------------------------------------------------------------------
keys = clustered_unit_rows(40 * 512, 64).view(40, 512, 64)
pools = keys.view(40, 128, 4, 64).sum(2)
ref_labels, reached = ref_ward.ward_rounds(pools.double().float(), torch.full((40, 128), 4.0), 64, 16)
# The port is device-agnostic: on the CPU it must reproduce the reference's arithmetic exactly. On the
# GPU a merge-cost near-tie can flip under another reduction order, which the GPU count reports.
ours = core.ward_labels(pools, 64, 4)
same = (ref_labels == ours).all(1)
check("ward partition == reference (CPU)", bool(same.all()) and bool(reached.all()),
      f"({int(same.sum())}/40 blocks identical)")
gpu_same = (ref_labels == core.ward_labels(pools.to(dev), 64, 4).cpu()).all(1)
check("ward partition on GPU == reference for >= 90% of blocks", gpu_same.float().mean() >= 0.9,
      f"({int(gpu_same.sum())}/40 blocks identical)")

# 2. Selection on a paged cache ---------------------------------------------------------------
T, BS = 24 * 512 + 77, 16
num_blocks = (T + BS - 1) // BS + 9
perm = torch.randperm(num_blocks)[: (T + BS - 1) // BS]
block_table = perm.int()[None, :].to(dev)
idx_keys = clustered_unit_rows(T, 64).to(torch.bfloat16)
tables = core.LayerTables.allocate(num_blocks * BS, dev, BS)
positions = torch.arange(T)
slots = (perm[positions // BS] * BS + positions % BS).to(dev)
core.write_index_keys(tables, idx_keys.to(dev), slots)
closed = T // 512
core.cluster_blocks(tables, block_table, torch.zeros(closed, dtype=torch.long, device=dev),
                    torch.arange(closed, device=dev) * 512, torch.ones(closed, dtype=torch.bool, device=dev), BS)
# The cache holds bf16 rows; the reference refuses rows off unit length by more than fp32 allows, and
# ours renormalizes the same way when it sums them, so hand it the bf16 values renormalized in fp32.
ref_keys = F.normalize(idx_keys.float(), dim=-1)
ref_buckets = reference.build_buckets(ref_keys, 4, 8, 16, 512)

design = core.Design(top_k=1024, top_p=256)  # small budget, so selection really selects
qpos = torch.tensor(sorted(set([600, 1023, 1024, 1500, 5000, 9999, T - 1] + torch.randint(512, T, (40,)).tolist())))
qrows = F.normalize(idx_keys.float()[qpos] + 0.3 * torch.randn(len(qpos), 64), dim=-1)
# The strict check: the reference's read over our own bucket tables (bf16 centroids, as upstream serves
# them, and our GPU Ward partition) - this is the selection and expansion port alone. The read over the
# reference's own fp32 buckets is reported beside it as key overlap.
anchors = torch.arange(closed * 64, device=dev) * 8
rows = core.bucket_rows(block_table, torch.zeros_like(anchors), anchors, BS)
own_members = tables.members[rows].long().cpu()
own_members = torch.where(own_members >= 0, own_members + (torch.arange(closed * 64) // 64 * 512)[:, None], -1)
own_buckets = ref_buckets._replace(bucket_centroids=tables.centroids[rows].float().cpu(),
                                   bucket_members=own_members.int(),
                                   bucket_sizes=(own_members >= 0).sum(1).int())
ref_sel, ref_len = reference.read(qrows, qpos, own_buckets, design.top_p, design.top_k, design.sink, design.window)
fp32_sel, fp32_len = reference.read(qrows, qpos, ref_buckets, design.top_p, design.top_k, design.sink, design.window)

n = len(qpos)
qsl = torch.tensor([0, n], dtype=torch.int32, device=dev)
treq = torch.zeros(n, dtype=torch.long, device=dev)
tpos = qpos.to(dev)
ce = torch.full((n,), closed * 512, device=dev)
tails = read.tail_starts(tpos, ce, design.window)
vis = read.visible_buckets(tpos, ce)
from hyperqwen_lsa import kernels  # noqa: E402
raw = kernels.score_buckets(qrows.to(dev), qsl, vis, block_table, tables.centroids, 0, n, T // 8 + 1, BS, 1, n)
# columns past a row's visible count are left unwritten by the kernel; the torch paths need them -inf
scores = torch.where(torch.arange(raw.shape[1], device=dev)[None] < vis[:, None], raw, float("-inf"))
best_p, picked_p = kernels.top_p(raw, vis, 0, design.top_p, parts=32)
order = torch.sort(-scores, dim=1, stable=True).indices[:, : design.top_p]
ok_top = all(
    torch.equal(picked_p[i, : min(int(vis[i]), design.top_p)].long(), order[i, : min(int(vis[i]), design.top_p)])
    for i in range(n))
check("top_p kernel == stable sort of the visible scores", ok_top)
sel, cnt = core.expand(tables, block_table, treq, tpos, tails, scores, BS, design)
agree = near = 0
for i in range(n):
    p = int(qpos[i])
    tail = int(tails[i])
    mine = set(range(min(design.sink, tail))) | set(sel[i, : int(cnt[i])].tolist()) | set(range(tail, p + 1))
    ref = set(ref_sel[i, : int(ref_len[i])].tolist())
    if mine == ref:
        agree += 1
    elif len(mine ^ ref) <= 2 * core.BUCKET_CAP:
        near += 1  # two buckets of near-equal score ranked the other way round, at the top-p or top-k cut
    else:
        print(f"      query {p}: mine {len(mine)} ref {len(ref)} sym-diff {len(mine ^ ref)}")
check("selection == reference over our buckets", agree + near == n and agree >= 0.9 * n,
      f"({agree}/{n} identical, {near} a near-tie swap at the cut)")
overlap = []
for i in range(n):
    a = set(ref_sel[i, : int(ref_len[i])].tolist())
    b = set(fp32_sel[i, : int(fp32_len[i])].tolist())
    overlap.append(len(a & b) / len(b))
print(f"      info: key overlap with the fp32-centroid reference: mean {sum(overlap)/n:.4f}, min {min(overlap):.4f}")

# 3. Attention --------------------------------------------------------------------------------
# The served path: the expansion written straight to physical slots, read by the slot kernel.
HQ, HKV, D = 24, 4, 256
q = torch.randn(n, HQ, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(T, HKV, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(T, HKV, D, device=dev, dtype=torch.bfloat16)
scale = D ** -0.5
width = design.sink + design.top_k + design.window + 512
slot_rows, slot_lens = kernels.expand_slots(picked_p, best_p, tables.members, block_table, treq, tpos, tails,
                                            design.top_k, design.sink, BS, width, cb=64)
lists = []
for i in range(n):
    p, tail = int(qpos[i]), int(tails[i])
    lists.append(list(range(min(design.sink, tail))) + sel[i, : int(cnt[i])].tolist() + list(range(tail, p + 1)))
check("expansion to slots == core.expand, in read order",
      all(slot_rows[i, : int(slot_lens[i])].tolist() == slots[torch.tensor(lists[i])].tolist() for i in range(n)))


def paged(x, dtype=torch.bfloat16, pad=0):
    cache = torch.zeros(num_blocks, BS, HKV, D + pad, device=dev, dtype=dtype)
    cache.view(-1, HKV, D + pad)[slots, :, :D] = x.to(dtype)
    return cache


def reference_out(kk, vv):
    """lsa.core.attention.topk_attention, one query at its absolute position, over its list."""
    outs = []
    for i in range(n):
        p = int(qpos[i])
        ids = torch.tensor(lists[i], dtype=torch.int32)[None, None]
        o = topk_attention(q[i:i + 1][None].float().cpu(), kk[: p + 1][None].float().cpu(),
                           vv[: p + 1][None].float().cpu(), ids, window=1, sink=0, scale=scale, query_start=p)
        outs.append(o[0, 0])
    return torch.stack(outs)


ref_o = reference_out(k, v)
kc, vc = paged(k), paged(v)
for splits in (1, 16):
    out = torch.empty_like(q)
    kernels.slot_attention(q, kc, vc, None, None, out, slot_rows, slot_lens, scale, num_splits=splits, block_n=16, num_warps=2)
    err = (out.float().cpu() - ref_o).abs().max().item()
    check(f"bf16 read == topk_attention (splits={splits})", err < 2e-2, f"(max err {err:.2e})")


def quant(x):
    s = x.float().abs().amax(-1) / 127
    return (x.float() / s[..., None]).round().clamp(-127, 127).to(torch.int8), s


kq, ks = quant(k)
vq, vs = quant(v)
kc8, vc8 = paged(kq, torch.int8, pad=4), paged(vq, torch.int8, pad=4)
ksc = torch.zeros(num_blocks, BS, HKV, device=dev)
vsc = torch.zeros(num_blocks, BS, HKV, device=dev)
ksc.view(-1, HKV)[slots] = ks
vsc.view(-1, HKV)[slots] = vs
ref_o8 = reference_out((kq.float() * ks[..., None]).to(torch.bfloat16), (vq.float() * vs[..., None]).to(torch.bfloat16))
for splits in (1, 16):
    out = torch.empty_like(q)
    kernels.slot_attention(q, kc8[..., :D], vc8[..., :D], ksc, vsc, out, slot_rows, slot_lens, scale,
                           num_splits=splits, block_n=16, num_warps=2)
    err = (out.float().cpu() - ref_o8).abs().max().item()
    check(f"int8_per_token_head read == topk_attention (splits={splits})", err < 3e-2, f"(max err {err:.2e})")

# A budget covering the context: the whole read path is exact dense causal attention
big = core.Design(top_k=T, top_p=4096)
out = torch.empty_like(q)
read.lsa_attention(q, torch.eye(HKV * 192, 64, device=dev), HKV, 64, tables, kc, vc, None, None, out, qsl,
                   block_table, treq, tpos, ce, 1, n, T // 8 + 1, scale, big, chunk=n, num_splits=4)
dense = []
for i in range(n):
    p = int(qpos[i])
    kk = k[: p + 1].float().repeat_interleave(HQ // HKV, 1).transpose(0, 1)
    vv = v[: p + 1].float().repeat_interleave(HQ // HKV, 1).transpose(0, 1)
    a = torch.softmax((q[i].float()[:, None, :] @ kk.transpose(1, 2)) * scale, -1)
    dense.append((a @ vv)[:, 0])
err = (out.float() - torch.stack(dense)).abs().max().item()
check("full budget, whole read path == dense causal attention", err < 2e-2, f"(max err {err:.2e})")

print("check_core:", "OK" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
