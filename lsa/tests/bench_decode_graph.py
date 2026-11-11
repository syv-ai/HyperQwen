# Decode-step cost of the LSA read under CUDA-graph replay, component by component.
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from hyperqwen_lsa import core, kernels, read
dev = torch.device("cuda"); T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
HQ, HKV, D, ROT, BS, N = 24, 4, 256, 64, 816, 4
nb = (T + BS - 1) // BS + 2
bt = torch.randperm(nb, device=dev)[None, : nb - 2].int().contiguous()
kc = torch.randint(-127, 127, (nb, BS, HKV, D + 4), dtype=torch.int8, device=dev)
vc = torch.randint(-127, 127, (nb, BS, HKV, D + 4), dtype=torch.int8, device=dev)
ks = torch.rand(nb, BS, HKV, device=dev) / 100; vs = torch.rand(nb, BS, HKV, device=dev) / 100
tables = core.LayerTables.allocate(nb * BS, dev, BS)
tables.centroids.copy_(torch.nn.functional.normalize(torch.randn_like(tables.centroids.float()), dim=-1))
tables.members.copy_(torch.randint(0, 512, tables.members.shape, device=dev, dtype=torch.int16))
w_q = torch.randn(HKV * (D - ROT), 64, device=dev)
q = torch.randn(N, HQ, D, device=dev, dtype=torch.bfloat16); out = torch.empty_like(q)
qsl = torch.tensor([0, N], dtype=torch.int32, device=dev); treq = torch.zeros(N, dtype=torch.long, device=dev)
tpos = torch.arange(T - N, T, device=dev); ce = torch.full((N,), (T - N) // 512 * 512, device=dev)
design = core.Design(); nb_max = 150000 // 8 + 1
splits = int(os.environ.get("SPLITS", 16))
qidx = core.index_queries(q, w_q, HKV, ROT); tails = read.tail_starts(tpos, ce, 512); vis = read.visible_buckets(tpos, ce)
state = {}
def s_qidx(): state["qidx"] = core.index_queries(q, w_q, HKV, ROT)
def s_score(): state["sc"] = kernels.score_buckets(qidx, qsl, vis, bt, tables.centroids, 0, N, nb_max, BS, 1, N)
s_score()
def s_topk(): state["tk"] = kernels.top_p(state["sc"], vis, 0, 1024, parts=32)
s_topk()
def s_expand(): state["sl"] = kernels.expand_slots(state["tk"][1], state["tk"][0], tables.members, bt, treq, tpos, tails, 4096, 4, BS, 4 + 4096 + 1024, cb=64)
s_expand()
def s_attn(): kernels.slot_attention(q, kc[..., :D], vc[..., :D], ks, vs, out, state["sl"][0], state["sl"][1], D ** -0.5, num_splits=splits, block_n=16, num_warps=2)
def s_all(): read.lsa_attention(q, w_q, HKV, ROT, tables, kc[..., :D], vc[..., :D], ks, vs, out, qsl, bt, treq, tpos, ce, 1, N, nb_max, D ** -0.5, design, chunk=N, num_splits=splits)
for name, fn in [("index query", s_qidx), ("score", s_score), ("topk", s_topk), ("expand", s_expand), ("attention", s_attn), ("whole read", s_all)]:
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(16): fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(10): g.replay()
    b.record(); torch.cuda.synchronize()
    print(f"{name}: {a.elapsed_time(b) / 160 * 1000:.1f} us a layer (graph), x16 = {a.elapsed_time(b) / 10:.2f} ms a step")
