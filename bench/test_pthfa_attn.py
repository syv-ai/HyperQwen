"""The pth-flash-attn op (pthfa/, PTH_FA=1) against an fp64 reference on the int8_per_token_head cache it reads.

Three parts, all on the cache views the CUSTOM backend passes to the op (PthFlashAttnImpl._caches over a packed
[blocks, ., ., 2 * (D + 4)] int8 cache, scales inline):
  1. shapes: head sizes 64/128/256, the target's GQA group at TP=1 (24 q / 4 kv) and TP=2 (12 / 2), decode,
     MTP verify (4 query tokens per request) and chunked prefill, both cache layouts, one and several KV splits,
     bf16 and fp16. Tolerance from the kernel's rounding-error analysis, per element:
     2^-8 * (p @ |v|) + u_out * |o| + 2^-24 (proven for L <= 4096).
  2. big pool (#86/#109): one request's blocks at the TOP of a 2,600-block pool of 816-token blocks (the hybrid
     group's block size), 4.4 GB of int8 cache, so block ids run past 2,383 and byte offsets past 2^32; every other
     block is poisoned, so a wrapped offset reads values ~1e5 off. L = 4,000 against the proven tolerance, and
     L = 20,000 / 131,072 against bench/test_spec_decode_bigpool.py's 0.05.
  3. CUDA-graph replay: a captured decode/verify launch, replayed after seq_lens grew on the device (as vLLM's
     FULL graphs replay), equals the eager launch bit for bit and the reference, with one and several splits.

Run inside the image or a venv with pthfa/install.sh done (production stopped; ~6 GB of GPU memory):
  venv/bin/python bench/test_pthfa_attn.py
Designed skips, exit 0: no CUDA device; pth_flash_attn not installed; a card other than sm_80/86/89 (the library
has no cubin for it, by design). Exit 0 = all OK, 1 = a case failed (named)."""
import math, sys
import torch
try:
    import pth_flash_attn.ops as ops
    from pth_flash_attn.planner import plan_launch, workspace_token_splits
    from pth_flash_attn.vllm_backend import PthFlashAttnImpl
except ImportError as e:
    print(f"pthfa: SKIP(by design): pth_flash_attn is not installed here ({e}); bash pthfa/install.sh"); sys.exit(0)
if not torch.cuda.is_available():
    print("pthfa: SKIP(by design): no CUDA device"); sys.exit(0)
cap = torch.cuda.get_device_capability()
if cap[0] != 8:
    print(f"pthfa: SKIP(by design): sm{cap[0]}{cap[1]}; the library has SASS for sm80/86/89 only"); sys.exit(0)
if not ops.is_available():
    print(f"pthfa: FAIL: the package is installed but its op library is not ({ops.LIBRARY_PATH})"); sys.exit(1)
DEV = "cuda"
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
OUT_ROUNDING = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11}
RESULTS = []


def report(ok, what):
    print(f"  {'OK  ' if ok else 'FAIL'}  {what}", flush=True)
    RESULTS.append(ok)


def impl_for(hq, hkv, d):
    return PthFlashAttnImpl(num_heads=hq, head_size=d, scale=d ** -0.5, num_kv_heads=hkv, alibi_slopes=None,
                            sliding_window=None, kv_cache_dtype="int8_per_token_head")


def make_cache(impl, nb, bs, hkv, d, layout, fill=True):
    shape = (nb, bs, hkv, 2 * (d + 4)) if layout == "NHD" else (nb, hkv, bs, 2 * (d + 4))
    kv = torch.empty(shape, dtype=torch.int8, device=DEV)
    if layout == "NHD":
        kv = kv.transpose(1, 2)
    k, v, ks, vs = impl._caches(kv)
    if fill:
        k.copy_(torch.randint_like(k, -127, 128)); v.copy_(torch.randint_like(v, -127, 128))
        ks.copy_(torch.rand_like(ks) * 0.5 / 127 + 0.5 / 127); vs.copy_(torch.rand_like(vs) * 0.5 / 127 + 0.5 / 127)
    return k, v, ks, vs


def reference(q, caches, bt, seq_lens, q_lens, scale):
    """fp64 causal attention on the dequantized cache; returns (o, p @ |v|)."""
    k, v, ks, vs = caches
    bs, hkv = k.shape[1], k.shape[2]
    g = q.shape[1] // hkv
    out = torch.empty(q.shape, dtype=torch.float64, device=DEV); absv = torch.empty_like(out)
    start = 0
    for r, (L, ql) in enumerate(zip(seq_lens, q_lens)):
        slots = torch.arange(L, device=DEV); blocks = bt[r, slots // bs].long(); offs = slots % bs
        vis = slots[None, :] <= (L - ql + torch.arange(ql, device=DEV))[:, None]
        for h in range(hkv):  # one KV head at a time keeps a 131k-token reference in a few hundred MB
            kd = k[blocks, offs, h].double() * ks[blocks, offs, h].double()[:, None]   # [L, D]
            vd = v[blocks, offs, h].double() * vs[blocks, offs, h].double()[:, None]
            qh = q[start:start + ql, h * g:(h + 1) * g].double()                        # [q, G, D]
            p = (torch.einsum("qgd,kd->gqk", qh, kd) * scale).masked_fill(~vis, -math.inf).softmax(-1)
            out[start:start + ql, h * g:(h + 1) * g] = torch.einsum("gqk,kd->qgd", p, vd)
            absv[start:start + ql, h * g:(h + 1) * g] = torch.einsum("gqk,kd->qgd", p, vd.abs())
        start += ql
    return out, absv


def launch(q, caches, bt, seq_lens_t, qsl, q_lens, scale, hkv, d, ws_ts, out=None):
    g = q.shape[1] // hkv; qmax = max(q_lens)
    plan = plan_launch(num_tokens=q.shape[0], num_reqs=len(q_lens), max_query_len=qmax, G=g, Hkv=hkv, D=d,
                       num_sms=NUM_SMS, is_prefill=qmax > 64 // g, ws_token_splits=ws_ts)
    ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=DEV)
    ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=DEV)
    out = torch.full_like(q, float("nan")) if out is None else out
    ops.pth_flash_attn_fwd(out, q, *caches, bt, seq_lens_t, qsl, qmax, plan.wm, plan.num_splits, scale, ws_o, ws_lse)
    return out, plan


def case(hq, hkv, d, seq_lens, q_lens, dtype, layout, bs, ws_ts):
    impl = impl_for(hq, hkv, d)
    nbr = [math.ceil(L / bs) for L in seq_lens]
    nb = sum(nbr) + 1
    caches = make_cache(impl, nb, bs, hkv, d, layout)
    perm = torch.randperm(nb, device=DEV, dtype=torch.int32)
    bt = torch.zeros(len(seq_lens), max(nbr), dtype=torch.int32, device=DEV); used = 0
    for r, n in enumerate(nbr):
        bt[r, :n] = perm[used:used + n]; used += n
    q = torch.randn(sum(q_lens), hq, d, dtype=dtype, device=DEV)
    qsl = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    out, plan = launch(q, caches, bt, sl, qsl, q_lens, impl.scale, hkv, d, ws_ts)
    torch.cuda.synchronize()
    ref, absv = reference(q, caches, bt, seq_lens, q_lens, impl.scale)
    tol = 2.0 ** -8 * absv + OUT_ROUNDING[dtype] * ref.abs() + 2.0 ** -24
    worst = ((out.double() - ref).abs() / tol).max().item()
    ok = bool(torch.isfinite(out).all()) and worst <= 1.0
    report(ok, f"D={d} Hq/Hkv={hq}/{hkv} q_lens={q_lens} L={seq_lens} {str(dtype)[6:]} {layout} bs={bs} "
               f"splits={plan.num_splits}: max err/tol {worst:.3f}")


print(f"== pthfa attention, {torch.cuda.get_device_name(0)} (sm{cap[0]}{cap[1]}, {NUM_SMS} SMs)", flush=True)
torch.manual_seed(0)
print("== 1. shapes against the proven tolerance", flush=True)
SHAPES = {"decode": ([1, 17, 64, 1000, 4096], [1, 1, 1, 1, 1]),
          "verify": ([4, 333, 4096], [4, 4, 4]),
          "prefill": ([512, 1500, 37], [512, 300, 37])}
for d in (64, 128, 256):
    for hq, hkv in ((24, 4), (12, 2)):
        for name, (sl, ql) in SHAPES.items():
            case(hq, hkv, d, sl, ql, torch.bfloat16, "NHD", 16, 1 << 20)
for layout in ("NHD", "HND"):
    for bs in (16, 816):
        for ws in (0, 1 << 20):  # ws 0 forces one split
            case(24, 4, 256, *SHAPES["verify"], torch.bfloat16, layout, bs, ws)
case(24, 4, 256, *SHAPES["verify"], torch.float16, "NHD", 816, 1 << 20)

print("== 2. big pool: block ids past 2,383, byte offsets past 2^32", flush=True)
NB, BS = 2600, 816
for hq, hkv in ((24, 4), (12, 2)):
    d = 256
    impl = impl_for(hq, hkv, d)
    caches = make_cache(impl, NB, BS, hkv, d, "NHD", fill=False)
    k, v, ks, vs = caches
    k.fill_(127); v.fill_(127); ks.fill_(1000.0); vs.fill_(1000.0)   # poison: a wrong block reads ~1.3e5
    ws_ts = workspace_token_splits(max_num_seqs=8, q_dec_max=4, Hkv=hkv, D=d, num_sms=NUM_SMS)
    for L, q_len, proven in ((4000, 4, True), (20000, 4, False), (131072, 4, False), (131072, 1, False)):
        nbr = math.ceil(L / BS)
        ids = torch.arange(NB - nbr, NB, device=DEV, dtype=torch.int32)
        idl = ids.long()
        k[idl] = torch.randint_like(k[idl], -127, 128); v[idl] = torch.randint_like(v[idl], -127, 128)
        ks[idl] = torch.rand_like(ks[idl]) * 0.5 / 127 + 0.5 / 127; vs[idl] = torch.rand_like(vs[idl]) * 0.5 / 127 + 0.5 / 127
        bt = ids.reshape(1, nbr)
        q = torch.randn(q_len, hq, d, dtype=torch.bfloat16, device=DEV)
        qsl = torch.tensor([0, q_len], dtype=torch.int32, device=DEV); sl = torch.tensor([L], dtype=torch.int32, device=DEV)
        out, plan = launch(q, caches, bt, sl, qsl, [q_len], impl.scale, hkv, d, ws_ts)
        torch.cuda.synchronize()
        ref, absv = reference(q, caches, bt, [L], [q_len], impl.scale)
        err = (out.double() - ref).abs()
        if proven:
            tol = 2.0 ** -8 * absv + OUT_ROUNDING[torch.bfloat16] * ref.abs() + 2.0 ** -24
            worst = (err / tol).max().item(); ok = worst <= 1.0; how = f"max err/tol {worst:.3f}"
        else:
            e = err.max().item(); ok = e < 0.05; how = f"max|ours-ref| {e:.4f} (< 0.05)"
        ok = ok and bool(torch.isfinite(out).all())
        byte_end = (int(ids[-1]) + 1) * k.stride(0)
        report(ok, f"Hq/Hkv={hq}/{hkv} L={L} q={q_len} blocks {NB - nbr}..{NB - 1} (last byte {byte_end:.3e} > 2^32 "
                   f"= {byte_end > 2 ** 32}) splits={plan.num_splits}: {how}")
        k[idl] = 127; v[idl] = 127; ks[idl] = 1000.0; vs[idl] = 1000.0
    del caches, k, v, ks, vs
    torch.cuda.empty_cache()

print("== 3. CUDA-graph replay after seq_lens grow on the device", flush=True)
for ws in (0, 1 << 20):
    hq, hkv, d, bs = 24, 4, 256, 816
    impl = impl_for(hq, hkv, d)
    nbr = math.ceil(4096 / bs)
    caches = make_cache(impl, 2 * nbr, bs, hkv, d, "NHD")
    bt = torch.arange(2 * nbr, dtype=torch.int32, device=DEV).view(2, nbr)
    q = torch.randn(8, hq, d, dtype=torch.bfloat16, device=DEV)
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEV)
    sl = torch.tensor([100, 200], dtype=torch.int32, device=DEV)
    out_g = torch.empty_like(q)
    _, plan = launch(q, caches, bt, sl, qsl, [4, 4], impl.scale, hkv, d, ws, out=out_g)   # warm-up, fixes the plan
    ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=DEV)
    ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=DEV)
    run = lambda o: ops.pth_flash_attn_fwd(o, q, *caches, bt, sl, qsl, 4, plan.wm, plan.num_splits, impl.scale,
                                           ws_o, ws_lse)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(out_g)
    sl.copy_(torch.tensor([1500, 4096], dtype=torch.int32))
    g.replay()
    out_e = torch.empty_like(q); run(out_e); torch.cuda.synchronize()
    same = torch.equal(out_g, out_e)
    ref, absv = reference(q, caches, bt, [1500, 4096], [4, 4], impl.scale)
    tol = 2.0 ** -8 * absv + OUT_ROUNDING[torch.bfloat16] * ref.abs() + 2.0 ** -24
    worst = ((out_g.double() - ref).abs() / tol).max().item()
    report(same and worst <= 1.0, f"splits={plan.num_splits}: L 100/200 -> 1500/4096 on the device; replay == eager "
                                  f"bit for bit: {same}; vs reference max err/tol {worst:.3f}")
ok = all(RESULTS)
print(f"PTHFA ATTN {'OK' if ok else 'FAIL'} ({sum(RESULTS)}/{len(RESULTS)})"); sys.exit(0 if ok else 1)
