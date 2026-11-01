"""The #86 pattern in the int8 prefill kernel (patches/prefill-attn-int8.patch): the block id is loaded from
the int32 block table and multiplied by the cache stride in int32, in three kernels (_k_stats_kernel,
_k_quant_kernel, _prefill_attn_kernel). On a pool larger than 2^31 elements / stride_kb (bf16, block 880:
901,120 elements per block, so block ids above 2,383) a request whose blocks sit high in the pool reads the
wrong memory. Places one request's blocks at the TOP of a 2,600-block pool and compares the kernel's output
with the same request placed at the BOTTOM of the same pool (identical K/V bytes, so the two must agree to
the bit whatever the int8 rounding does), plus a loose check against an fp32 reference on the exact-QK path.
Run inside the vLLM venv after the patches are applied (any card with ~10 GB free):
  venv/bin/python bench/test_prefill_attn_bigpool.py
Exit code 0 = all OK; a CUDA illegal access kills the process (that is the 'before' result)."""
import sys, torch
from vllm.v1.attention.backends.prefill_attn_hd256 import prefill_attn
torch.manual_seed(0); dev = "cuda"
Hq, Hkv, D, BS = 24, 4, 256, 880
scale = D ** -0.5
NB = 2600  # blocks in the pool: NB*BS*Hkv*D = 2.34e9 elements > 2^31
L, Q = 20000, 2048  # context tokens and the last chunk's query rows (chunked prefill: q_start_pos = L - Q)
nb_req = (L + BS - 1) // BS
kc = torch.empty(NB, BS, Hkv, D, device=dev, dtype=torch.bfloat16).normal_()
vc = torch.empty(NB, BS, Hkv, D, device=dev, dtype=torch.bfloat16).normal_()
top = torch.arange(NB - nb_req, NB, device=dev, dtype=torch.int32)     # the highest block ids
bottom = torch.arange(0, nb_req, device=dev, dtype=torch.int32)        # the same request, low in the pool
kc[bottom.long()] = kc[top.long()]; vc[bottom.long()] = vc[top.long()]  # identical bytes at both placements
q = torch.randn(Q, Hq, D, device=dev, dtype=torch.bfloat16)

def run(bt, int8_qk):
    out = torch.empty_like(q)
    prefill_attn(q, kc, vc, out, bt, L, L - Q, scale, int8_qk=int8_qk)
    torch.cuda.synchronize()
    return out

def reference():
    k = kc[bottom.long()].reshape(-1, Hkv, D)[:L].float(); v = vc[bottom.long()].reshape(-1, Hkv, D)[:L].float()
    G = Hq // Hkv; k = k.repeat_interleave(G, 1); v = v.repeat_interleave(G, 1)
    s_ = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    qpos = torch.arange(L - Q, L, device=dev)[:, None]; kpos = torch.arange(L, device=dev)[None, :]
    s_ = s_.masked_fill((kpos > qpos)[None], float("-inf")); p = torch.softmax(s_, -1)
    return torch.einsum("hqk,khd->qhd", p, v)

print("  (pre-fix expectation: the top-of-pool arm dies with a CUDA illegal memory access, or returns nan, or", flush=True)
print("   disagrees with the bottom-of-pool arm; measured both ways on two cards. That IS the bug, not a broken test)", flush=True)
ok = True
ref = reference()
for int8_qk in (False, True):
    lo = run(bottom, int8_qk)
    hi = run(top, int8_qk)
    same = torch.equal(lo, hi)
    e_lo = (lo.float() - ref).abs().max().item()
    e_hi = (hi.float() - ref).abs().max().item()
    tol = 0.05 if not int8_qk else 0.5   # the int8-QK path is approximate by design; the bit-equality is the sharp check
    row_ok = same and e_lo < tol and e_hi < tol
    ok = ok and row_ok
    print(f"  int8_qk={int8_qk}: pool {NB} blocks, request on blocks {NB-nb_req}..{NB-1} vs 0..{nb_req-1}, L={L}, chunk {Q}: "
          f"top==bottom {'yes' if same else 'NO'}; max|ours-ref| bottom {e_lo:.4f} top {e_hi:.4f} {'OK' if row_ok else 'FAIL'}", flush=True)
print("PREFILL BIGPOOL OK" if ok else "PREFILL BIGPOOL FAIL", flush=True)
sys.exit(0 if ok else 1)
