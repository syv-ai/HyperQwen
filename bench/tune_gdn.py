"""Microbenchmark of the Gated DeltaNet packed-decode kernel (vLLM's fused_recurrent) over
block sizes (BV) and warp counts, at batch 24 and 48 with Qwen3.8-27B's decode shapes. Each row
is one config's time per call and the state bandwidth it reached. A measurement, not a gate.
  venv/bin/python bench/tune_gdn.py    # GPU
"""
import torch, triton, sys, importlib

from vllm.third_party.flash_linear_attention.ops import fused_recurrent as fr

# Shapes from Qwen3.8-27B decode: H=16 qk heads, HV=48 v heads, K=V=128
H,HV,K,V = 16,48,128,128
SLOTS=80
dev="cuda"
torch.manual_seed(0)

def bench(N, BV, num_warps):
    mixed = torch.randn(N, 2*H*K+HV*V, dtype=torch.bfloat16, device=dev)
    a = torch.randn(N, HV, dtype=torch.bfloat16, device=dev)
    b = torch.randn(N, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev)
    o = torch.empty(N, HV*V, dtype=torch.bfloat16, device=dev)
    state = torch.randn(SLOTS, HV*V*K, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(1, N+1, dtype=torch.int32, device=dev).unsqueeze(1)
    NV = triton.cdiv(V, BV)
    grid = (NV, N*HV)
    args = dict(mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias, o=o,
        h0=state, ht=state, ssm_state_indices=idx, scale=K**-0.5,
        stride_mixed_qkv_tok=mixed.stride(0), stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0), stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0), stride_indices_seq=idx.stride(0),
        H=H, HV=HV, K=K, V=V, BK=K, BV=BV,
        SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=True)
    fn = lambda: fr.fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](**args, num_warps=num_warps)
    fn()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=10, rep=50)
    gb = 2*N*HV*V*K*2/1e9
    print(f"N={N} BV={BV} warps={num_warps}: {ms*1000:.1f}us  {gb/ms*1000:.0f} GB/s")
    return ms

for N in (24, 48):
    print(f"--- batch {N} ---")
    for BV,W in [(32,1),(32,2),(32,4),(64,2),(64,4),(64,8),(128,4),(128,8),(128,16)]:
        try: bench(N,BV,W)
        except Exception as e: print(f"BV={BV} warps={W}: FAIL {str(e)[:80]}")
