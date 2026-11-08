#!/usr/bin/env python3
"""Check native FlashInfer graph replanning against eager attention on sm75.

Run with the serving venv. Exercises causal verification and noncausal sliding
window drafting as KV lengths grow, shrink, and cross page boundaries.
"""
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

torch.manual_seed(0)
dev='cuda'; bs=32; max_pages=800
ws=torch.empty(64*1024*1024,dtype=torch.uint8,device=dev)
q=torch.randn(8,12,256,device=dev,dtype=torch.float16)
kv=torch.randn(max_pages,2,bs,2,256,device=dev,dtype=torch.float16)
qo=torch.tensor([0,8],dtype=torch.int32)
idx=torch.arange(max_pages,device=dev,dtype=torch.int32)
for causal,window in [(True,-1),(False,2047)]:
    wrapper=BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',use_cuda_graph=True,
        qo_indptr_buf=torch.empty(2,dtype=torch.int32,device=dev),
        paged_kv_indptr_buf=torch.empty(2,dtype=torch.int32,device=dev),
        paged_kv_indices_buf=torch.empty(max_pages,dtype=torch.int32,device=dev),
        paged_kv_last_page_len_buf=torch.empty(1,dtype=torch.int32,device=dev),backend='fa2')
    eager=BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',backend='fa2')
    out=torch.empty_like(q)
    def plan(w,n):
        pages=(n+bs-1)//bs
        w.plan(qo,torch.tensor([0,pages],dtype=torch.int32),idx[:pages],torch.tensor([(n-1)%bs+1],dtype=torch.int32),12,2,256,bs,causal=causal,window_left=window,q_data_type=torch.float16,kv_data_type=torch.float16)
    plan(wrapper,8)
    for _ in range(3): wrapper.run(q,kv,out=out)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): wrapper.run(q,kv,out=out)
    for n in [8,31,32,33,1024,23700,65,8]:
        plan(wrapper,n); graph.replay(); actual=out.clone()
        plan(eager,n); expected=eager.run(q,kv)
        torch.testing.assert_close(actual,expected,atol=0.003,rtol=0.003)
        print('PASS',causal,window,n,'maxerr',float((actual-expected).abs().max()),flush=True)
