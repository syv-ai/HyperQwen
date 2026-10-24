# Bounded FA2 prefill over the PTH INT8 cache

The upstream base remains pth-flash-attn v0.1.0,
`b0d4f7ee228448ecc3d85a2015db208ed8d21954`. CUDA/C++ sources are unchanged.
Two local Python modules implement the optional prefill adapter:
`python/pth_flash_attn/fa2_slab.py` and `fa2_slab_plan.py`.
`fa2-slab.patch` records the six-line integration into upstream `vllm_backend.py`.
`check_vendor.sh --source DIR` reproduces that integration over the pinned
export before comparing; unmodified upstream files still compare byte for byte.

## Enable

Install normally with `bash pthfa/install.sh` or build the standard Dockerfile.
The install stamp includes the source receipt so an existing venv is refreshed.
Set `PTH_FA2_SLAB_PREFILL=1` in the server environment alongside `PTH_FA=1`.
For example, in a prepared checkout:

```bash
PTH_FA=1 PTH_FA2_SLAB_PREFILL=1 SPEC=mtp CTX=long MAX_SEQS=1 \
  bash single-user/start_qwen.sh
```

The default is `PTH_FA2_SLAB_PREFILL=0`. The adapter is qualified for sm86,
BF16 queries, 24 query / 4 KV heads, head size 256, a single causal request,
no sliding window, and 17–2,048 query tokens outside CUDA graph capture.
Other devices, shapes, batches, decode, mixed prefill/decode and graph capture
retain native PTH. It never changes native PTH decode/verification kernels.

## Attention and memory

Paged INT8 K/V are gathered and dequantized into balanced BF16 slabs of at most
16,384 tokens. Each slab calls the installed, unchanged FA2 implementation.
The final slab alone uses FA2's bottom-right causal mask; earlier slabs are
fully visible. Normalized partial outputs are merged in FP32 using FA2's
log-sum-exp statistics. Every key participates; the context is not truncated.
Sequence lengths come from host metadata without GPU-to-host scalar reads.

| Persistent allocation shared across sequential layers | MiB |
| --- | ---: |
| BF16 K/V, 16,384 tokens | 64 |
| BF16 partial output, 2,048 query tokens | 24 |
| FP32 output accumulator | 48 |
| FP32 normalization accumulator | 0.1875 |
| Cumulative KV lengths | 8 bytes |
| **Total** | **136.1875 MiB + 8 bytes** |

The workspace is reserved at model construction, before KV-pool sizing, shared
across the serving stack, and restricted to one CUDA stream. It does not grow
with context length. FA2's small transient internal/LSE allocations are additional.
Chunks above 2,048 tokens fall back to native PTH. Floating-point rounding can
differ from native PTH or one full-buffer FA2 call, including generated tokens;
the numerical tests are not a model quality evaluation.

## Validation and measurements

```bash
python3 bench/test_fa2_slab_plan.py
python3 bench/test_fa2_slab_dispatch.py
bash bench/test_pthfa_install_toolchain.sh
bash pthfa/check_vendor.sh
# In an installed vLLM/PTH/FA2 environment, with the sm86 GPU free:
venv/bin/python bench/test_fa2_slab_gpu.py --allow-gpu
# Optional isolated attention timings:
venv/bin/python bench/test_fa2_slab_gpu.py --allow-gpu --benchmark
```

RTX 3090 numerical checks cover 17–153,088 KV tokens, HND/NHD physical layouts,
shuffled pages, padded table/output strides, poisoned unused tails, causal slab
boundaries, extreme LSE differences and empty partitions. Gathered K/V and
single-slab FA2 match the independent full-buffer reference bit for bit.
Multi-slab normalized RMS error is at most 0.2929% against full-buffer FA2
and 0.2684% against independent FP32 first/last-query attention. Allocation is
exactly 142,802,952 persistent bytes.

The 2026-10-23 full-model comparison used Qwen3.8-27B W4A16, INT8 KV, MTP4,
2,048-token chunks, one sequence, a 5,888 MiB KV pool, 153,600-token limit,
CPU vision offload, and 512 output tokens. Both arms used the same image and
differed only in `PTH_FA2_SLAB_PREFILL`. Order was stock/slab/slab/stock,
with 29 measurements per trial (116 requests total), after probes and warmup.
These measurements used the tested local integration image; upstream's other
open fixes were held constant across both arms.

| Cold input tokens | Native PTH prefill tok/s | FA2 slab prefill tok/s |
| ---: | ---: | ---: |
| 8,192 | 1,324 | 1,329 |
| 32,768 | 1,174 | 1,191 |
| 65,536 | 1,015 | 1,057 |
| 131,072 | 806 | 864 |
| 150,528 | 760 | 819 |

Weighted cold prefill throughput increased 6.53%; the 150,528-token row gained
7.79%. Aggregate request time across cold, cached and growing conversations
fell 5.20%. The slab arm used about 136 MiB more GPU memory; minimum free VRAM
was 2,594–2,596 MiB versus 2,730–2,732 MiB for stock, with no observed increase
in Windows shared GPU memory. All runs completed, text/tool/vision probes
passed and all 40 cold/cached recall checks passed.

Cold prompt tokens match between arms; generated outputs differ, so cached
continuations can differ too. The 24 growing-conversation pairs with identical
prompt and output tokens improved weighted prefill throughput by 5.05% and
request time by 3.57%. This is prefill acceleration; decode still uses PTH,
and differing MTP acceptance must not be claimed as a decode kernel speedup.
