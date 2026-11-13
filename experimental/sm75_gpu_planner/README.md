# SM75 native FA2 GPU planner

This opt-in integration moves per-step split-KV scheduling and exact-length
page metadata updates to a GPU kernel. Initial shape and module planning
remain on the CPU. The scheduler mirrors FlashInfer's
`PrefillSplitQOKVIndptr` and binary chunk-size search.

It updates request, query and KV tile indices; merge/output pointers; valid
masks; chunk size; and exact GPU page lengths. Exact attention lengths come
from GPU sequence lengths rather than CPU upper bounds.

## Supported path

Tested with vLLM 0.33.0, FlashInfer 0.7.2.post1, native FlashInfer speculative
graphs, FP16 Q/KV and TP2 on 68-SM RTX 2080 Ti cards. The integration covers
uniform eight-query speculative batches of one or two requests, full
attention and the DFlash2 sliding window. DCP, sinks, adaptive verification
and fixed split overrides are outside this path. Other configurations retain
the existing host planning path.

The implementation depends on FlashInfer's private 15-field FA2 plan layout.
Compatibility with other releases is not established.

## Recorded validation

- 352 exact attention-output cases covering growing/shrinking contexts,
  page/split boundaries, one/two requests, causal attention and window-2047
  attention, without CPU replanning after initialization.
- 240 exact integer-plan comparisons through length 262,136, including zero
  lengths and page sizes 16, 32, 64, 128 and 512.
- Paired serving output/counter comparisons, concurrent requests and cached
  prefix repeats on the tested NVLink and PCIe/NCCL hosts.

Attention-output checks extend through 23,700 tokens. Maximum-length metadata
coverage does not establish maximum-context attention correctness or speed.
See [planner results](results.json) and the
[complete bundle's serving measurements](../sm75_pr_review.md).

## Reproduce kernel checks

Use the matching environment and CUDA SDK:

```bash
CUDA_HOME=/opt/cuda venv/bin/python experimental/sm75_gpu_planner/check_attention.py
CUDA_HOME=/opt/cuda venv/bin/python experimental/sm75_gpu_planner/check_metadata.py
```

Results are written under `/tmp/hq-gpu-plan-tests`. Run with available GPU
memory and separately from timed serving measurements.

## Integration

In an isolated copy of the patched vLLM package, apply
`flashinfer_gpu_plan.patch` from the package root. Copy `hq_gpu_plan.py` beside
`v1/attention/backends/flashinfer.py`.

The wheel-only Python patch series does not enable this integration. Explicit
version/layout validation is required before adding it to that series.
