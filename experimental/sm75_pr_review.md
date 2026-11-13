# SM75 GPU planning and Marlin load pipeline

This change reduces CPU attention-planning work and overlaps Marlin weight
loads with tensor-core computation on the tested SM75 configuration.

## Changes

- Move per-step native FA2 split-KV scheduling and exact-length page metadata
  updates to a GPU kernel. Initial shape and module planning remain on the CPU.
- Launch 136 blocks instead of 68 for the tested gate/up verification shape,
  using the existing kernel and FP16 arithmetic.
- Use streaming-cache loads for small-batch W4 weights.
- For seven M8 W4 projection shapes, buffer the next tile's inputs, weights and
  scales in registers after the shared-memory barrier. Commit those values to
  shared memory after overlapping the loads with the current tensor-core work.

The accumulator K order, FP16 rounding, model weights and sampling are unchanged.
W8 output heads retain their original load policy. GDN and collective settings
are unchanged. These are opt-in integrations; the native changes require a
rebuild and are not installed by the wheel-only Python patch series.

## Measured decode performance

The complete bundle was compared with the existing vLLM 0.33 wheel and common
HyperQwen Python patches, with experimental changes disabled in the baseline.
Each context used three sampled seeds, cold prefixes, normal EOS and fixed
output caps. The trial order was baseline, candidate, returning baseline.

Throughput values below are arithmetic means across three seeds, in tokens/s,
excluding prompt processing. Baseline values average the before/after control
means.

| Input context | PCIe/NCCL baseline | PCIe/NCCL candidate | NVLink baseline | NVLink candidate |
|---|---:|---:|---:|---:|
| 0.07k (71 tokens) | 102.48 | 106.03 | 109.96 | 115.81 |
| 4k (4,102 tokens) | 122.69 | 126.44 | 131.76 | 138.25 |
| 8k (8,072 tokens) | 98.58 | 101.38 | 117.53 | 123.37 |
| 24k (24,098 tokens) | 145.99 | 150.26 | 149.22 | 156.41 |

The equal-context geometric mean gain was **3.06% on PCIe/NCCL** and **5.01% on
NVLink**. Gains use each seed's candidate rate divided by the geometric mean
of its two controls, then aggregate those ratios geometrically. They need not
equal ratios of the arithmetic means in the table.

NVLink context means ranged from 4.82% to 5.32%. The 5.01% aggregate does not
establish a 5% minimum or confidence bound. Control drift was at most 0.57%
on PCIe/NCCL and 0.24% on NVLink. Different contexts use different fixtures and
output caps; this is not a controlled context-length scaling study.

See [per-seed serving results](sm75_marlin_register_pipe/serving-results.json).
Serving fixtures and token IDs are not included in the source bundle.

## Validation

- All paired serving output token IDs and generation/speculation counters
  matched. Measurements required completed streams with `[DONE]`.
- Twelve native operator cases passed per variant and host, including negative
  scales, fallback batches, all selected geometries and the full W8 head.
  Comparisons used int16 views for exact FP16 bits.
- Local Compute Sanitizer checked the seven selected register-buffered
  projection shapes with zero memory errors and outputs matching baseline.
- Eight routed LiteLLM checks passed on the candidate, covering streaming,
  nonstreaming, cached prefixes, concurrent requests and tool calls.
- The unchanged GPU planner's recorded validation includes 352 exact attention
  output cases and 240 exact integer-plan comparisons through length 262,136.
  Attention-output cases extend through 23,700 tokens.

## Tested configuration and limits

Pairs of modified 22 GB RTX 2080 Ti cards at 250 W; TP2; FP16 model, KV and
recurrent state; seven draft tokens; two maximum concurrent requests; model
capacity 262,136. The software stack was vLLM 0.33.0, FlashInfer 0.7.2.post1,
torch 2.14/cu130 and CUDA 13.4.92. Native builds used C++20, Release and SM75.

The PCIe host used NCCL and the NVLink host used custom all-reduce. CPUs,
kernels and clocks differed, so compare variants within a host. Native build,
package and compiler-cache paths also differed from the installed wheel;
these measurements describe the complete build/profile comparison rather
than isolated contributions from each component.

The planner uses FlashInfer's private 15-field FA2 plan layout and requires
version/layout validation before broader integration. Maximum-context
attention performance, other model geometries and long-duration load have
not been qualified. Metadata checks at length 262,136 do not establish full
attention correctness or performance at that length.

## Integration and checks

- [GPU planner](sm75_gpu_planner/README.md)
- [Native occupancy dependency](sm75_marlin/README.md)
- [Register-buffered native pipeline and operator checks](sm75_marlin_register_pipe/README.md)
