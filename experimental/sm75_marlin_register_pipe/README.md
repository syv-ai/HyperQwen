# SM75 register-buffered Marlin loads

This native integration buffers the next tile's global input, weight and
scale loads in registers after the shared-memory barrier. Those loads can
overlap the current tensor-core MMAs before the buffered values are committed
to the next shared stage. Accumulator K order and FP16 arithmetic are unchanged.

## Dispatch scope

The register-buffered path requires SM75 with 68 SMs, M8, FP16
activations/scales/output, symmetric U4B8 weights, group size 128, no bias or
zero points, non-atomic reduction, FP32 global reduction and the
128x128/256-thread kernel with two shared stages. It selects these K/N shapes:

| K | N |
|---:|---:|
| 8,704 | 5,120 |
| 3,072 | 5,120 |
| 25,600 | 5,120 |
| 2,048 | 5,120 |
| 5,120 | 8,192 |
| 5,120 | 7,168 |
| 5,120 | 3,072 |

Gate/up uses the occupancy specialization and W4 streaming loads. W8 output
heads retain their original load policy. Other batches and unsupported paths
retain the existing kernel. GDN, collective settings, model weights, context
capacity and sampling are unchanged.

## Measured results

The complete GPU-planner/native bundle measured a geometric mean decode gain
of **5.01% on NVLink** and **3.06% on PCIe/NCCL** against matched stock controls.
Each context used three seeds. NVLink context means ranged from 4.82% to
5.32%; the aggregate does not establish a 5% minimum or confidence bound.
All paired output token IDs and generation/speculation counters matched.

See [serving-results.json](serving-results.json) for per-seed measurements and
[the measurement report](../sm75_pr_review.md) for token rates, configuration,
control methodology and limitations. Serving fixtures are not included.

Native validation passed 12 cases per variant and host, including negative
scales, fallback batches, all selected shapes and the full W8 head. Local
Compute Sanitizer checked all seven selected register-buffered shapes with
zero memory errors and outputs matching baseline. Eight routed LiteLLM checks
passed, covering streaming, nonstreaming, cached prefixes, concurrent requests
and tool calls.

[Kernel results](kernel-results.json) contain the isolated timings and bit
comparisons. Nominal weight bandwidth and read-loop calibration are not DRAM
hardware-counter measurements, and kernel gains are not serving gains.

## Apply and build

Tested source: vLLM 0.33.0 commit
`4ba787ec9959e566d3b587a796dff07673324da5`, torch 2.14/cu130, CUDA 13.4.92,
C++20/Release/SM75.

1. Start from that vLLM source in a separate checkout and environment.
2. Apply `../sm75_marlin/native-marlin-occupancy.patch` from the source root.
3. Apply `native-register-pipe.patch` from the source root. This includes W4
   streaming loads; do not also apply `streaming-load.patch`.
4. Rebuild the native core with the matching vLLM build instructions and
   dependencies, including the new `marlin_register_pipe.cu` source.
5. For the measured complete bundle, use the common HyperQwen Python patches
   and the [GPU-planner integration](../sm75_gpu_planner/README.md).

The wheel-only Python patch series does not install these native changes.

## Check native correctness

Run in separate baseline and modified environments, sharing the output directory:

```bash
/path/to/baseline/bin/python experimental/sm75_marlin_register_pipe/check_native.py --kind baseline
/path/to/modified/bin/python experimental/sm75_marlin_register_pipe/check_native.py --kind modified
```

The default output directory is `/tmp/hq-sm75-register-check`; override with
`--output-dir`. The checker expects 68 blocks for the first baseline kernel
and 136 for the modified occupancy kernel, and compares all 12 outputs using
int16 views for exact FP16 bits. Allow GPU memory for the full W8 case and run
checks separately from timed serving measurements.
