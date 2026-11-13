# SM75 Marlin occupancy

`native-marlin-occupancy.patch` changes the launch grid from 68 to 136 blocks
for the tested TP2 Qwen gate/up verification shape on a 68-SM GPU. It retains
the existing kernel and FP16 arithmetic.

The dispatch is restricted to M8, K5120, N17408, group size 128, FP16
activations/scales/output, symmetric U4B8 weights, no bias or zero points,
non-atomic reduction, FP32 global reduction and the 128x128/256-thread kernel
with two pipeline stages. Shared-memory validation guards the selection.

## Native build dependency

Tested source: vLLM 0.33.0 commit
`4ba787ec9959e566d3b587a796dff07673324da5`.
Apply the patch from that source root and rebuild with the matching vLLM
native-build instructions and dependencies. The tested build used CUDA
13.4.92, torch 2.14/cu130, C++20, Release and SM75.

For the complete measured bundle, apply the
[register-pipeline patch](../sm75_marlin_register_pipe/README.md) after the
occupancy patch. The wheel-only Python patch series does not install native
changes.

`streaming-load.patch` is an alternative cache-policy-only implementation.
It is not required by the register-pipeline build; do not apply both patches
to the same source tree.

## Native occupancy checks

Use separate baseline and occupancy-modified environments, sharing an output
directory:

```bash
/path/to/baseline/bin/python experimental/sm75_marlin/check_native.py --kind baseline
/path/to/modified/bin/python experimental/sm75_marlin/check_native.py --kind modified
```

The checker writes reference tensors, a kernel trace and scalar summaries
under `/tmp/hq-sm75-native-check`; override with `--output-dir`. It checks the
68/136-block launch grids, finite outputs and exact FP16 bits for six cases,
including negative scales, fallback shapes and the full W8 output head.
Run with available GPU memory and separately from serving measurements.

The [register-pipeline checker](../sm75_marlin_register_pipe/README.md) extends
coverage to the complete native candidate. Complete serving results and their
scope are documented in [sm75_pr_review.md](../sm75_pr_review.md).
