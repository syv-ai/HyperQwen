# SPDX-License-Identifier: Apache-2.0
"""Check the real native operator, including eligible and fallback shapes."""

import argparse, json
from pathlib import Path
import torch
from vllm import _custom_ops
from vllm.scalar_type import scalar_types

p = argparse.ArgumentParser()
p.add_argument("--kind", choices=["baseline", "modified"], required=True)
p.add_argument("--output-dir", type=Path, default=Path("/tmp/hq-sm75-native-check"))
a = p.parse_args()
BASE = a.output_dir
BASE.mkdir(parents=True, exist_ok=True)
outputs = {}
summary = []
cases = [
    (8, 5120, 17408, 4, False),
    (8, 5120, 17408, 4, True),
    (7, 5120, 17408, 4, True),
    (16, 5120, 17408, 4, True),
    (8, 8704, 5120, 4, True),
    (8, 5120, 124160, 8, True),
]
for i, (M, K, N, bits, negative) in enumerate(cases):
    torch.manual_seed(420 + i)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randint(
        -(2**31),
        2**31 - 1,
        (K // 16, N * 16 // (32 // bits)),
        device="cuda",
        dtype=torch.int32,
    )
    s = torch.rand(K // 128, N, device="cuda", dtype=torch.float16) * 0.002 + 0.0001
    if negative:
        s[:, ::2] *= -1
    locks = torch.zeros(
        torch.cuda.get_device_properties(0).multi_processor_count * 16,
        device="cuda",
        dtype=torch.int32,
    )
    typ = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128

    def run():
        return torch.ops._C.marlin_gemm(
            x,
            None,
            w,
            None,
            s,
            None,
            None,
            None,
            locks,
            typ.id,
            M,
            N,
            K,
            False,
            True,
            False,
        )

    if i == 0:
        run()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            y = run()
        trace = BASE / ("kernel-" + a.kind + ".trace.json")
        prof.export_chrome_trace(str(trace))
        kernels = [
            e
            for e in json.loads(trace.read_text())["traceEvents"]
            if e.get("cat") == "kernel" and "Marlin<" in e["name"]
        ]
        assert len(kernels) == 1, [(e["name"], e.get("args")) for e in kernels]
        grid = kernels[0]["args"]["grid"][0]
        assert grid == (68 if a.kind == "baseline" else 136), (
            "Wrong launch geometry",
            a.kind,
            grid,
        )
        print("NATIVE_GRID", a.kind, grid, flush=True)
    else:
        y = run()
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item()
    outputs[str(i)] = y.cpu()
    summary.append({"M": M, "K": K, "N": N, "bits": bits, "negative_scales": negative})
    del x, w, s, locks, y
    torch.cuda.empty_cache()
gold = BASE / "kernel-gold.pt"
if a.kind == "baseline":
    torch.save(outputs, gold)
else:
    expected = torch.load(gold, map_location="cpu", weights_only=True)
    for name, out in outputs.items():
        assert torch.equal(out.view(torch.int16), expected[name].view(torch.int16)), (
            "Native operator changed output",
            summary[int(name)],
        )
(BASE / ("kernel-" + a.kind + ".json")).write_text(
    json.dumps(
        {
            "cases": summary,
            "all_finite": True,
            "bit_exact_vs_baseline": a.kind == "modified",
        },
        indent=2,
    )
)
print("PASS", a.kind, len(cases), "eligible/fallback/negative-scale cases", flush=True)
