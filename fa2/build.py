"""Build the FP8 KV extension against installed CUDA and PyTorch dependencies."""

import argparse
import importlib.util
import os
from pathlib import Path
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", action="store_true")
parser.add_argument("--prefill", action="store_true", help="Also build bounded KV unpacking for native FA2 prefill")
parser.add_argument("--cutlass-include", type=Path,
                    default=Path(os.environ.get("CUTLASS_PATH", Path(__file__).resolve().parent / "third_party/cutlass/include")))
parser.add_argument("--flashinfer-include", type=Path)
args = parser.parse_args()
source = Path(__file__).resolve().parent
flashinfer_include = args.flashinfer_include
if flashinfer_include is None:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        parser.error("Install flashinfer-python or pass --flashinfer-include containing flashinfer/vec_dtypes.cuh")
    flashinfer_include = Path(next(iter(spec.submodule_search_locations))) / "data/include"
for directory, header in ((args.cutlass_include, "cute/tensor.hpp"),
                          (flashinfer_include, "flashinfer/vec_dtypes.cuh")):
    if not (directory / header).is_file():
        parser.error(f"Missing {directory / header}; install the dependency or set its include path")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
build = source / ("build-pipeline" if args.pipeline else "build-gqa")
build.mkdir(exist_ok=True)
print(load(
    name="fa2_fp8kv",
    sources=[str(source / "fp8_attn.cu")],
    extra_include_paths=[str(source / "headers"), str(args.cutlass_include.resolve()),
                         str(flashinfer_include.resolve())],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "--ptxas-options=-v",
        "-DFLASHATTENTION_DISABLE_DROPOUT", "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP", "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        f"-DFA2_FP8KV_PIPELINE={int(args.pipeline)}"],
    build_directory=str(build), is_python_module=False, verbose=True,
))
if args.prefill:
    prefill_build = source / "build-prefill"
    prefill_build.mkdir(exist_ok=True)
    print(load(name="fa2_fp8kv_prefill", sources=[str(source / "prefill.cu")],
               extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
               build_directory=str(prefill_build), is_python_module=False, verbose=True))
