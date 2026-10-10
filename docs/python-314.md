# Python 3.14, natively

*Contributed from a reproduction on an RTX 3090, Ubuntu 26.04, August 2026. The short
version: **nothing in this repo needs changing for 3.14.** One system package does.*

[← back to the main README](../README.md)

> **Status, 2026-10-10.** This page records a reproduction on vLLM 0.27.1. The repo now
> pins vLLM 0.31.0, and the patch series is cut against 0.31.0, so the series does not apply
> to 0.27.1. Nobody has re-run this page on 0.31.0 yet. vLLM 0.31.0 also publishes only
> `cp38-abi3` wheels with `requires_python <3.15,>=3.10`, so the abi3 reasoning below can
> still hold. The torch and FlashInfer set for 0.31.0 did install on 3.14.4 from
> [install.md](install.md), and the patched install served a `SPEC=dflash2` boot
> ([vllm-0.31.md](vllm-0.31.md)). Use the steps and the numbers here as history until someone
> re-runs them.

The README specifies Python 3.12, and the container freezes it there. That is a safe
default rather than a hard requirement: vLLM 0.27.1 publishes an **abi3 wheel** (tagged
`cp38`, `requires_python <3.15,>=3.10`) and torch 2.13 ships real `cp314` wheels, so
`pip install vllm==0.27.1` resolves and imports cleanly on 3.14.

Every patch in `patches/` applies unmodified. `verify.sh --no-server` passes. The
single-user benchmark reproduces the README's cohort table (numbers below).

## What actually blocks it

**`python3.14-dev`.** Triton JIT-compiles a small C launcher at runtime, so it needs the
Python development headers. Without them the server dies during model-architecture
inspection with a traceback whose useful line is buried:

```
/tmp/tmpXXXXXX/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
```

which vLLM then re-raises as the much less helpful:

```
Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected.
```

Fix, matching your interpreter version exactly:

```bash
sudo apt-get install -y python3.14-dev     # or python3.13-dev, etc.
```

`gcc` must also be present; Triton shells out to it.

## Three things that cost us time and are not version-specific

1. **The patches are SEQUENTIALLY DEPENDENT.** Applied individually, three of fifteen fail
   (`dflash2-lookup-drafting`, `spec-decode-int8-kv`, `vision-tower-cpu-offload`) because
   they build on hunks an earlier patch introduces. Applied in the README's loop order, all
   fifteen land. A dry-run over the set will look like an incompatibility but is not one.

2. **`patch -d` targets the `vllm` PACKAGE directory, not `site-packages`.** The README is
   correct; it is an easy misread. Pointing one level too high fails all fifteen, which
   looks exactly like "this vLLM is wrong."

3. **`bench/run_benchmarks.sh` needs `vllm[bench]`** (pandas) for the custom-dataset
   cohorts. The `random` dataset path works without it, so a partial install looks like a
   working one: the cohort logs contain `ModuleNotFoundError: No module named 'pandas'`
   while the harness still prints its `ROW` lines, with every field empty.
   And the bench client authenticates with **`OPENAI_API_KEY`**, not `VLLM_API_KEY`; with
   only the latter set it receives silent 401s and reports `0.00` in every field rather
   than failing.

## Install, start to finish

```bash
sudo apt-get install -y python3.14-dev gcc
python3 -m venv venv
venv/bin/pip install 'vllm[bench]==0.27.1' huggingface_hub hf_transfer ninja \
  flashinfer-python flashinfer-cubin==0.6.13
# model + requantization exactly as the README
VP=$(venv/bin/python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
bash patches/apply.sh "$VP"   # the order is patches/series, with --fuzz 0
bash verify.sh --no-server
```

## Reproduction

See [reproductions/native-3090.md](reproductions/native-3090.md).
