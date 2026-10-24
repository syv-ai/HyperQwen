# Setup: the bare-metal install

The venv path, for when you are not using the container. The Docker route in
the [main README](../README.md#quick-start) is the default and needs none of
this.

[← back to the main README](../README.md)

The default install is the container ([Quick start](../README.md#quick-start) — the
prebuilt image already contains everything this section builds), so this
manual venv path is for hacking on the stack, or running it bare-metal.

> **Python 3.14 works natively** — nothing in this repo needs changing, but
> `python3.14-dev` does need installing. See [python-314.md](python-314.md),
> with a full RTX 3090 reproduction of the benchmark tables in
> [docs/reproductions/native-3090.md](reproductions/native-3090.md).

Volta (V100, sm70) and older cards cannot run this stack at all: the CUDA 13.0 build of torch 2.13
that vLLM 0.31 pins carries kernels for sm_75 and newer only (`torch.cuda.get_arch_list()`), CUDA 13
dropped Volta, and vLLM's Marlin W4A16 kernel needs compute capability 7.5 or newer
([#349](https://github.com/syv-ai/HyperQwen/issues/349)). Turing (sm75) is untested.

You need: a 24 GB Ampere or newer NVIDIA card, a recent driver, Python 3.12,
~40 GB disk — and if the host has less than ~16 GB of free RAM, load the
weights with the streamer instead of the stock loader (gotcha 45: the stock
loader peaks at whatever RAM exists; the streamer is bounded and faster). Everything below is CPU-safe to run while the GPU does other
things; the container details live in [docker.md](docker.md).

```bash
git clone https://github.com/syv-ai/HyperQwen ~/qwen-serving
cd ~/qwen-serving

python3 -m venv venv
venv/bin/pip install vllm==0.32.0 huggingface_hub hf_transfer ninja \
  --extra-index-url https://flashinfer.ai/whl/ flashinfer-cubin==0.7.0.post1 pandas \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88
# The four nvidia-* pins hold the CUDA 13 compiler at the runtime's 13.0 (see "Any
# FlashInfer JIT needs nvcc ... to EQUAL" below): unpinned, pip now resolves nvcc 13.4
# beside the 13.0 runtime, and the first FlashInfer JIT fails.
# pandas is what `vllm[bench]` pulls in for the custom-dataset path: without it
# bench/prefill_ab.sh's decode guard dies with "Please install vllm[bench] for
# bench support" after the prefill rows have already run.
# flashinfer-python is NOT listed above on purpose: the vllm wheel pins it exactly
# (Requires-Dist: flashinfer-python==0.7.0.post1 on 0.32.0 and 0.31.0, ==0.6.18.post1 on
# 0.30.0, ==0.6.18 on 0.29.0), so
# naming it here can only fight that pin. Do not downgrade it to fix a cubin version
# mismatch: that drags torch back and breaks vLLM's C extension.
#
# flashinfer-cubin ships prebuilt kernels so they do not have to JIT. It is NOT on
# PyPI past 0.6.13 (upstream's requirements/cuda.txt said "not on PyPI since 0.6.14"
# and pinned it from https://flashinfer.ai/whl/), which is why the extra index is above
# and why `pip index versions flashinfer-cubin` alone reports 0.6.13 as the latest.
# vLLM 0.32.0 dropped that pin in favour of `vllm download-kernels` (vllm #58765), so
# the version here is flashinfer-python's: the two must match exactly. With the
# versions matched, FLASHINFER_DISABLE_VERSION_CHECK=1 is no longer needed; the
# launchers still export it, harmlessly.
# Cost to know before you install it: the 0.6.18 cubin package is ~6.3 GB unpacked
# (85,496 files). The Docker image deliberately does NOT carry it: the image ships
# nvcc 13.0.88, which satisfies the equality rule below, and 6.3 GB on a 9.5 GB image
# is disk the CI runner already has to free to build at all. The venv path installs
# it because a venv host may have no usable nvcc.
#
# It does NOT cover everything. No cubin release carries the vocab-wide top-k, so
# with VLLM_USE_FLASHINFER_SAMPLER=1 the DFlash2 candidate selector JITs on first
# use, and that JIT needs a working nvcc (see below). All three launchers export it
# as 0, so no shipped launcher reaches the sampler or selector JIT: a native
# SPEC=dflash2 boot of 0.31.0 from this file built no FlashInfer kernels.
#
# The fp8-KV attention kernel is the JIT the shipped launchers do reach. No cubin
# release carries FlashInfer's fp8-KV batch_prefill either, so the profiles that run
# FlashInfer attention on an fp8 KV cache (batch, whose KV=fp8 is the default, and
# single-user CTX=long unless PTH_FA=1 takes it, its default on an 8.6 GPU with pthfa/ installed) build it once per cold cache, ~31-36 s inside the first
# CUDA-graph capture, on 0.30.0 and 0.31.0 alike. The launchers point CUDA_HOME at
# the venv's nvcc 13.0.88 (the pins above) when the system nvcc is older, and that
# is the nvcc that builds it.
#
# If you have no usable nvcc, set VLLM_USE_FLASHINFER_SAMPLER=0 (the launchers
# already do): it now covers the selector as well as the sampler and falls back to
# torch.topk. Measured cost of the selector's fallback on a 3090 (vLLM 0.32.0,
# dflash2 CTX=fast, ~23k-token prompts teacher-forced against a SPEC=off target,
# three boots each way): the same tokens per step, 0.35% lower decode tok/s. The
# sampler's fallback was not measured; a request with a seed, or with neither
# top_k nor top_p, takes torch's sampler either way.
# Upstream's "roughly half the speed" is the vocab-wide op in isolation, not the step.
#
# Any FlashInfer JIT needs nvcc, and needs its version to EQUAL the CUDA headers'
# CUDART_VERSION -- a newer nvcc fails too, with "CUDA compiler and CUDA toolkit
# headers are incompatible". 0.6.18's JIT also emits --compress-mode=size, which
# nvcc older than 13.0 rejects outright ("nvcc fatal : Unknown option"); 0.6.16.post3
# does not emit it, which is why the 0.28 line is unaffected.
#
# If you set VLLM_USE_FLASHINFER_SAMPLER=1, the venv path also wants the CUDA
# curand *headers*. vLLM's DFlash2 sampling path then JIT-compiles a FlashInfer
# kernel that includes curand.h; without
# the headers the build fails with "fatal error: curand.h: No such file or
# directory" and it *silently falls back* -- you get correct output at a lower
# rate, not an error. A 4x 5060 Ti reporter measured 192.9 -> 202.1 tok/s
# (+4.8%, step 18.4 -> 17.6 ms) just from installing them (#105). The Docker
# path already covers this (Dockerfile:18). On Ubuntu with the CUDA repo,
# matching your CUDA minor:
#   sudo apt-get install -y libcurand-dev-13-0   # or libcurand-dev-13-3, etc.

# model, ~19.5 GB
HF_XET_HIGH_PERFORMANCE=1 venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound

# requantize lm_head + embeddings + the MTP draft module (CPU only, a few minutes)
venv/bin/python prepare/quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_mtp.py     models/Qwen3.8-27B-W4A16-AutoRound
# 40k-token draft head for single-user mode (uses the shipped id list)
venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound \
  --ids prepare/draft_vocab_ids.json
# single-user "fast" variant (~1 GB from the Hub, hardlinks the rest): int4-GPTQ
# lm_head + drafter; single-user/start_qwen.sh picks it up automatically
venv/bin/python prepare/fetch_fast_variant.py
# optional: the W4A16 DFlash2 block drafter (1.2 GB) for SPEC=dflash2 single-user mode
venv/bin/python prepare/fetch_dflash2.py
# optional: a third-party checkpoint instead of the base model (e.g. the uncensored
# build, ~18.6 GB, its own requant step; MODEL= serves it --
# see docs/third-party-checkpoints.md)
venv/bin/python prepare/fetch_thirdparty.py
venv/bin/python prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16

# patch vllm (all compatible patches are written against 0.32.0; reapply after upgrades).
# patches/apply.sh applies patches/series in order, with --fuzz 0, and stops at the first
# patch that does not apply, by name. The order matters: a few patches carry hunk context
# that an earlier patch adds. A new independent patch goes on the last line of
# patches/series; one that must apply before an existing patch is listed before it.
# the venv's own vllm directory: python3 -m venv names it after the interpreter (python3.12,
# python3.14, ...), so ask the interpreter rather than spelling the path (verify.sh and
# kvarn/install.sh find it the same way; tail -n1 because importing vllm can log to stdout)
SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
bash patches/apply.sh "$SP"
# optional: the KVarN 4/2-bit KV cache for 262k context (docs/long-context.md)
bash kvarn/install.sh
# optional: the fp8-KV FlashAttention-2 build for FA2_FP8KV=1 (fa2/README.md). It compiles two CUDA
# libraries with nvcc 13 (the pip nvcc above is found automatically), needs git and a host C++
# compiler, and takes ~3-5 minutes. Nothing uses it unless FA2_FP8KV=1.
bash fa2/install.sh
# optional: the pth-flash-attn int8-KV attention backend (pthfa/README.md). Once installed it is the default
# for SPEC=mtp CTX=long and SPEC=dflash2 CTX=long on a compute capability 8.6 GPU (PTH_FA=0 opts out; without
# the install those arms keep their own default and the launcher says so once). It compiles one CUDA library
# for sm80/86/89 with nvcc 13 and a host C++ compiler; no GPU and no network. Nothing else uses it.
bash pthfa/install.sh
# xgrammar built from source with the bounded-string fix for Qwen tool calls (xgrammar/README.md): without it, a
# maxLength in a tool schema drops decode to ~7 tok/s under TOOL_STRICT=parameter. Needs git, a host C++ compiler
# and about a minute.
bash xgrammar/install.sh

# api key — optional on one machine: with no key the launchers bind 127.0.0.1 only. To serve other
# machines, set a key (they then bind 0.0.0.0) or set HOST explicitly.
openssl rand -hex 24 > api_key.txt
```

Then `bash verify.sh --no-server` — it checks the venv and vLLM version, that
every compatible patch in `patches/` is actually applied, and that the model has been
requantized (lm_head, embeddings, MTP module, draft head). Then pick a mode
and follow its README:

- **[batch/](../batch)** — throughput. `bash batch/start_qwen.sh`
- **[single-user/](../single-user)** — latency. `bash single-user/start_qwen.sh`

First start takes a few minutes (torch.compile, CUDA graph capture, flashinfer
JIT). Test it:

```bash
OPENAI_API_KEY=$(cat api_key.txt 2>/dev/null)
curl http://localhost:18020/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "hej"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Qwen recommends temperature 0.7 / top_p 0.8 for instruct mode, and 1.0 / 0.95
with thinking enabled (the default).

Tool calling works over the same endpoint — send `tools` with `tool_choice:
"auto"` and the reply carries `tool_calls`. Both launchers set
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`; the parser has to
read Qwen's XML call format, which is what this model's chat template emits —
not the JSON that `hermes` reads. `TOOLS=0` turns it off.
