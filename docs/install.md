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

You need: a 24 GB Ampere or newer NVIDIA card, a recent driver, Python 3.12,
~40 GB disk — and if the host has less than ~16 GB of free RAM, load the
weights with the streamer instead of the stock loader (gotcha 45: the stock
loader peaks at whatever RAM exists; the streamer is bounded and faster). Everything below is CPU-safe to run while the GPU does other
things; the container details live in [docker.md](docker.md).

```bash
git clone https://github.com/syv-ai/HyperQwen ~/qwen-serving
cd ~/qwen-serving

python3 -m venv venv
venv/bin/pip install vllm==0.29.0 huggingface_hub hf_transfer ninja \
  --extra-index-url https://flashinfer.ai/whl/ flashinfer-cubin==0.6.18 pandas
# pandas is what `vllm[bench]` pulls in for the custom-dataset path: without it
# bench/prefill_ab.sh's decode guard dies with "Please install vllm[bench] for
# bench support" after the prefill rows have already run.
# flashinfer-python is NOT listed above on purpose: the vllm wheel pins it exactly
# (Requires-Dist: flashinfer-python==0.6.18 on 0.29.0, ==0.6.16.post3 on 0.28.0), so
# naming it here can only fight that pin. Do not downgrade it to fix a cubin version
# mismatch: that drags torch back and breaks vLLM's C extension.
#
# flashinfer-cubin ships prebuilt kernels so they do not have to JIT. It is NOT on
# PyPI past 0.6.13 -- upstream's own requirements/cuda.txt says "not on PyPI since
# 0.6.14" and pins it from https://flashinfer.ai/whl/, which is why the extra index
# is above and why `pip index versions flashinfer-cubin` alone reports 0.6.13 as the
# latest. With the versions matched, FLASHINFER_DISABLE_VERSION_CHECK=1 is no longer
# needed; the launchers still export it, harmlessly.
# Cost to know before you install it: the 0.6.18 cubin package is ~6.3 GB unpacked
# (85,496 files). The Docker image deliberately does NOT carry it: the image ships
# nvcc 13.0.88, which satisfies the equality rule below, and 6.3 GB on a 9.5 GB image
# is disk the CI runner already has to free to build at all. The venv path installs
# it because a venv host may have no usable nvcc.
#
# It does NOT cover everything. No cubin release carries the vocab-wide top-k, so
# the DFlash2 candidate selector still JITs on first use, and that JIT needs a
# working nvcc (see below). SPEC=dflash2 is the only line that reaches it, which is
# why SPEC=off and SPEC=mtp boot on a box where dflash2 does not.
#
# If you have no usable nvcc, set VLLM_USE_FLASHINFER_SAMPLER=0 (the launchers
# already do): it now covers the selector as well as the sampler and falls back to
# torch.topk. Measured cost of that fallback on a 3090, dflash2 CTX=fast,
# teacher-forced against a neutral SPEC=off target: -1.8% tok/step, -3.0% tok/s.
# Upstream's "roughly half the speed" is the vocab-wide op in isolation, not the step.
#
# Any FlashInfer JIT needs nvcc, and needs its version to EQUAL the CUDA headers'
# CUDART_VERSION -- a newer nvcc fails too, with "CUDA compiler and CUDA toolkit
# headers are incompatible". 0.6.18's JIT also emits --compress-mode=size, which
# nvcc older than 13.0 rejects outright ("nvcc fatal : Unknown option"); 0.6.16.post3
# does not emit it, which is why the 0.28 line is unaffected.
#
# On the venv path you also want the CUDA curand *headers*. vLLM's DFlash2
# sampling path JIT-compiles a FlashInfer kernel that includes curand.h; without
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

# patch vllm (all compatible patches are written against 0.29.0; reapply after upgrades)
# Order is patches/series, one basename per line: a few patches carry hunk context
# that an earlier patch adds, so the glob order of the directory is wrong. A new
# independent patch goes on the last line; one that must apply before an existing
# patch is listed before it.
sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' patches/series |
while IFS= read -r name; do
  case "$name" in
    dflash2-backport.patch) echo "skip $name (DFlash2 is native since vLLM 0.28.0)"; continue ;;
  esac
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < "patches/$name"
done
# optional: the KVarN 4/2-bit KV cache for 262k context (docs/long-context.md)
bash kvarn/install.sh

# api key — optional, but the server binds 0.0.0.0 and is open without one
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
curl http://localhost:18020/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt 2>/dev/null)" \
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
