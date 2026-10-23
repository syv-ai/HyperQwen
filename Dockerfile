# Same stack as the README's venv install, frozen: Python 3.12 venv at /app/venv,
# vLLM 0.31.0 (torch 2.13 / cu130 / Triton 3.7.1), every compatible patch in
# patches/ applied,
# the KVarN KV cache installed, the pth-flash-attn op library built (pthfa/, selected by
# PTH_FA=1, the default on an 8.6 GPU for SPEC=mtp or dflash2 CTX=long), verify.sh --install and the scratch-pool test run at build time.
#
# The base image is CUDA "base" + nvcc, not "devel": vLLM's wheels bring their own
# CUDA libraries, but FlashInfer JIT-compiles its fp8-KV attention kernel with nvcc
# on first use (batch mode, CTX=long) and Triton needs a C compiler for its launchers.
# The compiled kernels and the torch.compile cache live in the /cache volume, so
# that only happens once.
#
#   docker compose --profile single up -d      (see docs/docker.md)
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
      build-essential patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.12 -m venv venv && venv/bin/pip install --upgrade pip
COPY docker/requirements.txt docker/requirements.txt
RUN venv/bin/pip install -r docker/requirements.txt

# The pth-flash-attn op library (pthfa/, optional: PTH_FA=1, its default on an 8.6 GPU) compiles here, ahead of `COPY . .`, so an
# edit elsewhere does not recompile it: SASS for sm_80/86/89 with the image's nvcc and the venv's torch,
# installed into the venv with its vllm.general_plugins entry point. install.sh checks the cubins and runs
# pthfa/tests/gate_test.py against the pristine wheel; the RUN below runs it again on the patched tree.
COPY pthfa/ pthfa/
RUN PY=venv/bin/python bash pthfa/install.sh
# The fp8-KV FA2 libraries (fa2/, optional, FA2_FP8KV=1) compile here, ahead of `COPY . .`, so an edit
# elsewhere does not recompile them. git fetches CUTLASS at its pinned commit; the CUTLASS checkout and
# the build tree go in the same layer's /tmp, which is removed, so only the two small .so files remain.
COPY fa2/ fa2/
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && FA2_CACHE=/tmp/fa2-cache bash fa2/install.sh && rm -rf /tmp/fa2-cache

COPY . .
# patches/apply.sh reads the apply order from patches/series and applies it with --fuzz 0:
# a few patches carry hunk context that an earlier patch adds, so the glob order of
# patches/*.patch is wrong.
# bench/mq3d_scratch_pool_test.py imports the patched triton_attn.py and runs on CPU
# tensors, so the build is where CI can run it. pthfa/tests/gate_test.py (the CUSTOM
# backend's registration, inertness and limits against this vLLM), fa2/tests/gate_test.py (the
# FA2_FP8KV gate) and check_mixed_prefill.py --device cpu (the adapter's routing) do the same.
RUN set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    bash patches/apply.sh "$SP"; \
    bash kvarn/install.sh; \
    bash verify.sh --install; \
    venv/bin/python bench/mq3d_scratch_pool_test.py; \
    CUDA_VISIBLE_DEVICES= venv/bin/python pthfa/tests/gate_test.py; \
    venv/bin/python fa2/tests/gate_test.py; \
    venv/bin/python fa2/tests/check_mixed_prefill.py --device cpu; \
    CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu venv/bin/python bench/test_selective_mtp_streamer.py

# HOME is a volume: torch.compile cache (~/.cache/vllm), Triton (~/.triton),
# FlashInfer JIT (~/.cache/flashinfer), HF hub cache.
RUN mkdir -p /cache /app/models && chmod 1777 /cache
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_XET_HIGH_PERFORMANCE=1
VOLUME ["/cache", "/app/models"]
EXPOSE 18020
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["single"]
