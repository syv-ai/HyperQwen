#!/bin/bash
# HyperQwen on Turing (sm75 / RTX 20-series, e.g. 2x RTX 2080 Ti 22 GB).
#
# Turing has no usable bf16 (torch emulates it ~10x slower), so everything runs
# --dtype float16, with DFlash2 speculation and TP=2 by default. The attention
# kernels are picked with ATTN (see below): FlashInfer by default, because this
# Triton build emits no tensor-core code on sm75 and its attention slows with context.
# DFlash2 in fp16 needs patches/dflash2-fp16-residual-scale.patch: without it the
# drafter's residual overflows fp16, every draft is NaN, and acceptance is ~2%.
#
# Measured on 2x RTX 2080 Ti 22 GB (TP=2, 0.90 util, 262k context), single stream:
#   ATTN=flashinfer (default) 24k-token prompt: 185-208 tok/s decode (full graphs), ~14 s cold prefill
#   ATTN=triton               short prompt: ~200 tok/s; 24k prompt: 74 tok/s, 80 s prefill
#   SPEC=off ATTN=triton      54.8 tok/s short prompt
# TP=1 does not fit DFlash2 at the default KV pin; for TP=1 use
# SPEC=off (32.7 tok/s), or KV_MEM=3000000000 DFLASH_MAX_LEN=8192 with dflash2.
#
# Everything below can be overridden from the environment, e.g.
#   TP=1 MAX_LEN=32768 bash start-turing.sh
#   KV_DTYPE=fp8 bash start-turing.sh   # optional FP8 KV capacity trial
#   CUDA_VISIBLE_DEVICES=1 bash start-turing.sh
#   VLLM_API_KEY=... BIND=0.0.0.0 bash start-turing.sh   # expose to the LAN
#
# First boot JITs many Triton kernels for sm_75 (~10 min); later boots reuse
# ~/.cache/vllm and ~/.triton and are much faster.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This checkpoint has 24 attention heads / 4 KV heads, so TP must be 1, 2 or 4.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ -z "${TP:-}" ]; then
  case "$NGPU" in 0|1) TP=1 ;; *) TP=2 ;; esac
fi
case "$TP" in 1|2|4) ;; *) echo "TP=$TP: this checkpoint only supports 1, 2 or 4 (24 heads / 4 KV heads)" >&2; exit 1 ;; esac
# 262136, not 262144: DFlash2 needs 8 lookahead positions inside the model's 262144.
MAX_LEN=${MAX_LEN:-$([ "$TP" -gt 1 ] && echo 262136 || echo 32768)}
MAX_SEQS=${MAX_SEQS:-2}
GPU_UTIL=${GPU_UTIL:-0.90}

TP_ARGS=()
[ "$TP" -gt 1 ] && TP_ARGS=(--tensor-parallel-size "$TP")

export MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
export SPEC=${SPEC:-dflash2}   # SPEC=off for no speculation
# ATTN picks the attention kernels. Triton 3.7 emits no tensor-core (mma) code on sm75, so
# its attention runs on the FP32 units and slows with context; FlashInfer's CUDA kernels use
# Turing's tensor cores. Measured at TP=2 with DFlash2 (decode tok/s, 24k-token prompt):
#   ATTN=flashinfer (default): 129-185 short-prompt, 185-208 at 24k, ~14 s cold prefill /
#                              ~0.8 s cached, fp16 KV (full target + drafter CUDA graphs)
#   ATTN=triton:               ~200 short-prompt, 24k -> 74, 24k prefill 80 s cold / 7 s cached,
#                              int8 KV, 530k-token pool (split-KV verify kernel, SPEC_ATTN=1)
ATTN=${ATTN:-flashinfer}
case "$ATTN" in
  flashinfer)
    export VLLM_FLASHINFER_SM75=1   # vLLM gates FlashInfer to sm80+; see patches/flashinfer-sm75-optin.patch
    # Persistent native FA2 plans enable full graphs for target verification and
    # the noncausal DFlash drafter. Set to 0 to compare against eager attention.
    export VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=${VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS:-1}
    export SPEC_ATTN=0
    ATTN_ARGS="--attention-backend FLASHINFER --kv-cache-dtype ${KV_DTYPE:-auto}" ;;
  triton)
    export SPEC_ATTN=${SPEC_ATTN:-1}   # split-KV verify kernel (patches/spec-attn-fp16.patch)
    ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head" ;;
  *) echo "ATTN=$ATTN: use flashinfer or triton" >&2; exit 1 ;;
esac
# CUDA recurrent GDN prefill: fp32 state arithmetic, fp16 inputs. The launcher
# opts into patches/gdn-sm75-cuda-prefill.patch; 24k cold prefill measured ~14 s
# versus ~16 s for Triton, with identical deterministic smoke-test completions.
# Requires nvcc/C++; set VLLM_GDN_SM75=0 to use the original Triton prefill.
export VLLM_GDN_SM75=${VLLM_GDN_SM75:-1}
# Maya-style per-request terminal progress, also available to watch_turing.py.
export VLLM_TERMINAL_PROGRESS=${VLLM_TERMINAL_PROGRESS:-1}
export HYPERQWEN_LOG_FILE=${HYPERQWEN_LOG_FILE:-$REPO/logs/turing.log}
mkdir -p "$(dirname "$HYPERQWEN_LOG_FILE")"
exec > >(tee -a "$HYPERQWEN_LOG_FILE") 2>&1
export CTX=fast
# fp16 + a speculator is refused upstream (the split-KV verify kernel was bf16-only;
# patches/spec-attn-fp16.patch makes it follow the model dtype).
[ "$SPEC" = "dflash2" ] && export FP16_SPEC_OK=1
export MAX_LEN MAX_SEQS GPU_UTIL
# expandable segments need CUDA VMM; Turing + this allocator is where the
# sm75 reports see mapping failures. The 3090 profiles keep it on at 0.93+;
# at 0.90 there is enough headroom without it.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
# EXTRA_ARGS is applied last, so these replace the launcher's FLASH_ATTN/bf16
# CTX=fast defaults.
export EXTRA_ARGS="--dtype float16 $ATTN_ARGS ${TP_ARGS[*]} ${EXTRA_ARGS:-}"

exec bash "$REPO/single-user/start_qwen.sh" "$@"
