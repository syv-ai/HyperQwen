# Arete-27B: a prepared checkpoint for batch serving

[NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B) is a
post-trained Qwen3.8-27B checkpoint. An independently prepared, calibrated
compressed-tensors export is available as
[surak/HyperArete-27B-W4A16](https://huggingface.co/surak/HyperArete-27B-W4A16).
Its [reproduction repository](https://github.com/surak/HyperArete) contains the
AutoRound recipe, head preparation, validation script, licenses and build details.

This is a use of HyperQwen's existing preparation and batch-serving code with
Arete's own weights. No extra alignment training or merge with Qwen weights was
performed. It is not an official Nous Research or HyperQwen checkpoint release.

## Preparation and provenance

The released checkpoint has a symmetric W4A16 group-128 language-model body,
INT8 group-128 `lm_head` and embeddings, and all 333 vision tensors verified
unchanged. Its five weight shards total 16,100,314,792 bytes. The tokenizer and
model configuration come from Arete. **It has no MTP head**: do not run
`quant_mtp.py` or `build_draft_vocab.py` on it, or assume the default speculative
single-user profile works with it.

The measured build used:

- Arete revision `108f0afbdcfaa1ceeda4a635004304dfff6d4da2`;
- HyperQwen revision `ce9615c4f2309be727407918a6c979bd67968069`;
- AutoRound 0.16.1, compressed-tensors export, 200 iterations, 128 calibration
  samples of 2,048 tokens, batch size 4, seed 42, `NeelNanda/pile-10k`;
- `prepare/quant_lm_head.py` followed by `prepare/quant_embed.py`.

The downloadable checkpoint is already prepared. Do not rerun those in-place
preparation scripts on the release. For rebuilding from BF16, use the linked
recipe; the AutoRound export must be compressed-tensors `pack-quantized`, as
explained in [prepare/](../prepare/README.md).

## One RTX 3090, 24 GB

Use the patched HyperQwen environment from [installation](install.md). The
following is the tested batch profile, with image support retained:

```bash
hf download surak/HyperArete-27B-W4A16 --local-dir models/HyperArete-27B-W4A16
# Activate the prepared environment and set VLLM_API_KEY privately first.
MODEL="$PWD/models/HyperArete-27B-W4A16" \
MAX_LEN=150000 MAX_SEQS=64 GPU_UTIL=0.94 KV=fp8 VISION=1 \
INT8_ACT=int8 INT8_LAYERS=mlp \
EXTRA_ARGS='--dtype bfloat16 --served-model-name hyper-arete-27b --watermark-config {"algorithm":"gumbel","key":16386}' \
  bash batch/start_qwen.sh
```

The fixed watermark key reproduces the measured profile; it is public and is not
an attribution secret. Gumbel marks generated text, not image pixels. The batch
launcher also uses FP16 recurrent state. Stored W4A16 weights and runtime INT8
activation kernels are separate parts of the setup.

The service limit is **150,000 total tokens per request**, including output.
The checkpoint retains a 262,144-position configuration; that does not make 256K
fit this single-GPU profile. A separate two-GPU TP2 deployment passed a real
261,888-input-token retrieval test. Arete's original SFT used 64K context; these
retrieval checks are not comprehensive long-context evaluations.

For two independent replicas, start the profile twice with distinct
`CUDA_VISIBLE_DEVICES`, `PORT`, `VLLM_CACHE_ROOT`, and `TRITON_CACHE_DIR` values.
On a shared host set `VLLM_OFFLOAD_KEEP_SHM=1`, so startup does not clean another
engine's offload regions. Keep Arete's model IDs separate from ordinary Qwen.
The linked standalone recipe includes an optional chat template that normalizes
reasoning-effort names; this launch example uses the checkpoint's own template.

## Observations, 2026-11-08

The measured runtime was patched vLLM 0.31.0, Torch 2.13.0, Transformers 5.17.0
and compressed-tensors 0.17.0. Later HyperQwen revisions have not been retested
for these numbers.

Both independent RTX 3090 workers passed text, image comprehension, JSON, tools,
streaming, token-usage arithmetic, Gumbel correct/wrong-key detection, and a
149,744-input-token retrieval request (~125 seconds). Short-request decode was
about 47.6 output tokens/sec per worker. Each worker reached about 342 aggregate
output tokens/sec at eight concurrent requests and 1,257–1,270 at 64.

A simultaneous run across two workers measured:

| Requests | Output tokens | Wall time | Aggregate output tokens/sec |
| --- | --- | --- | --- |
| 16 | 2,048 | 3.531 s | 579.96 |
| 128 | 16,384 | 6.301 s | 2,600.25 |

These are synthetic short photosynthesis prompts, 128 output tokens per request,
temperature 0.7, thinking disabled, `ignore_eos=True`, and prefix caching enabled.
Repeated prefixes can benefit from caching. These are short bursts, not sustained
production capacity estimates, and are not directly comparable to different
prompt/output-length benchmarks. Sixty-four scheduler slots do not imply 64
resident long-context requests: startup estimated 34–45 at 1K, five at 32K or one
at 150K per worker.

No broad quality evaluation against BF16 Arete has been completed. Do not infer
preservation of Nous Research's accuracy or alignment results from functional
checks. Quantization, FP8 KV, FP16 recurrent state and INT8 activations can change
outputs. Preserve the upstream licenses and attribution when redistributing.
