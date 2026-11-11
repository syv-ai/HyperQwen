# lsa/: Latent Sparse Attention (D-LSA, flat) for vLLM

An opt-in port of **Latent Sparse Attention** (Kubicek et al., [arXiv 2610.42147](https://arxiv.org/abs/2610.42147),
[Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention), Apache-2.0) to this
repo's vLLM: the paper's **D-LSA (Flat)** design on Qwen3.8-27B's 16 full-attention layers, served with the paper's own
distilled map for this checkpoint. Upstream ships it as an SGLang plugin only; this is a vLLM attention backend.

```bash
bash lsa/install.sh                     # the plugin (pure Python + Triton, nothing compiles) and the 6 MB map
LSA=1 SPEC=mtp CTX=long bash single-user/start_qwen.sh
```

In Docker: `LSA=1` with `SPEC=mtp CTX=long` in `.env`; the image has the plugin and `prepare` fetches the map into
`./models/lsa/`. The boot log says `LSA armed: D-LSA flat on pth_flash_attn ...` when it is active; without that line
the server runs dense attention.

**Read [Results](#results) before turning it on.** On a 24 GB card the contexts that fit (≤150k) are well below the
1M-10M tokens the paper's speedups are measured at, and the shipped dense kernel is fast, so LSA is a long-prompt
prefill option here, not a general speed-up.

## What it does

Dense attention reads every key. LSA reads, for each query, a fixed budget whatever the context:

- the first 4 keys (sinks) and the last 512 (window);
- up to 4,096 more: the keys of the 1,024 *buckets* whose centroid scores best against the query's *index query*.

Buckets come from Ward clustering in a learned 64-dim index space. Keys are projected through the distilled map
`W_K` (`[768, 64]` a layer: the four KV heads' 192 position-free dims, past the 64 rotated ones, side by side) and
unit-normalized; every closed block of 512 keys is pooled in fours and merged by Ward into 64 buckets of about 8 keys
(at most 16). A query's index query is its six query heads per group, summed, through `W_Q`. The attention itself is
unchanged: same q, k, v, scale and softmax, over the selected keys. Below 4,612 tokens of context the budget covers
everything and the read is exact.

The defaults are the paper's Qwen3.8-27B `lsa_flat` configuration (`top_k` 4096, `top_p` 1024, window 512, sink 4,
pools of 4, buckets of 8 capped at 16, index dim 64) and its map,
`qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt` from release `d-lsa-weights-v1`, pinned by SHA-256 in `PIN`.

## How it maps onto vLLM

`hyperqwen_lsa.backend` registers as `--attention-backend CUSTOM` (only with `LSA=1`). It extends pthfa's int8 backend
(`pthfa/`) when installed, else vLLM's Triton backend, so the cache layout (`int8_per_token_head`), the KV writer and
every dense read are that backend's own code. Three things are added:

- **Index keys** are written beside the KV cache in `do_kv_cache_update`, into a table paged by the same slot ids.
  All 64 bucket rows of a cluster belong to the KV page containing its final key. A shared prefix page can
  therefore carry only buckets whose entire cluster is shared. Anchoring rows at their first keys would let
  re-clustering a divergent suffix overwrite another request's buckets when a 512-key cluster crosses a KV
  page boundary (for example, with 816-token pages). The allocator, prefix caching and preemption carry the
  side tables with their owning pages without a separate lifetime manager.
- **Clustering when a block is final.** A prefill chunk buckets the blocks it closes in its own forward, before its
  queries select. A block closed by decode tokens, which may be rejected drafts, is bucketed by the next step's
  metadata builder, once they are accepted. Until then the query's dense tail reaches down to the last bucketed
  boundary, so nothing goes unread in between: the read is a superset of the paper's for at most one step.
- **The read** (`read.py`, `kernels.py`): bucket scores (Triton), an exact top-1,024 by radix select (ties to the lower
  bucket, as the reference's stable sort), expansion of the picked buckets straight to physical KV slots best bucket
  first, and a FlashAttention-style online-softmax kernel over the slot list, split-KV for decode. The int8 per-token
  scales are folded into the logit and the probability, which is exact. Decode is capturable in a FULL CUDA graph
  (static shapes, no host sync); a prefill runs eager.

The MTP drafter's attention layer is not in the map and keeps the dense read.

## Verification

| check | what it holds | where |
|---|---|---|
| `tests/check_core.py` (GPU) | Ward partitions bit-identical to `lsa.core.ward` (CPU); selection identical to `lsa.core.reference.read` over our tables for every query; top-p kernel equal to a stable sort; sparse read within 1.2e-3 of `lsa.core.attention.topk_attention` on bf16 and int8 caches, split-KV or not; at a budget covering the context, equal to dense causal attention | `LSA_REF=<upstream checkout>/src python lsa/tests/check_core.py` |
| `tests/gate_test.py` (CPU) | the plugin is inert without `LSA=1` and owns `CUSTOM` with it; defaults are the paper's | Docker build |
| `tests/check_cache_ownership.py` (CPU; `--device cuda` optional) | forks sharing a prefix page cannot modify each other's private-cluster buckets; complete shared clusters keep their rows; page bounds/accounting and sparse-read output isolation | Docker build (CPU); GPU by hand |
| `tests/ruler_mk.py` | RULER-style multi-key needle retrieval against a running server | below |
| `tests/bench_read.py`, `bench_decode_graph.py` | one layer's cost, prefill and graph-replayed decode | |

## Knobs

| variable | default | |
|---|---|---|
| `LSA` | 0 | 1 turns it on; `SPEC=mtp CTX=long` only (refused elsewhere, and with `PTH_FA=1`, `FA2_FP8KV=1` or a KV offload tier) |
| `LSA_MAP` | `models/lsa/<PIN name>` | another distilled map (`<layer>.W_Q_idx`, `<layer>.W_K_idx`, `[768, 64]`) |
| `LSA_TOP_K`, `LSA_TOP_P` | 4096, 1024 | the budget; `LSA_TOP_P` must be a power of two |
| `LSA_DENSE_BELOW` | 4612 | an eager batch whose longest sequence is at most this reads densely |
| `LSA_BASE` | pthfa if installed | `triton` forces vLLM's Triton backend underneath |
| `LSA_DECODE_SPLITS`, `LSA_PREFILL_CHUNK` | 16, 256 | launch shape of the read |

Index keys take 128 bytes per token per layer. Bucket storage is page-rounded: each page reserves
`64 * ceil(block_size / 512)` rows of 160 bytes, plus small sink rows. At 816-token pages that is about
153 bytes per token per layer (2.4 KB over 16 layers against 33 KB of int8 KV), still roughly 7% of the KV pool.
Smaller pages can have substantially more padding; the boot log accounts for the actual page geometry.
An unset `GPU_UTIL` drops by 0.02 with `LSA=1`.

## Results

RTX 3090 at 250 W (the `syv` reference box), vLLM 0.32.0, `SPEC=mtp CTX=long`, launcher defaults for both arms
(dense is the shipped `PTH_FA=1` arm at GPU_UTIL 0.93, LSA the same int8 cache with `LSA=1` at 0.91), one cold request
per point (unique salt, no prefix hit), 512 greedy tokens, 2026-11-03:

| context | TTFT dense | TTFT LSA | decode dense | decode LSA |
|---|---|---|---|---|
| 31,331 | 30.7 s | 36.4 s | 105.6 tok/s | 101.0 tok/s |
| 62,779 | 70.3 s | 75.4 s | 88.0 | 106.8 |
| 98,722 | 126.1 s | 121.2 s | 92.0 | 92.8 |
| 136,667 | 196.4 s | 170.6 s (-13%) | 88.0 | 85.2 |

- **Prefill** wins from ~100k: the dense prefill is quadratic and LSA's reads a fixed ~4,612 keys a query. Below that
  the gather-per-query read is slower than the dense kernel (+19% at 31k).
- **Decode** is a wash within run-to-run noise (MTP acceptance differs with the generated text; the 63k row is
  mostly that). The profile at 137k says why: the 16 layers' dense verify read costs ~10 ms of a 30 ms step, LSA's
  read ~2.8 ms (graph replay: index query 16 µs, scores 14, top-p 33, expansion 6, attention 86 a layer), but the
  step is dominated by the int4 GEMMs either way; over 1,500 tokens both arms ran ~91 tok/s.
- **Accuracy** is the dense model's on everything measured: RULER-style multi-key retrieval (8 keys in vLLM's own
  source, `tests/ruler_mk.py`) 8/8 at 47-56k tokens and 6/6 at 96-109k for both arms on identical questions; the
  passcode needle at 100k at 10/50/90% depth for both. Below 4,612 tokens the read is exact.

So on this card LSA is a long-prompt prefill option, not a speed-up across the board; its decode advantage needs
contexts past what 24 GB holds, which is where the paper measures it (1M-10M on H200).

## Limits

- Qwen3.8-27B only: the map is per checkpoint. Another model needs its own distilled map (upstream's `lsa distill`).
- `SPEC=mtp CTX=long` only, single GPU (no TP or DCP split of the KV heads), int8_per_token_head or bf16 KV,
  layer-compact NHD (`LBNHC`) cache layout. Other layouts are refused during layout negotiation.
- The flat design only; the tree design (D-LSA Tree) and N-LSA are not ported.
