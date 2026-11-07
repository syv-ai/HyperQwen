# CMP 170HX decode tuning for HyperQwen (vLLM 0.32.0)

Bind-mounted vLLM sources plus the matching environment settings for the CMP
170HX (GA100, sm_80, 64 GiB). No image rebuild and no patch-series edit: the
four files here are mounted read-only over the copies inside
`ghcr.io/syv-ai/hyperqwen:latest` (`sha256:f706b092…`), and removing those
mounts returns the container to the stock image.

Everything was measured on 2026-11-07 with
`Qwen3.8-27B-W4A16-AutoRound` + the DFlash2 drafter (`k=7`), BF16 KV,
`CTX=fast`, 256K context, driver 610.57.04, card pinned at 180 W.

## Changes

### `envs.py`

Registers two knobs:

| knob | default | meaning |
| --- | --- | --- |
| `VLLM_SPEC_ATTN_NSEG` | `0` | `0` = adaptive segments; `>0` = fixed segment count (clamped to 16) |
| `VLLM_SPEC_ATTN_MIN_BATCH` | `4` | use split-KV verify attention only when `num_reqs >= this` |

### `spec_decode_attn.py`

- `_segments()`: picks the active KV segment count from the **captured batch
  size**, aiming at ~64 blocks (`num_reqs * ntile * Hkv * segments`), and rounds
  down to a power of two because the combine kernel's `tl.arange(0, NSEG)`
  requires one. An explicit `VLLM_SPEC_ATTN_NSEG` wins over the heuristic.
- `_plan()`: 8 warps whenever `D > 128`. At `D = 256` the `[BLOCK_M, D]`
  accumulator does not fit the 4-warp register budget.
- KV tile: 64 tokens at `D == 256` when `block_m <= 64`. GA100 has 163 KB of
  shared memory per block, not sm86's 99 KB, so the production 8-query shape
  fits a 64-token tile. The existing `OutOfResources` retry still halves the
  tile on any shape that does not.

### `flash_attn.py` and `triton_attn.py`

- `MIN_BATCH` gate in the split-KV branch: at 1–2 requests the extra combine
  launch loses to FA2, so the split path is skipped there. The request count is
  a captured constant under FULL cudagraph, which makes the branch replay-safe
  (a KV-length branch would freeze at capture).

## Deploy

`docker-compose.yml` on this branch mounts the files:

```yaml
volumes:
  - ./opt-170hx/envs.py:/app/venv/lib/python3.12/site-packages/vllm/envs.py:ro
  - ./opt-170hx/spec_decode_attn.py:/app/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/spec_decode_attn.py:ro
  - ./opt-170hx/flash_attn.py:/app/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py:ro
  - ./opt-170hx/triton_attn.py:/app/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py:ro
```

Relevant `.env` fragment (`170hx.env.example` is a ready-to-merge copy):

```sh
SPEC=dflash2
PREFIX_CACHE=1
CTX=fast

VLLM_SPEC_ATTN_NSEG=16
VLLM_SPEC_ATTN_MIN_BATCH=0

VERIFY=0

VLLM_MARLIN_REPACK_STAGED=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

GPU_UTIL=0.90
KV_MEM=
MAX_LEN=262136
MAX_SEQS=8
```

Why these values:

- **`MIN_BATCH=0`** is required for single-stream long context. With the
  default `4`, C1 falls back to FA2 and loses the TILE=64 gain; the short
  256/256 protocol is already equal between FA2 and split-KV, so `0` is safe
  there too.
- **`NSEG=16`** is fixed instead of adaptive. Adaptive is fine at C4 but at C8
  long context it creates only 2 segments → `8 × 1 × 4 × 2 = 64` blocks,
  leaving 44 of the 108 GA100 SMs idle (4.6% slower than fixed 16).
- **`VERIFY=0`** while the bind mounts are active: `verify.sh` matches patch
  lines and will report these files as modified.
- **`VLLM_MARLIN_REPACK_STAGED=1`** and
  **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** are independent
  GA100 fixes for Xid 31; they are required on this card, not optional tuning.

## Correctness

HyperQwen's `bench/test_spec_decode_attn.py` passes all 13 rows with the
adaptive segment path and with TILE=64: `max |ours - reference| <= 0.008`.

## Results

### Short protocol (256 in / 256 out, greedy, 3 samples, median)

Neutral — no regression. Step estimate = `TPOT × acceptance`:

| concurrency | stock step | tuned step |
| --- | --- | --- |
| C1 | 21.1 ms | 21.2 ms |
| C2 | 25.0 ms | 24.9 ms |
| C4 | 32.6 ms | 33.2 ms |
| C8 | 51.6 ms | 52.2 ms |

E2E differences at this protocol are acceptance noise, not kernel time.

### Single stream, fast mode, per verify step

The gain grows with context, because it comes from the KV tile:

| context | TILE=64 step | stock (TILE=32) step | gain | TILE=64 TTFT |
| --- | --- | --- | --- | --- |
| 4K | 21.4 ms | 21.5 ms | ~0% | 1.85 s |
| 32K | 24.2 ms | 25.8 ms | +6% | 15.95 s |
| 64K | 28.0 ms | 30.6 ms | +9% | 36.22 s |
| 96K | 30.8 ms | 35.5 ms | +15% | 60.92 s |
| 128K | 33.9 ms | 40.4 ms | +19% | 90.10 s |
| 192K | 40.3 ms | 50.2 ms | +25% | 161.95 s |
| 256K | 50.9 ms | 59.7 ms | +17% | 250.27 s |

Layer-level check at `kv=60k, q=8`: 455.9 µs vs 649.6 µs (tile 32), −30%.
Prefill is unaffected (it uses the FLASH_ATTN prefill path).

### C4 / C8, 32K input, steady-state verify step

Method: eight exact-token-ID prompts of 32K, prefix-cache warmed with a
`max_tokens=1` pass, then `C` concurrent streaming requests over the same
prompts and seed; step = median inter-chunk gap, median of three runs. Warming
first matters: cold simultaneous 32K prefills interleave with decode and make
the gaps meaningless for a kernel A/B.

| config | C4 step | C8 step |
| --- | --- | --- |
| stock TILE=32, NSEG=16 | 49.8 ms | 80.8 ms |
| TILE=64, NSEG=16 | 43.3 ms | 70.0 ms |
| TILE=64, NSEG=8 | 43.1 ms | 72.6 ms |
| TILE=64, NSEG=32 | 41.8 ms | 70.0 ms |
| TILE=64, adaptive (4 segs C4 / 2 segs C8) | 43.1 ms | 73.2 ms |

TILE=64 is worth ~13% at both C4 and C8. Adaptive segments cost 4.6% at C8.
NSEG=32 gains another 3.5% at C4 but is ~0.5% worse at C1 32K and ~2% worse at
C1 192K, so it is only worth using if C4 long-context traffic dominates.

## Caveats

- Not measured: C8 at 64K+ (8 × 64K ≈ 89% of the 589k-token pool) and more
  than 8 streams.
- The TILE=64 gain is a GA100 shared-memory fact; do not assume it on sm86.
  The `OutOfResources` retry keeps the kernel correct elsewhere, but there is
  no gain to expect there.
- The `MIN_BATCH` gate is batch-size based on purpose. A KV-length gate cannot
  be used under FULL cudagraph because KV length is not a captured constant.
- These files track the HQ 0.32.0 + ~50-patch + KVarN series. Re-diff them if
  the base image moves.

## Applying the patches without bind mounts

The `patches/` directory contains the same changes as unified diffs with paths
relative to the `vllm/` package root:

```sh
cd /app/venv/lib/python3.12/site-packages/vllm
for p in .../opt-170hx/patches/*.patch; do patch -p1 < "$p"; done
```

## Reproducing the measurements

`bench_stream_probe.py` (single stream) and `bench_multi_probe.py` (concurrent,
prefix-warm then timed) send exact token IDs through `/v1/completions` and
report TTFT plus per-chunk gap statistics. Both read `PORT`, `MODEL` and
`API_KEY` from the environment.
