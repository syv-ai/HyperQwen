# Reproductions

Other people's hardware, running this stack. The write-ups in this folder are
full reproductions; the list below collects the shorter reports from issues.

[← back to the main README](../../README.md)

- [native-3090.md](native-3090.md) — the reference bare-metal 3090 run
  (Python 3.14, no container): hardware table, README-parity benches, needle
  recall, the prompt-shape finding
- [../ubuntu-3090.md](../ubuntu-3090.md) — the later campaign on that same box:
  the offload-fix control arm, retention bisection, the MAX_SEQS ladder, and the
  batch arm only a headless box can run
- [../wsl2-4090.md](../wsl2-4090.md) — RTX 4090 under Windows 11 / WSL2, the
  cross-platform half of the same campaign

## Results from other hardware

Community reproductions of the single-user headline number, harness runs first.
`bench/run_benchmarks.sh single`, greedy, second run (the first reads low):

**Set the power limit before you compare anything.** Every number in this repo
is an RTX 3090 at 250 W, and on this card that is not a soft preference. A
sustained-load ladder from [#62](https://github.com/syv-ai/HyperQwen/issues/62)
(14 minutes per cell, same service): 200 W gives 57.5 tok/s at 781 MHz, 250 W
gives 85.6 at 978 MHz, and 280 W gives 86.7 — it hits 90 °C within two minutes,
pins the fan at 100% and throttles back to the same throughput. Prefill loses
about the same third at 200 W. So a quiet home box capped at 200 W is measuring
its power cap rather than this stack, and nothing above 250 W is worth the
noise.

**"Nothing above 250 W" is a sustained-load statement, and a harness cohort is
not sustained load.** The 280 W cell above is flat because 14 minutes of it
reaches 90 °C and throttles back. A WSL2 3090 running the harness at its stock
370 W cap measured C1 greedy decode at 125.9 tok/s against 110.7 at a hard
250 W (+14%), drawing 312-354 W in short bursts that never get hot enough to
throttle ([#156](https://github.com/syv-ai/HyperQwen/issues/156)). Both
readings are correct and they do not contradict each other: a benchmark cohort
is minutes of burst, a served box is hours of sustained decode. Every number in
this repo is at 250 W because that is what a box in service can hold; if you
compare against one of them, cap yours too, or you are measuring headroom you
will not have in production.

| card | power | C1 decode | notes | source |
|---|---|---|---|---|
| RTX 3090 (reference) | 250 W | 133 tok/s | pool 57,669 tok, ppl 8.09 | [main README](../../README.md) |
| RTX 4090 | 450 W | **135.5 tok/s** | pool 57,669 and ppl 8.0921 reproduce exactly; no-spec control 60.3 (DFlash2 worth 2.31x); +1.9% from ~8% more bandwidth — batch-1 decode is bandwidth-bound, the extra compute has nothing to bite on | [#32](https://github.com/syv-ai/HyperQwen/issues/32) |
| 2x RTX 3090 NVLink (TP=2) | 250 W | **182.8 tok/s** | setup B, greedy, GSM8K 0.965 over 200. Two arms, one variable: 171.8 with `--disable-custom-all-reduce` (NCCL carrying the collectives), 182.8 with custom all-reduce working, which on this box needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` -- the CUDA-graph capture crash at `custom_all_reduce.cuh:164 'invalid argument'` is gotcha 3, not NVLink. 3.32 tok/step in both arms | [#159](https://github.com/syv-ai/HyperQwen/issues/159), [#163](https://github.com/syv-ai/HyperQwen/issues/163) |
| RTX 3090, Windows 11 / WSL2 | 250 W | 110.7 tok/s | setup B, greedy, `tok/step` 3.28 — the reference profile's own acceptance, so this is the WSL2 tax on step *time*, not on the drafter. 125.9 at the card's stock 370 W cap, which is a burst effect: see the power note below | [#156](https://github.com/syv-ai/HyperQwen/issues/156) |
| RTX 4080 Super 32 GB (sm89), WSL2 | 250 W | 115.2 tok/s | setup B, greedy, GSM8K 0.960 over 200; a 32 GB card under that name is a board mod rather than a stock SKU, so the card line is as reported | [#149](https://github.com/syv-ai/HyperQwen/issues/149) |

Batch profile (setup A), `bench/run_benchmarks.sh batch`, 64 concurrent on
128 in / 512 out, aggregate decode:

| cards | power | C64 decode | notes | source |
|---|---|---|---|---|
| 1x RTX 3090 (reference) | 250 W | ~1,035 tok/s | 948 e2e; ~1,222 with every layer int8 | [main README](../../README.md) |
| 2x RTX 3090 NVLink (TP=2) | 250 W/card | **1,439 tok/s** | 1,344 e2e, median of three measured runs within 1%; documented batch defaults plus TP=2, KV pool 872,938 tokens, GSM8K 0.965 over 200; NCCL arm, so not the +6.4% custom-all-reduce path above | [#164](https://github.com/syv-ai/HyperQwen/issues/164) |

Measured with their own clients rather than the harness — comparable to each
other only loosely, and not rows for either table above:

- **CMP 170HX 40 GB (GA100, sm80)**: 133.7 tok/s median (3x900 tok, greedy) on
  the shipped fast target — the first sm80 datapoint, level with the 3090 —
  and 97.8 tok/s on their own w8a16 int8 target after the sm80 repack
  workaround in [#27](https://github.com/syv-ai/HyperQwen/issues/27)
  (gotcha 41).
- **RTX 5090 32 GB (sm120)**: ~410-449 tok/s on code and ~198 on prose at
  `CTX=fast`, 500 W cap, roughly flat out to `CTX=huge` at 240k — different
  prompts, output length and rate definition, so deliberately not in the table
  (their own insistence, and correct). Setup gotchas and the full ladder:
  [#35](https://github.com/syv-ai/HyperQwen/issues/35).
- **RTX 4090, Windows 11 / WSL2 (Docker path)**: reproduces with zero repo
  changes; CTX ladder incl. huge's pool byte-identical to the 3090 reference
  (268,169), concurrency ladder to N=8, and a measured both-ways case for
  leaving the `KV_MEM` pin alone — [wsl2-4090.md](../wsl2-4090.md).
- **RTX 4090, Windows 11 / WSL2, second box**: all three single-user profiles
  plus the experimental int4 one (230,830-token pool at 160k), and 135k
  real-task numbers on the MTP + FP8 daily-driver profile — 62 tok/s decode on
  QA over the document, TTFT 5.4 s → 0.33 s on a repeat turn. Also the
  `nvidia-smi dmon` detector for WSL2 host-backed memory now in gotcha 43 —
  [#61](https://github.com/syv-ai/HyperQwen/issues/61).
- **RTX 3090, Windows 11 / WSL2**: independent confirmation of the int8 prefill
  stack on Ampere — `INT8_ACT=int8` +59%/+57%/+37% at 5k/21k/66k, the int8-QK
  attention adding +1.8% at 21k and +6.3% at 66k on top, against this repo's
  +2.7% at 16k and +5.3% at 51k. Plus the power-limit ladder quoted above —
  [#62](https://github.com/syv-ai/HyperQwen/issues/62).
- **4x RTX 5060 Ti 16 GB (sm120, TP4)**: the canonical harness at TP4, and the
  cleanest multi-arm A/B this project has received --- four arms on one box,
  `PREFIX_CACHE=1` throughout, one variable apart. C1 greedy decode 72.8
  (`SPEC=mtp` k=3, fp8/FlashInfer) -> 137.6 (`SPEC=mtp`, int8 KV on
  `TRITON_ATTN`) -> 154.3 (`SPEC=dflash2` k=7, same KV). So the headline 2.37x
  from the first round is **1.89x KV dtype and attention backend, 1.12x
  drafter**, and the drafter's advantage is gone by C4. `DFLASH_TOKENS=15` at
  TP4 is a clear loss (154.3 -> 89.4 at C1, TTFT 3.3x at C8), which is the TP4
  half the launcher's keep-it-at-7 warning was missing. Also the source of the
  `curand` headers gotcha and the `NCCL_P2P_LEVEL=SYS` note ---
  [#105](https://github.com/syv-ai/HyperQwen/issues/105).
- **2x RTX 3060 12 GB**: 24 GB of VRAM in two 12 GB cards, which forces a
  tensor-parallel split the single-card path never takes. Two independent boxes
  in the thread, both on the Docker path at `SPEC=mtp CTX=long`, one reporting
  ~50 tok/s at 49k and ~35 at 80k context at a 130 W per-card cap. Measured with
  their own clients, so not comparable to the table above --- a harness run on
  this pair is the open ask in
  [#68](https://github.com/syv-ai/HyperQwen/issues/68).
- **3x RTX 3090**: confirms TP=3 is refused by the checkpoint rather than by this
  repo (4 KV heads, 64 layers: neither TP=3 nor an even PP=3 split exists), so
  the third card idles under `--tensor-parallel-size 2` by construction. The
  thread's workaround --- a second independent engine on the spare card, with
  `VLLM_OFFLOAD_KEEP_SHM=1` on both containers --- is the shape a triple-card
  owner wants: [#104](https://github.com/syv-ai/HyperQwen/issues/104).
- **2x RTX 3090 (TP=2), native Ubuntu**: batch harness at 917 decode / 861 e2e
  at 64 concurrent against the single card's ~1,035 / 948, and 60.1 tok/s at C1
  against 46 --- two cards giving less aggregate throughput and ~30% more
  single-stream, the same shape as [#40](https://github.com/syv-ai/HyperQwen/issues/40).
  Run with `KV=kvarn VISION=1` rather than the reference batch profile, so
  indicative rather than a controlled row:
  [#135](https://github.com/syv-ai/HyperQwen/issues/135).
- **Dual-GPU reports**: the controlled 1-vs-2×3090 A/B in
  [#40](https://github.com/syv-ai/HyperQwen/issues/40) (+16–35%,
  161.6 C1 greedy at 275 W, PCIe x8 without NVLink; independently reproduced
  in-thread at 153.6/250 W by a second dual-3090 box), the NVLink dual 3090 in
  [#7](https://github.com/syv-ai/HyperQwen/issues/7), dual 5060 Ti in
  [#22](https://github.com/syv-ai/HyperQwen/issues/22).
  [multi-gpu.md](../multi-gpu.md) has what transfers.
