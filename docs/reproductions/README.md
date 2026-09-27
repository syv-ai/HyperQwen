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
- [cmp-170hx-64gb.md](cmp-170hx-64gb.md) — the third sm80 box the README asked for:
  unlocked CMP 170HX 64 GB, Docker, vLLM 0.27.1 AND 0.29.0. The #72/#98 fault did
  not reproduce on either image (including a stock positive control and the
  maintainer's closing protocol past the 24-request boundary on main), plus the two
  boot-log lines the thread asked for and a 0.29.0 harness row

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
| RTX 4080 Super 32 GB (sm89), WSL2 | 250 W | 115.2 tok/s | setup B, greedy, GSM8K 0.960 over 200; a clamshell memory-modded board (the stock SKU is 16 GB, which this stack does not fit), confirmed by the reporter with `nvidia-smi` and the startup log | [#149](https://github.com/syv-ai/HyperQwen/issues/149) |
| 2x RTX 3060 12 GB (TP=2), setup D | 170 W stock (draws ~135) | 59.1 tok/s | the first 12 GB-card harness row: `SPEC=mtp CTX=long`, greedy (52.9 at the default temperature), `tok/step` 3.01, GSM8K 0.965 over 200. About 60% of a 3090 on the same setup (98.0): acceptance is normal, the gap is the PCIe all-reduce on every layer. That is vLLM 0.28; re-run on 0.29 it reads 58.3 greedy and 58.2 at the default temperature (tok/step 2.96 / 3.01), with the per-card KV pin lowered by ~105 MiB to boot. Run with `MAX_SEQS=1`, so only its C1 row is a measurement | [#68](https://github.com/syv-ai/HyperQwen/issues/68) |
| RTX 3090, setup D on vLLM 0.29 | 250 W | 95.4 tok/s | `SPEC=mtp CTX=long`, greedy (89.8 at the default temperature), `tok/step` 2.62, GSM8K 0.955 over 200. Two runs on one boot: the same harness uncapped (420 W) reads 99.4, +4%, the burst effect in the power note above. Docker image at 1cf8665 | [#192](https://github.com/syv-ai/HyperQwen/issues/192) |
| RTX 3090, `SPEC=dflash2 CTX=long` | 250 W | 113.4 tok/s | greedy (120.3 at the default temperature), `tok/step` 3.23 / 3.27, GSM8K 0.960 over 200, Docker at 73fd65d (vLLM 0.29). This profile is int8 KV on `TRITON_ATTN` (the launcher's DFlash2 long-context arm), not setup D's fp8 | [#194](https://github.com/syv-ai/HyperQwen/issues/194) |
| 2x RTX 3060 12 GB (TP=2), setup D, a second box | 170 W | 62.4 tok/s | greedy (56.7 at the default temperature), `tok/step` 2.73 / 2.58, GSM8K 0.955 over 200, vLLM 0.29 at 1cf8665. Peer-to-peer enabled by a community-patched driver (aikitoria's open-gpu-kernel-modules) but custom all-reduce off, `VISION=1`, `MAX_SEQS=4`, `GPU_UTIL=0.89`, `MAX_LEN=131072`: not the same launch as the row above, so read the +7% loosely | [#205](https://github.com/syv-ai/HyperQwen/issues/205) |
| 4x RTX 3060 Ti 8 GB (TP=4), setup E | 110 W/card | 118.1 tok/s | the only official profile that boots on 4x8 GB: `SPEC=dflash2 CTX=huge` at the launcher's own settings (380,218-token pool), greedy, `tok/step` 3.50 (113.9 / 3.45 at the default temperature), GSM8K 0.955 over 200. C8 lost 3/8 requests at the default temperature and 1/8 greedy. A, B, C and D all ran out of memory at startup; the adapted profiles that booted, and what each needed, are in the issue | [#210](https://github.com/syv-ai/HyperQwen/issues/210) |
| CMP 170HX 64 GB unlocked (sm80) | 180 W pinned | **164.7 tok/s** | setup D, `SPEC=dflash2 CTX=fast`, greedy (155.3 at the default temperature), `tok/step` 3.38, vLLM 0.29.0 at da8a8e9, Docker. At the pin this is a power-capped number (see the note above), yet still 3090-class. Also the #72/#98 non-repro on this card, the stock-image positive control, and the boot-log lines from the thread: [cmp-170hx-64gb.md](cmp-170hx-64gb.md) | this write-up |

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
  `curand` headers gotcha and the `NCCL_P2P_LEVEL=SYS` note. A TP2 follow-up
  on two of the same cards (one variable moved, TP 4 -> 2) puts the KV-path
  gap at **1.43x** at C1 greedy (71.4 -> 102.0) against TP4's 1.89x, and shows
  where it comes from: the fp8/FlashInfer arm does not care about TP at all
  (72.8 at TP4, 71.4 at TP2), while the int8/`TRITON_ATTN` arm gains 35% going
  from two cards to four, at ~2.7 tok/step in all four cells ---
  [#105](https://github.com/syv-ai/HyperQwen/issues/105).
- **2x RTX 3060 12 GB, a second box**: `SPEC=mtp CTX=long` at a 130 W per-card
  cap, ~50 tok/s at 49k context and ~35 at 80k by its own client, so not
  comparable to the table. The harness row above is the other dual-3060 box in
  the same thread ([#68](https://github.com/syv-ai/HyperQwen/issues/68)).
- **2x RTX 3060 12 GB, a third box, 0.27.1 against 0.29**: x4 slots, no
  peer-to-peer (NCCL over host shared memory), `SPEC=mtp CTX=long` at 128k,
  its own streaming client. Short-context decode 62.9 against 50 tok/s
  (1.26x), a tie at 48k, and 96k decode ~16% lower on 0.29 (the 0.27.1 side is
  a six-day-old single run). The finding that matters for 12 GB cards: at
  `GPU_UTIL=0.915` with no pin, 0.29 sized the fp8 pool at 3.01 GiB per card
  against 0.27.1's 2.41, and the MTP/FlashInfer workspaces then ran out of
  memory in warmup. The fix was the pin vLLM's own error suggests,
  `--kv-cache-memory=2975129437` (155,316 tokens, still enough for 128k).
  See the pin section of [multi-gpu.md](../multi-gpu.md)
  ([#68](https://github.com/syv-ai/HyperQwen/issues/68)).
- **2x RTX 3090 NVLink (TP=2), KVarN k4v2 + DFlash2 at 262k**: its own client
  and its own launch (DFlash2 k=3, 32 sequences, an OffloadingConnector tier
  over 32 GiB of RAM and NVMe), on a Swift-Qwen3.8 AutoRound build. A
  1,593,093-token pool (6.08 requests at 262k), C1 greedy 140.9 tok/s at depth
  0, 78.6 at 131k and 61.2 at 200k (39.3 with no drafter), 648 tok/s aggregate
  at 16 streams. GSM8K 0.970 over 200, verbatim reproduction from 200-227k cold
  prompts 40/40, needles at 131k and 240k 8/8, 24 conversations restored from
  the tier 24/24. It also carries eight patches: four KVarN fixes, including
  one for the "!!!!" output (a late KVarN flush into a block that now holds
  another request's mamba state), and four vLLM backports for evicted DFlash2
  conversations. They are not in this repo's series yet
  ([#208](https://github.com/syv-ai/HyperQwen/issues/208)).
- **2x RTX 3090, PCIe x8 without NVLink (TP=2)**: peer access through a
  community-patched driver lets vLLM's custom all-reduce run with the
  launcher's `expandable_segments:False` default. C1 greedy 207.5-211.9 tok/s
  against 194.9-197.6 with NCCL (+6.5 to +7.2%, `bench/real_rep.sh`, not the
  harness, so not a table row), `tok/step` 4.09 in both arms, GSM8K 0.970. A
  fragmentation soak with 200k prefills plus concurrent 60k prompts held 10/10
  up to `GPU_UTIL=0.96`, and both allocator settings failed identically at
  0.97 ([#163](https://github.com/syv-ai/HyperQwen/issues/163)).
- **2x RTX 4090 without peer-to-peer (x8 + x4, Docker on WSL2)**: TP=2 is
  slower than one card, single-user and batch, and two independent engines
  (`--data-parallel-size 2`) give 1.92x one card on a batch burst. The numbers
  and the NCCL knob that recovers part of TP=2's loss are in
  [multi-gpu.md](../multi-gpu.md)
  ([#190](https://github.com/syv-ai/HyperQwen/issues/190)).
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
