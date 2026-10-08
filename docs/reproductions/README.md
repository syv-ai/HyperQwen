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
- [a5000-230w.md](a5000-230w.md) — RTX A5000 24 GB (sm86) at a 230 W cap: the
  1× batch profile, the first 4× TP4 at 150k (0.28 and 0.29 tied within noise —
  the 0.28→0.29 regression is 1M-specific) and at 1M (0.29 regresses −10 to
  −22 %), the 24 GB mamba-align ceiling (~16–32k per single request) and the
  cudagraph-64 capture OOM, plus a **0.30 follow-up: the 1M regression is
  unchanged on vLLM 0.30, 1× stays GO** and a **TP=2 all-reduce-count test: TP2
  is worse than TP4 (−22 to −34 % prefill, −53 % C8, grows with size) and cannot
  hold 1M on 24 GB (~860k)** — so the regression is a 1M-profile effect, not an
  all-reduce-count effect — consolidates the #228 / #229 field reports

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
| CMP 170HX 64 GB unlocked (sm80) | 180 W pinned | **164.7 tok/s** | setup B (`SPEC=dflash2 CTX=fast`), greedy (155.3 at the default temperature), `tok/step` 3.38, vLLM 0.29.0 at da8a8e9, Docker. At the pin this is a power-capped number (see the note above), yet still 3090-class. Also the #72/#98 non-repro on this card, the stock-image positive control, and the boot-log lines from the thread: [cmp-170hx-64gb.md](cmp-170hx-64gb.md) | this write-up |
| 2x RTX 3090 (TP=2), no NVLink, peer-to-peer by a patched driver | 420 W (uncapped) | 90.9 tok/s | setup D as shipped: greedy (80.5 at the default temperature), `tok/step` 2.73 / 2.53, GSM8K 0.950 over 200, vLLM 0.29 at 1cf8665. The same profile with `EXTRA_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"` reads **156.0** greedy (146.2 default) at the same `tok/step` 2.59 / 2.57, so the gap is the fp8/FlashInfer path's step time, not the drafter (~31 to ~18 ms per step). `SPEC=dflash2` on the same box, 170.6 greedy / 166.9 default at 3.40 / 3.41, GSM8K 0.960. Uncapped, so read it against other uncapped rows | [#217](https://github.com/syv-ai/HyperQwen/issues/217) |
| 2x RTX 3080 20 GB (memory-modded, TP=2) | 220 W | 117.8 tok/s | `SPEC=dflash2 CTX=long` (int8 KV on `TRITON_ATTN`), greedy (114.8 at the default temperature), `tok/step` 3.28 / 3.25, GSM8K 0.970 over 200, on a third-party finetune (`ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound`) with the stock DFlash2 drafter. Native Fedora, Docker, vLLM 0.29 at 1cf8665. `CTX=fast` read 122.0 greedy in one indicative run | [#216](https://github.com/syv-ai/HyperQwen/issues/216) |
| RTX 5090 (sm120, Blackwell) | 400 W | **212.8 tok/s** | setup E (`CTX=huge`), single, Docker compose on CachyOS, driver 615.71.09 / CUDA 13.4, main at d5e2a01 (vLLM 0.30). Greedy (204.9 at the default temperature), `tok/step` 3.34 / 3.28, GSM8K 0.960 over 200. The first Blackwell row and the first above 200 tok/s at C1. Cohort ladder, greedy decode aggregate: C2 383.9, C4 781.2, C8 1,609.7 tok/s (e2e 340.5 / 348.9 / 394.5); mean TTFT 69 ms at C1 and about 4.2 s at C4, 8.1 s at C8 (the C4/C8 TTFT is the prefill queue of a huge-context profile, not a stall). Power is 400 W, not 250 W, so it is a headroom number: read it against the uncapped rows and against the power note above, not against the 3090 at 250 W. The reporter's follow-up confirms the settings (`SPEC=dflash2`, `PREFIX_CACHE=1`, `CTX=huge`, so KVarN `k4v2_g128` with the mamba align cache) and that 400 W is this card's **minimum** power limit (`nvidia-smi -pl 250` is refused: "should be between 400.00 W and 600.00 W"), so there is no 250 W run to ask for. The same box on a second configuration, `DFLASH_TOKENS=15` and `MAX_LEN=261744` with a hand-set `KV_MEM` of 5,798,205,824 bytes: C1 greedy 211.0 tok/s (192.3 at the default temperature), `tok/step` 3.63 / 3.35, GSM8K 0.950 over 200, C8 greedy decode 1,326.7 tok/s; that is about 26.5-27.5 GB in use with room left, and the reporter needed to override the launcher's pinned pool constant on a 32 GB card to get it to boot | [#247](https://github.com/syv-ai/HyperQwen/issues/247) |
| RTX 3090, 2x (TP=2), peer access and interconnect not stated | 250 W | 176.8 tok/s | setup B (`SPEC=dflash2 PREFIX_CACHE=1`, `MAX_LEN=262144`, `VISION=1`), greedy (166.1 at the default temperature), `tok/step` 3.62 / 3.40, Ryzen 9 5950X, driver 615.71.09 / CUDA 13.0, Docker, main at e1459c7. `--disable-custom-all-reduce`, so NCCL carries the collectives: read it against the 171.8 NCCL arm in the NVLink row above, not the 182.8 one. Decode aggregate, greedy: C2 317.5, C4 449.9, C8 614.0. No GSM8K line was reported | [#256](https://github.com/syv-ai/HyperQwen/issues/256) |
| RTX L40S 48 GB (sm89) | 350 W (the card's stock limit) | 150.4 tok/s | setup B (`SPEC=dflash2 PREFIX_CACHE=1`), greedy (133.9 at the default temperature), `tok/step` 3.46 / 3.08, GSM8K 0.965 over 200, main at e1459c7 (vLLM 0.30), Docker on Rocky 10. Decode aggregate, greedy: C2 278.2, C4 482.5, C8 792.9. The same box on setup E (`CTX=huge`): 133.3 greedy (129.7 default), 3.26 / 3.12, GSM8K 0.955, C8 mean TTFT 5.7 s (the huge-context prefill queue); on setup D (`SPEC=mtp CTX=long`): 106.8 greedy (105.6 default), 2.66 / 2.63, GSM8K 0.955. Not the stock launch: `GPU_UTIL=0.96`, `MAX_SEQS=16`, `VISION=1` with offload, `--max-num-batched-tokens 4096`. A second, smaller card sits in the box, so this is the case behind [#258](https://github.com/syv-ai/HyperQwen/issues/258) | [#257](https://github.com/syv-ai/HyperQwen/issues/257), [#253](https://github.com/syv-ai/HyperQwen/issues/253), [#252](https://github.com/syv-ai/HyperQwen/issues/252) |
| RTX PRO 4000 Blackwell SFF 24 GB (sm120) | 70 W (the SKU's default) | 60.1 tok/s | setup D (`SPEC=mtp CTX=long`), greedy (59.4 at the default temperature), `tok/step` 2.60 / 2.59, GSM8K 0.965 over 200, main at e1459c7. Bare-metal venv on Ubuntu 26.04, driver 595.91.07, torch 2.13.0+cu130. Decode aggregate, greedy: C2 121.4, C4 213.6, C8 331.5; mean TTFT 233 ms at C1, 1.2 s at C8. The first 70 W card in the table, so the power note above does not apply: 70 W is what this card is | [#266](https://github.com/syv-ai/HyperQwen/issues/266) |
| 2x RTX 3060 12 GB (TP=2), a fourth box, x16 + x4 | 100-170 W per card (ladder in the list below) | 64.5 tok/s at 170 W | `SPEC=mtp CTX=long`, `MAX_LEN=131072`, `MAX_SEQS=4`, `GPU_UTIL=0.89`, greedy (60.2 at the default temperature), `tok/step` 2.64 / 2.50, GSM8K 0.950 over 200, main at d5e2a01, Ubuntu 26.04 VM on Proxmox, driver 580.178.04. C1 reads 63.9 greedy at 100 W: see the ladder, power is not what limits this pair | [#255](https://github.com/syv-ai/HyperQwen/issues/255) |

Batch profile (setup A), `bench/run_benchmarks.sh batch`, 64 concurrent on
128 in / 512 out, aggregate decode:

| cards | power | C64 decode | notes | source |
|---|---|---|---|---|
| 1x RTX 3090 (reference) | 250 W | ~1,035 tok/s | 948 e2e; ~1,222 with every layer int8 | [main README](../../README.md) |
| 2x RTX 3090 NVLink (TP=2) | 250 W/card | **1,439 tok/s** | 1,344 e2e, median of three measured runs within 1%; documented batch defaults plus TP=2, KV pool 872,938 tokens, GSM8K 0.965 over 200; NCCL arm, so not the +6.4% custom-all-reduce path above | [#164](https://github.com/syv-ai/HyperQwen/issues/164) |
| 1x A5000 24 GB (sm86) | 230 W | **958.6 tok/s** | 64 conc, 128 in / 512 out, second run; ~0.92x the 250 W 3090 reference — the power-cap ratio, not an A5000 anomaly. Cohorts 39.7/75.2/142/278 tok/s (C1/C2/C4/C8), 0.84–0.92x the 3090 throughout; 149k prefill 1.68k tok/s; PPL 8.44, GSM8K 94.0% | [#228](https://github.com/syv-ai/HyperQwen/issues/228) |
| 3x RTX 3090 + 1x RTX 3090 Ti (TP=4), mixed cards, two NUMA domains, PCIe 3.0, no NVLink | 250 W/card (matched cap for the Ti) | 942 tok/s (689 at 256 in / 256 out) | 64 conc, 128 in / 512 out, the second of two passes, median TPOT 68.0 ms; e2e 863.4 tok/s. No speculation. Not the untouched setup A: `GPU_UTIL=0.90` against 0.95, `MAX_LEN=150000`, FP8 KV, a pinned KV of 17,146,314,752 bytes (pool 2,022,680 tokens), vision off. Image digest `sha256:65f399a1...` (vLLM 0.30.0 in the worker log). Cohorts (default temperature, greedy not run): C1 84.7, C2 142.7, C4 253.5, C8 370.7 decode. **Slower than one 3090 on this profile** (942 against ~1,035 decode, 863 against 948 e2e), which fits the dual-4090 report below (TP=2 slower than one card without peer-to-peer); the follow-up gives a TP=2 pair from the same box (the same-NUMA 3090s at PCIe 3.0 x8 and x16, PHB, no NVLink, 250 W, custom all-reduce disabled for failed peer access, so PYNCCL): C64 128 in / 512 out **961.1 e2e / 1,036 decode** (789 decode at 256 in / 256 out), C1 real prompts 64.6 e2e / 65.1 decode, against 863 / 942 at TP=4. So two cards beat four here, and match a single 3090. The two arms differ in `GPU_UTIL`, KV pin and topology together, and the retained logs do not show whether NCCL used P2P or SHM, so this is not a causal scaling or P2P result; the reporter plans matched P2P off/on runs at TP2 and TP4 and will post them. Prefix caching was on and saw zero hits in 106,920 queries (random prompts, as intended). No quality check was run, so no quality claim | [#135](https://github.com/syv-ai/HyperQwen/issues/135) |

Measured with their own clients rather than the harness — comparable to each
other only loosely, and not rows for either table above:

- **CMP 170HX 40 GB (GA100, sm80)**: 133.7 tok/s median (3x900 tok, greedy) on
  the shipped fast target — the first sm80 datapoint, level with the 3090 —
  and 97.8 tok/s on their own w8a16 int8 target after the sm80 repack
  workaround in [#27](https://github.com/syv-ai/HyperQwen/issues/27)
  (gotcha 41).
- **4x A5000 24 GB (sm86, 230 W/card), 1M context, TP=4**: the first 4× TP4 1M
  the repo has. 0.29 vs 0.28, identical command line except the image, both
  pods 230 W: prefill −22 % (16k/32k) → −14 % (512k) → −10.5 % / −10.2 % (1M,
  30:55/539 vs 27:41–28:21/602 tok/s), C8 decode −35 % raw (232 vs 356-385) and
  −12…−19 % with `max_cudagraph_capture_size:16` + `silu`-only (312-313 tok/s);
  reproduced over two independent days. No config lever closes the prefill gap
  (KV layout, mamba-cache-mode, GDN backend, int8 prefill attention, 8192
  chunks, custom-all-reduce — all negative or no-ops); quality is untouched
  (PPL 8.41, GSM8K 95.5%). The 1M usable budget is ≈ 999.8k (the chat template
  is counted on top). Full A/B and levers:
  [#229](https://github.com/syv-ai/HyperQwen/issues/229), write-up in
  [a5000-230w.md](a5000-230w.md).
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
- **2x RTX 3060 12 GB (TP=2), a power ladder from 100 W to 170 W**: one
  boot per step, the harness at each, greedy decode aggregate. C1 63.9 / 64.1 /
  64.4 / 64.5 / 64.5 / 64.5 / 64.5 / 64.5 tok/s at 100 / 110 / 120 / 130 / 140
  / 150 / 160 / 170 W (+0.9% across the whole range), `tok/step` 2.64 at every
  step; C8 357-366 tok/s with no trend, and a C8 mean TTFT of 12.6-13.2 s
  throughout. So on this pair the cap is not the lever: single-stream decode
  is bandwidth- and all-reduce-bound, and the reporter's cooling throttled the
  top card to about 115 W above a 120 W cap in any case, so the steps from
  120 W up are not independent readings of the watt setting. What the data
  supports is 100 W costing about 1% against 170 W on this setup; it does not
  say that every 12 GB pair is cap-insensitive. The reporter did not log
  per-card draw, so the 3060 row's "(draws ~135)" in the table above is still
  the only measured draw for this card
  ([#255](https://github.com/syv-ai/HyperQwen/issues/255)).
- **4x RTX 5060 Ti 16 GB (TP4), `NCCL_P2P_LEVEL`**: the cause behind the
  `NCCL_P2P_LEVEL=SYS` note, with the A/B and a micro-benchmark. NCCL's default
  allows P2P only up to `PXB` (cards behind a PCIe switch), so cards on CPU
  root ports used host shared memory (`via SHM`) even with peer access on all
  12 pairs; `NCCL_P2P_LEVEL=PHB` is the lowest setting that turned P2P on.
  Same container, only the NCCL line changed, two harness runs per arm: C1
  greedy decode 137.3 -> 145.7 (+6.1%), C2 +10.1%, C4 +13.3%, C8 within noise;
  prefill unchanged at ~1,080-1,090 tok/s on a 28k prompt. The mechanism is
  latency on the small allreduces of a decode step (an 8-token allreduce is
  13% faster, a 2048-token one 3%). One box, one boot without P2P, arms run in
  sequence and not interleaved, so quote the ranges, not the means. The point
  to take away is the pitfall: an A/B of "P2P on" that does not check the NCCL
  log for `via P2P` can have both arms on SHM. Probe scripts are in the
  report; the practical note is in [multi-gpu.md](../multi-gpu.md)
  ([#254](https://github.com/syv-ai/HyperQwen/issues/254)).
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
  conversations ([#208](https://github.com/syv-ai/HyperQwen/issues/208)). The
  "!!!!" fix is reworked in #222; the other seven are not in the series yet.
  The reporter then tested #222 on 0.29 and on the 0.30 series: 0 NaN requests
  with it on DFlash2 k=3 (320 replies, `bench/concurrent_collapse.py` 0 of 30,
  544 pages dropped) and on the no-drafter config (480 replies, 262 pages
  dropped). Without it, the no-drafter config still gave 1 NaN request in 480
  on 0.29's V2 runner, much rarer than the ~0.8% of replies on 0.28's V1 runner,
  and DFlash2 did not reproduce at all; why is not explained yet.
- **2x RTX 3090, PCIe x8 without NVLink (TP=2)**: peer access through a
  community-patched driver lets vLLM's custom all-reduce run with the
  launcher's `expandable_segments:False` default. C1 greedy 207.5-211.9 tok/s
  against 194.9-197.6 with NCCL (+6.5 to +7.2%, `bench/real_rep.sh`, not the
  harness, so not a table row), `tok/step` 4.09 in both arms, GSM8K 0.970. A
  fragmentation soak with 200k prefills plus concurrent 60k prompts held 10/10
  up to `GPU_UTIL=0.96`, and both allocator settings failed identically at
  0.97 ([#163](https://github.com/syv-ai/HyperQwen/issues/163)).
- **2x RTX 3080 20 GB (TP=2), on a box that also holds an RTX 5080**: the
  table row above, plus three setup notes. On a mixed box CUDA orders devices
  fastest first, so `CUDA_VISIBLE_DEVICES=0,1` paired the 5080 with a 3080;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes the indices match `nvidia-smi`. The
  AutoRound export loads through vLLM 0.29's GPTQ path after two `config.json`
  edits (`quant_method` to `gptq`, and `desc_act: false` added), because 0.29
  has no `auto-round` entry. The harness client's SSE reader gave up on
  prompts of 48k and more while the server completed them, so those prefill
  rows come from the engine log
  ([#216](https://github.com/syv-ai/HyperQwen/issues/216)).
- **2x RTX 3090 (TP=2) with peer-to-peer from aikitoria's patched driver, no
  NVLink**: setup D's fp8/FlashInfer path is the slow part at TP=2. Moving the
  same profile to int8 KV on `TRITON_ATTN` took C1 from 80.5 to 146.2 tok/s at
  the default temperature with acceptance unchanged, which is what #105 found
  at TP4 and #156 found under WSL2. `SPEC=dflash2` stayed ahead at 166.9, and
  it was also the reporter's best result in real coding traffic (over 100
  tok/s, against 40-50 for the MTP arms)
  ([#217](https://github.com/syv-ai/HyperQwen/issues/217)).
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
