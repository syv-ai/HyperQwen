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

| card | power | C1 decode | notes | source |
|---|---|---|---|---|
| RTX 3090 (reference) | 250 W | 133 tok/s | pool 57,669 tok, ppl 8.09 | [main README](../../README.md) |
| RTX 4090 | 450 W | **135.5 tok/s** | pool 57,669 and ppl 8.0921 reproduce exactly; no-spec control 60.3 (DFlash2 worth 2.31x); +1.9% from ~8% more bandwidth — batch-1 decode is bandwidth-bound, the extra compute has nothing to bite on | [#32](https://github.com/syv-ai/HyperQwen/issues/32) |

Measured with their own clients rather than the harness — comparable to each
other only loosely, and not rows for the table above:

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
- **Dual-GPU reports**: the controlled 1-vs-2×3090 A/B in
  [#40](https://github.com/syv-ai/HyperQwen/issues/40) (+16–35%,
  161.6 C1 greedy at 275 W, PCIe x8 without NVLink; independently reproduced
  in-thread at 153.6/250 W by a second dual-3090 box), the NVLink dual 3090 in
  [#7](https://github.com/syv-ai/HyperQwen/issues/7), dual 5060 Ti in
  [#22](https://github.com/syv-ai/HyperQwen/issues/22).
  [multi-gpu.md](../multi-gpu.md) has what transfers.
