# Reproduction: CMP 170HX 64 GB (GA100, sm80), Docker — vLLM 0.27.1 pin and 0.29.0 main

*September 2026. Docker (`--runtime nvidia`, IPC host), unlocked CMP 170HX 64 GB at a
pinned 180 W (the card's community power-limit service; see the power note in the
README's reproductions page — at 180 W these decode numbers measure the cap, not the
stack). `verify.sh --no-server` clean on both images.*

[← back to the reproductions README](README.md)

This box exists because of
[#72](https://github.com/syv-ai/HyperQwen/issues/72) /
[#98](https://github.com/syv-ai/HyperQwen/issues/98): two sm80 owners mid-bisect on a
speculation fault only their cards produce, and the README asking for a third sm80 box
to separate the card from the build. This is that third box. **The fault did not
reproduce — on either vLLM, in either the stock or a locally patched image, under the
reporter's exact repro shape and under the maintainer's closing protocol.**

## Hardware and stack

| | |
|---|---|
| GPU | NVIDIA CMP 170HX, VRAM-unlocked to 64 GiB (GA100, sm_80), PCIe Gen2 x4 (riser), driver **610.57.04**, CUDA UMD 13.x |
| Power | pinned 180 W (not raised) |
| OS | Ubuntu, headless; Docker everywhere |
| Images | `ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0` (vLLM **0.27.1**) and a local build of **main** `da8a8e9` (vLLM **0.29.0**, the 46-patch series + KVarN, verify.sh clean) |
| models | `Qwen3.8-27B-W4A16-AutoRound-fast` + DFlash2 drafter W4A16 (and the Huihui abliterated INT4-AWQ-GPTQ variant on the 0.27.1 image) |

One difference from both reporters worth flagging: **driver 610.57.04, not
610.43.02.** Both #72 and #98 report 610.43.02; this is the largest untested delta
between this box and the failing ones.

## The two boot-log lines the #72 maintainer asked for

Read off a 0.27.1 `CTX=fast` boot (Huihui INT4-AWQ-GPTQ, `KV_MEM=34e9`, GPU_UTIL 0.93):

```
Setting attention block size to 448 tokens to ensure that attention page size is >= mamba page size.
Initial free memory 63.08 GiB, reserved 35.39 GiB memory for KV Cache ...
Sliding-window bucket(s) are the smallest; using group_size 8 ...
```

`num_blocks = 35.39 GiB / (448 * 4096 * 8) ≈ 2,560` — against the maintainer's measured
int32 overflow wall at `floor(2^31 / (BS*Hkv*D))` blocks (≈ 4,369 at BS=480, scale-
equivalent here), the highest block id reaches **~59% of the wall at util 0.93**,
**~71% at util 1.0 with zero overhead**. Same `group_size 8` on the 0.29.0 boot (block
size 448, 5.2 GiB pinned pool). **The sensitive input the thread flagged is confirmed: the wall
is unreachable on this card at any boot this stack produces** — on both vLLM versions.

## #72 repro attempts — all clean

The reporter's shape: `--dataset-name random --input-len 8192 --output-len 512
--num-prompts 12 --max-concurrency 4 --temperature 1 --ignore-eos`, rounds
back-to-back; k=7, `VLLM_SPEC_DECODE_ATTN=1`, `VLLM_DFLASH2_LOOKUP=1`,
`PREFIX_CACHE=1`, `QMAX=8`.

| image | vLLM | geometry | rounds / requests | result |
|---|---|---|---|---|
| 69ba4d0 + int64 cast (local) | 0.27.1 | MAX_SEQS=1 | 22 rounds ≈ 26k verify steps | clean, acc 60.5%, 0 Xid |
| 69ba4d0 + int64 cast (local) | 0.27.1 | MAX_SEQS=8, conc 8 | 6 rounds ≈ 50k verify steps, prefix hit 69.7% | clean, acc 48-54%, 0 Xid |
| **stock 69ba4d0 (positive control)** | 0.27.1 | MAX_SEQS=8, conc 8 | 8 rounds ≈ 34k verify steps | **clean**, 0 Xid, 0 restarts |
| **stock 69ba4d0, reporter's exact command** (`--backend openai-chat`, `/v1/chat/completions`) | 0.27.1 | conc 4 | 3 rounds | **clean**, 12/12 each, 0 Xid |

The positive control is the important row: **the unpatched stock image survives the
repro on this box**, so whatever kills the reporters' cards is not in the patch
differential and not in the harness. Remaining deltas vs the failing setups: driver
610.57.04 vs 610.43.02, PCIe Gen2, and (for #72) a vanilla wheel + hand-applied
patches instead of the official image.

## The maintainer's closing protocol, on main (0.29.0)

`k=7`, `VLLM_SPEC_DECODE_ATTN=1`, `CTX=fast` (max_model_len 65,536 as #72's serving
config), `MAX_SEQS=4`, the reporter's bench shape, one boot:

- 3 rounds x 12 requests, then a 48-request round on the same boot — **84 requests
  total, past the 24-request boundary**, every one successful.
- 0 Xid 31, 0 restarts, acceptance 50-60% throughout.

## Harness rows (main `da8a8e9`, 0.29.0, `Qwen3.8-27B-W4A16-AutoRound-fast`)

`bench/run_benchmarks.sh single` on setup B (`SPEC=dflash2 CTX=fast`), first row is the README-protocol second run:

| cohort | e2e | decode | tok/step |
|---|---:|---:|---:|
| C1 (default T) | **151.4** | 155.3 | 3.21 |
| C1 (greedy) | **159.9** | 164.7 | 3.38 |
| C4 (greedy) | 358.5 | 424.6 | 3.31 |

At 180 W pinned, this card delivers a 3090-class C1 decode on 0.29.0 — the earlier
40 GB field report (133.7 greedy) reads as power-capped the same way. Not run:
`quality_battery.py` (needs the wikitext/GSM8K data pre-staged; skipped, and the C1
row is the only headline number claimed here).

## What this does and does not establish

Established: the #72/#98 signature needs something this box lacks — most plausibly
the 610.43.02 driver (both reporters) or the hand-patched-wheel build (#72). The
int32 block-id wall is unreachable in every boot this stack makes on 64 GB, so #72 is
not a hidden case of #86 on this card.

Not established: that the reporters' fault is gone on 610.57.04. The single highest-
value follow-up is theirs to run: **same repro, driver bumped to 610.57.04** — a
one-variable A/B on hardware that actually dies.

Raw logs: 0.27.1 campaigns and the stock controls in this box's
`~/services/qwen38-cmp/phase0-logs/`; the 0.29.0 protocol and harness output in the
description of the pull request that added this page (#218).
