# The vLLM 0.29.0 pin

What the move from 0.28.0 to 0.29.0 changed in this repo, and what was re-measured.

[← back to the main README](../README.md)

## Dependencies

`vllm==0.29.0` and `huggingface_hub==1.28.0` (0.29 requires it; 1.27 conflicts). torch 2.13.0 and
compressed-tensors 0.17.0 are unchanged. `verify.sh` checks the 0.29.0 pin.

## Patch series

Retired, because upstream carries the change:

- `vllm-pr54282-draft-gumbel-salt.patch` (in 0.29.0)
- `xgrammar-spec-terminated.patch` (in 0.29.0)
- the graph-memory reserve hunk of `hybrid-kv-groups-v2-cudagraph.patch` (0.29 profiles graph memory
  natively; the kv_cache_utils hunks stay)
- the padded-page view hunk of `int4-kv-per-token-head.patch` (0.29's layout strides cover it; the
  triton kernel hunks stay)

Regenerated against the 0.29 tree, same behaviour:

- `dflash2-prewarm.patch` (the launch path gained the context-parallel arguments)
- `dflash2-lookup-drafting.patch` (32 hunks; import placement and the selector-walk anchor moved)

Carried from #100's branch: the launcher defaults `expandable_segments` off when a KV connector is
configured. Main never had it, so the offload profile cannot boot on a native box from main (vLLM refuses a
KV connector under the VMM allocator); WSL2 does not see it because its default is already off.

Carried after the port, because the 0.29.0 tag does not have them and the fork's open PRs do (#100, #101):

- `offload-mtp-serve.patch` (upstream #52771 and #52807, with the finished-request store watermark clamp):
  all seven hunks apply to the 0.29 tree unchanged. With it the offload profile serves stored hits under
  MTP/EAGLE on 0.29 as it does on the #100 branch; `bench/replay_offload_serve.py` is the oracle.
- `mamba-align-retire-null-gaps.patch` (upstream #55450): the `_remove_blocks_in_range` override applies
  unchanged; the field init and the free-path pop are re-anchored around 0.29's `_num_checkpoint_blocks`.
  What it buys is peak pool pressure during long align-mode prefills (about one pool token per prompt token
  on 0.28, both boxes); the long profile's peak rows are the check.

Both are on the fork branch as commits (`cpuchip/vllm` `qwen38/0.29` at 1bedc9ecb) and exported from there,
so a pin that carries the upstream change retires the file by dropping the commit.

Every other file in `patches/` was regenerated the same way in this port's last pass. Before that, the shipped
0.29 image placed nine of them by fuzz (GNU patch's default of two lines of slack when the context does not
match: hybrid-sw-block-promote, mamba-align-checkpoint-order, offload-dflash-eagle-groups, offload-wsl2-devptr,
spec-decode-attn, spec-sampler-prewarm, speed-knobs-envs, prefill-attn-int8 (then named triton-prefill-attn-int8), vision-tower-cpu-offload;
found by replaying the apply loop against a pristine wheel on the native 3090), and nothing reported it because the
check counted fuzz and offset under one word. The files now apply to v0.29.0 with exact context (30 clean, 1
at an offset, 0 with fuzz), the tree they produce is byte-identical to the fork branch, and both the Dockerfile
and `check_vllm_series.sh` run `patch --fuzz 0`, so the next drift fails the build and names the patch.
The two `kvarn/` files are exported from the fork the same way, from commits that sit after the whole
series (the order `kvarn/install.sh` applies them in), so they apply at `--fuzz 0` too; before this pass four
of their hunks landed by fuzz behind the series and the installer's `|| true` would have hidden a rejected
hunk in any file without a `port(kvarn-v2)` marker. The installer now applies at `--fuzz 0` and stops on
a rejected hunk.

Every knob the launchers export is registered in `envs.py` and read through `envs` (`speed-knobs-envs.patch`
carries the registry: `VLLM_PREFILL_ATTN`, `VLLM_SPEC_DECODE_ATTN_QMAX`, the `VLLM_DFLASH2_*` lookup and chain
family, `VLLM_SPEC_DECODE_ATTN`, `VLLM_SPEC_ATTN_BLOCK_M`, `VLLM_INT4_MQ_3D` and its debug switch,
`VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS`, `VLLM_DRAFT_TEMP_SCALE`, `VLLM_MARLIN_REPACK_STAGED`, and the Marlin int8 and tune
knobs that were registered but still read raw); a boot on the production line prints no "Unknown vLLM environment
variable detected" line, and no fork knob is read with a raw `os.environ` anywhere in the tree.

The KVarN modules under `kvarn/` read 20 knobs of their own (`KVARN_*`, Huawei CSL's names, one of them
`KVARN_POOL_MEM_FRAC` exported by both launchers); they are registered in `envs.py` under those names by
`kvarn-0.29.0.patch` and read through `envs` at all 25 sites, with the same defaults and the same
present-or-absent semantics where the reader branches on presence. One limit stays: vLLM's "Unknown vLLM
environment variable" warning fires only for names that start `VLLM_`, so a misspelt `KVARN_` knob is still
silent. Renaming them `VLLM_KVARN_*` would close that and diverge from KVarN's own documentation; it is offered
as a follow-up rather than done here. The launcher no
longer exports `VLLM_V2_CUDAGRAPH_MEM_MIB`: nothing on 0.29 reads it, since the graph-reserve hunk retired when vLLM
started profiling graph memory itself, so the export was a dead knob that every production boot warned about.

`VLLM_PREFILL_ATTN` is registered in `envs.py` (`speed-knobs-envs.patch`) and read through `envs` at both
sites in `flash_attn.py`; before, it was a raw `os.environ.get` and every boot with `PREFILL_ATTN` set printed
"Unknown vLLM environment variable detected: VLLM_PREFILL_ATTN" (the 0.28 line still does).

Adjusted for 0.29 API changes:

- `ngram-chains`: the `propose` override takes and forwards `dp_sync` (new runner signature).
- `hybrid-sw-block-promote`: `AttentionSpec.indexes_kv_by_block_stride` is gone; the "can this
  layer pad" check now mirrors upstream's own pad branch, any non-MLA attention layer pads. Reading
  the removed flag with a default of False refused every promotion, so the int4 profile padded the
  five drafter layers at block 16 and could not fit 120k tokens; that is the failure to look for if
  the promotion lines stop appearing at boot.

KVarN (`kvarn/`, `CTX=huge`) is ported: `kvarn-0.29.0.patch` and `kvarn-v2-runner-0.29.0.patch`. The
layout refactor removed the per-backend shape and stride hooks, so the backend now declares its layout
(`LBHNC`: heads outside tokens within a block) and folds the runner's 4D per-layer view back into one
tile per block and head with a `view`, so a wrong layout fails at the first KV update instead of
returning wrong numbers (the first port declared `LBNHC` and did exactly that; the guard caught it).
The old strided-view hunk and the four block-size hunks are retired; their reasons are in the patch
preambles and in `kvarn/README.md`.

On WSL2 the ttft/prefill/decode split of a single long request is unstable across builds while the request
total is not: the same 25k prompt on the huge profile moved from 9.75 s to first token and 20 tok/s decode
(0.29 without the backports) to 15.68 s and 78 tok/s (with them) with the total within 8 percent and equal
quality, and the native 3090 shows neither the split nor the move (16.37 s vs 16.46 s, totals within 1 percent).
Read totals and counters on WSL2; the split is where the first content byte lands relative to the work, not
compute (read on the native 3090, 2026-09-13).

Two layout strings one letter apart appear in 0.29 boot logs and both are right: the fast profile's FLASH_ATTN
path logs "Using LBNHC KV cache layout", KVarN (`CTX=huge`) logs "Using LBHNC". The letters are the physical
order of the cache tensor for that backend; 0.28 logged no layout line at all. The failure to watch for is the
reverse, KVarN declaring LBNHC, which the guard refuses at the first KV update.

## Acceptance (WSL2 4090, card 1, 2026-09-12)

Same script on the 0.28.0 image and the 0.29.0 image, fresh cache volume per run.

| profile | 0.28.0 | 0.29.0 |
|---|---|---|
| fast (dflash2 k=7, prefix caching): ppl / GSM8K n=100 | 10.8437 / 0.95 | 10.8437 / 0.95 |
| fast depth 25k: prefill / decode tok/s | 2678 / 105.0 | 2685 / 103.2 |
| fast depth 47k: prefill / decode tok/s | 2324 / 89.7 | 2323 / 97.9 |
| long (mtp, align, prefix caching): prefix ladder and 60k/160k peaks | pass | pass, same numbers |
| int4 (`kv_cache_dtype=int4_per_token_head`, 120k): KV tokens, mq3d oracle | 179,701, 8/8 | 173,134, 8/8 |
| int4 depth 25k / 90k decode tok/s | 42.1 / 19.5 | 43.2 / 19.1 |
| offload (12 GiB tier, mtp): served after eviction, tier guard | pass | pass |
| huge (KVarN k4v2_g128, dflash2, 262k): block / KV tokens | 2176 / 268,169 | 2176 / 268,169 |
| huge: ppl en / da, GSM8K n=100 | 10.7674 / 10.9097, 0.89 | 10.7691 / 10.9085, 0.93 |
| huge: needle at 32k / 90k / 200k (thinking off) | see note | retrieved at all three |
| huge: request time at 25k / 90k, WSL2 4090 (256 output tokens) | 30.2 s / 125.2 s | 22.4 s / 67.8 s |
| huge on the native 3090: prefill / decode at 25k | 1206 / 74.5 | 1210 / 73.7 |
| huge on the native 3090: prefill / decode at 90k | 1046 / 38.1 | 1047 / 38.6 |

The 47k decode column is bimodal per prompt slice on both images (rows land near 89 or near 104), so
the medians differ by draw, not by version; with prefix caching off both images read 102 to 104.

The huge-context rows: both pins compute the same KV geometry on both cards, and the native 3090 shows no
timing difference at all. On the WSL2 4090 the 0.28 boot delivers its first token about 16 s after the
engine's prefill and then streams fast, while the 0.29 boot delivers it early and streams slower, finishing
sooner at both depths with equal quality (three 0.28 boots across two images agree; the first stream delta
is content on both pins). Not understood, WSL2-only, not a regression. The needle passcode is retrieved at
32k, 90k and 200k on both pins (probe with thinking off).

One knob worth knowing: 0.29 defaults `prefix_cache_retention_interval` to dense checkpointing for
hybrid models with a draft model (the same behaviour 0.28 had). Setting it to 0 on this model halves
the 47k prefill (1303 vs 2323 tok/s) on this card. Leave it at the default. (2026-09-23: the halving is a
WSL2 4090 reading. On the native reference 3090, 0 and dense prefill within 0.5% of each other on 0.29 and
on 0.30, and 0.30's default is no longer dense; see [vllm-0.30.md](vllm-0.30.md).)

## Porting the next pin (the procedure this port settled on)

Three steps, in this order; skipping one is how this port lost most of a day.

1. **Triage with a dry run.** Install the new wheel in a throwaway container, `patch --dry-run` every file in
   `patches/` and `kvarn/` against it, and list three things: hunks that fail, hunks that apply with fuzz or a
   large offset, and files that apply cleanly but sit in a part of the tree the release notes say moved. Only
   the first list is visible; the other two are where the silent faults live (the block-promotion patch applied
   cleanly on 0.29 and refused every promotion).
2. **Read the commits, not the tree.** For every file on any of the three lists, `git log <old-tag>..<new-tag> --
   vllm/<file>` in a clone of vLLM, then read the commit that broke the hunk. Re-derive the hunk from what
   upstream changed and write the commit number into the patch preamble; retire a hunk only when a named commit
   carries the behaviour, and look for the in-tree sibling that had to make the same move (TurboQuant's change
   inside the layout refactor was the template for KVarN's). A hunk that regenerates cleanly against the new tree
   without this step is a guess that happens to apply.
3. **Boot with a control that must fail.** Build the image, run the acceptance profiles on two boxes against a
   0.28 image built from the same fork commit (the merge base, so the pair differs by the pin alone), and include
   one boot that is expected to go red (a deliberately wrong layout, a removed flag left in place). A green boot
   after a red one is evidence; a green boot alone is a build log.

What to read in the numbers: quality and geometry compare across versions; prefill compares by paired rows;
decode does not at n=3 even on byte-identical prompts, because the continuations differ. Perplexity lanes that
read the image's own source (the code corpus) never compare across pins. Prompts must be deterministic per row
(no timestamps in the salt), or no two runs share an input.

## Decisions made in this port, one line each (reject any by name)

Every one of these is a judgment call, not a consequence of the pin. Each is reversible on its own.

1. **KVarN keeps Huawei's knob names** (`KVARN_*`), registered under them; the `VLLM_KVARN_*` rename is a follow-up.
2. **Every fork knob is registered and read through `envs`**, including nineteen in patches that predate this port
   (the DFlash2 lookup and chain family, the split-KV `QMAX` and `BLOCK_M`, the Marlin int8 and tune knobs, the
   align-mode checkpoint flag); the reads are one-for-one with the raw expressions they replace.
3. **`VLLM_V2_CUDAGRAPH_MEM_MIB` is no longer exported by `single-user/start_qwen.sh`** on this line: nothing on
   0.29 reads it since the graph-reserve hunk retired.
4. **`hybrid-sw-block-promote` pads any non-MLA attention layer**, mirroring upstream's own pad branch, after the
   removed `indexes_kv_by_block_stride` flag; the 0.28 decision (promote instead of pad) is kept.
5. **`ngram-chains`' `propose` override forwards `dp_sync`** (the 0.29 runner signature).
6. **KVarN declares `LBHNC`** and folds the runner's 4D view into tiles with a `view` that fails on a wrong layout.
7. **The three KVarN commits sit last on the fork branch**, in the order `kvarn/install.sh` applies them, so the
   exported files apply at `--fuzz 0`; `install.sh` stops on a rejected hunk instead of `|| true`.
8. **`vllm-pr54282-draft-gumbel-salt` and `xgrammar-spec-terminated` are retired** (both in 0.29.0), and the
   graph-memory reserve hunk of `hybrid-kv-groups-v2-cudagraph` and the int4 padded-page view hunk are dropped.
9. **`--fuzz 0` everywhere** (Dockerfile, `check_vllm_series.sh`, `kvarn/install.sh`), so a drifted hunk fails the
   build by name instead of landing by guess.
10. **The two backports (#100, #101) are carried** as fork commits and exported patches, and retire when a pin
    carries the upstream change.
11. **Three patch preambles were cut to prose** (`dflash2-prewarm`, `dflash2-z-adaptive-emitted`,
    `prefill-attn-int8` carried a whole diff a second time above the first file header).
12. **The int64 casts from #91 and #109** are in the fork commits, not fixups on top.

## 0.28 against 0.29 on every run setting (WSL2 4090, card 1, 2026-09-18)

The README's five setups plus the production line, each booted on main's own CI image and on this branch's image, same box, same harness, one after the other in one session.


Arms: 028 = `ghcr.io/syv-ai/hyperqwen:sha-684e927` (main at 684e927, vLLM 0.28.0); 029 = this branch at 7a6b00f (vLLM 0.29.0, before the memory-profile fix). GPU_UTIL=0.90 both.
Per boot: bench/run_benchmarks.sh in the mode's own mode, run twice, second kept; bench/quality_battery.py --gsm-n 50; the bogus-knob control
(one made-up `VLLM_` name exported: the unknown-variable count must be exactly 1 and name it). Decode = C x 1000 / mean TPOT for cohorts,
C x 1000 / median TPOT for the 64-concurrent rows, as the script prints them. Same bench scripts and prompts (7a6b00f checkout) for both arms.

### B, single default (dflash2, CTX=fast)

- pool: 028 **66,692** tokens, 029 **77,872** tokens
- boot to health: 028 231s (200); 029 228s (000), 152s (200)  (a 000 is a cold boot that failed the KV-memory check and was retried; see the note at the end)
- unknown-variable lines (bogus knob named): 028 1 (1), 029 1 (1)
- perplexity (windows): 028 en 10.7602 (11175) / da 10.9098 (13917); 029 en 10.7623 (11175) / da 10.9109 (13917)
- GSM8K: 028 0.980 (n=50), 029 0.980 (n=50)

| row | 028 decode tok/s | 029 decode tok/s | delta | 028 TTFT ms | 029 TTFT ms | 028 tok/step | 029 tok/step |
|---|---|---|---|---|---|---|---|
| cohort C1 real prompts T=default | 121.8 | 138.1 | +13.4% | 128.49 | 102.42 | 2.70 | 2.80 |
| cohort C2 real prompts T=default | 234.5 | 264.2 | +12.7% | 212.45 | 183.68 | 2.75 | 2.78 |
| cohort C4 real prompts T=default | 434.8 | 473.4 | +8.9% | 338.15 | 321.60 | 2.82 | 2.78 |
| cohort C8 real prompts T=default | 666.1 | 782.0 | +17.4% | 764.43 | 644.46 | 2.78 | 2.80 |
| cohort C1 real prompts T=0 | 140.6 | 150.8 | +7.3% | 121.29 | 118.76 | 2.85 | 2.96 |
| cohort C2 real prompts T=0 | 269.5 | 286.9 | +6.5% | 189.95 | 193.82 | 2.89 | 2.99 |
| cohort C4 real prompts T=0 | 476.2 | 486.6 | +2.2% | 261.10 | 280.32 | 2.84 | 2.81 |
| cohort C8 real prompts T=0 | 738.7 | 810.5 | +9.7% | 693.18 | 755.35 | 2.81 | 2.87 |

### C, reproduction (DFLASH_TOKENS=15)

- pool: 028 **66,692** tokens, 029 **77,872** tokens
- boot to health: 028 148s (200); 029 148s (200)
- unknown-variable lines (bogus knob named): 028 1 (1), 029 1 (1)
- perplexity (windows): 028 en 10.7602 (11175) / da 10.9098 (13917); 029 en 10.7623 (11175) / da 10.9109 (13917)
- GSM8K: 028 0.980 (n=50), 029 0.980 (n=50)

| row | 028 decode tok/s | 029 decode tok/s | delta | 028 TTFT ms | 029 TTFT ms | 028 tok/step | 029 tok/step |
|---|---|---|---|---|---|---|---|
| cohort C1 real prompts T=default | 127.4 | 138.3 | +8.6% | 127.70 | 101.58 | 2.86 | 2.80 |
| cohort C2 real prompts T=default | 241.0 | 264.2 | +9.6% | 197.83 | 169.36 | 2.84 | 2.78 |
| cohort C4 real prompts T=default | 441.0 | 473.4 | +7.3% | 328.70 | 277.79 | 2.85 | 2.78 |
| cohort C8 real prompts T=default | 670.0 | 788.2 | +17.6% | 763.10 | 738.74 | 2.76 | 2.85 |
| cohort C1 real prompts T=0 | 141.0 | 150.8 | +7.0% | 105.08 | 123.18 | 2.86 | 2.96 |
| cohort C2 real prompts T=0 | 263.5 | 287.4 | +9.1% | 203.41 | 200.15 | 2.83 | 2.99 |
| cohort C4 real prompts T=0 | 468.9 | 486.6 | +3.8% | 257.24 | 323.00 | 2.78 | 2.81 |
| cohort C8 real prompts T=0 | 727.3 | 810.5 | +11.4% | 719.08 | 705.73 | 2.77 | 2.87 |

### M, Mads's production line (dflash2 fast, prefix cache, k=15, INT8_ACT, PREFILL_ATTN int8)

- pool: 028 **57,669** tokens, 029 **57,669** tokens
- boot to health: 028 356s (200); 029 299s (200)
- unknown-variable lines (bogus knob named): 028 1 (1), 029 1 (1)
- perplexity (windows): 028 en 11.0833 (11175) / da 11.3487 (13917); 029 en 11.0252 (11175) / da 11.3427 (13917)
- GSM8K: 028 0.960 (n=50), 029 0.920 (n=50)

| row | 028 decode tok/s | 029 decode tok/s | delta | 028 TTFT ms | 029 TTFT ms | 028 tok/step | 029 tok/step |
|---|---|---|---|---|---|---|---|
| cohort C1 real prompts T=default | 144.1 | 141.6 | -1.7% | 104.45 | 109.86 | 3.22 | 3.15 |
| cohort C2 real prompts T=default | 269.5 | 281.7 | +4.5% | 173.37 | 171.02 | 3.16 | 3.36 |
| cohort C4 real prompts T=default | 509.6 | 537.6 | +5.5% | 2360.67 | 2188.06 | 3.21 | 3.37 |
| cohort C8 real prompts T=default | 993.8 | 1040.3 | +4.7% | 7338.54 | 7112.88 | 3.08 | 3.26 |
| cohort C1 real prompts T=0 | 157.7 | 163.1 | +3.4% | 102.84 | 103.72 | 3.54 | 3.61 |
| cohort C2 real prompts T=0 | 298.5 | 306.7 | +2.7% | 174.72 | 178.28 | 3.51 | 3.59 |
| cohort C4 real prompts T=0 | 561.0 | 586.5 | +4.5% | 2238.36 | 1891.32 | 3.48 | 3.67 |
| cohort C8 real prompts T=0 | 1125.2 | 1151.1 | +2.3% | 6972.51 | 6462.25 | 3.48 | 3.61 |

### D, long context (SPEC=mtp CTX=long)

- pool: 028 **163,010** tokens, 029 **161,479** tokens
- boot to health: 028 226s (200); 029 246s (200)
- unknown-variable lines (bogus knob named): 028 1 (1), 029 1 (1)
- perplexity (windows): 028 en 10.7666 (11175) / da 10.9060 (13917); 029 en 10.7635 (11175) / da 10.9108 (13917)
- GSM8K: 028 0.960 (n=50), 029 0.980 (n=50)

| row | 028 decode tok/s | 029 decode tok/s | delta | 028 TTFT ms | 029 TTFT ms | 028 tok/step | 029 tok/step |
|---|---|---|---|---|---|---|---|
| cohort C1 real prompts T=default | 94.4 | 101.8 | +7.8% | 128.18 | 142.17 | 2.57 | 2.61 |
| cohort C2 real prompts T=default | 192.7 | 203.9 | +5.8% | 221.26 | 204.78 | 2.60 | 2.56 |
| cohort C4 real prompts T=default | 358.4 | 385.7 | +7.6% | 299.73 | 301.63 | 2.60 | 2.57 |
| cohort C8 real prompts T=default | 696.3 | 698.7 | +0.3% | 696.68 | 817.44 | 2.69 | 2.51 |
| cohort C1 real prompts T=0 | 103.6 | 105.7 | +2.0% | 130.66 | 138.14 | 2.61 | 2.64 |
| cohort C2 real prompts T=0 | 206.2 | 209.9 | +1.8% | 217.71 | 184.83 | 2.58 | 2.60 |
| cohort C4 real prompts T=0 | 406.1 | 402.8 | -0.8% | 303.66 | 282.55 | 2.72 | 2.66 |
| cohort C8 real prompts T=0 | 730.6 | 732.6 | +0.3% | 794.27 | 659.60 | 2.64 | 2.60 |

### E, huge context (CTX=huge, KVarN)

- pool: 028 **221,238** tokens, 029 **281,415** tokens
- boot to health: 028 283s (200); 029 286s (000), 148s (200)  (a 000 is a cold boot that failed the KV-memory check and was retried; see the note at the end)
- unknown-variable lines (bogus knob named): 028 1 (1), 029 1 (1)
- perplexity (windows): 028 en 10.7612 (11175) / da 10.9105 (13917); 029 en 10.7612 (11175) / da 10.9105 (13917)
- GSM8K: 028 0.940 (n=50), 029 0.960 (n=50)

| row | 028 decode tok/s | 029 decode tok/s | delta | 028 TTFT ms | 029 TTFT ms | 028 tok/step | 029 tok/step |
|---|---|---|---|---|---|---|---|
| cohort C1 real prompts T=default | 91.5 | 107.2 | +17.2% | 140.69 | 135.19 | 2.55 | 2.56 |
| cohort C2 real prompts T=default | 182.0 | 202.8 | +11.4% | 216.89 | 227.25 | 2.54 | 2.44 |
| cohort C4 real prompts T=default | 337.3 | 377.4 | +11.9% | 304.50 | 356.52 | 2.59 | 2.54 |
| cohort C8 real prompts T=default | 618.7 | 627.9 | +1.5% | 814.10 | 689.88 | 2.66 | 2.41 |
| cohort C1 real prompts T=0 | 100.9 | 110.9 | +9.9% | 135.11 | 140.94 | 2.60 | 2.58 |
| cohort C2 real prompts T=0 | 196.3 | 211.9 | +7.9% | 213.72 | 230.39 | 2.53 | 2.51 |
| cohort C4 real prompts T=0 | 367.3 | 389.9 | +6.2% | 336.51 | 338.38 | 2.65 | 2.59 |
| cohort C8 real prompts T=0 | 643.1 | 693.8 | +7.9% | 738.92 | 729.78 | 2.58 | 2.62 |

### A, batch mode

- pool: 028 **152,319** tokens, 029 **168,556** tokens
- boot to health: 028 283s (200); 029 252s (200)
- unknown-variable lines (bogus knob named): 028 1 (1), 029 1 (1)
- perplexity (windows): 028 en 10.8609 (11175) / da 11.0488 (13917); 029 en 10.8714 (11175) / da 11.0813 (13917)
- GSM8K: 028 0.940 (n=50), 029 0.940 (n=50)

| row | 028 decode tok/s | 029 decode tok/s | delta | 028 TTFT ms | 029 TTFT ms | 028 tok/step | 029 tok/step |
|---|---|---|---|---|---|---|---|
| 64conc 128in/512out | 1976 | 1904 | -3.6% | 5777.21 | 3780.91 | - | - |
| 64conc 256in/256out | 1634 | 1548 | -5.3% | 4169.22 | 3062.46 | - | - |
| cohort C1 real prompts T=default | 55.0 | 54.1 | -1.6% | 93.79 | 94.60 | - | - |
| cohort C2 real prompts T=default | 103.0 | 99.9 | -3.0% | 208.77 | 195.74 | - | - |
| cohort C4 real prompts T=default | 200.8 | 194.1 | -3.3% | 334.95 | 279.56 | - | - |
| cohort C8 real prompts T=default | 402.8 | 397.6 | -1.3% | 521.50 | 542.19 | - | - |

### Notes read from the logs

- Two 029 first boots (B default, E huge) exited with vLLM's KV-memory check on a fresh cache volume (cold torch compile): B needed 4.76 GiB for max_model_len 65536 against 4.73 available (equivalent utilization 0.8522); E read available KV memory as -1.86 GiB (equivalent 0.6733). The warm retry on the same volume passed at the same equivalent utilization. The 028 arms booted cold on equally fresh volumes and passed on this card (on the native 3090, 0.28.0's default boot refuses cold and warm alike: 4.63 GiB against 4.76 needed; falsifier section). The fix is `memory-profile-after-warmup.patch`; the falsifier rows are below.
- The code perplexity row is not shown: bench/quality_battery.py globs its code corpus from the installed vllm/v1/core, so each arm perplexed its own engine source (7586 windows on 0.28, 7525 on 0.29). The corpus is pinned to one tree for later runs. en and da are fixed parquet corpora and agree to three decimals on every pair.
- The quality battery could not write its result json because the data mount was read-only (OSError in the log); the printed PPL and GSM8K lines are the record.
- No VRAM column: Windows nvidia-smi does not see the WSL2 container; the engine's own memory lines are the record.

## 0.28 against 0.29 on the native 3090 (2026-09-18, the same twelve-boot protocol)

Same driver as the card-1 campaign above (`GPU_UTIL=0.90`, per-arm cache volume, bench twice with the second kept, the quality corpus pinned to the fifteen files whose per-file hashes match the card-1 copy), one GPU with production stopped, 13:37 to 16:44Z. Nine of twelve boots came up; the three that did not are 0.28 refusals with numbers, not failed measurements: B and C at 4.63 GiB available against 4.76 needed (cold and warm alike; the C arm inherited B's warm cache and read the same figure in 46 s), A at 4.74 needed for max_model_len 150,000. Those three settings have a 0.29 row and no A/B on this card. Single-stream decode on the C1 cohort at the model-default temperature (C over mean TPOT), tokens per step, pool in tokens:

| setting | 0.28 tok/s | 0.29 tok/s | 0.28 tok/step | 0.29 tok/step | 0.28 pool | 0.29 pool |
|---|---|---|---|---|---|---|
| B, single default | refused (4.63 GiB available, 4.76 needed) | 115.1 | | 2.67 | | 73,631 |
| C, reproduction (`DFLASH_TOKENS=15`) | refused (same 4.63, warm cache) | 115.1 | | 2.67 | | 73,631 |
| the production line | 126.4 | 134.0 | 3.19 | 3.42 | 57,669 | 57,669 |
| D, `SPEC=mtp CTX=long` | 88.7 | 92.6 | 2.60 | 2.66 | 156,122 | 179,846 |
| E, `CTX=huge` (KVarN) | 83.8 | 93.6 | 2.54 | 2.53 | 237,168 | 283,185 |
| A, batch mode, C1 single stream | refused (4.74 GiB needed for max_model_len 150,000) | 48.1 | | no tok/step reported | | 183,247 |

Three readings. The gains come from different places: E gains 11.7% with tokens per step flat (2.54 to 2.53), so that is raw decode plus a 19.4% larger pool, not better speculation; the production line gains 6.0% with tokens per step up 7.2% on an identical pool, which is speculation. The native gains are smaller than the WSL2 4090's and the production line flips sign (4090 -2%, within noise; 3090 +6.0%), with D and E at roughly half the 4090's figures (+4.4% and +11.7% against +8% and +17%); the two boxes are printed as two rows rather than averaged. B and C are identical on 0.29 on both boxes (115.1 and 115.1 here, 138.1 and 138.3 on the 4090), so `DFLASH_TOKENS=15` changes nothing at the default temperature. Batch on the 3090 (0.29 only, 64 concurrent 128 in / 512 out): 1213 tok/s decode by C over median TPOT (a different estimator from the C1 rows), 991 e2e, median TPOT 52.76 ms, mean TTFT 4.24 s. Under greedy (T=0) the three settings with both arms read: the production line 133.3 to 140.3 (+5.3%), D 95.5 to 93.5 (-2.1%), E 88.3 to 99.4 (+12.6%); D flips sign under greedy where the production line flipped between boxes at the default temperature, so the small deltas move under sampling and the headline deltas above are the default-temperature ones.

## The cold-boot falsifier for `memory-profile-after-warmup` (2026-09-18)

Same image with and without the patch, `GPU_UTIL=0.90`, "cold" = a fresh cache volume (torch compile from scratch), "warm" = the same volume booted again.

| box | profile | torch.compile | available KV | pool | result |
|---|---|---|---|---|---|
| native 3090 | B default, cold, no fix | 44.00 s | 4.42 GiB | none | refused at 207 s |
| native 3090 | B default, warm, no fix | 0.73 s | 5.37 GiB | 73,631 | health at 86 s |
| native 3090 | B default, cold, with fix | 43.98 s | 5.37 GiB | 73,631 | health at 221 s |
| native 3090 | B default, warm, with fix | 0.66 s | 5.37 GiB | 73,631 | health at 80 s |
| WSL2 4090 | B default, cold, no fix | 40.79 s | 4.76 GiB (4.73 after alignment, 4.76 needed) | none | refused at 228 s |
| WSL2 4090 | B default, cold, with fix | 40.79 s | 5.68 GiB | 77,872 | health at 242 s |
| WSL2 4090 | B default, warm, with fix | 0.00 s | 5.68 GiB | 77,872 | health at 126 s |
| WSL2 4090 | E huge, cold, no fix | 102.58 s | -1.86 GiB | none | refused at 286 s |
| WSL2 4090 | E huge, cold, with fix | 104.24 s | -0.97 GiB | none | refused at 280 s |
| WSL2 4090 | E huge, warm (campaign retry, no fix) | 0.00 s | positive | 281,415 | health at 148 s |
| native 3090 | E huge, cold, with fix | 43.85 s + 3.97 s | 4.38 GiB | 283,185 | health at 333 s |
| native 3090 | E huge, warm, with fix | 0.75 s + 0.06 s | 4.52 GiB | 292,035 | health at 81 s |
| WSL2 4090 | E huge, warm, with fix | 0.73 s + 0.04 s | 4.31 GiB | 279,646 | health at 132 s |
| native 3090 | B default, cold, 0.28.0 (main's CI image `sha-684e927`) | 51.53 s | 4.63 GiB (4.76 needed) | none | refused at 182 s |
| native 3090 | B default, warm, 0.28.0 (the C arm on the same volume) | 0.69 s | 4.63 GiB (4.76 needed) | none | refused at 46 s |
| WSL2 4090 | E huge, cold, with fix, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | warm-up 102.20 s | 4.47 GiB (estimate 5.44 computed, not applied; actual graph pool 0.0) | 290,265 | health at 283 s |
| WSL2 4090 | E huge, warm, with fix, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | warm-up 1.23 s | 4.68 GiB (estimate 0.37, actual 0.0) | 302,654 | health at 127 s |
| WSL2 4090 | E huge, cold, both fixes (allocator-measured pool), estimate on | warm-up 103.13 s | 4.19 GiB (estimate 0.28, actual 0.23; driver delta at the estimate capture 5.44) | 270,796 | health at 288 s |
| WSL2 4090 | B default, cold, both fixes | warm-up 40.92 s | 5.77 GiB (estimate 1.05, actual 0.09) | 79,414 | health at 226 s |
| WSL2 4090 | E huge, warm, both fixes | warm-up 1.50 s | 4.41 GiB (estimate 0.28, actual 0.23) | 284,955 | health at 137 s |
| native 3090 | B default, cold and warm, both fixes | not read | 5.48 GiB both (estimate 1.03, actual 0.09; driver 0.11) | 75,173 both | booted |
| native 3090 | E huge, cold, both fixes | not read | 4.48 GiB (actual 0.21; driver 0.24) | 290,265 | booted |
| native 3090 | E huge, warm, both fixes | not read | 4.62 GiB (actual 0.21; driver 0.24) | 299,115 | booted |

On the default profile the fix makes the cold boot measure exactly what the warm boot measures, on both boxes: same available KV, same pool, same equivalent utilization (0.8514 on the 3090 in all four rows, 0.8522 on the 4090). 0.28.0 is a different case on that card: at `GPU_UTIL=0.90` its default profile does not boot on the native 3090 at all, cold or warm (4.63 GiB available against 4.76 needed in both cache states, main's CI image; only the compile time differs, 51.53 s against 0.69 s), while it passes on the WSL2 4090 (the campaign's B arm, pool 66,692). So 0.28.0's shortfall there is flat and cache-independent; what 0.29.0's profiling change introduced is a cache-state dependence 0.28.0 never had (4.42 GiB cold against 5.37 warm), and the fix removes it: 5.37 GiB in both states, 0.74 above 0.28.0's figure, clearing a requirement 0.28.0 does not clear on that card in any state. (The falsifier for this reading is one 0.28.0 boot of the default profile passing at 0.90 on that box in either cache state; two independent boots read 4.63.)

On `CTX=huge` the profiling pass alone was not enough on the WSL2 4090: cold went from -1.86 GiB available to -0.97 and still refused, while the native 3090 passed cold (4.38 GiB) and warm (4.52 GiB), a 0.14 GiB cold penalty against 5.28 GiB on the 4090 (warm with the fix 4.31 GiB, pool 279,646). Where the 4090's residual sits was found by cutting the cache: five boots on card 1 of the fix image, `CTX=huge`, `GPU_UTIL=0.90`, every one a clone of the failed cold boot's own volume (a sixth, at `GPU_UTIL=0.80` to build a warmer state, refused at 1.91 GiB and is void):

| arm | cache state | available KV | pool | result |
|---|---|---|---|---|
| A | the failed cold boot's volume as it was left | 4.31 GiB | 279,646 | health at 132 s |
| C2 | the same volume a third time | 4.31 GiB | 279,646 | health at 131 s |
| D | `torch_compile_cache` deleted, everything else kept | -0.97 GiB | none | refused at 264 s |
| E1 | only the 37 directories holding a KVarN kernel kept (282 files, 41 MB), the other 1,967 compile-cache files deleted; Dynamo recompiled in 9.2 s | 4.31 GiB | 279,646 | health at 236 s |
| E2 | everything kept except those 37 directories | -0.97 GiB | none | refused at 160 s |

The KVarN kernels' compiled binaries and autotune results live inside `torch_compile_cache` (inductor's Triton directory; `TRITON_CACHE_AUTOTUNING=1` is vLLM's default), and they alone decide the outcome; the compiled graph does not. vLLM's own profiling breakdown (`VLLM_LOGGING_LEVEL=DEBUG`, `gpu_worker.py:636`) on the E2 state against the warm state: total consumed 16.79 GiB against 16.58, torch peak increase 0.32 against 0.32, so the profile itself is 0.21 GiB colder and would still grant about 4.5 GiB. The rest is the CUDA-graph memory estimate (`profile_cudagraph_memory`, on by default in 0.29): 5.44 GiB cold against 0.37 GiB warm (the "equivalent to --gpu-memory-utilization" line reads 0.6732 against 0.8845; every refusing E boot on this card carried 0.673, every passing one 0.8845). The estimate warms up and captures the largest graph first and reads the free-memory drop across that as graph memory, and on a cold cache that warm-up is where the decode-shaped KVarN kernels JIT and run their twelve-config autotune. On the native 3090 the same cold boot's line read 0.8847, so the estimate is not inflated there; the 0.14 GiB it pays is the profile-side term the 4090 also pays (0.21 GiB), the driver-side footprint of the freshly loaded kernels. Two terms, one trigger: the small one is on both boxes and survives the fix; the large one is the estimate pass and shows on the WSL2 4090 only. Why the same warm-up costs the estimate 5 GiB on that box and nothing on the other is not identified (WSL2's free-memory reporting under the driver is the suspect, not a claim).

Warming the estimate pass does not cure it: a build that ran `profile_cudagraph_memory` once uncounted before the counted call (tried on card 1, fresh volume) still refused at -1.2 GiB with the counted estimate at 5.3 GiB (equivalent utilization 0.6778), so on this box the cost is paid on every estimate call of a cold-compiled process, not once; and the same build measured 0.24 GiB less on a warm boot (4.07 GiB, pool 263,716 against 4.31 and 279,646), so the extra pass leaves state behind. That build is not on the branch. What the patch on the branch does is the profiling pass only (fork tip `512a9699c`, the same tree as the earlier export under a corrected message), and the refusal it left is closed by the second fix below; the `CTX=huge` cold-to-warm gap is not.

`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` boots that profile on that box (cold on a fresh volume 4.47 GiB, pool 290,265, health at 283 s, a request served; warm 4.68 GiB, pool 302,654, health at 127 s), and the knob is part of the compile-cache key (a second `torch_compile_cache` directory appears when it flips, so a boot with it back on is cold again). It is not the fix, because the pool it stops reserving is real. Read at the same two points the driver is read, by the allocator (`torch.cuda.memory_reserved`, a bind-mounted `gpu/model_runner.py`, card 1, warm boots): the real capture after KV allocation reserves 0.232 GiB on `CTX=huge` and 0.092 GiB on the default profile, while the driver's free-memory figure reads 0.000 GiB before and after it, so the logged "actual 0.0" on this box is the driver reading pinned at zero once the KV cache fills the budget, not the pool. On the native 3090, both readings from the same capture of the same boot: the real capture reserves 0.09 GiB on the default profile and 0.21 on `CTX=huge` by the allocator while the driver delta reads 0.11 and 0.24, and the estimate's throwaway capture 1.03 and 0.21 against 1.13 and 0.33 (native 3090, image built from this branch at f07554d). And the cold `CTX=huge` estimate on card 1, read the same way: the throwaway capture reserves 0.232 GiB, the same as warm, while the driver's free-memory figure falls from 5.436 GiB to 0.000 across it; that 5.44 is the "estimate" the profiler subtracted from the KV budget. The 0.29 estimator is built on the driver's free-memory delta, and under WSL2's paravirtual driver that reading collapses during the first compile of the KVarN kernels. The estimate's own capture also over-reserves about 1 GiB on the default profile on both platforms (1.03 GiB reserved against 0.09 real on card 1; 1.14 estimated against 0.11 actual on the native 3090): that is the estimate pass's shape, not WSL2, and it is unchanged by anything here.

So the branch carries a second fix, `cudagraph-memory-from-allocator.patch`: `capture_model` returns the allocator's reserved delta and logs the driver delta beside it, and the FULL-graph samples the estimate extrapolates from are read the same way, so the estimate never mixes an allocator total with driver samples. What that reading knowingly leaves unreserved is the driver-side residue of a capture (graph executables, kernel module loads, per-kernel local memory, non-torch workspaces): measured on the native 3090 as the driver delta minus the allocator delta on the same real capture, 0.02 GiB on the default profile and 0.03 GiB on `CTX=huge` (0.10 and 0.12 on the throwaway capture), the same sign on both profiles; on the WSL2 box the driver cannot measure it at all. Verified on card 1 with the estimate on: the cold `CTX=huge` boot on a fresh volume passes (4.19 GiB, pool 270,796, health at 288 s, a request served; the estimate reads 0.28 GiB for a 0.23 GiB pool while the driver's delta across the same capture still reads 5.44), the warm boot reads 4.41 GiB (pool 284,955), and the cold default boot 5.77 GiB (pool 79,414, 0.09 above the profiling fix alone). The 0.22 GiB between cold and warm on `CTX=huge` is the profile-side term above, and it survives both patches on both boxes: 0.14 GiB on the native 3090 (cold 4.48 GiB, pool 290,265; warm 4.62, pool 299,115) where the default profile is now exactly flat (5.48 GiB and 75,173 tokens in both states). Neither box has an explanation for it; on `CTX=huge` cold does not equal warm, and that stays open. The default profile's estimate is still about a GiB above its real pool (1.05 against 0.09), which answers which half of the estimate over-reserves: the throwaway capture itself reserves 1.03 GiB where the real capture reserves 0.09, so the extrapolation from FULL-graph samples is not the term. The logged "actual" now reads the pool (0.23 and 0.09) instead of 0.0. Two things the numbers say that the patch does not claim: the estimate itself is not made accurate, only the actual is measured (on the default profile 1.03 GiB estimated against 0.09 actual after the patch on both boxes, the log's own 1026.6%; on `CTX=huge` 0.28 against 0.23), and correcting the actual returns the difference to the KV budget on every boot, not only the failing ones (native 3090 default profile 5.37 to 5.48 GiB, pool 73,631 to 75,173; `CTX=huge` pool 283,185 to 290,265 cold and 292,035 to 299,115 warm; the native estimate on the default profile fell 1.14 to 1.03, the same 0.1 as on card 1, with four of five native profiles running FULL graphs, so the FULL-sample extrapolation contributes nothing there either). On the default profile the equivalent-utilization line is identical across failing and passing boots and is not the explanation; on `CTX=huge` it separates them, because the term it reports is the estimate, which is the thing that moves.


## Batch mode's `GPU_UTIL` on 0.29 (reference 3090, 2026-09-23)

The shipped batch default, `GPU_UTIL=0.972`, OOMs in warmup on 0.29 on a native
3090 (`Tried to allocate 96.00 MiB`, 47-69 MiB free), with the vision tower on
and off, on the reference checkpoint and on an asymmetric AWQ one alike (#182).
0.28 boots and serves the same settings. The cause is this port's own two memory
fixes working: with `memory-profile-after-warmup` and
`cudagraph-memory-from-allocator`, 0.29 stops over-reserving about 1.5 GiB (KV
6.09 GiB at 0.972 on 0.28, 7.63 GiB on 0.29, same box, same settings), and at
0.972 that over-reservation was the headroom batch mode's unprofiled warmup
transients lived in (64 seats, the int8 activation workspace). The campaign
tables above ran every arm at 0.90, so they never met it. The single-user
profiles are unaffected: they pin `KV_MEM`.

Each cell below is a boot, then 128 requests at 64-way concurrency (1024 in /
256 out), `KV=fp8`, `MAX_LEN=150000`, int8 MLP activations, and a check that
the log has no `OutOfMemoryError` after the burst:

| `GPU_UTIL` | `VISION=1`: KV, pool | `VISION=0`: KV, pool | requests |
|---|---|---|---|
| 0.93 | 6.48 GiB, 204,896 | 6.64 GiB, 210,309 | 128/128 both |
| 0.94 | 6.72 GiB, 212,628 | 6.88 GiB, 217,268 | 128/128 both |
| **0.95 (new default)** | 6.96 GiB, 219,587 | 7.11 GiB, 225,000 | 128/128 both |
| 0.95, cold compile cache | 6.95 GiB, 219,587 | | 128/128 |
| 0.96 | 7.19 GiB, 227,319 | | 128/128 |
| 0.972 (0.28's default) | OOM in warmup | OOM in warmup | |
| 0.28 at 0.972, for reference | 6.96 GiB, 220,360 | 6.09 GiB, 192,525 | 128/128 both |

The new default is 0.95: one full 0.01 step below the highest value that passed,
the same pool cold as warm, and at least the pool 0.28 had at 0.972, so batch
users lose no capacity in the upgrade. `KV=kvarn` (293,444 tokens) and
`KV=int4pth` (405,948) boot and serve at their existing 0.93 defaults and are
unchanged.

## Reading greedy tok/step across the two versions

A greedy harness cohort is not a like-for-like comparison between 0.28 and 0.29,
and its rows should not be read below about ±11%. 0.28 reproduces itself exactly
at temperature 0 (the same text and the same tok/step on every prompt, across
boots), but 0.29's numerics differ, so it writes different text on every one of
the eight real prompts. Per prompt that moves acceptance by up to ±25% in both
directions: on the production line, prompt 0 goes 3.23 to 4.03 while prompt 3
goes 4.10 to 3.34. The aggregate of the same eight requests is 3.43 against 3.41.
With a per-prompt spread of ~0.55 tok/step, the standard error of an eight-prompt
cohort is ~0.19 tok/step, about 6%, so a −5.6% greedy row is one standard error
and not a regression. It was first reported as one on the PR and retracted. To
compare drafting between versions, the lookup lane's own number is exact: a
verbatim-copy prompt at 15 drafts runs 15.45 tok/step on both.
