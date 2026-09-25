# The vLLM 0.30.0 pin

What the move from 0.29.0 to 0.30.0 changed in this repo, what was re-measured, and what the port taught the
procedure.

[← back to the main README](../README.md)

## Dependencies

`vllm==0.30.0`, `huggingface_hub==1.32.0` (0.30.0 requires `>=1.31.0`; the 0.29 pin of 1.28.0 made the image
unresolvable), FlashInfer `0.6.18.post1` (from 0.6.18; the wheel pins `flashinfer-python==0.6.18.post1`, and
`docs/install.md`'s `flashinfer-cubin` pin moved with it). torch 2.13.0 is unchanged. `verify.sh` checks the
0.30.0 pin and the `kvarn-0.30.0` patch names.

## Patch series

One fork branch, `cpuchip/vllm` `qwen38/0.30` (v0.30.0 + 45 topic commits), with an annotated tag at every export
point (`qwen38/0.30-cut1` ... `-cut5`), so a later rewrite of the branch never orphans a hash a patch file names.
The files apply to v0.30.0 at `--fuzz 0` (42 series patches: 39 clean, 3 at an offset), and the series plus both
KVarN patches reproduce the branch's `vllm/` tree with 0 differing files.

Retired, because 0.30.0 carries the change:

- `offload-mtp-serve.patch` (vllm #52771, #52807 and #54288)
- `mamba-align-retire-null-gaps.patch` (vllm #55450)

Both retirements are measured, not only read from ancestry (reference 3090, the acceptance harness's profiles D and
B, one frozen corpus, both images hashed against their commits). The offload tier serves an evicted prefix back by load
on both pins: serve ratio 0.16 (11.0 -> 1.78 s), 486,932,480 bytes CPU to GPU, the same 4.52 GB stored. Peak pool
during a long align-mode prefill is identical on both pins (0.16736 at 60k characters, 0.36402 at 160k), and the
control that shows the check can see the defect, 0.29 with the patch removed, reads 0.26778 and 0.64017 (60% and
76% higher: 1.85 pool tokens per prompt token at 160k against 1.05).

Re-cut against upstream code that moved, same behaviour:

- `hybrid-sw-block-promote`: upstream #53007 changed hybrid grouping; the #142 divisor condition is carried.
- `spec-decode-attn`: `flash_attn.py` resolved against #55768; `envs.py` from the 0.29 line.
- `mamba-align-checkpoint-order`, `offload-dflash-eagle-groups`, `sampler-small-topk-fast-softmax`: re-cut
  unchanged in intent. `offload-dflash-eagle-groups` no longer needs its #33 hunk: with no annotated draft group,
  0.30's offloading scheduler treats every group as non-draft (the opposite of the fallback #33 fixed). One known
  difference, bounded and not measured: on 0.29 the hunk flagged the DFlash2 drafter's sliding-window group ("EAGLE/MTP
  draft attention groups [8] detected"), so the drafter's volatile trailing chunk was not stored while decoding. On 0.30
  nothing is flagged ("no KV-cache group is annotated as a drafter group"), and that chunk is stored. Upstream's own
  comment on that path bounds the effect: a drafter chunk served one chunk stale can only lower speculative acceptance,
  since the target verifies every draft. It applies to DFlash2 with the CPU offload tier, on CPU-tier hits; MTP is the
  same on both pins. A resend probe (GPU-tier hit) reads the same on both pins: 22,400 of ~23.3K tokens for dflash2 k7,
  22,464 for mtp.
- `spec-sampler-prewarm`: kept. #56323 warms the V1 sampler's kernels, not the V2 runner's that this patch warms.
- `cudagraph-memory-from-allocator`: both readings sit inside #54646's `freeze_gc_for_cudagraph_capture()` block.
- `kvarn-0.30.0`, `kvarn-v2-runner-0.30.0`: re-cut with #54713's `replay_boundaries`; `KVARN` is `KVQuantMode`
  value 11, because upstream took 10 (every use is by name).

Adapted to upstream #54809, which removed GPTQ activation ordering (`has_g_idx`, `g_idx_sort_indices`, and
`marlin_gemm`'s `g_idx`/`perm`/`is_k_full`):

- `marlin-int8-negative-scales`: the `has_g_idx` guard is gone. **Without this, every `INT8_ACT=int8` boot (batch
  mode's default, and single-user with `INT8_ACT`) died at load** with an AttributeError.
- `marlin-repack-staged-sm80`: the staged repack no longer passes `perm` (sm80, or `VLLM_MARLIN_REPACK_STAGED=1`).
- `marlin-tune-table`: the standalone tuned build keeps its 0.27.1 schema and gets `None, None` and
  `is_k_full=True` (off by default; not booted, it needs the standalone build).

Carried by hand into upstream's rewrite of `_largest_kernel_block_within` (#53007): `kvarn-v2-runner`'s divisor
rule for the drafter's padded sliding-window block. Without it, the drafter's group took block 432 against a 2176
primary, and `CTX=huge SPEC=dflash2 PREFIX_CACHE=1` (#179) was refused at boot ("prefix-cacheable KV cache group
block sizes must be divisible by prefix_match_unit", 128). With it the group is 128 and the pool is 268,169
tokens, as on 0.29.

Merged from main during the port: `auth-deny-default` (#169), cut against 0.29.0; it applies to 0.30.0 as cut
(`authenticate.py` did not change; the help-text hunk lands at an offset), and the routes 0.30 adds fall under its
deny-by-default rule. #187's retention-cost table sits in gotcha 60 ahead of the 0.30 note.

Carried from the 0.29 line into the topics the port re-cut: the knob sweep (readers through `vllm.envs`), the
`KVARN_*` registration, and #86's int64 block-id cast. All of them post-date the resolution the re-cut started
from.

## Acceptance (reference 3090 native headless, and the WSL2 4090)

0.29 arm: an image built from syv main 73fd65d, the commit this port was cut from, so the pair differs by the pin
and its port alone. Every mode at its shipped defaults, fresh volume per arm.

| mode | pool 0.29 / 0.30 | result |
|---|---|---|
| single default | 84,811 / 84,811 | pass both |
| dflash2 k7 / k15 | 68,605 / 68,605, 57,669 / 57,669 | pass both |
| CTX=long SPEC=mtp | 202,040 / 202,040 | pass both |
| CTX=huge | 336,283 / 336,283 | pass both |
| alternative.sh | 302,094 / 302,094 | pass both; groups [1696 x9] |
| batch fp8 (0.95), burst 128 at 64-way offered | 225,000 / 225,000 | 95.7 s / 95.8 s, 128/128 |
| batch kvarn (0.93) | 297,357 / 297,357 | 108.5 s / 109.0 s, 128/128 |
| batch int4pth (0.93) | 407,446 / 407,446 | 97.6 s / 97.6 s, 128/128 |
| batch fp8 with `INT8_ACT=` (W4A16, 0.95) | 231,958 / 231,958 | 140.9 s / 141.0 s, 128/128 |
| CTX=huge SPEC=dflash2 PREFIX_CACHE=1 (#179) | 268,169 / 268,169 | pass both; groups [2176 x8, 128]; before cut4, refused on 0.30 |

The batch burst is KV-limited on fp8 and kvarn (57 and 37 running at most, the same on both pins); int4pth reaches
64. Burst wall time is the comparable number: the stats line's generation throughput alternates between prefill and
decode windows. On both pins the int8 activation path buys 32% of burst wall time on this card (95.7 s against
140.9 s W4A16) for about 7K tokens of pool (its workspace). A W4A16 gap measured on the WSL2 4090 (0.30 about 15%
slower) does not reproduce here.

Speculative decoding at C1, shipped defaults, tokens per step (greedy / default temperature): every delta is inside
the ±11% band an 8-prompt cohort shows between runs.

| arm | 0.29 | 0.30 |
|---|---|---|
| mtp | 2.93 / 2.85 | 2.89 / 2.81 |
| dflash2 k7 | 3.41 / 3.05 | 3.43 / 3.28 |
| dflash2 k15 | 3.40 / 3.30 | 3.21 / 3.29 |

First-request JIT on a cold volume (`--jit-monitor-verbose`) depends on what the first traffic is. For greedy traffic,
or single requests with the model's default sampling (`top_k` set), it is 0 on both pins, default and `SPEC=mtp
CTX=long`. For a sampled batch (the bench's own warmup: 16 requests at concurrency 8, default sampling) it is still 0
on 0.29, with no cold-vs-warm cost, but on 0.30 the default profile makes five compiles of 0.30's three split top-p
kernels (`_topp_sb_stats`, `_topp_sb_step` x3, `_topp_sb_mask`, all at `S=4`), and the first request's TTFT reads
3,433 ms cold against 1,579 ms warm (n=1). The kernels are real but unwarmed on CUDA: the V2 runner's sampler never
registers them, and vllm #58465 limited their registration to ROCm because on CUDA it "adds ~2 min to every engine
start". The cost is once per cold Triton cache (the cache lives on the `qwen-cache` volume), and the #155 follow-up
carries vllm #58092's registration for CUDA, on a branch stacked on this one. The monitor also counts compiled kernels
loaded from the disk cache, so on a warm volume its count is not the cost; read the latency. On the WSL2 4090, the
int8 prefill profile logs the same four in-request compiles on both pins, of three kernels (`_k_quant`, `_k_stats`,
`_prefill_attn` twice), which is the positive control that the counter works.

A new 0.30 warning, "Speculative decoding (method=...) is enabled but no KV cache group could be identified as the
draft model's", appears on MTP and DFlash2 profiles. 0.29 computes the same condition without logging it. Prefix reuse
is unaffected: an identical ~14.7k-token resend hits 13,824 tokens on both pins (default) and 13,312 (long), exactly
`floor((N-1)/A)*A - A` for the boot's attention block A.

The retention knob reaches the engine on 0.30. On the #179 profile, `PREFIX_RETENTION=13057` is refused by the
retention validator ("must be non-negative and a multiple of scheduler_block_size (2176)"), and 13056 boots with a
resend of 17,706 tokens hitting 15,232 = floor((17706-1)/2176)*2176 - 2176. Before cut4 the same boot was refused
earlier, by prefix_match_unit, so the knob was unproven there.

Native install at this branch's head (`docs/install.md`'s own pip line and patch loop, a fresh venv, Python 3.14,
system `nvcc` 12.4 on the reference 3090): all patches applied, `verify.sh --no-server` 68 PASS, and default,
`INT8_ACT=int8` and dflash2 k7 boot and serve with the Docker pools (84,811, 80,956 and 68,605). The launcher points
`CUDA_HOME` at the pip CUDA 13 toolkit when the `nvcc` on PATH is older (logged), and
`VLLM_USE_FLASHINFER_SAMPLER=0` keeps the vocab-wide top-k on `torch.topk`, so nothing JIT-compiles with the old
`nvcc`. The patch loop in `docs/install.md` still spells `venv/lib/python3.12`, which breaks on any other Python;
that fix is its own change (the loop asks the venv's python for the path), not part of this port.

## Prefix-cache retention: 0.29's default is not 0.30's

0.29.0 resolved an unset `prefix_cache_retention_interval` to dense for hybrid models with an EAGLE-family draft
(vllm #55760). That change merged into `releases/v0.29.0` only; 0.30.0 does not have it, and its default is 0 (the
replay boundaries only). Main fixed the mechanism #55760 worked around instead: #53945 lands the replay boundary after
the EAGLE tail-block drop, and #54713 retains both the resend and the extension boundary. So 0.30's default still
hits, but nothing past the prompt is retained.

Measured on the reference 3090 with the interval unset, one conversation of ~20K tokens, four turns of 1,000-token
replies and ~1,100-token user turns, and A the boot's attention block (432 MTP, 448 DFlash2 k7, both pins). Every
cell is exact to the token against the formula beside it:

| | identical resend | turn k (k = 2..4) |
|---|---|---|
| 0.29 (dense) | `floor((N-1)/A)*A - A` | `((P + R)//A)*A - A`: into the previous reply |
| 0.30 (0) | the same | `(P//A)*A - A`: the previous prompt's boundary |

Here P is the previous turn's prompt and R its reply. Neither reuses much of the reply: 0.29's hit reaches a few
hundred tokens into it and both prefill the rest. What 0.30 at 0 prefills in addition is the span between the two
formulas, 864 tokens at A=432 (896 at A=448), and that span is mostly the previous prompt's last blocks: 571-700
prompt tokens and 164-293 reply tokens in reruns at ~8K, ~20K and ~40K (two replicates each, every turn on its
formula). It costs about 0.7-0.8 s at this card's ~1,150 tok/s prefill, and the hit rate drops from 91-94% to 87-90%.
Measured turn times differ by 0.7 s a turn on MTP and 0.7-1.8 s on DFlash2 k7; the excess over the re-prefill is
decode on replies that differ between the pins (greedy still diverges across versions). An identical resend cannot
tell the two apart (the EAGLE drop caps both at the same boundary); only an extension can. With
`--prefix-cache-retention-interval None` (dense), 0.30 prefills a fresh ~20-22K and 37-48K prompt within 0.5% of both
0.30 at 0 and 0.29 dense.

So both single-user launchers pass the interval on every draft profile: the measured one for `CTX=huge SPEC=dflash2`
(13056 at 7 drafts, 14592 at 15), else `None`, which is 0.29's behaviour. `PREFIX_RETENTION=0` asks for boundaries
only, and a flag in `EXTRA_ARGS` wins. Batch mode runs no draft, so #55760 never applied to it and nothing changes.
Verified on the reference 3090 with the launchers read out of the image against the commit: the default MTP profile
shows `'prefix_cache_retention_interval': None` in the engine's arguments and its turn 2 hits S1 (20,304, 92.0%);
alternative.sh shows None and keeps its 302,094 pool; `CTX=huge SPEC=dflash2` keeps 13056 (pool 268,169) and still
refuses 13057 at the retention validator; `PREFIX_RETENTION=0` on dflash2 k7 hits S0 (19,264, 87.3%). An explicit 0
never appears in the engine's non-default arguments on 0.30, because 0 is the parser default there, so the proof for
0 is the S0 hit, not the log line.

Retention is prefill-neutral on the reference 3090: the acceptance A shape (dflash2 k7, `CTX=fast`, prefix caching),
fresh ~20-22K and 37-48K prompts, reads 1221-1225 and 1140-1145 tok/s on 0.29 dense, 0.29 at 0, 0.30 at 0 and 0.30 at
None alike (per-rep TTFTs within ~0.1 s). `docs/vllm-0.29.md`'s "0 halves the 47k prefill" (1303 vs 2323 tok/s) was
measured on the WSL2 4090 and does not reproduce natively on either pin. The reason to pass None is the multi-turn
reuse above, not prefill.

## Porting the next pin (what this port added to the procedure)

`docs/vllm-0.29.md`'s three steps stand. The replay oracle ("the patch files reproduce the fork branch") was green
at every cut of this port, and still four defects shipped into a built image, because a patch that applies is not a
patch that still means what it did. The procedure now starts with four checks that need no card:

- `scripts/port-triage.sh`: cherry-picks each topic onto the new tag and reports clean, CONFLICT (with files) or
  EMPTY, plus a RETIRE? column from each row's upstream PR ancestry.
- `scripts/pin-bump.py`: moves every mechanical pin in one pass, checks every exact pin in
  `docker/requirements.txt` against the new vLLM's floors (the hub pin would have failed a 20-minute build), and
  lists every leftover mention of the old version, line by line in the files it edits and as a count in every other
  tracked doc and script. The install doc's `vllm==0.29.0` and the README badge survived this port's first pass; this
  is what now reports them.
- `scripts/port-removed-names.py`: every identifier the series' added lines use that upstream deleted between the
  pins. The #54809 names were invisible to the replay and to a clean apply, because they sit in our lines and never
  in hunk context. At cut2 it reports all three topics.
- `scripts/port-drift.sh --show`: each topic's added lines compared with the same topic on the old line, **printing
  the lost lines**. A count ("attention.py:18") was filed as a documented adaptation at cut2; the lines were the
  divisor rule behind #179.

Two lessons for the reading:

- A re-cut from a resolution made on another branch (here `qwen38/main-track`, cut 09-13) loses whatever the old line
  gained after that date. List it (`port-drift`), then carry it.
- Run the install doc's commands as extracted from the file, on a host whose Python is not the image's. That is how
  the stale pin and the Python 3.12 path were found.

## Decisions made in this port, one line each (reject any by name)

1. **One fork branch per pin, with an annotated tag at every export point** (`qwen38/0.30-cutN`), instead of suffixed
   branches.
2. **`spec-sampler-prewarm` is kept** over #56323, which warms V1 kernels and not the V2 runner's.
3. **The #54809 adaptations** pass the values activation ordering's removal implies (`None`, `None`,
   `is_k_full=True`) to the standalone tuned build instead of dropping `marlin-tune-table`.
4. **`kvarn-v2-runner`'s divisor rule applies only to a padded sliding-window page** (`divisor_of`), so #53007's
   unpadded path is unchanged.
5. **`huggingface_hub==1.32.0`** in the image (0.30.0's floor is 1.31.0; 1.32.0 is the only version tested). The
   native install leaves it unpinned and resolves it from vLLM's own requirement (1.32.0 on the reference 3090).
6. **`offload-dflash-eagle-groups`' #33 hunk is not carried**: 0.30 fails toward non-draft on its own. The residual
   (the drafter's trailing chunk is stored under DFlash2 + CPU tier) is stated, not fixed. The port-faithful fix would
   be a narrow carry of the old connector hunk. Annotating the drafter at upstream's annotation site would also narrow
   the GPU tier's EAGLE drop, which 0.29 never did.
7. **Draft profiles pass `--prefix-cache-retention-interval None` unless measured or set**, keeping 0.29's multi-turn
   reuse. Boundaries-only (0.30's default) is one `PREFIX_RETENTION=0` away, and would be the better choice for many
   alternating long conversations on a small pool (gotcha 60); this port does not change that trade.
