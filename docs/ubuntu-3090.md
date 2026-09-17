# RTX 3090 on native Ubuntu: campaign notes

The native half of the two-box campaign behind PR #46. Companion to
[wsl2-4090.md](wsl2-4090.md) (the WSL2/4090 half) and to
[reproductions/native-3090.md](reproductions/native-3090.md) (the earlier
full reproduction on this box: hardware table, README-parity benches, needle
recall, the prompt-shape finding). Numbers below are the 3090 seat's five
run-books (2026-08-28), relayed cross-box with artifact hashes; the seat
reviewed every row as it entered the MR draft.

Stack: Ubuntu 26.04, system Python 3.14.4, native venv (no container), RTX 3090
@ 250 W, vLLM 0.27.1 + the full patch stack. See the reproduction doc for the
detailed table.

## Offload fix validation (the native control arm)

The [OffloadingConnector WSL2 fix](wsl2-4090.md#cpu-offload-tier-under-wsl2-the-offloadingconnector-fix-2026-08-28)
must be a no-op on native Linux, and "no-op" was validated as three arms, not
assumed:

| arm | decode | token sha | engine |
|---|---:|---|---|
| baseline (no fix) | 52.4 tok/s | `6ffe0154…` | alive |
| fix applied | 52.4 tok/s | `6ffe0154…` byte-identical | alive |
| fix removed | 52.4 tok/s | `6ffe0154…` byte-identical | alive |

Execution was confirmed, not inferred: the loader resolves at tier 0
(`CDLL(None)`), the live `cudaHostGetDevicePointer` call returns rc=0 delta=0,
and the CPU tier moved **805 MB** during the arm (`kv_offload_store_bytes`), so
the patched function ran live with translated pointers and changed nothing.
The v1 lesson from this box: v1's arms had "passed" **inert**; a bare
`CDLL("libcudart.so")` throws on standard pip installs, so its happy path never
executed; v3 raises on every failure branch, making a live engine itself the
proof the query ran.

Also found in the same logs: **#33's drafter handling is active here**: nine
EAGLE/MTP draft attention groups detected at runtime (an earlier
"deliberately skipped" read came from patch-application state, not runtime;
`verify.sh` checks the former).

## Retention: the trio, the bisection, and the model kill

Same pool, same contexts, opposite answers by design (int8/long, pool 136,429,
tier off):

```
ROUND-ROBIN  (recheck A,B,C in order)   0/3 — every recheck ~1.0x cold
CAPACITY     (recheck newest only)      RETAINED — 2.24s vs 50.35s cold (22x)
```

The round-robin sweep evicts each context via the next recheck's own prefill;
0/3 was never a capacity measurement. The L-bisection (per-length salts,
measured `prompt_tokens` per rung, boundary rungs re-run **alone in fresh
boots**; both reproduce to the decimal):

| L (tok) | K=2 result | isolated re-run |
|---:|---|---|
| 34,124 | 0/2 | — |
| 29,340 | 0/2 | **29,344 → 0/2**, 35.7s / 37.1s |
| **24,865** | **2/2** (2.5s / 2.2s vs 44s cold) | **24,869 → 2/2**, 2.5s / 2.2s |

Both boundary rungs were re-run **alone in fresh boots** and reproduce to the
decimal. That matters because the emptiness argument below leans on *both*
sides: pool residue from the rung above could in principle manufacture the
eviction at 29,340, and if the true 0/2 boundary were higher, `F` would be
smaller and additive would survive.

Per-context retention cost ≈ **2.3-2.7× its token count** at ~25-29k. The
pre-registered K=3 discriminator (12,142 tok, fresh solo boot) returned
**3/3**, which kills the additive cost model on this geometry by an emptiness
argument (K=2 demands F ∈ (38,874, 43,350]; K=3 demands F ≤ 33,334; no such F).
The 4090 ran the mirror rung and got the mirror result (0/3, killing the
multiplicative model there): **no one-parameter cost model survives both
boxes**; see the MR's §5 for the affine candidate and why it is a fit, not a
finding.

## Designed-config ladder: MAX_SEQS is geometry, not a knob

CTX=long, identical instrument, only MAX_SEQS differs:

| N | MS=8 agg | MS=4 agg | preempts |
|---:|---:|---:|---:|
| 1 | 128.0 | 127.8 | 0 |
| 2 | 236.0 | 241.6 | 0 |
| 4 | 316.0 | **335.4 (+6.1%)** | 0 |

Not over-provisioning slots is worth +6.1% at the design point, and the
designed configs take zero preemptions where the uniform sweep took two.

**Scope, because this ladder cannot see the constraint that binds mixed load.**
Every stream here carries a uniform ~4k prompt, so every prefill is small and
comparable: `conc_ladder.py` measures *decode* concurrency and is structurally
blind to **prefill head-of-line blocking**. The 4090 measured the case it
misses: one 72.6k whale against 4k minnows, where the deep prefill monopolises
the engine for its full ~65 s and admitted minnows ration to ~1 tok/s. So
everything in this section is scoped to **uniform short-prompt load**; under
mixed depth the binding constraint is prefill admission and none of this advice
transfers.
CTX=huge at its designed MS=2 shows 44.7% KV at N=2, but 2 × 245,760 >
the 268,169 pool, so MAX_SEQS there is a **worst-case admission bound**
deliberately overcommitted for realistic prompts, not a throughput setting.
The kv% gauge understates that risk while the concurrency banner overstates
multi-context capacity; neither is safe to tune against.

## Batch wing at GPU_UTIL=0.972: the arm only a headless box can run

CTX=fast, MAX_SEQS=64, SPEC=off, pool 86,353 (0.972 boots and serves here;
WSL2 refuses it):

| N | per-stream | agg | ms/pass | preempts | kv% | TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 130.7 | 131.7 | 23.3 | 0 | 14.3 | 3,172 |
| 2 | 97.8 | 238.8 | 26.0 | 0 | 26.5 | 5,722 |
| 4 | 53.8 | 332.4 | 37.9 | 0 | 56.1 | 9,794 |
| 8 | 34.7 | 187.1* | 25.7 | 0 | 98.7 | 16,899 |
| 16 | 23.6 | 282.3* | 28.2 | 0 | 96.9 | 31,981 |

(*tail measurement.) ms/pass spreads 21% over a 16× batch, with **zero
preemptions at every level**, where the spec profile took two at N=8. The
wings differ in what they are willing to have go wrong, not only in speed.
(Not comparable to the 4090's fp8-geometry batch rows: different profile;
separate rows, never a ratio.)

## Needle at max length: the banner is honest about one

CTX=huge, single request, climbing: 159,262 / 194,121 / 217,221 / **234,158**
tokens; every level completes with exact needle recall, the last at 95.3% of
max_model_len in 355.2s. The final 4.7% is untested because the corpus ran
out, not the engine. So the "Maximum concurrency 1.09x" banner is **true for
one live request**, but misleading only for concurrent or cached contexts,
where the retention tax above applies and nothing on the line says so.

## Operator notes from this box

- The 8 GB `/dev/shm/vllm_offload_*.mmap` **survived a teardown kill** and
  held that RAM until removed by hand: clean stale regions at shutdown as
  well as startup.
- `verify.sh` verifies patch application, not runtime behavior; the nine
  draft groups above were the proof.
- **`single-user/alternative.sh` silently ignores `SPEC`.** Line 71 hardcodes
  `--speculative-config` with no guard, so `SPEC=off` still serves n7. Two arms
  labelled spec-on and spec-off came back *bit-identical*, and the tell was the
  **emitted/step receipt: 2.29 on an arm that must be 1.00 if speculation were
  off.** Anyone A/B-ing speculation on the int4 profile gets two spec-on arms and
  concludes speculation does nothing: one inference from a conclusion this PR
  had already retracted once by a different road. An unrecognised `SPEC` should
  refuse, not proceed.
- Instrument ledger entries paid for here: basic `sed` has no alternation
  without `-E` (a display filter printed nothing while the run was fine, so
  instruments persist `--json`); a hardcoded K=2 label printed "3/2" with the
  count right and the label wrong; a descending-L ladder with fixed salts is
  contaminated at exactly the boundary rung (smaller contexts are prefixes of
  larger ones: salt per rung); estimates place rungs, only measured
  `prompt_tokens` place conclusions.
