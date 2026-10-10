# The vLLM 0.31.0 pin

What the move from 0.30.0 to 0.31.0 changed in this repo, what was re-measured, and what it costs.

[← back to the main README](../README.md)

## Dependencies

`vllm==0.31.0` (0.31.0rc5 is the release; it is six commits past rc3, HiSparse fixes and a transformers cap). FlashInfer
`0.7.0.post1` (from 0.6.18.post1; `docs/install.md`'s `flashinfer-cubin` pin moved with it). vLLM 0.31.0 caps
transformers below 5.18 and pins `xgrammar==0.2.7`; `docker/requirements.txt`'s transformers 5.15.0 and huggingface_hub
1.32.0 already sit inside its ranges. torch 2.13.0 is unchanged. `verify.sh` checks the 0.31.0 pin and the
`kvarn-*-0.31.0` patch names, and `scripts/pin-bump.py` moved every mechanical pin with 0 problems.

## Patch series

All 52 files (48 in the series, 4 KVarN) are re-exported from a vLLM branch that carries the series as one commit per
topic on v0.31.0. Each keeps its preamble and table headers unchanged except `Cut-against`. They apply to v0.31.0 at
`--fuzz 0`, 10 hunks in 8 patches at an offset, and the series plus the KVarN files and patches reproduce that branch's
`vllm/` tree with 0 differing files. Nothing retires: `sampler-warmup-cuda`'s #58092 is in 0.31.0 but still gated to
ROCm by #58465, which the patch removes.

Re-implemented, because upstream rewrote the code under them:

- `dflash2-lookup-drafting`: upstream moved DFlash2's candidate selection into a shared `CandidateSampler` (#57934,
  also used by LiLiCorr). The topic is ported onto it rather than kept on the old structure: `CandidateSampler` gains
  `cache_steps` and optional per-request top-p/top-k, both defaulting to upstream's behaviour.
- `dflash2-ngram-chains`: follows `CandidateSampler`'s members, keeps #261's context-parallel arguments with the CP size
  now read from `dcp_size` (#56723), and calls `prepare_context_anchor` after the hidden-states copy, which #57934 needs.
- `offload-wsl2-devptr`: upstream #51081 registers host memory in 64 GiB chunks. The chunking is kept; the device-pointer
  translation now queries every chunk and refuses to start if they map at different offsets, which can only happen
  under WSL2 above 64 GiB.

Re-cut where upstream lines moved beside ours: `spec-sampler-prewarm` (#57428's renamed GEMM warmup),
`speed-knobs-envs`, `engine-completion-log`, `auth-deny-default` (a docstring), `pinned-kv-empty-cache` (#58411's
`randomize_inputs`; 0.31's pinned path still has no cache release), `kvarn-fp16-dequant-0.31.0` (`envs.py`), and
`kvarn-v2-runner-0.31.0`, whose `attention.py` keeps upstream's new `kv_cache_spec` beside our `divisor_of`. The
runner topic keeps its `has_prefill` decode-graph guard beside #58400 (FULL decode graphs for one-token prompt tails); on
the drafter profiles the guard is never reached, and dropping it is a separate change.

`kvarn-fp16-dequant-0.31.0`'s preamble and one comment in its hunk still name the `-0.30.0` KVarN files. Both come
from its fork commit, so they change at the next cut rather than by hand here.

Two checks that need no card ran on the result: `scripts/port-removed-names.py` (0 removed names in our added lines; 3
docstring words match removed identifiers and are passed with `--accept`) and `scripts/port-changed-signatures.py` (0
calls that no longer bind, over the 188 it can resolve).

## Acceptance (native 3090, headless)

0.30 arm: vLLM 0.30.0 with the series as of main 10bb488. 0.31 arm: vLLM 0.31.0 installed from `docs/install.md` at this
branch's first commit; the later commits add `api-root-health` (four keyless routes) and `qwen3_5-embed-uva` (opt-in),
change no other patch, and are covered by the final-head checks below. One boot per cell unless stated, so no same-pin
spread was measured and small gaps are not claims.

- **Install from `docs/install.md`** (Python 3.14.4, system nvcc 12.4, no curand headers; the file's commands run as
  extracted): pip resolves in 141 s, `patches/apply.sh` and `kvarn/install.sh` exit 0, `verify.sh --no-server` reports
  0 failures, and a `SPEC=dflash2` boot serves greedy and sampled requests.
- **Every profile boots its shipped draft count**, and 0.31 boots 40-60 s faster cold (183-252 s against 229-313 s).
- **Acceptance, tokens per step, greedy / default temperature, 0.30 then 0.31:** MTP fast k4 3.666 / 3.662 and 3.695 /
  3.645; MTP `CTX=long` k3 3.201 / 3.074 and 3.259 / 3.116; DFlash2 k7 4.600 / 4.889 and 4.814 / 4.939; DFlash2 k15
  4.642 / 5.155 and 4.858 / 4.962; `alternative.sh` k7 4.826 / 4.560 and 4.777 / 4.884. No row moves the same way at
  both temperatures except DFlash2 k7; copy-heavy output reads the same tokens per step on both pins.
- **Prefix-cache correctness** (`bench/residue_sweep.py`, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1`): 0 broken of 128 and
  0 broken cold on both pins; all 256 answers have one hash.
- **First-request JIT** (default profile and `SPEC=mtp CTX=long`): 0 compiles during inference on both pins. The fp8-KV
  profiles (batch, `CTX=long`) build FlashInfer's fp8-KV `batch_prefill` once per cold cache, ~31-36 s inside the first
  CUDA-graph capture, on both pins (`docs/install.md`).
- **Two long conversations' reuse:** `alternative.sh` 93.5% for both on both pins, as on 0.30 before; `CTX=fast` reuses
  97.4-98.8% for one conversation and 0 for two, which is its dense retention by design; `CTX=huge` (KVarN pool 268,169
  tokens on both pins) reuses 94.5% for one ~60K conversation on both pins, and two in flight at once reuse nothing on
  either pin (#299 corrects gotcha 60 to say so).
- **Env knobs:** all 12 the launchers export reach the engine on 0.31. An invalid value is not always refused, on 0.30
  as well: the boolean knobs read anything but `1` as off, so `VLLM_DFLASH2_LOOKUP=true` runs without lookup (copy 3.992
  tokens per step against 7.701). That is a separate fix.

## Batch: `GPU_UTIL` 0.94, not 0.95

vLLM's profiling run skips the GDN layers, so batch's KV sizing never budgets the worst prefill step: 64 new short
prompts landing in one step. That step needs ~624 MiB over rest on 0.30.0 and ~650 on 0.31.0, and 0.31.0 also rests ~40
MiB higher, which comes with FlashInfer 0.7. At 0.95 on a native 3090, 0.30.0 survived that step with ~12-17 MiB to
spare, and 0.31.0 ran out of memory: its 64-way burst of 128 failed on all 5 boots run without a memory hook. Forcing
the step (one `/v1/completions` request carrying 64 chat-formatted prompts), two boots each:

| `GPU_UTIL` | 0.31.0 | margin | KV pool |
|---|---|---|---|
| 0.95 | out of memory, 2/2 | | 225,000 tokens |
| 0.945 | survives, 2/2 | ~54 MiB | 221,134 tokens |
| 0.94 | survives, 2/2 | ~174 MiB | 217,268 tokens |

At 0.94 the burst serves 128/128 at 1,208 tok/s and 0.30.0 clears the same step by ~232 MiB. The launcher's native
default is 0.94; the pool costs 3.4%. Margins are nvidia-smi peaks sampled at 50 ms, so upper bounds. The WSL2 defaults
(0.91, 0.88 for kvarn) are lower and unchanged. Budgeting the GDN prefill in vLLM's profiling run would be the real fix.

## Tool calls: `--tool-strict-level parameter`

vLLM 0.31 adds `--tool-strict-level` (#56268). Its default, `auto`, enforces a tool's argument schema only when the
tool sends `"strict": true`, which is OpenAI's rule. With `tool_choice=required` on a tool with no parameters, 0.31.0
invented arguments in 20 of 20 seeds, all different (`{"year": "2026", ...}`, `{"hour": "14", ...}`), where 0.30.0
sent `{}` in 20 of 20. The schema builder is identical on both pins; the change is the new flag's default. With
`parameter`, 0.31.0 sends `{}` in 20 of 20, auto tool calls work with thinking on and off, and the `alternative.sh`
checks above are unchanged. The three launchers pass `parameter`; `TOOL_STRICT=auto` restores upstream's default, and a
`--tool-strict-level` in `EXTRA_ARGS` still wins.

## WSL2 4090 (the image)

The image this branch builds (at 175b5f2; the commits after it change docs and comments only), on an RTX 4090 under
WSL2 with Docker Desktop, each launcher at its WSL2 defaults, one boot per row. Spill to system RAM is read from the
card's Shared Usage counter, sampled about every 3 s, as in `docs/wsl2-4090.md`:

| profile | up in | KV pool | checked |
|---|---|---|---|
| batch, `GPU_UTIL` 0.91 | ~285 s | 207,216 tokens (0.30.0: 208,762) | the forced 64-prompt step serves; the 64-way burst serves 128/128 at 1,614 and 1,650 tok/s; Shared Usage ends at 382 MB, the level 0.30.0 read before its spill stepped it to 632 |
| single-user default | ~225 s | 88,666 tokens | no spill |
| `alternative.sh`, `SPEC=dflash2 PREFIX_CACHE=1 MAX_SEQS=4 MAX_LEN=160000` | ~240 s | 226,184 tokens | two ~60K conversations at once reuse 55,968 tokens each on turn 2 (93.5%, as on 0.30.0); Shared Usage reaches 1.3 GB during model load, and is back to ~100 MB before the first request and through the traffic |
| `alternative.sh`, an 8 GB CPU tier | ~270 s | 226,184 tokens | boots and serves; the tier stores but never serves (below) |

Tool calls on every row: `tool_choice=auto` with thinking on and off, and `required` on a tool with no parameters
returns `{}` (one call per row). 0 tracebacks and 0 out-of-memory lines in any log.

## A regression: the CPU KV tier stores and never reads back

`alternative.sh` with `EXTRA_ARGS="--kv-offloading-size 8"` (dense retention, which the launcher picks beside a tier),
two ~59.9K conversations at once, the WSL2 4090, one boot per row:

| | turn 2 cached | GPU to CPU | CPU to GPU |
|---|---|---|---|
| 0.30.0 (the series without #260 and #285) | 57,664 and 57,664 (96.3%), 2.7-3.0 s | 8.62 GB | 1.08 GB |
| 0.31.0 | 0 and 0, 45-62 s | 9.60 GB | 0 |
| 0.31.0, repeated | 0 and 0 | 9.48 GB | 0 |

0.31.0 stores, but the tier's usage gauge (`vllm:kv_offload_cpu_cache_usage_perc`) stays at 0 through the whole run and
the lookups find nothing, where 0.30.0's rises and its lookup loads. No transfer error is logged. A native 3090
reproduces it: 0.31.0 reuses 0 and 0, reads nothing back and stores the same 9.60 GB, where 0.30.0 on the same card
reuses 96.3% for both and reads 2.16 GB back. So it is not WSL2's. Reversing `offload-dflash-eagle-groups`, the one
series patch inside the offload scheduler, changes nothing (0 and 0 again), and stock 0.31.0 cannot load this
checkpoint's quantized embedding, so whether the cause is upstream or elsewhere in the series is not settled. No other
profile was run with a tier. Until this is fixed, a tier on `alternative.sh` does worse than none on 0.31.0: beside a
tier the launcher picks dense retention for the tier to serve from, so the two conversations get 0 and 0, where without
one its default (retention 0) keeps 93.5% for both (above).

## What is still unproven

- One boot per cell on both cards, so small gaps are not claims and no same-pin spread was measured.
- Batch margins are nvidia-smi peaks sampled at 50 ms, so upper bounds. Budgeting the GDN prefill in vLLM's profiling
  run is the real fix, and it is not in this port.
- A forced tool call sometimes keeps decoding after a correct `{}` until `max_tokens`: 2 of 20 seeds on 0.31.0 batch
  (with one tool offered or two), 1 of 20 on the single-user default, and 5 of 20 on 0.30.0 batch (`max_tokens` 2048,
  one boot each), so it predates this port. The extra tokens are whitespace inside the `qwen3_coder` tool-call XML,
  after `<function=get_time>`, which xgrammar's `disable_any_whitespace` (JSON only) does not reach: 2 of 20 with it
  and without. Not fixed here.
- The `has_prefill` guard in `kvarn-v2-runner` is kept; on the drafter profiles it is never reached, and dropping it is
  a separate change.
- Under WSL2, `alternative.sh`'s Shared Usage rises to 1.3 GB while the model loads and is back to ~100 MB before the
  first request. Whether 0.30.0 does the same was not measured.
- Pre-existing on 0.30.0 as well, found during these runs and not fixed here: the boolean env knobs read anything but
  `1` as off (`VLLM_DFLASH2_LOOKUP=true` runs without lookup), and `verify.sh`'s fix-it hint suggests GNU `patch`'s
  default fuzz rather than `apply.sh`'s `--fuzz 0`.

## Decisions made in this port, one line each (reject any by name)

- Batch's native `GPU_UTIL` default 0.95 -> 0.94 (above).
- `--tool-strict-level parameter` in all three launchers, `TOOL_STRICT=auto` to opt out (above).
- `dflash2-lookup-drafting` ported onto `CandidateSampler` rather than restoring the old selector structure.
- `kvarn-v2-runner`'s `has_prefill` guard kept for this port; dropping it is its own change.
- `offload-wsl2-devptr` refuses to start when 64 GiB host chunks map at different device offsets, rather than
  translating per chunk.
