# Long context: 262k with KVarN, and the built-in KV modes

How to get past the ~200k the fp8 KV cache allows, and what vLLM's own per-token-head quantization modes are worth on this card.

[← back to the main README](../README.md)

## 262k context with KVarN

With fp8 KV the pool holds ~200k tokens, so 150k is the shipped max and ~195k
the ceiling; the model's full 262,144 is out of reach because 16 attention
layers × 4 KV heads × 256 dims × 2 bytes (K+V, fp8) is 2 KB per token. The
way past that is a smaller cache, not a different engine, and
[KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL) has the best one we
know of: Hadamard rotation + iterative variance normalization + 4-bit keys /
2-bit values per 128-token tile, at ~840 B/token/layer here. It ships as a
fork of vLLM 0.23; [kvarn/](../kvarn/) is our port of its dense backend onto
the 0.32.0 this repo runs (`bash kvarn/install.sh`, then `KV=kvarn` in batch mode
or `CTX=huge` in single-user mode). The rows below were measured on 0.28.0; the
0.29.0 port computes the same cache geometry and matched quality on a WSL2 4090
(see [vllm-0.29.md](vllm-0.29.md)), and the 0.30.0 port the same pools as 0.29 on the reference 3090
(see [vllm-0.30.md](vllm-0.30.md)), and the 0.31.0 port the same `CTX=huge` pool as 0.30
(see [vllm-0.31.md](vllm-0.31.md)), and the 0.32.0 port the same `CTX=huge` pool as 0.31
(see [vllm-0.32.md](vllm-0.32.md)).

Measured on the 3090 (`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`,
fp16 recurrent state, batch defaults otherwise). **Every row in this table is the
batch config, which runs no speculative decoding** — `ms/token` is `vllm bench
serve`'s mean TPOT, so one model step per output token. Single-user mode
speculates, and its numbers are in the next table; do not compare across the two.

| batch mode (no speculation) | fp8 KV (default) | KVarN k4v2 |
|---|---|---|
| KV pool, batch mode | ~205-225k tokens (150k max, ~195k ceiling) | 302-344k tokens with 64 slots, **420k with 4 slots — 262k fits with room for 1.6 such requests** |
| KV pool, single-user mode (MTP-3, `CTX=long`/`huge`) | 150k max | 200k max |
| needle-in-a-haystack, greedy | — | correct at 4k / 16k / 30k / 100k / 240k, both depths |
| perplexity (en/da/code, 33k tokens) | 8.223 | 8.236 (+0.16%) |
| prefill, 1k / 16k / 100k inputs | 1,812 / 1,595 / 997 tok/s | 1,741 / 1,569 / 1,050 tok/s (same within ±5%) |
| single stream at 100k context | TTFT 99 s, 27 ms/token | TTFT 94 s, 33 ms/token (1.22×) |
| 4 × 60k-token requests, 1,024 out | only 3 fit → 256 s total, ITL 33 ms | all 4 resident → 242 s total, ITL 49 ms |
| 64 concurrent short requests (128/512) | 876 tok/s | 692 tok/s (38 resident: 2048-token blocks cost as much per short request as fp8's 800-token block) |

### What it costs in single-user mode, which is where it hurts

The 1.22× above is batch mode at 100k. Single-user mode speculates, and the tax is
much larger there — reported in [#11](https://github.com/syv-ai/HyperQwen/issues/11)
and reproduced here. MTP-3, one request, 112,648-token prompts, `PREFIX_CACHE=1`,
two general tasks (summarize, answer-a-question), streamed so prefill is excluded
(`bench/labd_bench.py <tag> --ctx 100000 --corpus ~/bench/labd_corpus_long.txt
--tasks qa,summary`) — the only variable is `--kv-cache-dtype`:

| single-user, MTP-3, 112k context | fp8 (`CTX=long`) | KVarN (`CTX=huge`) |
|---|---|---|
| KV pool | 174,489 tokens | 292,035 |
| decode, summarize | 71.4 tok/s | 33.1 |
| decode, answer a question | 64.9 | 30.8 |
| **decode, both** | **68.1 tok/s — 14.7 ms/token** | **32.0 tok/s — 31.3 ms/token (2.13×)** |
| accepted tokens per step | 2.56 | 2.38 |
| TTFT, cold | 152.2 s | 146.7 s |

Two things worth separating out of that 2.13×. About 1.98× is raw step time. The rest
is acceptance: KVarN costs ~7% of it (2.38 against 2.56 tokens per step), because the
quantized cache shifts the target's logits enough that the draft head agrees less often.
Speculation stays exact — the sampled distribution is unchanged, and perplexity moves
+0.16% — but acceptance feeds straight back into throughput, so "quality-neutral" does
not imply "speed-neutral" for a speculating server.

For a short-prompt chat load the gap nearly closes: MTP-3 on real prompts reads
84 / 89 tok/s at fp8 against 79 / 88 with KVarN (base variant, earlier draft vocab),
about 1.06×. The tax is a function of context length, not a constant.

So: same VRAM, 1.6-2× the tokens, full 262k context, output quality intact, prefill
unchanged — and a decode tax that runs from ~6% on short single-user prompts through
1.22× in batch mode at 100k to **2.13× in single-user mode at 112k**. Past ~100k of
context you are buying 1.7× the context for less than half the decode rate, which is
worth it when the alternative is not fitting the request at all and a bad trade
otherwise. Which is why it's a mode and not the default — nothing changes unless you
set `KV=kvarn` (batch) or `CTX=huge` (single-user); the KV-cache format is an
engine-level choice in vLLM, so it can't be switched per request. Port notes and what
to watch when bumping vLLM are in [kvarn/README.md](../kvarn/README.md).

(vLLM 0.28.0 also has TurboQuant built in — `--kv-cache-dtype turboquant_4bit_nc`
gives a similar 413k-token pool here and about 15% slower decode, but its
chunked-prefill path allocates O(context) scratch outside the memory profile
and OOMs at 32k+ prompts on this card at 0.972 utilization, and at 128k even
at 0.90. KVarN's prefill path is bounded and did 240k.)

### The built-in per-token-head modes

vLLM 0.28.0 also ships `int8_per_token_head`, `fp8_per_token_head` and
`int4_per_token_head` (dynamic per-token, per-head scales; the int4 one with a
rotation and asymmetric zero-points), all only in the Triton attention
backend. Measured on the 3090 in the batch config at 0.93 utilization, same
script for every column (`fp8_per_token_head` does not start on sm86: Triton's
fp8 KV needs SM89+). **Batch config again, so no speculation** — see the table
above for what these caches cost a speculating single-user server:

| batch mode (no speculation) | fp8 (FlashInfer) | int8_per_token_head (Triton) | int4_per_token_head (Triton) | KVarN k4v2 |
|---|---|---|---|---|
| KV pool at 0.93 util | 164k tokens | 178k | **355k — 262k fits (1.35×)** | 302-420k |
| perplexity (same battery) | 8.235 | 8.231 | 8.257 (+0.3%) | +0.16% |
| needle, greedy | 100k ok | 100k ok | 100k ok, **240k ok** | 4k…240k ok |
| prefill 1k / 16k | 1,773 / 1,601 tok/s | 1,739 / 1,187 | 1,710 / 1,194 | 1,741 / 1,569 |
| 100k context, single stream | TTFT 100 s, 26.8 ms/token | 231 s, 40.8 ms | 220 s, 41.4 ms | 94 s, 33 ms |
| 64 concurrent short (128/512) | 839 tok/s | 850 | 835 | 692 |

Reading: `int8_per_token_head` buys nothing over fp8 here (same byte per
element, quality already neutral) and costs the Triton backend's long-context
speed. `int4_per_token_head` is a genuine zero-install alternative to KVarN for
the 262k use case — it fits, passes the 240k needle, and keeps short-request
throughput that KVarN's 2048-token blocks lose — at 2.3× the prefill time and
1.5× the decode time at 100k, because vLLM's Triton attention is that much
slower than FlashInfer/FlashAttention on this card at long context (the same
backend tax the single-user mode avoids by staying on FlashAttention). If the
Triton backend catches up, it becomes the simpler choice; today KVarN is
faster at long context and `int4_per_token_head` is faster on many short
requests. To try it: `--kv-cache-dtype int4_per_token_head --attention-backend
TRITON_ATTN --max-model-len 262144` (batch/start_qwen.sh: `KV=int4pth`).

## DFlash2 past 64k (`SPEC=dflash2 CTX=long`)

The target context plus DFlash's fixed bonus-and-draft lookahead must fit the
configured drafter context capacity. Startup refuses an oversized combination
before GPU inference; the error names the supported target limit. Reduce
`MAX_LEN` / `DFLASH_MAX_LEN`, or use a drafter explicitly configured for the
larger context. Fetching a drafter does not extend its positional configuration.
The shipped profiles fit the 262,144-token drafter with room for lookahead.


The block drafter was pinned to `CTX=fast` — bf16 KV on FlashAttention, 64k at
`DFLASH_TOKENS=7` and 56k at 15 — because bf16 KV is 64 KB per token and the pinned
5.2 GiB pool is exactly that much. `CTX=long` moves it to an `int8_per_token_head` cache
on the Triton backend (`FA2_FP8KV=0`; on an 8.6 GPU the default is now the same cache read by the pth-flash-attn kernel, `PTH_FA=1`, or else fp8 KV on FlashAttention-2, both below) and
roughly doubles the context:

| | bf16 (`CTX=fast`) | int8 (`CTX=long`) |
|---|---|---|
| context, `DFLASH_TOKENS=7` | 69,758 tokens | **138,696** (136,429 with prefix caching) |
| context, `DFLASH_TOKENS=15` | 57,669 | **114,224** |

Two patches make that work, and neither changes anything at bf16:

- **[hybrid-sw-block-promote.patch](../patches/hybrid-sw-block-promote.patch)** — without it,
  int8 costs *more* memory than bf16, not less: 6.82 GiB to serve 32,768 tokens. vLLM equalizes
  KV page sizes by scaling a layer's block size up by an integer ratio, and the drafter's five
  sliding-window layers are born at the smallest kernel block, 16. That ratio is an integer at
  bf16 only by coincidence — the target's 4 KV heads × 256 and the drafter's 8 × 128 both come
  to 4096 B per token per layer — and a per-token-head cache breaks it by adding one fp32 scale
  per head. The drafter's layers then keep a 16-token block while their page is padded to the
  full 1.71 MiB primary page: 385 blocks at 1.88% utilisation, a constant 5.2 GiB. The patch
  rounds those layers' block up (16 → 864) so their page covers the maximum instead.
- **[spec-decode-int8-kv.patch](../patches/spec-decode-int8-kv.patch)** — the split-KV verify
  kernel reads the quantized cache and is wired into the Triton backend, which otherwise cannot
  split KV for a multi-query verify at all (`use_3d` is off whenever `max_seqlen_q > 1`, and
  every DFlash2 step is a verify). Per attention layer at 128k, 8 query tokens: 1.3 ms for this
  kernel against 7.4 ms for vLLM's unified attention and 10.1 ms for FA2.
- **[triton-spec-attn-fp8-kv.patch](../patches/triton-spec-attn-fp8-kv.patch)** (sm89 and up) — the
  same split-KV verify kernel reads vLLM's per-tensor fp8 cache (`--kv-cache-dtype fp8`), so the
  fp8 pool can run with `TRITON_ATTN` on both sides and keep FULL CUDA graphs: on a 4090 at 120k,
  DFlash2 fp8 decodes 108.7 / 88.6 / 81.0 tok/s at 24k / 49k / 88k tokens against 71.4 / 50.9 /
  35.4 with vLLM's unified attention on the same route, and 84.4 / 81.7 / 83.5 for FlashInfer with
  PIECEWISE (issue #87). The route is `SPEC=dflash2 CTX=fast` with
  the exact launch line, verbatim (the `"attention_backend":"TRITON_ATTN"` inside the speculative
  config is the part that is easy to drop, and dropping it silently costs the FULL graphs):

  ```
  SPEC=dflash2 CTX=fast EXTRA_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype fp8 --speculative-config '{\"method\":\"dflash\",\"model\":\"/app/models/Qwen3.8-27B-DFlash2-W4A16\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\",\"attention_backend\":\"TRITON_ATTN\"}'" bash single-user/start_qwen.sh
  ```

  Measured on both a WSL2 4090 (Docker Desktop) and a native one (Ubuntu 24.04, driver 580.159, this
  image): on native, prose prompts of ~24k / ~50k / ~89k tokens decode 100.6 / 96.0 / 87.4 tok/s with
  this patch against 68.3 / 49.8 / 34.6 with vLLM's unified attention (median of 3, greedy, 256 out,
  fresh prefill, `MAX_LEN=120000 PREFIX_CACHE=1 MAX_SEQS=4`), prefill unchanged by the patch on either
  box (2.4k / 2.0k / 1.6k tok/s native). Not for the 3090: Triton has no fp8 conversion on sm86.
  `INT8_ACT=int8` on the fp8 route is slow with or without this patch (a Marlin variant choice,
  tracked separately): do not stack the two until the memory-pressure question behind it is
  understood.

### fp8 KV on FlashAttention-2 (`FA2_FP8KV`, the 8.6 default when `PTH_FA` does not apply)

On a compute capability 8.6 GPU with the pth-flash-attn package installed and the measured setup, `PTH_FA=1` is this arm's default instead
([below](#int8-kv-on-the-pth-flash-attn-kernel-pth_fa1-specmtp-or-specdflash2-with-ctxlong): int8 KV read by that kernel, not fp8 on FA2);
this section describes the arm that runs without it (`PTH_FA=0`, a missing install, or a setup the kernel was not measured with).

`SPEC=dflash2 CTX=long` runs on fp8 KV on `FLASH_ATTN` by default, with the fp8 kernels from the
`fa2/` plugin (`bash fa2/install.sh` builds and installs it once, the Docker image has it, and the
launcher refuses to start without it instead of falling back). `FA2_FP8KV=0` restores the int8 +
Triton arm described above. The build targets sm86 only, so the default is on only when
`nvidia-smi` reports compute capability 8.6 (an RTX 3090-class GPU); any other GPU, or no answer
from `nvidia-smi`, resolves an unset knob to 0 with a note. The sm86 build also runs on sm89, and
`FA2_FP8KV=1` opts in on a 4090 (unmeasured); on sm120 it fails at boot.
The drafter also selects `FLASH_ATTN`, `SPEC_ATTN` does nothing on this arm (the plugin has its
own split-KV verify), and `MAX_LEN` and `KV_MEM` keep the defaults above. An unset `FA2_FP8KV`
resolves to 0, with a note, when `KV_OFFLOAD_GB` or a KV connector in `EXTRA_ARGS` is set: the
plugin's backend supports only the LBNHC KV layout, vLLM's CPU-tier connector asks for LBHNC or
BLHNC, and the CPU tier on this backend is untested, so the offload tier keeps int8 + Triton. An
explicit `FA2_FP8KV=1` refuses that combination, and any other `SPEC` or `CTX`. `PTH_FA=1` (the
pth-flash-attn kernel, below) is the other backend for this arm, and on an 8.6 GPU with the package
installed it is this arm's default when the launch is the measured one: then an unset `FA2_FP8KV` stays
0, and an explicit `FA2_FP8KV=1` with an explicit `PTH_FA=1` is refused. `PTH_FA=0`, an explicit
`FA2_FP8KV`, a missing install or any setup the kernel was not measured with keeps the fp8 + FA2
default described here. The evidence for that default is the
T4 battery on the native 3090 ([#153](https://github.com/syv-ai/HyperQwen/issues/153)):
on 2026-10-18 (bare metal, 250 W, vLLM 0.31.0) the fp8 + FA2 arm passed every bar against the shipped int8 + Triton arm: GSM8K 0.950 vs 0.955 (n=200), perplexity +0.04%, 0 of 128 residues broken, a 20-minute soak with no restart, decode 54.0 vs 48.2 tok/s at ~66k and 48.2 vs 37.9 tok/s at ~100k (+27%), cold TTFT 75.5 vs 132.3 s and 54.7 vs 130.5 s, and a 3.3% larger pool (140,902 vs 136,429 tokens).

### What it is actually good for

Measured against the option that already covers this context, on 112,655-token prompts
(`bench/labd_bench.py --ctx 100000 --corpus ~/bench/labd_corpus_long.txt`), both with
`PREFIX_CACHE=1`; the `SPEC=dflash2` column is the int8 + Triton arm (`FA2_FP8KV=0`), measured
before the default changed:

| task | `SPEC=dflash2 CTX=long` (k=15) | `SPEC=mtp CTX=long` |
|---|---|---|
| reproduce the document verbatim | 14.19 tok/step, **154.8 tok/s** | 3.81, 101.4 |
| list every command | 5.32, 69.8 | 3.52, **93.6** |
| rewrite but keep | 5.09, 66.8 | 3.79, **100.6** |
| quote and explain | 2.17, 34.6 | 2.75, **72.9** |
| summarize | 2.17, 34.8 | 2.68, **71.4** |
| answer a question | 2.01, 32.1 | 2.34, **61.9** |
| all six | 3.10, 47.0 | 2.95, **78.6** |
| TTFT, first turn / cached turn | 316.8 s / 6.1 s | **151.9 s / 2.4 s** |

**So: +53% where the model reproduces its context, and about 2:1 behind everywhere else, with
twice the TTFT.** `SPEC=mtp CTX=long` remains the better general long-context server, and it
reaches 150k rather than 114k. Reach for `SPEC=dflash2 CTX=long` when the workload is a RAG
front-end quoting sources or a coding assistant applying edits to a large file it has already
loaded — where the answer is mostly text that is already in the prompt — and not otherwise.

The lookup drafter itself does not decay with context: it accepts 14.19 of a possible 16
tokens per step at 112k, against 15.0 at 25k and 50k. What costs the mode is the step, at
91.7 ms against bf16's 48.2 ms at 50k — the Triton backend, the drafter's own five layers, and
the int8 kernel's padded-stride penalty (its head dim is 260 B, so odd KV heads start off a
16-byte boundary; ~13% end to end, and reading the cache as int32 instead would recover most
of it). Prefill is the larger cost and this kernel cannot help there — a prefill chunk is 2048
query tokens, far above the block sizes it is for.

## int8 KV on the pth-flash-attn kernel (`PTH_FA=1`, `SPEC=mtp` or `SPEC=dflash2` with `CTX=long`)

`SPEC=mtp CTX=long` runs fp8 KV on FlashInfer, the one fp8 backend sm86 has (gotcha 40), and the
flashinfer-free escape from #34 is `int8_per_token_head` on `TRITON_ATTN`, which decays past 25k.
`PTH_FA=1` keeps the int8 cache and reads it with [pth-flash-attn](../pthfa/README.md) instead: a CUDA
kernel that consumes the int8 bytes and their inline scales directly, registered through vLLM's plugin
slot as the backend `CUSTOM`. It is built into the image (`pthfa/install.sh`, SASS for sm80/86/89) and
selected only by this knob (set, or its default on an 8.6 GPU below); no vLLM file changes, so `patches/`
is the same with or without it.

```bash
SPEC=mtp CTX=long PTH_FA=1 bash single-user/start_qwen.sh
```

What that resolves to, the settings that matter (the full argv: `PRINT_ARGV=1`):

```
--attention-backend CUSTOM --kv-cache-dtype int8_per_token_head
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","attention_backend":"CUSTOM"}'
```

The `"attention_backend":"CUSTOM"` inside the speculative config is the setting that is easy to drop: a
drafter takes its backend from there, not from `--attention-backend`. `PTH_FA=1` writes both, so the MTP
layer reads the same cache with the same kernel. Decode and verify batches are captured FULL (the backend
declares uniform-batch graphs) with vLLM's default mode; nothing else in the argv changes
(`bench/test_pthfa_knob.sh` checks it). `VLLM_SPEC_DECODE_ATTN` is not read on this route: on vLLM 0.31.0
the `FLASH_ATTN` and `TRITON_ATTN` impls read it (`patches/spec-decode-attn.patch`,
`patches/spec-decode-int8-kv.patch`), and here both models run `CUSTOM`. Refused at startup: any other
`SPEC`/`CTX`, `--attention-backend`, `--kv-cache-dtype` or `--speculative-config` in `EXTRA_ARGS`, a KV
offload tier (untested on this backend), a venv without the package, `FA2_FP8KV=1` alongside it. Limits
are gotcha 63.

**Default on an 8.6 GPU.** An unset or empty `PTH_FA` resolves to 1 on this arm, as on
`SPEC=dflash2 CTX=long`, when `nvidia-smi` reports compute capability 8.6, the package is installed, and
the launch is the measured one: no explicit `FA2_FP8KV`, no KV offload tier or connector, none of the three
flags above in `EXTRA_ARGS`, no `DYNAMIC_K`, no `VISION=1` and `DRAFT_TOKENS` unset or 3. Then the argv is
that of `PTH_FA=1` and the effective-config line says `PTH_FA=1 (default; ...)`; otherwise fp8 on
FlashInfer runs as before, with one line if the package is missing. `PTH_FA=0` opts out. On the native
RTX 3090 (TP=1, 250 W, vLLM 0.31.0, the launcher's own settings, four boots in the order fp8, default, default, fp8;
the maintainer's A/B on [#345](https://github.com/syv-ai/HyperQwen/pull/345)), against fp8 on FlashInfer:
decode (`bench/labd_bench.py` total) at ~66k 84.8 and 84.7 against 75.6 and 75.5 tok/s (+12.2%) and at ~100k 78.5 and 78.5
against 68.9 and 68.8 tok/s (+14.0%), tokens per step equal within 2%, cold TTFT at ~66k 75.0 and 75.2 against 74.8 and 75.0 s,
GSM8K n=200 0.965 against 0.960, KV pool 198,241 against 202,040 tokens (1.9% smaller), and on the default arm a residue sweep with
0 of 128 broken and a 20-minute soak that exits 0. The step-time gain is the kernel and the CUDA-graph mode together: with speculative
decoding vLLM falls back to piecewise graphs on FlashInfer and the kernel arm captured full graphs, and the kernel arm was not run forced to
piecewise. An earlier single-boot run on the same card ([#325](https://github.com/syv-ai/HyperQwen/pull/325)) read 84.5 against 77.5 tok/s
at ~66k (+9%), with `bench/test_pthfa_attn.py` 37/37 on the card. On two RTX 3080 20 GB (TP=2, 220 W, one sitting, N=1, #345) the default
took 0.449 / 0.393 / 0.375 of fp8's time per decode pass at 128k / 32k / 8k (decode 110.4 against 46.5 tok/s at 128k) with a 1.6% smaller KV pool;
its tables and caveats are on [pth-flash-attn's benchmarks page](https://github.com/noon-at-cgn/pth-flash-attn/blob/v0.1.0/docs/benchmarks.md).
The tables below are from two RTX 3080 at TP=2.

**Measured on two RTX 3080 20 GB (sm86) at TP=2, the PR author's box, not on the reference 3090.**
Linux in a Proxmox LXC container, driver 615.71.09, PCIe without P2P (`NCCL_P2P_DISABLE=1`,
`--disable-custom-all-reduce`), checkpoint Swift-1.5 Qwen3.8-27B MTP W4A16 AutoRound (a Qwen3.8-27B
fine-tune, same architecture), `SPEC=mtp` k=3, `PREFIX_CACHE=1 VISION=1 MAX_SEQS=2`, `MAX_LEN=262144`,
`GPU_UTIL=0.94`, `--max-num-batched-tokens 8192 --long-prefill-token-threshold 1024`. That is this
knob's argv with `EXTRA_ARGS="--tensor-parallel-size 2 --disable-custom-all-reduce --max-num-batched-tokens 8192 --long-prefill-token-threshold 1024"`.
**vLLM 0.30.0** (HyperQwen 3acb93f plus the plugin, image `hyperqwen-local:3acb93f-pth-c47adf3`), with the
backend passed through `EXTRA_ARGS`; the vendored kernel is SASS-identical to that image's, function by
function on sm80/86/89. Old arm: the same HyperQwen image and cache on `TRITON_ATTN`, back to back in one
power state. "stock" is 320 W unlocked. Distinct 8k / 32k / 128k prompts (no prefix hits), 256 tokens out,
greedy, medians with n in parentheses:

| stock, 2x RTX 3080 20 GB, TP=2, vLLM 0.30.0 | `TRITON_ATTN` (int8) | `CUSTOM` (int8, `PTH_FA=1`) | ratio |
|---|---|---|---|
| TTFT 8,217 / 31,977 tokens, warm | 6.03 s (7) / 31.74 s (5) | 5.55 s (7) / 23.99 s (5) | 0.92 / 0.76 |
| TTFT 128,016 tokens, cold / warm | 252.9 s (1) / 250.2 s (2) | 129.3 s (1) / 129.2 s (2) | 0.51 / 0.52 |
| 1 stream decode at 8k / 32k / 128k | 96.9 / 78.5 / 45.8 tok/s (5/3/3) | 101.4 / 98.3 / 85.0 tok/s | 1.05 / 1.25 / 1.86 |
| ms per forward pass, same | 29.7 / 35.3 / 59.4 | 27.9 / 28.2 / 31.3 | 0.94 / 0.80 / 0.53 |
| 2 streams, per stream, same | 74.6 / 61.5 / 33.0 tok/s | 80.3 / 79.2 / 59.0 tok/s | 1.08 / 1.29 / 1.79 |
| MTP tokens per step (GSM8K window) | 3.642 | 3.631 | 0.997 |
| GSM8K, n=200, greedy | 0.965 | 0.960 | |
| KV pool at `MAX_LEN=262144` | 445,722 tokens | 474,503 tokens | 1.065 |

The 8k decode ratio's 95% interval is 0.99-1.13: no difference at short context, as expected for a
kernel whose win is reading less memory per token. In the capped state (220 W, 1905 MHz lock), the new arm
on the same box read 133.2 s for the 128k TTFT, 90.5 tok/s and 31.7 ms per pass at 128k, against the
capped Triton baseline's 265.2 s, 47.8 tok/s and 61.1 ms; those two were measured two days apart, not
back to back. Also on this box, new arm only: a 3 h soak with mixed families up to 198,733-token prompts
and needles at 32k/128k/250k (517 of 517 checks), a sync-check boot, and teacher-forced logprobs at 32k
and 128k that sit no further from a bf16-KV reference than the Triton arm's (mean |delta| 0.0219 against
0.0220 nats at 32k).

**On vLLM 0.31.0 (this branch's pin when measured), with that pin's image.** Same box (two RTX 3080 20 GB, TP=2,
our hardware, not the reference 3090), **capped state** (220 W, 1905 MHz lock), same launch shape as
above. Four arms in one sitting, back to back: `PTH_FA=1` on this branch's image (NEW-on), the image
built the same way from `main` on `TRITON_ATTN` int8 (OLD), this branch's image with `PTH_FA` unset on
`TRITON_ATTN` int8 (NEW-off, the control), then NEW-on again. Medians of 3 reps (8k: 5), 95% CI of the
ratio in brackets; the second NEW-on arm agrees within 1-2%:

| capped, 2x RTX 3080 20 GB, TP=2, vLLM 0.31.0 | OLD (`TRITON_ATTN` int8) | NEW-on (`PTH_FA=1`) | NEW-on / OLD | NEW-off / OLD |
|---|---|---|---|---|
| TTFT 128k / 32k / 8k, warm | 261.8 / 26.89 / 6.15 s | 127.8 / 18.98 / 5.61 s | 0.488 (n=2) / 0.706 [0.59-0.89] / 0.912 [0.910-0.918] | 1.001 / 0.999 / 1.000 |
| 1 stream decode at 128k / 32k / 8k | 44.1 / 71.9 / 93.2 tok/s | 86.2 / 100.6 / 100.6 tok/s | 1.95 [1.87-2.02] / 1.40 [1.16-1.50] / 1.08 [1.02-1.10] | 0.997 / 1.004 / 0.992 |
| ms per forward pass, same | 62.79 / 36.74 / 30.60 | 32.17 / 29.12 / 28.48 | 0.512 / 0.793 / 0.931 | 1.003 / 1.003 / 1.001 |
| 2 streams, per stream, same | 22.4 / 39.0 / 57.9 tok/s | 42.4 / 51.6 / 61.8 tok/s | 1.89 / 1.32 / 1.07 | 0.999 / 1.000 / 0.999 |
| 2-stream pair TTFT at 128k | 502.6 s | 245.3 s | 0.488 | 1.001 |
| MTP tokens per step (1 stream, 128k) | 2.783 | 2.783 | 1.000 | 1.000 |
| GSM8K n=200 / tokens per step (GSM8K window) | 0.960 / 3.637 | 0.970 / 3.635 (second arm 0.965 / 3.634) | +1.0 pt | NEW-off 0.960 / 3.634 |

No preemption in any arm. The control column is this branch's image with the knob unset: every row
0.987-1.005 of `main`'s image, with intervals spanning 1, and both log `TRITON_ATTN` and the same
445,722-token pool. On the same box and pin, with `PTH_FA=1`: the kernel check `bench/test_pthfa_attn.py`
passes 37/37 (including a request on blocks 2,439-2,599), all 128 prompt residues come back verbatim
with prefix caching, needles at 128k and 250k are retrieved 9/9, a 3 h soak passes 531/531 checks, and
teacher-forced logprobs at 32k and 128k sit closer to a bf16-KV reference than the `TRITON_ATTN` arm's
on all four measures (32k all: mean |delta| 0.0223 against 0.0232 nats).

**Not measured, so not claimed:** any card but the RTX 3080 and, on this arm, the maintainer's RTX 3090
run above, so sm86 only. sm80 and sm89 have cubins and no
datapoint, and the `SKIP(by design)` guard on a non-sm8x card has not run on one. TP=1 is measured only
in that 3090 run.
The GPU checks are `bench/test_pthfa_attn.py` (kernel against an fp64 reference, a pool past block id
2,383, CUDA-graph replay) and the README's field-test protocol with `PTH_FA=1`.

### `SPEC=dflash2 CTX=long PTH_FA=1`: the target and the drafter on the kernel

`SPEC=dflash2 CTX=long` with `FA2_FP8KV=0` runs `int8_per_token_head` on `TRITON_ATTN` (see "DFlash2 past
64k" above; on an 8.6 GPU the FA2 fp8 arm is the default only where the kernel's default below does not apply), and its decode decays with context.
`PTH_FA=1` keeps the int8 cache and moves both models onto the kernel: the target's 16 full-attention
layers and the DFlash2 drafter's five sliding-window layers.

```bash
SPEC=dflash2 CTX=long PTH_FA=1 bash single-user/start_qwen.sh
```

What that resolves to, the settings that matter (the full argv: `PRINT_ARGV=1`):

```
--attention-backend CUSTOM --kv-cache-dtype int8_per_token_head
--speculative-config '{"method":"dflash","model":"<repo>/models/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7,"draft_sample_method":"probabilistic","attention_backend":"CUSTOM"}'
```

As on mtp, the `"attention_backend":"CUSTOM"` inside the speculative config is what puts the drafter on
the kernel: a drafter takes its backend only from there. The speculative config is the arm's own with
that one key appended; nothing else in the argv or the environment changes (`bench/test_pthfa_knob.sh`
checks it, also against a launcher with the target-only configuration, where the same line has no
`attention_backend` and the drafter autoselects `TRITON_ATTN`). It names no `kv_cache_dtype`, so the drafter's cache keeps
the target's `int8_per_token_head` (vLLM 0.31.0 overrides the drafter's dtype only when that key is set).

**What the kernel serves for the drafter, and nothing more.** The drafter's layers are sliding-window
(2048) and non-causal at head size 128, 32 query / 8 KV heads (16 / 4 per rank at TP=2), with 8 query
tokens per request (the bonus token and 7 drafts): row `r` at position `p` sees keys
`[max(0, p - 2047), min(L - 1, p + 2047)]`, the same mask vLLM's `TRITON_ATTN` applies to these layers.
The kernel has windowed, non-causal variants at head size 128 only; a sliding window or non-causal layer
at head size 64 or 256 is refused at backend selection, and ALiBi, sinks, soft cap, a bidirectional
multimodal prefix and R-SWA stay refused (gotcha 63, `pthfa/tests/gate_test.py`). Blocks below a
request's window, which vLLM points at its null block, are never read: the kernel starts each row tile's
key range at the window's lower edge and zero-fills below it. The causal kernels the target runs compile
to the same SASS as the previous pin's (pth-flash-attn's `tools/sass_diff.py`: 105 kernels identical,
18 new, on sm80/86/89), so the target's half of this arm runs the same machine code as the target-only
configuration.
`VLLM_SPEC_DECODE_ATTN` stays the arm's 1; with both models on `CUSTOM` nothing reads it (only the
`FLASH_ATTN` and `TRITON_ATTN` impls do). Refused at startup, besides the mtp arm's refusals: a drafting
setup other than the measured one, i.e. `DRAFT_METHOD` other than `dflash` (DSpark is another drafter),
`DFLASH_TOKENS` other than 7 (it changes the target's verify block, and above 7 makes its length
adaptive), `LOOKUP=0`, and `DYNAMIC_K` (a draft count per batch size, so the verify block's length varies
with the batch). `VISION=1` is accepted and untested on this arm (gotcha 63).

**On an 8.6 GPU this is the arm's default.** An unset or empty `PTH_FA` resolves to 1 on this arm when
all of these hold: `nvidia-smi` reports compute capability 8.6 (the same query as the FA2 default; the only
card family measured), the `pth_flash_attn` package and its op library are in the venv (the check
`PTH_FA=1` makes), and nothing the user set picks something else or leaves the measured setup: no explicit
`FA2_FP8KV` (1 is the FA2 arm, 0 the `TRITON_ATTN` int8 arm), no KV offload tier or connector, no
`--attention-backend`, `--kv-cache-dtype` or `--speculative-config` in `EXTRA_ARGS`, no `DRAFT`, the DFlash2
drafter at `DFLASH_TOKENS=7` with lookup on, no `DYNAMIC_K` and no `VISION=1`. Then the argv and the
environment are those of an explicit `PTH_FA=1`, and the effective-config lines read
`FA2_FP8KV=0 (PTH_FA=1 overrides the 8.6 default)` and `PTH_FA=1 (default; ...)`. Otherwise the arm's own
default runs (fp8 KV on FlashAttention-2 on 8.6) with no message and never a refusal of `PTH_FA`'s;
a missing install prints one line saying so. `PTH_FA=0` opts out. With an explicit `PTH_FA=1` the
FA2 default is overridden the same way on any card the kernel serves, and an explicit `FA2_FP8KV=1` with
it is refused before either plugin is probed (`resolve_config.sh`); `FA2_FP8KV=0` is accepted.

**Boot log, this branch's image (`PTH_FA=1`, the 2026-10-17 daytime sitting below):** `Using AttentionBackendEnum.CUSTOM
backend.` four times, for the target and for the drafter on each of the two ranks, and no
`Using TRITON_ATTN attention backend` line (the target-only arm has one, the drafter's autoselect); the
engine's speculative config carries `'attention_backend': 'CUSTOM'` and the cache is
`int8_per_token_head`; the backend reserves its workspace for 16 target layers (`Hq=12, D=256`) and 5
drafter layers (`Hq=16, D=128`) per rank; FULL CUDA graphs are captured for the target and for the
`dflash2` head; the KV pool is 504,770 tokens, as with the target only; no `NotImplementedError` or
`does not support`. On one of the GPUs, before the sitting, `bench/test_pthfa_attn.py` passes 51 of 51
cases on this image, including the 14 of part 4 (the drafter's shapes, windowed and non-causal, against
an fp64 reference, with blocks below the window on vLLM's null block; max error 0.52 of the tolerance),
and `pthfa/tests/gate_test.py` passes all 34 checks with the GPU visible.

**Target on the kernel, drafter on `TRITON_ATTN` (the target-only configuration), measured on two RTX 3080
20 GB (sm86) at TP=2, the PR author's box, not on the reference 3090:** the same box as above (Linux in a
Proxmox LXC container, driver 615.71.09, PCIe without P2P). Capped state (220 W, 1905 MHz lock), vLLM
0.31.0, text-only, k=7 with lookup drafting on, `PREFIX_CACHE=1 MAX_SEQS=2 MAX_LEN=262144 GPU_UTIL=0.94
NCCL_P2P_DISABLE=1` and
`EXTRA_ARGS="--tensor-parallel-size 2 --disable-custom-all-reduce --max-num-batched-tokens 8192 --long-prefill-token-threshold 1024"`.
Checkpoints are the launcher's defaults: the target `models/Qwen3.8-27B-W4A16-AutoRound-fast`, which
`prepare/fetch_fast_variant.py` assembles from `dbirks/Qwen3.8-27B-W4A16-AutoRound` (its shards, hardlinked)
and `syvai/qwen3.8-27b-3090-fast-variant` (int4-GPTQ lm_head and MTP module, draft vocabulary), and the
drafter `models/Qwen3.8-27B-DFlash2-W4A16`, fetched by `prepare/fetch_dflash2.py` from
`syvai/Qwen3.8-27B-DFlash2-W4A16`.

Two sittings on 2026-10-17, in the morning (04:56-08:45 box time) and during the day (10:30-16:35), each with
`main`'s image on `TRITON_ATTN` (the arm without the knob, built
from 95ee3b6) as the same-sitting control, run first and repeated last for drift. Morning sitting: an
image of the target-only configuration built at 8b55d4a with `PTH_FA=1`. Daytime sitting: the same
configuration built at a61f0a6, which adds refusals, tests and docs only (the argv of this arm and the
`pthfa/` files are the same), the same launch. Distinct 8k / 32k / 128k prompts (the same prompts in both), 256 tokens out, greedy;
medians of 3 reps (8k: 5) unless the row says n; ratio `CUSTOM` / `TRITON_ATTN` of the same sitting, 95%
CI in brackets:

| capped, 2x RTX 3080 20 GB, TP=2, vLLM 0.31.0, `SPEC=dflash2 CTX=long` | 2026-10-17 morning: `TRITON_ATTN` (int8) | morning: `CUSTOM` target (`PTH_FA=1`, image at 8b55d4a) | morning: ratio | 2026-10-17 daytime: `TRITON_ATTN` (int8) | daytime: `CUSTOM` target (`PTH_FA=1`, image at a61f0a6) | daytime: ratio |
|---|---|---|---|---|---|---|
| TTFT 128k cold (n=1) / 32k warm (n=5) / 8k warm (n=7) | 249.3 / 30.99 / 5.93 s | 122.5 / 23.24 / 5.44 s | 0.491 / 0.750 [0.60-0.89] / 0.918 [0.913-0.921] | 249.6 / 31.01 / 5.93 s | 122.4 / 23.21 / 5.47 s | 0.491 / 0.748 [0.60-0.89] / 0.922 [0.920-0.925] |
| 1 stream decode at 128k / 32k / 8k | 58.9 / 101.9 / 138.5 tok/s | 108.1 / 126.3 / 148.1 tok/s | 1.84 [1.77-1.96] / 1.24 [1.05-1.28] / 1.07 [0.95-1.10] | 59.1 / 102.2 / 138.1 tok/s | 107.3 / 127.2 / 147.3 tok/s | 1.81 [1.76-1.95] / 1.24 [1.05-1.28] / 1.07 [0.95-1.11] |
| ms per forward pass, same | 55.0 / 31.0 / 25.2 | 29.2 / 24.6 / 23.6 | 0.531 [0.528-0.536] / 0.794 [0.793-0.801] / 0.939 [0.933-0.957] | 54.9 / 30.9 / 25.1 | 29.1 / 24.5 / 23.7 | 0.531 [0.527-0.537] / 0.793 [0.786-0.801] / 0.944 [0.935-0.952] |
| 2 streams, per stream, same | 32.8 / 54.1 / 79.7 tok/s | 54.8 / 69.6 / 81.4 tok/s | 1.67 [1.58-1.75] / 1.29 [1.13-1.41] / 1.02 [1.00-1.11] | 32.8 / 54.0 / 78.9 tok/s | 54.8 / 70.1 / 81.3 tok/s | 1.67 [1.59-1.73] / 1.30 [1.14-1.42] / 1.03 [1.00-1.09] |
| 2-stream pair TTFT at 128k | 480.0 s | 239.4 s | 0.499 [0.497-0.499] | 482.1 s | 239.3 s | 0.496 [0.490-0.499] |
| accepted length per draft (of 7), GSM8K window | 4.789 | 4.811 | 1.005 | 4.796 | 4.812 | 1.003 |
| GSM8K n=200, greedy | 0.955 | 0.960 | +0.5 pt | 0.955 | 0.960 | +0.5 pt |
| KV pool at `MAX_LEN=262144` | 477,599 tokens | 504,770 tokens | 1.057 | 477,599 tokens | 504,770 tokens | 1.057 |

No preemption in any arm; needles at 50% and 90% depth of a 128k prompt retrieved and determinism at
1k/8k/32k passing in every arm with the full gate set. The control repeated last moved by at most 1.4%
on any median in the 2026-10-17 morning sitting (decode and TTFT 0.992-1.003, ms per pass 1.001-1.014)
and by about 1% in the daytime one (ms per pass 0.988-1.002, decode 0.990-1.006, all CIs including 1). The two `CUSTOM`
images agree within that drift (ms per pass 29.2 / 24.6 / 23.6 against 29.1 / 24.5 / 23.7). As on the
mtp arm, the gain grows with context: at 8k the one-stream decode intervals include 1. Acceptance does
not depend on the backend: per-position rates agree within 0.01. Prefix caching with the arm's FULL
graphs: every one of the 128 prompt-length residues, sent twice so the second is a full prefix-cache
hit, comes back verbatim with `PTH_FA=1` on both images, 0 broken of 128 (`bench/residue_sweep.py`; the
corpus was a stand-in, this repo's `README.md` and `docs/*.md`, because the sweep's usual corpus file was
not on this box; the test depends on the prompt length mod 128, not on the text).

**Against the FA2 fp8 default, same sitting.** On an 8.6 GPU this arm's default is fp8 KV on
FlashAttention-2 (`FA2_FP8KV=1`, above), not `TRITON_ATTN` int8; `PTH_FA=1` replaces it with the int8 cache
on this kernel. They were measured back to back in a third sitting, 2026-10-17 20:11 to 2026-10-18 03:08, same box, capped
state, launch shape and prompts as above: `main`'s image before the FA2 merge (built from 95ee3b6) on
`TRITON_ATTN` int8, the FA2 change's own image (built from its head 71253a4) with no knob set, which
resolves to the FA2 default on this box, the target-only image built at a61f0a6 and this branch's image
(target and drafter), both with `PTH_FA=1`. Each had the full gate set; medians of 3 reps (8k: 5), cold
TTFT n=1, 95% CI in brackets:

| capped, 2x RTX 3080 20 GB, TP=2, vLLM 0.31.0, `SPEC=dflash2 CTX=long`, 2026-10-17 | `TRITON_ATTN` int8 (`FA2_FP8KV=0`) | `FLASH_ATTN` fp8 (the 8.6 default) | `CUSTOM` target only (`PTH_FA=1`) | `CUSTOM` target and drafter (`PTH_FA=1`) | target only / FA2 | target and drafter / FA2 | FA2 / `TRITON_ATTN` |
|---|---|---|---|---|---|---|---|
| 1 stream decode at 128k / 32k / 8k | 58.6 / 102.0 / 137.1 tok/s | 81.5 / 110.8 / 134.2 tok/s | 108.4 / 127.5 / 148.5 tok/s | 115.1 / 129.5 / 151.6 tok/s | 1.33 [1.18-1.41] / 1.15 [1.09-1.25] / 1.11 [0.99-1.14] | 1.41 [1.23-1.45] / 1.17 [1.13-1.29] / 1.13 [1.00-1.19] | 1.39 [1.33-1.60] / 1.09 [0.89-1.12] / 0.98 [0.92-1.03] |
| ms per forward pass, same | 55.0 / 31.0 / 25.1 | 36.8 / 28.8 / 25.0 | 29.0 / 24.6 / 23.4 | 28.4 / 24.0 / 22.8 | 0.789 [0.788-0.793] / 0.854 [0.844-0.857] / 0.936 [0.932-0.950] | 0.770 [0.769-0.779] / 0.832 [0.818-0.837] / 0.913 [0.908-0.920] | 0.669 [0.664-0.678] / 0.931 [0.921-0.938] / 0.995 [0.989-1.004] |
| 2 streams, per stream, same | 32.7 / 50.5 / 78.9 tok/s | 39.3 / 54.4 / 77.0 tok/s | 54.6 / 69.9 / 82.0 tok/s | 55.7 / 67.1 / 85.3 tok/s | 1.39 [1.32-1.41] / 1.29 [1.17-1.38] / 1.07 [1.01-1.15] | 1.42 [1.32-1.50] / 1.23 [1.19-1.39] / 1.11 [1.05-1.18] | 1.20 [1.16-1.23] / 1.08 [0.90-1.13] / 0.98 [0.92-1.03] |
| ms per forward pass, 2 streams | 63.3 / 35.9 / 29.3 | 54.1 / 38.6 / 30.7 | 39.0 / 30.3 / 28.0 | 38.2 / 29.3 / 27.2 | 0.720 [0.717-0.729] / 0.785 [0.777-0.788] / 0.912 [0.908-0.916] | 0.707 [0.702-0.716] / 0.759 [0.754-0.769] / 0.887 [0.883-0.897] | 0.855 [0.842-0.863] / 1.073 [1.064-1.074] / 1.048 [0.988-1.058] |
| TTFT cold, 128k / 32k / 8k | 249.6 / 27.8 / 6.22 s | 114.1 / 18.3 / 5.85 s | 122.3 / 18.7 / 5.45 s | 122.1 / 18.7 / 5.62 s | 1.071 / 1.024 / 0.931 | 1.070 / 1.022 / 0.961 | 0.457 / 0.657 / 0.940 |
| KV pool at `MAX_LEN=262144` | 477,599 tokens | 517,656 tokens | 504,770 tokens | 504,770 tokens | 0.975 | 0.975 | 1.084 |
| accepted length per draft (of 7), GSM8K window | 4.794 | 4.838 | 4.811 | 4.835 | 0.994 | 0.999 | 1.009 |
| GSM8K n=200, greedy | 0.955 | 0.955 | 0.955 | 0.965 | 0 pt | +1.0 pt | 0 pt |
| residue sweep, 128 prompt-length residues | 0 broken | 0 broken | 0 broken | 0 broken | | | |

Needles at 50% and 90% of 128k retrieved, determinism at 1k/8k/32k passing and no preemption in all four.
Where FA2 is ahead: the cold 128k prefill is 6.7% shorter than with the target only on the kernel and
6.5% shorter than with both (the ladder's 128k TTFT ratios are 1.071 [1.070-1.072] and 1.072
[1.071-1.074]), and its pool is 2.6% larger. Per decode pass at one stream the target-only configuration
takes 21% less time than FA2 at 128k and 6% less at 8k, the target and drafter configuration 23% and 9%
less; the two kernel configurations differ by 1.9-3.4% per pass (0.966-0.981), as in the 2026-10-17
daytime sitting; at two streams FA2 takes longer per pass than `TRITON_ATTN` at 32k (1.073
[1.064-1.074]) and at 8k (1.048, CI including 1). On the ladder's 128k
prompt FA2 accepts 3.012 tokens per step against 3.108 (target only) and 3.200 (target and drafter), the
same in both FA2 boots, so the 128k
tok/s ratio carries that prompt's acceptance; ms per pass does not. A second boot of the FA2 image
reproduced its ms per pass within about 1%, and the control repeated last within 2%.

**Accuracy against a bf16 cache, same images.** Teacher-forced prompt logprobs of a 32k and a 128k document
(30,616 and 127,937 scored positions) against a reference with bf16 KV on `FLASH_ATTN` (the launcher's
`CTX=fast` on `main`'s image, same checkpoints), all boots at `MAX_LEN=135168 GPU_UTIL=0.88`, 2026-10-18.
Mean |delta logprob| of the actual token, 32k / 128k: `TRITON_ATTN` int8 0.0219 / 0.0194, `PTH_FA=1` int8
0.0207 / 0.0193 (target only and target and drafter, identical on every position: the drafter does not
touch prompt logprobs), FA2 fp8 0.0360 / 0.0338 (p99 0.156 / 0.152, 0.150 / 0.150, 0.268 / 0.274). A
non-inferiority test (the new arm's mean |delta| at most the old arm's plus max(10% of it, 0.002 nats),
on the point estimates over all scored positions and over the last 2048, with a 95% CI of the paired
per-position difference from a non-overlapping block bootstrap, blocks of 256 and 1024 positions):
`PTH_FA=1` passes against `TRITON_ATTN` int8, and FA2 fp8 fails against both int8 arms, by +0.014 to
+0.015 nats over all scored positions in both documents, CIs excluding 0. GSM8K n=200 cannot resolve a difference of this size.

**The T4 battery** (`bench/t4_battery.py`, its own configuration: `MAX_SEQS=4 GPU_UTIL=0.93`, max model
length 131072, 2048 batched tokens, ~60k and ~90k prompts, so not comparable with the tables above),
2026-10-18, same box and state, three arms: `TRITON_ATTN` int8 and FA2 fp8 on the image of the FA2
change, and the int8 kernel with the target and the drafter on it (this branch's measured image; the
target-only configuration was not run through the battery). Every bar passes for all
three: GSM8K 0.960 / 0.955 / 0.965, perplexity over the whole set 8.4013 / 8.4020 / 8.4022, 0 of 128
residues broken, soak exits 0 (FA2 60 min, the kernel 240 min; the battery soaks no int8 Triton arm).
Decode at ~60k 55.3 / 69.5 / 81.7 tok/s and at ~90k 44.2 / 62.2 / 76.1 tok/s, i.e. the
kernel +17.6% and +22.3% over FA2; cold TTFT at ~60k 76.0 / 46.8 / 49.1 s (FA2 4.9% shorter than the
kernel). One run per arm, no CIs. The corpus was a stand-in, this repo's markdown, because the battery's
usual corpus was not on this box; the arms share it, absolute numbers are not comparable with other
machines'.

**Target and drafter on the kernel (`PTH_FA=1` on this branch's image), measured in the 2026-10-17
daytime sitting above** (10:30-16:35, same box, capped state, launch shape and prompts): `main`'s image on `TRITON_ATTN`
(built from 95ee3b6, the control), this branch's image with `PTH_FA=1`, the target-only image at
a61f0a6 with `PTH_FA=1`, each with the full gate set; then a second boot of each of the two
`CUSTOM` images (ladders and TTFT only, 5 reps), and the control's ladders again last. The measured image
vendors pth-flash-attn 07e5063; the branch then pinned a31d8ab, which differs from it in comments and
docstrings only (the Python modules have equal syntax trees without docstrings, the C++/CUDA files are
token for token the same without comments, and the exported sources compile to the same SASS), and `pthfa/` now pins the public
release v0.1.0 (b0d4f7e), which differs from a31d8ab in comments, docstrings, packaging metadata, two error messages and one sentence of
`NOTICE`, with the same SASS (123 of 123 kernels on sm80/86/89; pth-flash-attn's `tools/compare_vendor_trees.py` and `tools/prove_sass_equal.sh`), and its
launcher gives the measured launch the same argv as this branch's (which adds the target-only
configuration's `DYNAMIC_K` refusal and 8.6 override; `DYNAMIC_K` was unset, and with `PTH_FA=1` the
override leaves the argv as measured on an 8.6 GPU). Medians of the first boot of each arm (3 reps, 8k: 5); the "both / target
only" column pools the two boots of each `CUSTOM` arm (8 reps, 8k: 10), the "both / `TRITON_ATTN`"
column is the first boots; 95% CI in brackets:

| capped, 2x RTX 3080 20 GB, TP=2, vLLM 0.31.0, `SPEC=dflash2 CTX=long` | `TRITON_ATTN` (int8, `main`'s image) | `CUSTOM` target only (image at a61f0a6) | `CUSTOM` target and drafter (`PTH_FA=1`) | both / target only | both / `TRITON_ATTN` |
|---|---|---|---|---|---|
| TTFT 128k cold (n=1) / 32k warm (n=5) / 8k warm (n=7) | 249.6 / 31.01 / 5.93 s | 122.4 / 23.21 / 5.47 s | 122.2 / 23.10 / 5.43 s | 0.998 / 0.996 [0.80-1.24] / 0.992 [0.989-0.996] (first boots) | 0.490 / 0.745 [0.60-0.89] / 0.915 [0.913-0.919] |
| 1 stream decode at 128k / 32k / 8k | 59.1 / 102.2 / 138.1 tok/s | 107.3 / 127.2 / 147.3 tok/s | 115.8 / 129.4 / 151.3 tok/s | 1.054 [1.002-1.087] / 1.033 [0.992-1.081] / 1.025 [0.945-1.115] | 1.96 [1.86-2.05] / 1.27 [1.09-1.32] / 1.10 [0.97-1.16] |
| ms per forward pass, same | 54.9 / 30.9 / 25.1 | 29.1 / 24.5 / 23.7 | 28.2 / 24.1 / 22.6 | **0.975 [0.964-0.986] / 0.974 [0.932-0.984] / 0.967 [0.955-0.978]** | 0.514 [0.511-0.517] / 0.780 [0.769-0.786] / 0.900 [0.891-0.912] |
| 2 streams, per stream, same | 32.8 / 54.0 / 78.9 tok/s | 54.8 / 70.1 / 81.3 tok/s | 55.3 / 71.0 / 83.7 tok/s | 1.013 [0.976-1.064] / 1.014 [0.974-1.088] / 1.031 [0.971-1.064] | 1.69 [1.63-1.80] / 1.32 [1.17-1.45] / 1.06 [1.03-1.11] |
| ms per forward pass, 2 streams | 63.9 / 36.2 / 29.5 | 39.0 / 30.2 / 28.2 | 38.4 / 29.6 / 27.3 | 0.986 [0.979-0.994] / 0.979 [0.966-0.983] / 0.971 [0.967-0.984] | 0.601 [0.596-0.609] / 0.816 [0.804-0.827] / 0.924 [0.913-0.939] |
| 2-stream pair TTFT at 128k | 482.1 s | 239.3 s | 237.8 s | 0.994 [0.993-0.996] (first boots) | 0.493 [0.487-0.497] |
| accepted length per draft (of 7), GSM8K window | 4.796 | 4.812 | 4.827 | 1.003 | 1.006 |
| GSM8K n=200, greedy | 0.955 | 0.960 | 0.960 | 0 pt | +0.5 pt |
| KV pool at `MAX_LEN=262144` | 477,599 tokens | 504,770 tokens | 504,770 tokens | 1.000 | 1.057 |

**What it buys over the target-only arm: 2.5-3.3% less time per decode pass at one stream (1.4-2.9% at
two), and no established tok/s gain.** Every ms-per-pass interval of "both / target only" excludes 1,
and the profile below shows where the time goes. The per-stream tok/s ratios are all above 1 as point
estimates, but their intervals include 1 in five of six cells; the one that excludes 1 (one stream at
128k) is inflated by about 3% by acceptance on that particular prompt: the ladder's 128k prompt gets
3.200 tokens per step with both models on the kernel and 3.108 with the target only, the same in both
boots, so per prompt the acceptance differs while on GSM8K it differs by 0.3% (per position within
0.0025). Per boot, the two boots of each `CUSTOM` image agree within about 1% in ms per pass, and all
12 per-boot ratios (6 rows, 2 boot pairs) are below 1 (0.952-0.991); the pooled intervals treat reps
as independent and are narrower than that boot-to-boot spread. SM clock and power were the same in both
arms. Prefill does not change (TTFT ratios 0.99-1.01: the drafter's attention is not on the prefill
path). The control's ladders, repeated last, moved by about 1% (ms per pass 0.988-1.002). No preemption
in any arm; needles at 50% and 90% depth of a 128k prompt retrieved, determinism at 1k/8k/32k passing,
and the residue sweep 0 broken of 128 with both models on the kernel (same stand-in corpus as above).

Per engine step, from a torch profile of one rank (4 windows of 40 decode steps per depth, one stream,
the same images and launch; median over windows, ms per step; the other rank within 2%):

| rank 0, ms per step | target only, 8k | both, 8k | target only, 128k | both, 128k |
|---|---|---|---|---|
| drafter attention, 5 layers (`TRITON_ATTN` unified kernel, or this kernel's windowed kernel + combine) | 0.878 | 0.098 | 0.918 | 0.102 |
| target attention + combine, 16 layers | 0.746 | 0.755 | 5.631 | 5.664 |
| everything else | 21.47 | 21.57 | 22.09 | 22.22 |
| step wall (median spacing of consecutive steps) | 23.49 | 22.73 | 29.03 | 28.40 |

The drafter's attention falls from 175 / 183 us per layer call on `TRITON_ATTN` to 13 + 7 us (window
kernel + combine) at 8k / 128k, saving 0.78 ms per step at 8k (3.3% of the target-only step) and 0.82 ms
at 128k (2.8%); the step wall shortens by 0.75 / 0.62 ms, which matches the ladders' ms-per-pass ratios.
The traces with both models on the kernel contain no `TRITON_ATTN` attention kernel.

**Not measured, so not claimed:** any card but the RTX 3080, TP=1, and `VISION=1` on this arm.

## DFlash2 at 240k: `CTX=huge` (KVarN) also combines with `SPEC=dflash2`

```bash
bash kvarn/install.sh                # applies the v0.31.0 KVarN + V2-runner ports
SPEC=dflash2 CTX=huge PREFIX_CACHE=1 bash single-user/start_qwen.sh
```

Where `CTX=long` doubles the DFlash2 pool with int8 KV (138k), the KVarN cache
takes the same idea further: 268k tokens of pool at 245760 max-model-len, on the
same pinned budget. No kernel work — the KVarN Triton kernels run unmodified on
the V2 runner; the fixes in `kvarn/kvarn-v2-runner-0.31.0.patch` are allocator and
geometry logic (the patch header walks through them, including an upstream vLLM
bug in the mamba align resume path, and a NaN path in the DFlash2 candidate
selector that KVarN noise exposes on verbatim-reproduction content). Two
machines, both RTX 3090 at 250 W, `bench/labd_bench.py --ctx 20000` — the
contributor's WSL2 box and this repo's bare-metal one, which do not agree on
decode rate and do agree on everything else:

| `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` | WSL2 | bare metal |
|---|---|---|
| copy (reproduction) | 130 tok/s, 7.8 tok/step | 164 tok/s, 7.83 tok/step |
| code / edit / quote / summary / qa | 89 / 65 / 44 / 38 / 36 | 109 / 83 / 58 / 51 / 43 |
| all six tasks together | 53 tok/s, 3.0 tok/step | 67 tok/s, 3.15 tok/step |
| verbatim reproduction, 25k document | correct | 1,150 / 1,150 chars |
| KV capacity at 245760 max-model-len | 268,169 tokens | 268,169 tokens |
| GSM8K exact-match (thinking off) | 97.0% (n=200) | 95.2% (n=600), 95.0% (n=200) |
| 100k-deep needle, both turns | correct | — |
| turn 2 over a 100k cached prefix | 4.7 s (vs 169 s cold) | — |

<sub>Context for the GSM8K column: every configuration this repo already ships
reads 95.0-96.5% on the same 200-question harness ([quality.md](quality.md)),
and 95.0% is the batch-mode default. 95.2% at n=600 (±0.9 points) therefore sits
inside the band rather than below it — which is the useful comparison, since this
mode inherits KVarN's lossy 4/2-bit cache and should be judged against the other
lossy configurations rather than against bf16. Repeat runs of the reproduction
check on bare metal are bit-identical (same step count, same 1,150 characters),
which is the property that was missing before `PIECEWISE` — see below.</sub>

One caveat to the "all of it is lossless" paragraph above: the speculation here
is still exact, but this mode inherits KVarN's 4/2-bit KV cache, which is lossy —
the same trade `CTX=huge` already makes (deep-needle retrieval passes at 200k).

The WSL2 column's ~20% deficit is a WSL2 tax, not a Windows tax, and leaving
WSL for native Windows does not recover it: the same contributor ran
`aivrar/vllm-windows-build` (0.27.1, 18 of 19 patches apply after a CRLF→LF
pass) on the same box and measured native Windows *slower* than WSL2 — 66.0
vs 76.2 tok/s across the task mix, a 4.4× longer warm boot, and the same
WDDM paging behavior underneath ([#25](https://github.com/syv-ai/HyperQwen/issues/25)).
The bare-metal column is reachable from a Windows box only by putting Linux
on the metal.

**On WSL2, every `SPEC=dflash2` profile needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`** —
not just `CTX=huge`. The drafter's architecture forces vLLM's V2 model runner
(`_is_dflash2_draft()` in `config/vllm.py`), the V2 runner allocates UVA buffers
before the weights load, and vLLM leaves pinned memory off by default under WSL2,
so a clean venv aborts with `RuntimeError: UVA is not available` before it prints
anything model-shaped. Those buffers work fine on the paravirt driver. Note the
name: `VLLM_WSL_PIN_MEMORY` is **not** a vLLM variable and setting it does
nothing — this README named it for 22 minutes on 2026-08-21 (`589daae`, fixed in
`27f51fa`), so a tree cloned in that window will have it.

**On WSL2 the usable dedicated memory is about half a gigabyte less than the
same card on bare metal, and the shipped `SPEC=dflash2` boot sits about 50 MiB
under it.** Anything larger (a wider verify block, a bigger drafter, a raised
pin, a boot that recompiles a graph) runs two to six times slower instead of
failing, and the log does not say so; `nvidia-smi` looks the same either way.
Gotcha 58 has the counters to read, the four costs that turned out to be this,
and the profile that gives it room (`KV_MEM=3000000000 DFLASH_MAX_LEN=8192`,
free for the shipped head at width 7).

**Running a DSpark drafter.** vLLM 0.28.0 can serve RadixArk/Qwen3.8-27B-DSpark
(bf16, seven drafts per step like the shipped head) once two things are in
place: `patches/dspark-draft-quant-config.patch` (the loader refuses a bf16
drafter beside the quantized target without it), and a copy of the checkpoint
whose `config.json` names the architecture `Qwen3DSparkModel` instead of
`DSparkDraftModel` (the registry maps the published name to the DeepSeek V4
class; the weights are unchanged, so hard-link the safetensors). Then
`DRAFT=/path/to/that/copy DRAFT_METHOD=dspark KV_MEM=3000000000
DFLASH_MAX_LEN=8192 SPEC=dflash2 CTX=fast bash single-user/start_qwen.sh`. It
serves, and loses to the shipped head on the same requests: 3.19 against 3.77
tokens per step (accepted drafts plus the bonus token) and 115 against 160 tok/s on a 4090, 3.37 against 3.83
and 114 against 150 on a 3090 (issue #25, items 15 and 16). Documented so nobody
re-derives the two errors, not as a recommendation.

One knob this mode used to set for you, and now sets only for MTP:
`cudagraph_mode=PIECEWISE`. Prefix caching and a *captured* (FULL) verify step
did not mix on this path. On WSL2 that showed up as acceptance collapsing to
about one token per step; on bare metal it also **corrupted the output** —
special-token ids leaking into the stream, 1 of 1,176 characters matching the
source instead of all of them. It is the capture rather than the drafter: eager
is clean, `LOOKUP=0` is not, forcing a fixed verify-block length is not, and
PIECEWISE — which keeps the compiled graphs and leaves only the multi-query
verify uncaptured — restored both the speed and the correctness on both machines.

`dflash2` has its FULL graphs back (`a75ee4b` fixed the residue, `b356e31` then
swept **all 128** residues under FULL with 0 broken), so at HEAD
`SPEC=dflash2 CTX=huge PREFIX_CACHE=1` runs captured. `SPEC=mtp CTX=huge PREFIX_CACHE=1`
still forces PIECEWISE, and that one is a correctness constraint rather than a
preference: under FULL it breaks at one prompt length in 128 and nobody has
fixed it. Unset `PREFIX_CACHE` means 1, so unset gets PIECEWISE too. On 0.31 a sweep of
all 128 residues found 0 broken at an 86% prefix-cache hit rate (2026-10-17), but it
measured the path unset ran before the launcher passed the flag both ways (vLLM's
default capture and caching, without the launcher's safeguards). It does not lift the
pin. The pinned default path swept 0 of 128 broken on 0.31 as well (2026-10-17). In both
cases prefix caching is on, which is what the mode is for — turn 2 over a
cached 100k document costs 4.7 s against 169 s cold.

`CUDAGRAPH_MODE=FULL_AND_PIECEWISE` overrides the MTP `PREFIX_CACHE=1` line for anyone
hunting the root cause. Treat that as unsafe rather than merely slower.

What that trade costs, re-measured at HEAD. The numbers this README used to carry here
had `FULL_AND_PIECEWISE` at 38 tok/s (1.97 per step) on the 25k copy task against
PIECEWISE's 132, and called it 3.5x. That was not the capture mode — it was the residue
bug, which `a75ee4b` fixed. With the same server and only the capture toggled
(`bench/labd_bench.py --ctx 20000`, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1`, decode tok/s):

| | copy | code | edit | quote | summary | qa | all six |
|---|---|---|---|---|---|---|---|
| FULL (the default now) | 167.1 | 111.1 | 84.7 | 55.0 | 47.8 | 43.4 | 65.7 (3.03/step) |
| PIECEWISE | 166.3 | 111.3 | 83.0 | 62.4 | 48.6 | 43.1 | 67.6 (3.18/step) |

They are the same. Five of the six are within 2%; `quote` differs by 13% in PIECEWISE's
favour, which is greedy divergence on the task that diverges most, and it is what puts
PIECEWISE 3% ahead overall. So at this context length the capture mode is not a
performance decision at all, in either direction.

Short prompts are where a difference was measured, and that measurement is older:
78/125/202 tok/s captured against 74/102/176 piecewise on de/en/code, i.e. **13-18%**.
Treat that as an upper bound — @mjungnickel18 measures 0.2-2.3% for the same comparison
when only runs with identical step counts are compared, and he is right that greedy runs
which take a different number of steps are not comparable. Past 8k the two are within
noise on bare metal (111.8 vs 109.3 tok/s at 8k, 78.2 vs 86.1 at 16k, 68.9 vs 73.3 at
32k, 58.4 vs 56.0 at 50k, unique prompts, one server per mode). Under GPU passthrough on
a VM the same comparison costs 2-3x, reported in
[#13](https://github.com/syv-ai/HyperQwen/pull/13) and consistent with the
uncaptured verify being launch-bound: launches that are nearly free here are not free
there.

Two limits worth knowing before you point this at anything: what it does with
more than one user, and what the long verify block costs.

**It is a one-stream mode, and the limit is the pool rather than `MAX_SEQS`.** An
earlier version of this paragraph said the knob was the seat count and that
`MAX_SEQS=8` lifts it. It does not, and @mjungnickel18 was right to push back in
[#25](https://github.com/syv-ai/HyperQwen/issues/25). A *resident* request
reserves 1+k = 8 recurrent-state slots — **15.8% of the 69,758-token `CTX=fast` pool**,
~0.82 GiB of its pinned 5.20, which is the 0.88 GiB [gotcha 33](gotchas.md) fitted
from the memory model — before it holds one token of context. Seven fit with 128-token
prompts, five with 4k-token ones and two with 16k ones. MTP's k=4 costs 0.44 GiB, so
eight fit, four of them at 16k. Past that the extras queue, and once the pool is full
something has to be preempted and recomputed to make room — a ramp of tiny requests
hits that at the seventh. `MAX_SEQS` decides how many requests are *admitted*, not how
many can run. The pool itself is **unchanged** by the setting (268,169 tokens at
`CTX=huge` either way, ~8 MiB total between 1 slot and 8 —
[gotcha 33](gotchas.md)), which is the part of the old paragraph that was right.

What concurrency actually costs, measured with `bench/conc_ladder.py` on distinct
4k-token prompts — each salted so nothing is served out of the prefix cache — 256-token
answers, `MAX_SEQS=8`, `CTX=fast`, 250 W:

| streams | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| **dflash2** per-stream decode tok/s | 137 | 97 | 46 | 33 |
| **dflash2** aggregate decode tok/s | 137 | 225 | 309 | *5 resident, no steady state* |
| **dflash2** ms per forward pass | 25.9 | 32.8 | 49.1 | — |
| **mtp** per-stream decode tok/s | 126 | 103 | 46 | 23 |
| **mtp** aggregate decode tok/s | 124 | 212 | 280 | **383** |
| **mtp** ms per forward pass | 24.8 | 29.8 | 43.1 | 62.3 |

Two things to read off it. The verify step *does* batch — aggregate throughput keeps
climbing for both speculators, and no preemption happens at any of these points — so
"the block verify does not batch" is not what is going on. What it costs is latency:
each additional resident request adds about 7 ms to every forward pass under DFlash2
(5 ms under MTP), so the second user roughly halves your tokens per second and the
fourth roughly quarters them. That 7 ms is not attention over their context — with
128-token prompts the slope is the same (25.2 → 45.8 ms from one resident to four) —
it is the per-request recurrent state, the same thing that limits residency. And MTP keeps scaling to 8 streams where DFlash2 runs
out of pool at 5, which is the whole of its C8 advantage. Point one person at DFlash2;
point a team at `SPEC=mtp` or batch mode.

**On `CTX=huge`, raising `MAX_SEQS` is worse than not raising it.** That profile
defaults to 2 seats, and the reason is the same state page against a smaller pinned pool
(4.90 GiB): five residents with a short prompt, four with a 16k one. Force it
to 8 and feed it eight independent 16k-token streams and the scheduler starts evicting
— **10 preemptions** in one run, peak occupancy 99.4%, per-stream 3 / 7 / 72 tok/s,
end-to-end aggregate down to 10.3 from 13.0 at a single stream. The same eight streams
against the shipped 2 seats: **0 preemptions and 14.4 tok/s**, i.e. 40% more work done
by admitting fewer requests. An earlier version of this README recommended exactly that
override. Leave the seats where they are.

Make the streams long and independent and it stops being about decode at all. Eight
16k-token prompts with nothing shared between them: end-to-end aggregate **15.8 tok/s**
against 131 for a single stream, mean TTFT 71.7 s, two requests resident — but the
decode-only aggregate over the same run is 183 tok/s, tokens per step is unchanged at
3.71 and nothing is preempted. The run is 131k tokens of prompt at ~1,600 tok/s and
2,048 tokens of answer, so it is a prefill measurement wearing a decode measurement's
units, and `SPEC=mtp` reads the same 15.0-15.9 there. If your clients each bring their
own long document, that is the number you get, and no speculator changes it.

What *does* change it is sharing the document. The same eight 16k streams with one
shared prefix and `PREFIX_CACHE=1` (`bench/conc_ladder.py --shared`):

| streams | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| end-to-end aggregate tok/s | 15.5 | 128.4 | **147.8** | 68.3 |
| mean TTFT | 14.6 s | 1.3 s | 2.6 s | 11.5 s |

One stream pays the prefill; everyone after it hits the cache, and four concurrent
readers of the same document get 148 tok/s end-to-end against 15.9 when the documents
differ. That is the shape of workload this mode is for — a chat client or a coding
front-end against one codebase — and it is nearly a 10x difference from the same server
on the same prompt length.

`DFLASH_TOKENS=15` doubles the state page to 1.66 GiB: three residents with an empty
context, **two** with 4k-token prompts, against five. That mode is single-user in the
literal sense, and its launcher default of `MAX_SEQS=4` is already the tighter number —
do not raise it. `DFLASH_TOKENS=15 MAX_SEQS=8` used to boot, answer `/health` and then
die on the first concurrent batch (`torch.OutOfMemoryError` in the engine, every request
500); the launcher now caps the captured-graph size so that configuration degrades to
piecewise instead ([gotcha 38](gotchas.md)). It does boot at 240k since `82bd62d`,
which caps `max_model_len` to 221,184 above 7 drafts rather than letting the server fail
to come up.


The capture mode is fixed at boot, and for `SPEC=mtp CTX=huge PREFIX_CACHE=1` the trade
is not optional. What FULL does there is corrupt one prompt length in every 128, and only
for a request that hits the prefix cache. The broken residue is a function of the
draft count (`R = 117 + k`, the same for both speculators), which is why scoping
the workaround to `dflash2` was wrong the first time — `SPEC=mtp CTX=huge` shipped
with the same bug at residue 4. Piecewise costs MTP nothing measurable
(87.8/86.1/70.4/63.5 tok/s captured against 93.5/83.8/70.3/59.6 piecewise over
8k-50k), so it keeps PIECEWISE until residue 4 comes back verbatim under a full
sweep of that path.

**Do not test that residue by its symptom.** The location is deterministic and the
damage is not: the same `mtp` residue has returned an empty answer, a one-character
answer, and 400 tokens of fluent Danish inventing a task the prompt never asked for
(2 of 1,146 characters matching the document). A detector keyed on "it repeats" or
"it came back empty" passes at least one of those. Gotcha 37 in
[gotchas.md](gotchas.md) has the residue table; `bench/residue_sweep.py`
sweeps all 128 residues and judges every answer on how much of the document came
back, and `bench/verbatim.py` self-tests that rule against all three shapes.

## 256k the stock way: int4 KV (`single-user/alternative.sh`, experimental)

```bash
bash single-user/alternative.sh      # TRITON_ATTN + --kv-cache-dtype int4_per_token_head
```

Where KVarN reaches 268k with its own kernels, vLLM's stock
`int4_per_token_head` cache now combines with the DFlash2 drafter too:
**314,915 tokens of pool at 256000 max-model-len** (1.23× concurrency) on one
24 GB card, no `kvarn/install.sh`. Three boot blockers stood in the way — a
padded-page view error under the hybrid block-promotion geometry, and a
causal-only assert plus missing per-seq-causal plumbing in the int4 Triton
kernel, which the drafter's 8-row draft block needs
(`patches/int4-kv-per-token-head.patch`, contributed in
[#42](https://github.com/syv-ai/HyperQwen/pull/42) by @lachhabw).

The trade: the Triton attention backend plus the per-step int4 unpack cost
about 20% of decode against the shipped config on short prompts (~86 vs ~104
tok/s e2e on the same probe), and — unlike KVarN, which has GSM8K and
100k-needle numbers above — int4-KV quality at depth now reads:

| metric | result |
|---|---|
| GSM8K exact-match (200 questions, greedy, thinking off) | **96.0%** |
| 100k-token needle at 90% depth (`bench/needle_test.py`) | **retrieved** |

Measured on an RTX 4090 (24 GB) with `bench/quality_battery.py int4kv --gsm-only
--gsm-n 200` and `bench/needle_test.py 100000 0.9`; the 96.0% sits inside the band
the other configurations read (95.0-96.5%, docs/quality.md). Tool calling
round-trips correctly and the lookup lane works; the rest of this configuration
is still experimental.


### Sparse CPU offload with a sliding-window drafter

For a hybrid Mamba target with an Eagle-family drafter, the reusable recurrent
state can precede the prompt's final hash boundary. Sparse CPU retention must
also keep the drafter's sliding-window history at that earlier boundary.
`offload-mamba-swa-replay-boundaries.patch` retains that history as well as the
existing prompt/shared-prefix tails. Without it, a stored prefix can miss the
joint lookup despite fitting in the CPU tier (#405). Dense retention is unchanged.
