# More than one GPU

What transfers to a second card and what does not, plus the tensor-parallel
shapes this model's head count allows.

[← back to the main README](../README.md)

Everything here is written for one 24 GB card; multi-GPU goes through untouched
via `EXTRA_ARGS`:

```bash
SPEC=dflash2 PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2" bash single-user/start_qwen.sh
```

**`--tensor-parallel-size 3` is not valid for this model.** The checkpoint's
text config has 4 KV heads and 64 layers:

```bash
curl -sL https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/raw/main/config.json \
  | python3 -c "import json,sys; t=json.load(sys.stdin)['text_config']; print(t['num_key_value_heads'], t['num_hidden_layers'])"
# 4 64
```

Tensor parallelism needs the KV-head count and the TP size to divide one
another, and 4 and 3 do neither way; pipeline parallelism needs an even layer
split, and 64 does not divide by 3. So on a three-card box the third card
cannot join the engine — run a TP=2 engine on two cards plus a second
standalone engine on the third
([#104](https://github.com/syv-ai/HyperQwen/issues/104)), and if both
serve containers share one host, set `VLLM_OFFLOAD_KEEP_SHM=1` on both: each
launcher's stale-offload-region reaper only sees its own container's processes
and would delete the other engine's live region
([#33](https://github.com/syv-ai/HyperQwen/issues/33)).

Under Docker, the same two knobs live in `.env` — `GPU_COUNT` says how many
cards the container gets, `EXTRA_ARGS` says how many the engine uses, and
nothing in `docker-compose.yml` needs editing:

```
GPU_COUNT=2
EXTRA_ARGS="--tensor-parallel-size 2"
```

(The launchers already pass `--language-model-only` unless `VISION=1`, so it does not
belong in `EXTRA_ARGS`; an earlier version of this example repeated it.)

`GPU_COUNT` defaults to 1, which is *the first card the runtime enumerates* —
GPU 0, not necessarily the card you meant. On a mixed box, expose them all and
pin inside the container with `GPU_COUNT=all` plus `CUDA_VISIBLE_DEVICES=1,2`
(the device reservation decides what is visible, so `NVIDIA_VISIBLE_DEVICES`
in `.env` alone was not enough; `CUDA_VISIBLE_DEVICES` is read inside the
container and is what gotcha 53 recommends). Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
as well: by default CUDA numbers the fastest card first, so on a box that mixes
generations `CUDA_VISIBLE_DEVICES=0,1` can name a different pair than `nvidia-smi`
shows. That happens on native Linux too, not only under WSL2: a 5080 was paired
with a 3080 that way in [#216](https://github.com/syv-ai/HyperQwen/issues/216).

**Pin the KV pool before you measure anything, or before you trim `MAX_LEN`
until the OOMs stop.** Under `--tensor-parallel-size > 1` the launcher
deliberately skips the single-card `KV_MEM` pin (below), and `SPEC=mtp` has no
profile pin at all — so the pool is sized from `GPU_UTIL`, and the headroom it
is carved out of moves by ~0.92 GiB depending on whether the torch.compile
cache was warm (`docs/gotchas.md` 48). That is the "booted at 146k yesterday,
OOMs today" failure. To pin it, boot once at a context that works and read the
two lines the engine prints:

```bash
docker compose logs single | grep -E "Available KV cache memory|GPU KV cache size"
# INFO ... Available KV cache memory: 3.36 GiB
# INFO ... GPU KV cache size: 161,280 tokens
```

then put a byte count a little under that `GiB` figure into `.env` as
`KV_MEM=` (3.36 GiB ≈ 3607772528 bytes; round down). `KV_MEM` overrides the
TP skip — the launcher only declines to pin one *for* you. A pinned pool
either fits at boot or refuses at boot, instead of OOMing on the first
request, and it is the only way two benchmark arms are comparable.

**On 8-12 GB cards, vLLM 0.29 can size a pool that does not leave room for
warmup.** 0.29's memory accounting returns memory that 0.27/0.28 held back, so
the same `GPU_UTIL` gives a bigger pool. On two 12 GB 3060s at `GPU_UTIL=0.915`
the fp8 pool went from 2.41 GiB to 3.01 GiB per card, and `SPEC=mtp CTX=long`
then ran out of memory in sampler warmup. The pin vLLM's own error message
suggests (`--kv-cache-memory=2975129437`, 155,316 tokens) booted and serves
128k ([#68](https://github.com/syv-ai/HyperQwen/issues/68)). On four 8 GB 3060
Tis at TP=4, setups A, B, C and D all ran out of memory at startup at their
shipped settings, and only E booted as shipped
([#210](https://github.com/syv-ai/HyperQwen/issues/210)). Until
[#214](https://github.com/syv-ai/HyperQwen/pull/214), the launcher passes
`KV_MEM` to vLLM only under `SPEC=dflash2`. Under `SPEC=mtp` or `SPEC=off`,
pass the pin as `EXTRA_ARGS="... --kv-cache-memory=<bytes>"`.

What the second card is worth is now measured, not assumed —
[#40](https://github.com/syv-ai/HyperQwen/issues/40) ran a controlled
1-vs-2×3090 A/B on this harness (same box, same install, PCIe 4.0 x8, **no
NVLink**, 275 W):

- **+16–35% decode across C1–C8** (DFlash2 greedy: 127.4 → 161.6 at C1). Worth
  it at batch 1: decode is bandwidth-bound here, and TP=2 is a second memory
  system, not idle compute.
- **The DFlash2 residency ceiling is a 24 GB property, not a drafter
  property.** On one card DFlash2 collapses at C8 (3.9 s TTFT — the
  recurrent-state pool exhaustion documented above) and MTP wins; on two cards
  DFlash2 wins at every concurrency measured. On 2×24 GB, point everyone at
  DFlash2.
- **The launcher no longer pins `KV_MEM` under TP>1** (their finding, this
  fix): the pin is a single-card constant applied per worker, and it stranded
  ~16 GiB across two cards — 137,210 tokens of pool where `GPU_UTIL` sizing
  gets 302,223, at no measured decode cost. Export `KV_MEM` to pin anyway.
- **Keep `DFLASH_TOKENS=7` at TP>1** for now: the one T=15 datapoint at TP=2
  (same issue) lost 27% at C1 with the lookup-filled tail accepting nothing,
  which is under diagnosis. The launcher warns.
- NVLink appears to buy little:
  [#7](https://github.com/syv-ai/HyperQwen/issues/7)'s NVLink box at
  330 W and #40's PCIe-x8 box at 275 W land within a few percent of each other
  at C1.
- **Past two cards, or on a consumer board with the cards on separate PCIe root
  ports, you may need `NCCL_P2P_LEVEL=SYS`** and
  `EXTRA_ARGS="--disable-custom-all-reduce"`. Reported from a 4x RTX 5060 Ti box
  on a community-patched P2P driver
  ([#105](https://github.com/syv-ai/HyperQwen/issues/105)): NCCL would
  not bring up peer-to-peer across four separate root ports until P2P was forced
  down to `SYS`, and vLLM's custom all-reduce faulted on that driver even at
  TP=2. Neither is reproducible on this repo's single-card box, so treat both as
  field reports, not as defaults — try TP first without them.
- **With a P2P-patched consumer driver, NCCL usually will not use P2P until
  you set `NCCL_P2P_LEVEL`, and the cause is wider than separate root ports.**
  NCCL's default allows P2P only up to `PXB`, cards behind a PCIe switch.
  Consumer boards put the cards on CPU root ports, which NCCL classes as `PHB`
  or `SYS`, so it falls back to host shared memory even when every pair reports
  peer access; the one exception built into NCCL 2.29 is two GPUs on an AMD
  host. Boot once with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P` and look
  for `via P2P` or `via SHM`. If it says SHM, `NCCL_P2P_LEVEL=PHB` turns P2P on
  for cards under one host bridge, and `SYS` across sockets. On 4x RTX 5060 Ti
  at TP4 that gave 2-6% more decode at C1 and 10-16% at C4 with prefill
  unchanged, because P2P cuts the latency of the small allreduces in each
  decode step ([#254](https://github.com/syv-ai/HyperQwen/issues/254), one box;
  the numbers are in [reproductions](reproductions/README.md)). When you A/B
  P2P yourself, read the transport out of the log for each arm.

```bash
NCCL_P2P_LEVEL=SYS SPEC=dflash2 PREFIX_CACHE=1 \
  EXTRA_ARGS="--tensor-parallel-size 4 --disable-custom-all-reduce" \
  bash single-user/start_qwen.sh
```

**2x RTX 3090 with NVLink, both profiles, one box.** The fullest dual-card
report this repo has ([#159](https://github.com/syv-ai/HyperQwen/issues/159),
[#163](https://github.com/syv-ai/HyperQwen/issues/163),
[#164](https://github.com/syv-ai/HyperQwen/issues/164): NV4 link, 250 W per
card, driver 595.91.07, Docker, vLLM 0.28.0, harness runs with a discarded
warmup and three measured repeats).

- **Start by turning the allocator off, not custom all-reduce off.** This box
  first came up only with `--disable-custom-all-reduce`; the crash behind that
  is the expandable-segments/IPC interaction in `docs/gotchas.md` 3, and
  clearing it properly is worth +6.4% at C1 (171.8 -> 182.8 tok/s greedy,
  3.32 tok/step in both arms) and ~9% at C4. Both launchers now default
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` when `EXTRA_ARGS` asks
  for TP>1 without `--disable-custom-all-reduce` or `--enforce-eager`, and say
  so at boot; set the variable yourself to override either way. That 182.8 is the single-user
  row in [docs/reproductions](reproductions/README.md), and it is the fastest
  C1 decode reported on Ampere here.
- **Batch, setup A, is where the second card pays.** `GPU_COUNT=2` with
  `EXTRA_ARGS="--tensor-parallel-size 2"` on the documented batch defaults
  (base checkpoint, `KV=fp8`, `INT8_ACT=int8 INT8_LAYERS=mlp`, `MAX_SEQS=64`,
  `MAX_LEN=150000`, no speculation, no prefix cache) gives 1,439 tok/s decode
  and 1,344 e2e at 64 concurrent on 128 in / 512 out, against ~1,035 / 948 on
  one card: **+39% aggregate**, with an 872,938-token KV pool and GSM8K 0.965
  over 200. Three measured repeats landed within 1% of each other.
- **That contradicts the other dual-3090 batch report, and the profile is
  why.** [#135](https://github.com/syv-ai/HyperQwen/issues/135) measured 917
  decode at 64 concurrent on two cards -- *less* than one card -- but ran
  `KV=kvarn VISION=1` rather than the reference batch profile. Two cards do
  not make a batch server slower; a different KV dtype and a loaded vision
  tower do. Match the profile before you compare aggregates.
- **Single-user C1 does not scale the same way**, and that is expected: batch
  1 decode is bandwidth-bound, so TP=2 buys the ~35% in
  [#40](https://github.com/syv-ai/HyperQwen/issues/40) and this box's +37%
  over the 133 tok/s reference, while the batch profile gets a second memory
  system *and* a second set of SMs to fill.

- **Without NVLink, custom all-reduce needs peer access, not the bridge.** A
  2x3090 box on PCIe x8 (no NVLink) with peer access enabled by a
  community-patched driver ran vLLM's custom all-reduce under the launcher's
  `expandable_segments:False` default. It measured +6.5 to +7.2% at C1 greedy
  over NCCL (`bench/real_rep.sh`, 4 reps per arm, identical `tok/step`), about
  the NVLink box's +6.4%. A fragmentation soak (200k prefills with concurrent
  60k prompts) held to `GPU_UTIL=0.96`, and both allocator settings failed
  the same way at 0.97, so the `:False` default costs nothing measurable
  there ([#163](https://github.com/syv-ai/HyperQwen/issues/163)).

**Without peer-to-peer at all, TP=2 can be slower than one card, and data
parallel is what the second card buys.** 2x RTX 4090 (Ada, where NVIDIA
disables P2P on consumer cards), one at PCIe 4.0 x8 and one at x4, Docker on
WSL2 ([#190](https://github.com/syv-ai/HyperQwen/issues/190)). NCCL runs the
all-reduce over host shared memory, and the custom all-reduce cannot run
without P2P:

| | one card | TP=2 | DP=2 (`--data-parallel-size 2`) |
|---|---|---|---|
| C1 greedy, default `SPEC=mtp` profile | 151.1 tok/s | 118.7 | |
| batch burst, 128 requests at 64-way, ~1,478 in / 256 out | 76.2 s | 110.9 s | **39.6 s** (1.92x one card) |
| KV pool | 215,721 (batch) | 896,134 | 215,721 per engine |

(Both cards had no other GPU clients for these runs. A Windows app holding a
context on the same card slowed TP=2's batch burst by another 17%.) So on a
no-P2P pair, run `--data-parallel-size 2`: one endpoint, one engine per card,
and no per-layer all-reduce. Use TP=2 only when a single request needs the
~4x pool. If you do, `NCCL_PROTO=LL` recovered about a third of TP=2's C1
loss there (113.7 -> 127.0), and `NCCL_SHM_USE_CUDA_MEMCPY=1` hung after
compile. Pipeline parallelism is not an option with this checkpoint: vLLM
refuses `--pipeline-parallel-size` because the multimodal
`Qwen3_5ForConditionalGeneration` class does not declare `SupportsPP`, even
with `--language-model-only`. (One later report did boot PP=2 on vLLM 0.28;
see [Pipeline parallelism](#pipeline-parallelism-one-report-160) below.)
The same box's native-Linux numbers are not
measured, so part of the TP=2 loss may be WSL2's.

Also reported working: **2× RTX 5060 Ti 16 GB**
([#22](https://github.com/syv-ai/HyperQwen/issues/22)) — the "would
not fit on one card" case — and **4× RTX 5060 Ti 16 GB** (TP4, sm120, PCIe 4.0
x8, 180 W, community-patched P2P driver,
[#105](https://github.com/syv-ai/HyperQwen/issues/105)). **4× RTX 3060 Ti 8 GB** (TP4, 110 W per
card, [#210](https://github.com/syv-ai/HyperQwen/issues/210)) runs setup E as
shipped at 118.1 tok/s C1 greedy, and needs adapted profiles for the rest (see
the pin note above). **2× RTX 3060 12 GB with a P2P-patched driver**
([#205](https://github.com/syv-ai/HyperQwen/issues/205)) runs setup D at
62.4. The graph budget
and `MAX_SEQS` defaults are still single-card calibrations; more A/Bs like
#40's are the most useful numbers you can send.

**With `SPEC=mtp` at TP>1, try int8 KV on `TRITON_ATTN` before fp8 on
FlashInfer.** #105's re-run separated what its first round had moved together,
one variable per arm, on 5060 Ti (sm120) at 180 W per card. MTP k=3, C1
greedy decode:

| | fp8 / FlashInfer | int8_per_token_head / `TRITON_ATTN` | gap |
|---|---|---|---|
| TP2 | 71.4 | 102.0 | 1.43x |
| TP4 | 72.8 | 137.6 | 1.89x |
| TP2, 2x RTX 3090 (sm86, uncapped, [#217](https://github.com/syv-ai/HyperQwen/issues/217)) | 90.9 | 156.0 | 1.72x |

`tok/step` is 2.5-2.7 in every cell, so this is step time, not acceptance —
and the fp8/FlashInfer path does not scale with TP at all while the int8 path
does. The KV pool costs ~5-9% for it. This inverts the single-card picture,
where int8 on `TRITON_ATTN` is a long-context capacity trade that costs ~25% of
decode at depth (`docs/gotchas.md` 40), so it is not a launcher default yet. The
third row is the Ampere answer: two 3090s at TP=2 (peer-to-peer from a patched
driver, no NVLink) show the same shape, 2.53 against 2.57 `tok/step` at the
default temperature and roughly 31 against 18 ms per step.
It is at least not a correctness trap on Ampere: at TP1 on the reference 3090
(vLLM 0.29, async scheduling on) it passes the concurrent-garbage check from
[#121](https://github.com/syv-ai/HyperQwen/issues/121) — three concurrent
~48K prompts three times over, then three alone, 9/9 and 3/3 valid, same as
the fp8/FlashInfer control.
The arm, for anyone who wants to check it on their own cards:

```
SPEC=mtp CTX=long EXTRA_ARGS="--tensor-parallel-size 2 --attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
```

## Pipeline parallelism (one report, #160)

Nothing in the repo sets `--pipeline-parallel-size`, and no profile is tested
with it. One field report ([#160](https://github.com/syv-ai/HyperQwen/issues/160))
measured it on vLLM 0.28.0, and it is worth knowing before you try: **PP=2 with
`SPEC=off` is healthy, and PP=2 with MTP is a net loss.** A later 0.29 re-run
found PP=2 slower than TP=2 on fp8 KV as well (below). It is one box.

Hardware: an RTX 3090 24 GB and an RTX A4000 16 GB, PCIe, no peer-to-peer.
Serving: `--pipeline-parallel-size 2 --distributed-executor-backend mp`,
`VLLM_PP_LAYER_PARTITION=48,16` (48 layers on the 3090, 16 on the A4000),
`--attention-backend TRITON_ATTN --kv-cache-dtype int4_per_token_head`,
`MAX_LEN=262144`, a RedHatAI int4 checkpoint with the head requantized to int8.
Greedy, 256-token generations, median of three:

| configuration | single stream | 2 streams | 4 streams | MTP acceptance | KV pool |
|---|---|---|---|---|---|
| TP=2 + MTP (baseline) | 75.6 tok/s | 132.7 | | 45.6% | 581,403 |
| PP=2 + MTP | 19.7-19.8 | 48.6 | 72.3 | **7.5-8.4%** | 726K-792K |
| PP=2 + `SPEC=off` | 32.9 | 61.4 | 55.6 | | 778,942 |
| PP=2 + async + `SPEC=off` | 36.9 | 68.8 | **134.2** | | **844,852** |

- **A current TP=2 baseline on the same two cards** (RTX 3090 + RTX A4000, no
  peer-to-peer; reporter's follow-up, 2026-10). A vLLM 0.29-based recipe, MTP
  k=4 with probabilistic draft sampling, `MAX_LEN=262144` per slot,
  `MAX_SEQS=2`, FlashInfer attention with **fp8** KV (not the int4 KV of the
  table above), 2048 batched tokens, prefix caching on: KV pool 342,065 tokens,
  single stream about 54.6 tok/s, two streams 94.0 aggregate, TTFT about
  111 ms, and 99-123 tok/s streaming decode across 16K-190K prompts once warm.
  It has run in production for two weeks with no repetition or corruption, so
  MTP's acceptance at TP=2 is normal and the collapse above is specific to
  pipeline parallelism. The lower single-stream number against the 75.6 above
  is most likely the different KV dtype and attention backend rather than a
  regression: the fp8/FlashInfer path is the slow one on sm86 (see the 3090
  rows in the reproductions page), though the report did not A/B it
  ([#160](https://github.com/syv-ai/HyperQwen/issues/160)).
- **PP=2 + `SPEC=off` on the 0.29 stack** (same cards, same reporter, fp8 KV,
  FlashInfer, `VLLM_PP_LAYER_PARTITION=48,16`, 2048 batched tokens, async
  scheduling on, `MAX_SEQS=2`; the only changes from the TP=2 baseline above are
  `--pipeline-parallel-size 2` and no speculative config). It is healthy (a
  long-prompt needle passes, no corruption) but slower than TP=2 + MTP on
  every throughput row, and the pool advantage is small on fp8 KV:

  | | TP=2 + MTP | PP=2 + `SPEC=off` |
  |---|---|---|
  | KV pool | 342,065 tokens | 405,773 (1.19x) |
  | single stream | 52.9 tok/s | 39.1 (-26%) |
  | 2 streams, aggregate | 95.0 | 74.2 (-22%) |
  | TTFT (short) | 108 ms | 134 ms |
  | prefill 33K | 19.9 s | 22.6 s |
  | prefill 66K | 33.0 s | 30.4 s |

  The 3.22x pool above was measured on the int4 KV stack, where the cost per
  token is different, so it does not carry over to fp8 KV. Caveats from the
  reporter: the 66K prefill rows are partly warm (the first 33K tokens are a
  cached prefix from the preceding probe), the prefill rows are single samples,
  and the TP=2 column has MTP on, with no TP=2 `SPEC=off` control. Both PP and
  TP runs were at stock power limits (3090 350 W, A4000 140 W, its maximum); the
  A4000's 7,001 MHz memory against the 3090's 9,751 MHz bounds bandwidth-bound
  TP decode on that pair
  ([#160](https://github.com/syv-ai/HyperQwen/issues/160)).
- **PP=2 + MTP.** It only boots with a local patch: upstream's
  `_pp_broadcast_prev_sampled_token_ids` asserts one sampled token per request,
  which speculation breaks. With the patch, acceptance falls from 45.6% to about
  8% at every `MAX_SEQS` (2, 3 and 4), and single-stream decode is 40% *slower*
  than PP=2 with no drafter (19.8 against 32.9): the drafter's four chained
  forwards and the 5-token verify are paid for almost nothing.
- **PP=2 + async scheduling + MTP** printed degenerated repetition from the first
  token. The maintainer's read of vLLM 0.28.0: `Qwen3_5MTP` does not declare
  `SupportsPP` (four other MTP models in that release do), and under
  speculation a request accepts 1 to k+1 tokens per step, so the one-column
  broadcast hands the first pipeline stage the wrong prefix. That is a gap in
  vLLM's contract that a local patch cannot close; the reporter filed it
  upstream as [vllm-project/vllm#57755](https://github.com/vllm-project/vllm/issues/57755).
- **PP=2 + `SPEC=off`** is healthy: a 246K needle passes, the KV pool is
  778,942 to 844,852 tokens (2.97x to 3.22x concurrency at 262k) against
  581,403 (2.22x) at TP=2, a 32K prefill takes 15.9 s against 23.7 s at TP=2,
  and async scheduling with four slots reaches 134 tok/s aggregate.

On the 0.28 int4-KV stack, PP=2 with `SPEC=off` was the way to get a large pool
from two unequal cards; on 0.29 with fp8 KV, TP=2 with MTP was better on every
row measured, so start there. Do not combine PP with `SPEC=mtp`. The reporter's
TP=2 baseline was not confirmed to run at the same `MAX_LEN` and `MAX_SEQS` as
the PP rows, and the two cards had different power limits (an A4000 is a 140 W
part), which matters little for a PP comparison because the ranks run in turn.
The earlier note above that vLLM refuses PP for this checkpoint's multimodal
class was written for a different setup; the two are not reconciled here.
