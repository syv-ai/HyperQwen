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
EXTRA_ARGS="--tensor-parallel-size 2 --language-model-only"
```

`GPU_COUNT` defaults to 1, which is *the first card the runtime enumerates* —
GPU 0, not necessarily the card you meant. On a mixed box, expose them all and
pin inside the container with `GPU_COUNT=all` plus `CUDA_VISIBLE_DEVICES=1,2`
(the device reservation decides what is visible, so `NVIDIA_VISIBLE_DEVICES`
in `.env` alone was not enough; `CUDA_VISIBLE_DEVICES` is read inside the
container and is what gotcha 53 recommends).

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

```bash
NCCL_P2P_LEVEL=SYS SPEC=dflash2 PREFIX_CACHE=1 \
  EXTRA_ARGS="--tensor-parallel-size 4 --disable-custom-all-reduce" \
  bash single-user/start_qwen.sh
```

Also reported working: **2× RTX 5060 Ti 16 GB**
([#22](https://github.com/syv-ai/HyperQwen/issues/22)) — the "would
not fit on one card" case — and **4× RTX 5060 Ti 16 GB** (TP4, sm120, PCIe 4.0
x8, 180 W, community-patched P2P driver,
[#105](https://github.com/syv-ai/HyperQwen/issues/105)). The tok/s
numbers in #105 are not quoted here: its two arms moved drafter, KV dtype and
prefix cache together, so the ratio is a profile delta rather than a drafter
delta. The graph budget and `MAX_SEQS` defaults are still single-card
calibrations; more A/Bs like #40's are the most useful numbers you can send.
