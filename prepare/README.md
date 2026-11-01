# prepare/ — one-time model preparation

The published W4A16 quant of Qwen3.8-27B is not servable on 24 GB as it ships: two
2.5 GB bf16 embedding matrices and an unquantized MTP draft module. These scripts
fix that in place, on the CPU, once. They are the [Setup](../docs/install.md) steps,
and `docker compose run --rm prepare` (see [docker/prepare.sh](../docker/prepare.sh))
runs exactly them, each skipped when its result is already in the model dir.

## Which checkpoints these scripts modify

They **extend** a quantization config, they do not build or convert one: each reads
`config.json`, clones `config_groups.group_0` and appends a group of its own (int8
`lm_head`, int8 `embed_tokens`, int8/int4 `mtp.*`). The checkpoint they are pointed at
therefore has to already be in **compressed-tensors `pack-quantized`** form.

| what you have | what to do |
|---|---|
| the base W4A16 quant they were written for — `quant_method: "compressed-tensors"`, `ignore`, `config_groups.group_0` | run them, as below |
| a checkpoint **already prepared** for HyperQwen (int8 heads + drafter) | none of `prepare/` — download it and start the server |
| a **native AutoRound export** — `quant_method: "auto-round"`, `bits`/`group_size`/`sym`/`data_type` | not usable as it ships. It has no `group_0` and no `ignore` list to extend, so it has to be **converted** to compressed-tensors first; neither these scripts nor vLLM do that |
| a **single-shard** export too big to read into RAM | `quant_heads_stream.py`, see [A different checkpoint](#a-different-checkpoint) |

The directory name is not a guide either way: the base checkpoint these scripts were
written for is itself a compressed-tensors re-export that keeps `-AutoRound` in its name.
Ask `config.json` instead:

```bash
python -c "import json,sys; q=json.load(open(sys.argv[1]))['quantization_config']; print(q.get('quant_method'), 'ignore' in q, sorted(q.get('config_groups', {})))" \
  models/YOUR-MODEL/config.json
# compressed-tensors True ['group_0']   -> these scripts work
# auto-round False []                   -> convert it first; prepare/ cannot
```

Every in-place script runs that check before it opens a shard, and exits with one message
if it does not match (`prepare/quant_schema.py`). A wrong checkpoint costs a second and
writes nothing, instead of a `KeyError: 'ignore'` after the shard has been replaced (#241).

Run from the repo root, in order — `quant_lm_head.py` first, because
`build_draft_vocab.py` slices its rows:

```bash
V=venv/bin/python; M=models/Qwen3.8-27B-W4A16-AutoRound
$V prepare/quant_lm_head.py $M      # lm_head -> int8 group-128, in place: ~1.3 GB freed
$V prepare/quant_embed.py   $M      # embed_tokens likewise (untied): another ~1.3 GB
$V prepare/quant_mtp.py     $M      # the mtp.* draft module (~850 MB bf16) -> int8
$V prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json
$V prepare/fetch_fast_variant.py    # optional, ~1 GB: the single-user "fast" variant
$V prepare/fetch_dflash2.py         # optional, 1.2 GB: the DFlash2 drafter (SPEC=dflash2)
```

`build_draft_vocab.py` writes a 40,960-row slice of `lm_head` for the MTP drafter to
score instead of the full 248k vocabulary; `draft_vocab_ids.json` is the shipped id
list, and `--corpus` counts your own instead. It needs
[patches/qwen3_5-mtp-draft-vocab.patch](../patches/qwen3_5-mtp-draft-vocab.patch).

Two flags exist because that list is language-specific (gotcha 61): `--stats` prints its
coverage per script against your checkpoint's tokenizer, and `--add-scripts` writes a
*variant* list with whole-UTF-8 rows of the scripts you name, ranked by `--corpus` frequency and
capped at `--script-budget` rows (default 16,384, +41.9 MB of head on this checkpoint; 0 means uncapped: every eligible row, which for CJK is ~66k added rows and +169 MB, for a 107k-id variant list). The shipped list and the default head are untouched; a variant is a second run with `--ids` over the
file it writes. Measurements and the reasoning behind the cap: gotcha 61, and the
syv-ai/HyperQwen#196 thread.

## A different checkpoint

`quant_heads_stream.py` does the work of `quant_lm_head.py` + `quant_embed.py` +
`quant_mtp.py` in one pass, for **single-shard** checkpoints those three cannot open
(they read a shard into RAM whole; the uncensored build ships one 18.6 GB
`model.safetensors`). Same math, same output tensors and config groups (all four use
`prepare/quant_schema.py`), peak RSS well under the shard size (9.7 GB measured on the
18.6 GB example here -- still not a low-RAM tool). An asymmetric AWQ body is fine for
all four: the groups they add always declare the symmetric, zero-point-free tensors
they write.

```bash
$V prepare/fetch_thirdparty.py                          # ~18.6 GB (or: fetch_thirdparty.py <hf-repo>)
$V prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16
$V prepare/build_draft_vocab.py  models/Qwen3.8-27B-Uncensored-W4A16 \
  --ids prepare/draft_vocab_ids.json
```

Then `MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 bash single-user/start_qwen.sh`.
`SPEC=dflash2` additionally needs its pinned pool resized, because this checkpoint is
~1 GB heavier than the one those constants were measured on — the command and the
numbers are in the main README, "A different checkpoint: the uncensored build".
`--mtp-bits 4` and `--keep-fc` exist for experimenting with the draft module; the
defaults (int8, `mtp.fc` quantized) are what was verified.

The two `fetch_*` scripts only download: the fast variant is the int4-GPTQ lm_head and
drafter plus a draft vocabulary counted over the model's own outputs (worth ~15% in
single-user mode), and `fetch_dflash2.py` is the W4A16 DFlash2 block drafter. Both are
rebuildable from scratch — that is what [drafter/](../drafter/) is.

`bash verify.sh --no-server` checks every step above against the model dir and names
the script to run for whatever is missing. Each in-place script backs up what it
rewrites next to the original (`.bak*`), so a step can be undone without re-downloading
19.5 GB. Why each one is worth doing, with measurements:
[docs/optimizations.md](../docs/optimizations.md).

`docker/prepare.sh` runs on every container start, so a file that a killed run leaves
half-written stops every later start (#195). `atomic_publish.py` holds the write protocol
for these scripts: a temp file and a rename, the first backup kept, the safetensors index
written last. `bench/test_prepare_crash.py` kills the prepare sequence after each write,
on a small synthetic model, and checks that the next start completes it:

```bash
venv/bin/python bench/test_prepare_crash.py harden translate   # CPU, Linux
```
