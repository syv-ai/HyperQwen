# GDN checkpoints inside a prefill

`VLLM_GDN_PREFILL_CHECKPOINTS=1` opts the CUDA Triton/FLA GDN backend into exact
checkpoints at periodic retention boundaries inside each prefill. It is off by
default. It requires `mamba_cache_mode="align"` and either dense retention
(`prefix_cache_retention_interval=None`) or a positive interval that is a
multiple of `mamba_block_size`. Retention 0, other cache modes and unsupported
backends fail before allocating checkpoint pages. On vLLM 0.33 the opt-in remains Triton/FLA only: selecting FlashInfer with
this knob raises before allocation. The upstream FlashInfer single-checkpoint
planner, kernel emission and exporter are preserved with the knob off; this
patch does not extend periodic export to that backend. Other Mamba/KDA backends
keep their existing behavior.

The single-checkpoint version in #425 saved only the final reusable hash
boundary, which the default-off scheduler already materialized. This version
also saves earlier periodic boundaries. A later request that diverges inside
the preceding prefill can therefore resume from an otherwise absent state.

The manager and worker use the same position helper. Periodic states occupy
their normal block-table columns; a separate final hash checkpoint can borrow
one column, which is excluded from periodic export. The planner never writes
the query's initial-state page or a null destination. Already assigned worker
columns retain their physical page when speculative scratch becomes a
checkpoint. Hashes refer to the exported positions, not the end of the full
query. Admission probes leave the pending exports unchanged.

Each export saves raw convolution history and recomputes the selected
normalized recurrent prefix using the existing FLA kernel. The ordinary
full-query forward still produces the output and running state. Prefixes from
the same source row are supported. Temporary FLA calls contain at most the
current prefill batch's token count, rather than concatenating every duplicated
prefix into one potentially quadratic allocation. The persistent page budget
includes all checkpoints for each in-flight batch. The planner reads the small
destination list back once per metadata build; it adds no per-layer readback.

This adds recomputation, pages and latency to cold prefill. Cache reuse and
correctness must be measured separately from that cost; it is not a general
throughput optimization. More frequent retention boundaries increase both the
number of reusable states and the export work. Production defaults are unchanged.

CPU regressions, also run in the Docker build:

```bash
python bench/test_gdn_checkpoint_safety.py
python bench/test_gdn_multi_checkpoint.py
python bench/test_gdn_checkpoint_export.py
```

The metadata-only CPU cache comparison cannot establish numerical correctness.
Run the actual FLA prefix-state/resumed-suffix checks on a free CUDA GPU:

```bash
python bench/test_gdn_checkpoint_export.py --device cuda:0
```

For full-model TP=2 validation, run each arm in a fresh process against the same
cached weights, with two free GPUs. The model probe sets dense retention in
both arms, greedy sampling, a fixed GPU KV budget and frozen prompt token IDs.
Its default prompt contains 2000 material tokens; the second request shares
1200, then changes the suffix, inside the first shipped 2048-token chunk.
These lengths are fixed before inference and are not tuned to exported offsets.
Use `--chunk-size 8192 --material-tokens 8000 --shared-tokens 4800` to reproduce
the historical larger-chunk profile, and `--tensor-parallel-size 1` on a single
card with a suitably sized KV budget. `--require-multiple-export` separately
requires multiple states in one prefill; a smaller chunk may correctly export
only one, which is not an inference error.
The default pool is 4,000,000,000 bytes per GPU (`--kv-cache-bytes` overrides it);
the GDN checkpoint pages and DFlash scratch must fit as well as attention KV.

```bash
python bench/test_gdn_checkpoint_model.py --variant enabled --output-dir /tmp/gdn-replay
python bench/test_gdn_checkpoint_model.py --variant default-off --output-dir /tmp/gdn-replay
python bench/test_gdn_checkpoint_model.py --variant enabled --output-dir /tmp/gdn-replay --compare-only --require-off-miss
```

Repeat with `enabled-dflash` / `default-off-dflash` for actual DFlash2 k=7.
The comparison checks identical parameters, model/tokenizer file hashes,
probe source hashes, prompt hashes and output tokens, correct
passcodes, multiple checkpoints on every TP rank, and an enabled-only hit.
It records elapsed request time, cache hits and peak GPU memory separately;
a warm repeat is not the deep-divergence verdict.
