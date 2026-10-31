# Exact interior GDN prefill checkpoints

`VLLM_GDN_PREFILL_CHECKPOINTS=1` enables the GDN checkpoint plumbing introduced
in PR #392, re-cut for the current vLLM 0.32 pin. It is off by default. It requires `mamba_cache_mode=align` and the
CUDA Triton/FLA GDN prefill backend. Unsupported configurations fail before
checkpoint allocation. It does not enable checkpoints for KDA or other backends.

The allocator selects a position strictly inside the current prefill query.
The exporter saves the raw convolution history at that position, including the
initial history when the prefix is shorter than the convolution window. It
recomputes the selected recurrent prefix with the existing FLA chunk kernel and
stores its final state in the reserved checkpoint page. The ordinary full-query
forward still produces the output and current request's end state.

This implementation recomputes prefixes to provide exact checkpoint states.
It adds GPU work and temporary tensors; it is not a throughput optimization
claim. The planner reads the small list of destination page IDs back once per
metadata build to exclude null pages that the scheduler did not reserve, such
as intermediate prefill chunks. There is no such synchronization per layer.
Metadata containing
checkpoint destinations is not shared across distinct cache groups.

Run `python bench/test_gdn_checkpoint_safety.py` for allocation/dispatch source
regressions and `python bench/test_gdn_checkpoint_export.py` in the patched image
for planner and convolution history checks. Add `--device cuda:0` to run the
actual FLA prefix-state and resumed-suffix numerical checks.

For the real two-GPU model replay, run
`python bench/test_gdn_checkpoint_model.py --variant enabled --output-dir /tmp/gdn-replay`
then the same command with `--variant default-off` in a fresh process.
Both arms must see the same cached weights and tokenizer. Add
`--variant enabled-dflash` / `default-off-dflash` for seven DFlash2 drafts.
Use `--compare-only` to check the saved input hashes and output tokens without
loading a model. `--require-off-miss` additionally requires the baseline to miss;
it fails when both arms already hit, so identical hits are not a claimed gain.
