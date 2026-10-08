# The patch series, one line each

What every file in `patches/` (and `kvarn/`) is, where it came from, and what retires it. `patches/apply.sh` applies
them in the order of `patches/series` onto the installed vLLM wheel; `verify.sh` checks each one is in place. Kinds:

- **backport**: a merged or open upstream change carried early. Retires when the pin carries it.
- **fix**: a defect in upstream or in this stack, fixable upstream. Retires when upstream takes it.
- **feature**: something upstream does not have. Stays until upstreamed as a feature.
- **local**: this hardware or environment (WSL2, sm80, a tuned build, env knobs). Stays.
- **own**: a fix to a feature this repo introduced. Rides with that feature.

Cut against: the pin the current hunks were generated on. The files are the source. Each contributor keeps the topics as commits (subject `[qwen38] <topic>`) in any vLLM 0.30.0 checkout, and `scripts/export-patch.sh` turns a commit into a file whose `--- exported from` line names that commit, so the series applies to the 0.30.0 tree with exact context; `patches/apply.sh` (which the Dockerfile and the install pages call),
`patches/check_vllm_series.sh`, `kvarn/install.sh` and `verify.sh` apply and check with `--fuzz 0`, and a hunk whose context has moved fails the
build by name instead of landing by guess. Regenerate a file with `bash scripts/export-patch.sh <vllm checkout>
<commit> patches/<topic>.patch`; do not edit the files by hand. A patch that reads an env knob registers it in
`envs.py` in its own hunk (so the knob is in the torch.compile cache key), and reads it through `vllm.envs`.

The table. Each file ends its preamble with its row as headers: `Kind:`, `What:`, `Upstream:`, `Cut-against:` and
`Retires-when:`, one per line, and a long value continues on lines that start with spaces. `python3 scripts/patches_md.py`
writes the table below from them, in apply order, and CI fails when the table is not current. To change a row, change the
headers in the patch file (the files are the source) and run the script.

<!-- table:begin. scripts/patches_md.py writes this table from the patch headers; do not edit it -->
| patch | kind | what | upstream | cut against | retires when |
|---|---|---|---|---|---|
| qwen3_5-embed-quant | fix | pass `quant_config` to the token embedding (main model and MTP module) | none yet | 0.30.0 | upstream PR |
| marlin-int8-layer-select | local | env vars to pick which layers run W4A8 with the Marlin kernel | none | 0.30.0 | stays |
| marlin-int8-negative-scales | fix | Marlin W4A8 reads group scales as unsigned; AutoRound exports negative ones | none yet | 0.30.0, adapted to #54809 (activation ordering removed: g_idx, perm, is_k_full gone) | upstream PR |
| qwen3_5-mtp-draft-vocab | feature | vocab-truncated draft head for MTP | none | 0.30.0 | upstreamed |
| vllm-pr50021-gdn-spec-bounds | backport | bounds checks in GDN/KDA spec-decode state lookups | vllm #50021 (open) | 0.30.0 | the pin that carries #50021 |
| sampler-small-topk-fast-softmax | feature | sort-free top-k/top-p for small k, multi-block row softmax | none | 0.30.0: re-cut from the main-track resolution | upstreamed or superseded |
| spec-decode-attn | feature | split-KV verify attention on FLASH_ATTN with query-row tiling | none | 0.30.0: main-track resolution, envs.py from the 0.29 line (#114), flash_attn.py hand-resolved against #55768 | upstreamed |
| speed-knobs-envs | local | register this repo's env knobs in `envs.py` | none | 0.30.0 | stays while the knobs exist |
| hybrid-kv-groups-v2-cudagraph | fix | KV group sizing when the smallest bucket is the drafter's sliding-window layers | none yet | 0.30.0 | upstream PR |
| dflash2-lookup-drafting | feature | lookup-augmented drafting for DFlash2 (n-gram search over the context) | none | 0.30.0 | upstreamed |
| hybrid-sw-block-promote | fix | promote a draft SW layer's block to a divisor of the primary block instead of padding its page | none yet (upstream pads) | 0.30.0: re-cut from the main-track resolution against upstream #53007; #142's divisor condition carried | upstream PR |
| spec-decode-int8-kv | feature | split-KV verify attention over an int8 per-token-head cache | none | 0.30.0 | rides with spec-decode-attn |
| vision-tower-cpu-offload | local | Qwen3 vision tower bulk weights in host RAM | none | 0.30.0 | stays |
| int4-kv-per-token-head | feature | int4 per-token-head KV cache with the DFlash2 drafter | none | 0.30.0 | upstreamed |
| marlin-repack-staged-sm80 | local | one grow-only staging buffer for the sm80 Marlin repack (fork #27), opt-in with `VLLM_MARLIN_REPACK_STAGED=1` | none | 0.30.0, adapted to #54809 (activation ordering removed: g_idx, perm, is_k_full gone) | stays |
| offload-dflash-eagle-groups | fix | OffloadingConnector under dflash flagged every KV group as draft attention (fork #33) | none yet | 0.30.0: re-cut from the main-track resolution | upstream PR |
| dflash2-ngram-chains | feature | quantized candidate chains for the drafter; `propose` override | none | 0.30.0 | upstreamed |
| offload-wsl2-devptr | local | CPU offload tier device pointers on WSL2 | none | 0.30.0 | stays |
| spec-decode-int4-kv-mq3d | feature | multi-query 3D int4 verify path | none | 0.30.0 | rides with int4-kv-per-token-head |
| dflash2-prewarm | fix | compile every DFlash2 rung at boot instead of at first request | none yet | 0.30.0 | upstream PR |
| marlin-tune-table | local | wiring for a locally built tunable Marlin extension, off by default | none | 0.30.0, adapted to #54809 (activation ordering removed: g_idx, perm, is_k_full gone) | stays |
| prefill-attn-int8 | feature | int8-QK Triton prefill attention for head_dim 256 | none | 0.30.0 | upstreamed |
| spec-sampler-prewarm | fix | compile the rejection sampler's Triton kernels at boot (fork #48) | none yet | 0.30.0, hand-resolved: upstream #56323 warms the V1 sampler's kernels, not the V2 runner's this patch warms | upstream PR |
| mamba-align-checkpoint-order | fix | keep reachable Mamba state snapshots alive until request end (fork #52) | vllm #45238 (not merged) | 0.30.0: re-cut from the main-track resolution | check against upstream #52789 (internal prefill checkpoints, in 0.29) at each pin |
| spec-decode-scratch-token-units | own | mq3d scratch sized in tokens, not sequences (fork #46, #57) | none | 0.30.0 | rides with mq3d |
| mamba-chunked-prefill-align | fix | state loss and NaN during chunked prefill on Mamba/GDN | none yet | 0.30.0 | upstream PR |
| spec-decode-scratch-within-budget | own | mq3d scratch allocated inside the memory budget (fork #57) | none | 0.30.0 | rides with mq3d |
| dflash2-z-adaptive-emitted | fix | adaptive z counts emitted tokens, not sampling slots | none yet | 0.30.0 | upstream PR |
| dspark-draft-quant-config | fix | bf16 DSpark drafter beside a quantized target (callable `hf_overrides`) | none yet | 0.30.0 | upstream PR |
| engine-completion-log | feature | one log line per completed engine step, so a stalled core is visible without scraping stats gaps | upstream PR (syv-ai #94/#110) | 0.30.0 | upstreamed |
| engine-stall-sentinel | feature | daemon thread warns once per episode when no step completes for `VLLM_ENGINE_STALL_SENTINEL_S` while requests are live | upstream PR (syv-ai #94/#110) | 0.30.0 | upstreamed |
| triton-spec-attn-fp8-kv | feature | split-KV verify attention on the per-tensor fp8 KV cache (TRITON_ATTN, sm89+); registers `VLLM_SPEC_ATTN_DEBUG` | none | 0.30.0 | upstreamed |
| topk-honour-flashinfer-sampler-switch | fix | `VLLM_USE_FLASHINFER_SAMPLER=0` also covers the drafter's candidate top-k, which `_flashinfer_topk()` did not gate | none yet (syv-ai #106 B1) | 0.30.0 | upstream takes it |
| memory-profile-after-warmup | fix | run `profile_run` once before the memory-profiling window, synchronize and empty the allocator cache, so a cold compile cache's scratch is not counted as transient peak and the KV cache the warm boot grants is not refused | none yet | 0.30.0 | upstream profiles after warmup |
| cudagraph-memory-from-allocator | fix | measure captured CUDA-graph memory by the allocator's reserved bytes and log the driver's free-memory delta beside it; under WSL2's driver that delta reads zero once the KV cache fills the budget and collapses by 5.44 GiB during a cold compile, which the graph estimate subtracted from the KV budget and refused the CTX=huge first boot | none yet | 0.30.0, hand-resolved against #54646 (both readings inside the gc-freeze block) | upstream measures by the allocator |
| bench-probe-errors | fix | `vllm bench serve`'s /tokenize alignment probe sends the API key (Bearer from OPENAI_API_KEY, --header wins) and classifies its failure (404 route-or-name vs 401 vs unreachable vs timeout) instead of one "endpoint unavailable" line for every cause; the /metrics scrapes (`fetch_spec_decode_metrics`, `fetch_diffusion_metrics`) send the benchmark's headers too, so a keyed server no longer reports the spec-decode block as absent | vllm #59888 | 0.30.0 | upstream PR |
| serve-404-served-names | fix | the model-not-found 404 lists the served names (`Served models: ...`) so a misnamed model is a one-read response body | vllm #59889 (merged to main as `7867d6c52d`, not in a release yet; upstream says `Valid aliases: ...`) | 0.30.0 | the pin carries vllm #59889 |
| serve-model-path-match | fix | a model name equal to a served model's root path or its basename is accepted (exact matches only): /v1/models publishes the root, and echoing it back used to 404 | vllm #59890 | 0.30.0 | upstream PR |
| tokenize-v1-route | feature | /tokenize and /detokenize also served under /v1 for OpenAI-SDK base_urls; operation ids stay unique (name+path+method) | none (vllm #59891 was closed: upstream keeps `/v1` for the official OpenAI endpoints) | 0.30.0 | stays |
| marlin-int8-asym-zp | fix | the Marlin int8-activation path (`INT8_ACT=int8`) accepts zero-point `uint4` weights, so asymmetric AWQ exports (compressed-tensors `symmetric: false`) run W4A8 like the symmetric ones; the `kS8 x kU4` kernel is already compiled, only two asserts refused it | none yet | 0.30.0 | upstream PR |
| compile-key-runtime-knobs | fix | keeps this repo's runtime-only env knobs (`VLLM_ENGINE_STALL_SENTINEL_S`, `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS`, `VLLM_DFLASH2_CHAIN_LOG_SEC`, `VLLM_MARLIN_TUNE_DIR`) and the deprecated `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` out of `compile_factors()`, so changing one no longer forces a cold torch.compile (#183) | none | 0.30.0 | stays while the knobs exist |
| auth-deny-default | fix | --api-key guards every path except /health, /ping, /load and /version (deny by default). The old prefix list left /tokenize, /detokenize, /metrics and the docs open without the key. /metrics now needs the key: a scraper that cannot send it must use a separate listener, and the in-tree scrapes send it (bench-probe-errors). The allowlist ignores a trailing slash, so a probe on /health/ does not get a 401. Only a real CORS preflight (OPTIONS with Origin and Access-Control-Request-Method) skips the token; a bare OPTIONS needs it, because /metrics answers any method. The --api-key help text describes the allowlist | vllm #59892 | 0.30.0: cut against 0.29.0 and applies as cut (authenticate.py is unchanged; the cli_args.py help-text hunk lands at an offset) | upstream PR |
| spec-attn-smem-fit | fix | the split-KV verify attention sizes its KV tile to the device's shared memory: halve the KV tile when Triton reports OutOfResources, so it launches on Turing (sm75, 64 KB per block) | none | 0.30.0: hunk 1 (the import) re-placed by hand, hunks 2-4 at offset -1 | upstream with spec-decode-attn |
| bench-sse-keepalive | fix | `vllm bench serve` no longer fails a request whose server sends an SSE keep-alive before the first token: the request functions stripped each network chunk, which deleted the blank line between SSE messages and glued the `: keep-alive` comment to every message after it. With the launchers' `--sse-keep-alive-interval 30`, that failed every prompt whose prefill took over 30 s (`run_benchmarks.sh --long` read zeros; #216 lost its 48k+ rows). The audio request function also skips SSE comments now | none yet | 0.30.0 | upstream PR |
| sampler-warmup-cuda | backport | registers the V2 runner's top-k/top-p sampler JIT warmups on CUDA too (vllm #58092 without #58465's ROCm-only gate): 0 in-request compiles, ~71-78 s of warmup once per cold cache volume, ~0.2 s warm | vllm #58092 (merged after 0.30.0) | 0.30.0 | the pin that carries #58092 with CUDA registration (upstream gates it to ROCm; an opt-in is the ask) |
| pinned-kv-empty-cache | fix | with `kv_cache_memory_bytes` pinned (the launchers' `KV_MEM`), synchronize and empty the allocator cache after the profile run, so a cold compile cache's scratch (1.2 GiB on a 24 GiB card) does not stay under the KV cache and over-commit the card, which under WSL2's driver moves GPU memory to system RAM and slows the KVarN kernels; `memory-profile-after-warmup` does the same on the measured path only | vllm #59893 | 0.30.0 | upstream empties the cache on the pinned path |
| api-root-health | feature | GET/HEAD /, /api/status, /api/experimental/model-recommendations, and /api/show for `ollama launch codex` compatibility. Matches served models dynamically, returns 404 for unserved models (the body names the requested model only, never a served name or path), reads context length from model_config.max_model_len, and adds the routes to UNGUARDED_PATHS | none yet | 0.30.0 | upstream PR |
| kvarn/kvarn-0.30.0 | feature | KVarN cache dtypes, quant mode, backend registration, page size | none (KVarN is Huawei CSL's, Apache-2.0) | 0.30.0: re-cut from the main-track resolution, with its #54713 replay_boundaries fixup | upstreamed |
| kvarn/kvarn-v2-runner-0.30.0 | own | KVarN with the V2 runner and DFlash2 (SW groups, Mamba block index, selector guards) | none | 0.30.0: re-cut from the main-track resolution, with its #54713 replay_boundaries fixup; #53007 rewrote _largest_kernel_block_within and the SW divisor rule is carried into it by hand | rides with KVarN |
| kvarn/kvarn-recycled-pages-0.30.0 | own | both runners hand KVarN each step's block ids, so it drops (never flushes) what it still holds for a page another KV-cache group has taken: a late flush of a finished request's last block, or of an evicted retired sink, into another request's mamba state was the "!!!!" output (#208); the KVarN half is in `kvarn/files` | none | 0.30.0 | rides with KVarN |
| kvarn/kvarn-fp16-dequant-0.30.0 | own | registers `KVARN_FP16_DEQUANT` (#240's fp16 dequant in the fused KVarN decode kernels, default off) in `envs.py`, so `kvarn/files` reads it through `vllm.envs` and it is in the torch.compile cache key | none | 0.30.0 | rides with KVarN |
<!-- table:end -->

Retired at 0.30.0 and removed from the tree: `offload-mtp-serve` (vllm #52771, #52807 and #54288, all in 0.30.0) and `mamba-align-retire-null-gaps` (vllm #55450, in 0.30.0).

Retired at 0.28.0 and removed from the tree later, with the older KVarN ports: `dflash2-backport` (the DFlash2
speculator on 0.27.1, vllm #52816, native since 0.28.0; every apply site skipped it), `kvarn/kvarn-0.27.1.patch` and
`kvarn/kvarn-v2-runner.patch` (the 0.27.1 KVarN ports; `kvarn/install.sh` applied neither). Git history keeps
them.

Retired at 0.29.0 and removed from the tree: `vllm-pr54282-draft-gumbel-salt` (vllm #54282, in 0.29.0),
`xgrammar-spec-terminated` (in 0.29.0), and `sse-keep-alive` (vllm 585bb07c7, in 0.29.0 and not in
0.28.0; the `--sse-keep-alive-interval` flag is unchanged, so nothing that sets it needs to change).

Retired on 0.29 for a different reason, and temporarily: `int4-mq3d-envs`: its two registrations
(`VLLM_INT4_MQ_3D`, `VLLM_INT4_MQ_3D_DEBUG`) already exist on this line in `speed-knobs-envs`, and
this line's readers already go through `vllm.envs`, so applying it duplicates them and fails at
`--fuzz 0`. The #114 restructure recreates it as its own topic, moving those registrations out of
`speed-knobs-envs` rather than adding a second copy, before the pin-flip PR. Until then the 0.28
and 0.29 shapes differ here by design.

Six files have one line of prose or none above their headers, because their fork commit bodies were that short.
`dflash2-prewarm` and `dflash2-z-adaptive-emitted` have none. `dflash2-lookup-drafting`, `offload-wsl2-devptr`,
`spec-decode-int4-kv-mq3d` and `vllm-pr50021-gdn-spec-bounds` have one line. Their `What:` headers describe them, and
`docs/wsl2-4090.md` ("CPU offload tier under WSL2") explains `offload-wsl2-devptr`. Prose comes from the fork commit
body and a re-export, not from an edit to the file.
