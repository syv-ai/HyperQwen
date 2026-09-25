# The patch series, one line each

What every file in `patches/` (and `kvarn/`) is, where it came from, and what retires it. The Dockerfile applies
them in the order of `patches/series` onto the installed vLLM wheel; `verify.sh` checks each one is in place. Kinds:

- **backport**: a merged or open upstream change carried early. Retires when the pin carries it.
- **fix**: a defect in upstream or in this stack, fixable upstream. Retires when upstream takes it.
- **feature**: something upstream does not have. Stays until upstreamed as a feature.
- **local**: this hardware or environment (WSL2, sm80, a tuned build, env knobs). Stays.
- **own**: a fix to a feature this repo introduced. Rides with that feature.

Cut against: the pin the current hunks were generated on. Every file except the retired `dflash2-backport` is exported from its commit on one fork branch, `cpuchip/vllm` **`qwen38/0.30`** (v0.30.0 + one commit per row, in series order, subject `[qwen38] <topic>`; export point tagged `qwen38/0.30-cut5`, so a later rewrite of the branch never orphans a hash these files name), so the series applies to the 0.30.0 tree with exact context; the Dockerfile, `patches/check_vllm_series.sh`,
`kvarn/install.sh` and `verify.sh` apply and check with `--fuzz 0`, and a hunk whose context has moved fails the
build by name instead of landing by guess. Regenerate a file with `bash scripts/export-patch.sh <fork checkout>
<commit> patches/<topic>.patch`; do not edit the files by hand. A patch that reads an env knob registers it in
`envs.py` in its own hunk (so the knob is in the torch.compile cache key), and reads it through `vllm.envs`.

| patch | kind | what | upstream | cut against | retires when |
|---|---|---|---|---|---|
| auth-deny-default | fix | --api-key guards every path except /health, /ping, /load and /version (deny by default). The old prefix list left /tokenize, /detokenize, /metrics and the docs open without the key. /metrics now needs the key: a scraper that cannot send it must use a separate listener, and the in-tree scrapes send it (bench-probe-errors). The allowlist ignores a trailing slash, so a probe on /health/ does not get a 401. Only a real CORS preflight (OPTIONS with Origin and Access-Control-Request-Method) skips the token; a bare OPTIONS needs it, because /metrics answers any method. The --api-key help text describes the allowlist | vllm #58028 | 0.30.0: cut against 0.29.0 and applies as cut (authenticate.py is unchanged; the cli_args.py help-text hunk lands at an offset) | upstream PR |
| bench-probe-errors | fix | `vllm bench serve`'s /tokenize alignment probe sends the API key (Bearer from OPENAI_API_KEY, --header wins) and classifies its failure (404 route-or-name vs 401 vs unreachable vs timeout) instead of one "endpoint unavailable" line for every cause; the /metrics scrapes (`fetch_spec_decode_metrics`, `fetch_diffusion_metrics`) send the benchmark's headers too, so a keyed server no longer reports the spec-decode block as absent | vllm #58024 | 0.30.0 | upstream PR |
| dflash2-backport | backport, RETIRED | DFlash2 speculator on 0.27.1 | vllm #52816 (in 0.28.0) | 0.27.1 | done; kept for history, skipped by the Dockerfile |
| dflash2-lookup-drafting | feature | lookup-augmented drafting for DFlash2 (n-gram search over the context) | none | 0.30.0 | upstreamed |
| dflash2-ngram-chains | feature | quantized candidate chains for the drafter; `propose` override | none | 0.30.0 | upstreamed |
| dflash2-prewarm | fix | compile every DFlash2 rung at boot instead of at first request | none yet | 0.30.0 | upstream PR |
| dflash2-z-adaptive-emitted | fix | adaptive z counts emitted tokens, not sampling slots | none yet | 0.30.0 | upstream PR |
| dspark-draft-quant-config | fix | bf16 DSpark drafter beside a quantized target (callable `hf_overrides`) | none yet | 0.30.0 | upstream PR |
| hybrid-kv-groups-v2-cudagraph | fix | KV group sizing when the smallest bucket is the drafter's sliding-window layers | none yet | 0.30.0 | upstream PR |
| hybrid-sw-block-promote | fix | promote a draft SW layer's block to a divisor of the primary block instead of padding its page | none yet (upstream pads) | 0.30.0: re-cut from the main-track resolution against upstream #53007; #142's divisor condition carried | upstream PR |
| int4-kv-per-token-head | feature | int4 per-token-head KV cache with the DFlash2 drafter | none | 0.30.0 | upstreamed |
| mamba-align-checkpoint-order | fix | keep reachable Mamba state snapshots alive until request end (fork #52) | vllm #45238 (not merged) | 0.30.0: re-cut from the main-track resolution | check against upstream #52789 (internal prefill checkpoints, in 0.29) at each pin |
| mamba-chunked-prefill-align | fix | state loss and NaN during chunked prefill on Mamba/GDN | none yet | 0.30.0 | upstream PR |
| marlin-int8-asym-zp | fix | the Marlin int8-activation path (`INT8_ACT=int8`) accepts zero-point `uint4` weights, so asymmetric AWQ exports (compressed-tensors `symmetric: false`) run W4A8 like the symmetric ones; the `kS8 x kU4` kernel is already compiled, only two asserts refused it | none yet | 0.30.0 | upstream PR |
| marlin-int8-layer-select | local | env vars to pick which layers run W4A8 with the Marlin kernel | none | 0.30.0 | stays |
| marlin-int8-negative-scales | fix | Marlin W4A8 reads group scales as unsigned; AutoRound exports negative ones | none yet | 0.30.0, adapted to #54809 (activation ordering removed: g_idx, perm, is_k_full gone) | upstream PR |
| marlin-repack-staged-sm80 | local | one grow-only staging buffer for the sm80 Marlin repack (fork #27) | none | 0.30.0, adapted to #54809 (activation ordering removed: g_idx, perm, is_k_full gone) | stays |
| marlin-tune-table | local | wiring for a locally built tunable Marlin extension, off by default | none | 0.30.0, adapted to #54809 (activation ordering removed: g_idx, perm, is_k_full gone) | stays |
| offload-dflash-eagle-groups | fix | OffloadingConnector under dflash flagged every KV group as draft attention (fork #33) | none yet | 0.30.0: re-cut from the main-track resolution | upstream PR |
| offload-wsl2-devptr | local | CPU offload tier device pointers on WSL2 | none | 0.30.0 | stays |
| qwen3_5-embed-quant | fix | pass `quant_config` to the token embedding (main model and MTP module) | none yet | 0.30.0 | upstream PR |
| qwen3_5-mtp-draft-vocab | feature | vocab-truncated draft head for MTP | none | 0.30.0 | upstreamed |
| sampler-small-topk-fast-softmax | feature | sort-free top-k/top-p for small k, multi-block row softmax | none | 0.30.0: re-cut from the main-track resolution | upstreamed or superseded |
| spec-decode-attn | feature | split-KV verify attention on FLASH_ATTN with query-row tiling | none | 0.30.0: main-track resolution, envs.py from the 0.29 line (#114), flash_attn.py hand-resolved against #55768 | upstreamed |
| engine-completion-log | feature | one log line per completed engine step, so a stalled core is visible without scraping stats gaps | upstream PR (syv-ai #94/#110) | 0.30.0 | upstreamed |
| engine-stall-sentinel | feature | daemon thread warns once per episode when no step completes for `VLLM_ENGINE_STALL_SENTINEL_S` while requests are live | upstream PR (syv-ai #94/#110) | 0.30.0 | upstreamed |
| topk-honour-flashinfer-sampler-switch | fix | `VLLM_USE_FLASHINFER_SAMPLER=0` also covers the drafter's candidate top-k, which `_flashinfer_topk()` did not gate | none yet (syv-ai #106 B1) | 0.30.0 | upstream takes it |
| memory-profile-after-warmup | fix | run `profile_run` once before the memory-profiling window, synchronize and empty the allocator cache, so a cold compile cache's scratch is not counted as transient peak and the KV cache the warm boot grants is not refused | none yet | 0.30.0 | upstream profiles after warmup |
| cudagraph-memory-from-allocator | fix | measure captured CUDA-graph memory by the allocator's reserved bytes and log the driver's free-memory delta beside it; under WSL2's driver that delta reads zero once the KV cache fills the budget and collapses by 5.44 GiB during a cold compile, which the graph estimate subtracted from the KV budget and refused the CTX=huge first boot | none yet | 0.30.0, hand-resolved against #54646 (both readings inside the gc-freeze block) | upstream measures by the allocator |
| compile-key-runtime-knobs | fix | keeps this repo's runtime-only env knobs (`VLLM_ENGINE_STALL_SENTINEL_S`, `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS`, `VLLM_DFLASH2_CHAIN_LOG_SEC`, `VLLM_MARLIN_TUNE_DIR`) and the deprecated `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` out of `compile_factors()`, so changing one no longer forces a cold torch.compile (#183) | none | 0.30.0 | stays while the knobs exist |
| serve-404-served-names | fix | the model-not-found 404 lists the served names (`Served models: ...`) so a misnamed model is a one-read response body | vllm #58025 | 0.30.0 | upstream PR |
| serve-model-path-match | fix | a model name equal to a served model's root path or its basename is accepted (exact matches only): /v1/models publishes the root, and echoing it back used to 404 | vllm #58026 | 0.30.0 | upstream PR |
| tokenize-v1-route | feature | /tokenize and /detokenize also served under /v1 for OpenAI-SDK base_urls; operation ids stay unique (name+path+method) | vllm #58027 | 0.30.0 | upstream PR |
| triton-spec-attn-fp8-kv | feature | split-KV verify attention on the per-tensor fp8 KV cache (TRITON_ATTN, sm89+); registers `VLLM_SPEC_ATTN_DEBUG` | none | 0.30.0 | upstreamed |
| spec-decode-int4-kv-mq3d | feature | multi-query 3D int4 verify path | none | 0.30.0 | rides with int4-kv-per-token-head |
| spec-decode-int8-kv | feature | split-KV verify attention over an int8 per-token-head cache | none | 0.30.0 | rides with spec-decode-attn |
| spec-decode-scratch-token-units | own | mq3d scratch sized in tokens, not sequences (fork #46, #57) | none | 0.30.0 | rides with mq3d |
| spec-decode-scratch-within-budget | own | mq3d scratch allocated inside the memory budget (fork #57) | none | 0.30.0 | rides with mq3d |
| spec-sampler-prewarm | fix | compile the rejection sampler's Triton kernels at boot (fork #48) | none yet | 0.30.0, hand-resolved: upstream #56323 warms the V1 sampler's kernels, not the V2 runner's this patch warms | upstream PR |
| speed-knobs-envs | local | register this repo's env knobs in `envs.py` | none | 0.30.0 | stays while the knobs exist |
| prefill-attn-int8 | feature | int8-QK Triton prefill attention for head_dim 256 | none | 0.30.0 | upstreamed |
| vision-tower-cpu-offload | local | Qwen3 vision tower bulk weights in host RAM | none | 0.30.0 | stays |
| vllm-pr50021-gdn-spec-bounds | backport | bounds checks in GDN/KDA spec-decode state lookups | vllm #50021 (open) | 0.30.0 | the pin that carries #50021 |
| kvarn/kvarn-0.30.0 | feature | KVarN cache dtypes, quant mode, backend registration, page size | none (KVarN is Huawei CSL's, Apache-2.0) | 0.30.0: re-cut from the main-track resolution, with its #54713 replay_boundaries fixup | upstreamed |
| kvarn/kvarn-v2-runner-0.30.0 | own | KVarN with the V2 runner and DFlash2 (SW groups, Mamba block index, selector guards) | none | 0.30.0: re-cut from the main-track resolution, with its #54713 replay_boundaries fixup; #53007 rewrote _largest_kernel_block_within and the SW divisor rule is carried into it by hand | rides with KVarN |

Retired at 0.30.0 and removed from the tree: `offload-mtp-serve` (vllm #52771, #52807 and #54288, all in 0.30.0) and `mamba-align-retire-null-gaps` (vllm #55450, in 0.30.0).

Retired at 0.29.0 and removed from the tree: `vllm-pr54282-draft-gumbel-salt` (vllm #54282, in 0.29.0),
`xgrammar-spec-terminated` (in 0.29.0), and `sse-keep-alive` (vllm 585bb07c7, in 0.29.0 and not in
0.28.0; the `--sse-keep-alive-interval` flag is unchanged, so nothing that sets it needs to change).

Retired on 0.29 for a different reason, and temporarily: `int4-mq3d-envs`: its two registrations
(`VLLM_INT4_MQ_3D`, `VLLM_INT4_MQ_3D_DEBUG`) already exist on this line in `speed-knobs-envs`, and
this line's readers already go through `vllm.envs`, so applying it duplicates them and fails at
`--fuzz 0`. The #114 restructure recreates it as its own topic, moving those registrations out of
`speed-knobs-envs` rather than adding a second copy, before the pin-flip PR. Until then the 0.28
and 0.29 shapes differ here by design.

Two files still carry raw `diff -ruN` headers with timestamps instead of a preamble (`dflash2-z-adaptive-emitted`,
`offload-wsl2-devptr`); their descriptions live in `docs/gotchas.md` and `docs/MR-DRAFT.md` until they get one.
