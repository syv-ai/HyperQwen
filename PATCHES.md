# The patch series, one line each

What every file in `patches/` (and `kvarn/`) is, where it came from, and what retires it. `patches/apply.sh` applies
them in the order of `patches/series` onto the installed vLLM wheel; `verify.sh` checks each one is in place. Kinds:

- **backport**: a merged or open upstream change carried early. Retires when the pin carries it.
- **fix**: a defect in upstream or in this stack, fixable upstream. Retires when upstream takes it.
- **feature**: something upstream does not have. Stays until upstreamed as a feature.
- **local**: this hardware or environment (WSL2, sm80, a tuned build, env knobs). Stays.
- **own**: a fix to a feature this repo introduced. Rides with that feature.

`fa2/` carries a diff against another repository (the fp8-KV FA2 kernel against vllm-project/flash-attention at the commit vLLM pins), so it is not in `patches/` or this table. `fa2/README.md` has its provenance, and the `fa2-upstream` job in `patch-integrity.yml` checks it against that commit.

Cut against: the pin the current hunks were generated on. The files are the source. Check out the tag `v<version>` for the vLLM version pinned in `docker/requirements.txt` before authoring a topic; historical version numbers in benchmark notes are not the patch base. Each contributor keeps the topics as commits (subject `[qwen38] <topic>`) in that pinned checkout, and `scripts/export-patch.sh` turns a commit into a file whose `--- exported from` line names that commit, so the series applies to the pinned tree with exact context; `patches/apply.sh` (which the Dockerfile and the install pages call),
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
| qwen3_5-embed-quant | fix | pass `quant_config` to the token embedding (main model and MTP module) | none yet | 0.32.0 | upstream PR |
| marlin-int8-layer-select | local | env vars to pick which layers run W4A8 with the Marlin kernel | none | 0.32.0 | stays |
| marlin-int8-negative-scales | fix | Marlin W4A8 reads group scales as unsigned; AutoRound exports negative ones | none yet | 0.32.0 | upstream PR |
| qwen3_5-mtp-draft-vocab | feature | vocab-truncated draft head for MTP | none | 0.32.0: the draft_lm_head skip follows #60152's is_unused_checkpoint_weight check in remap_weight_names | upstreamed |
| vllm-pr50021-gdn-spec-bounds | backport | bounds checks in GDN/KDA spec-decode state lookups | vllm #50021 (open) | 0.32.0 | the pin that carries #50021 |
| sampler-small-topk-fast-softmax | feature | sort-free top-k/top-p for small k, multi-block row softmax | none | 0.32.0 | upstreamed or superseded |
| spec-decode-attn | feature | split-KV verify attention on FLASH_ATTN with query-row tiling | none | 0.32.0: the split-KV branch re-indented after #61126 removed cascade attention; its helpers follow the kernel singleton | upstreamed |
| speed-knobs-envs | local | register this repo's env knobs in `envs.py` | none | 0.32.0 | stays while the knobs exist |
| hybrid-kv-groups-v2-cudagraph | fix | KV group sizing when the smallest bucket is the drafter's sliding-window layers | none yet | 0.32.0 | upstream PR |
| dflash2-lookup-drafting | feature | lookup-augmented drafting for DFlash2 (n-gram search over the context) | none | 0.32.0: the lookup's draft count is clamped to the step's num_spec_tokens, which #57053 (dynamic K) now slices the drafts to; with dynamic K off the two are equal. A None from next_num_draft_tokens() (all of them) stays None, which set_draft_tokens reads as the sliced width | upstreamed |
| hybrid-sw-block-promote | fix | promote a draft SW layer's block to a divisor of the primary block instead of padding its page | none yet (upstream pads) | 0.32.0: re-anchored after #57169 deleted the GLM5-next helpers it sat beside; only _promote_indivisible_block_sizes is added | upstream PR |
| spec-decode-int8-kv | feature | split-KV verify attention over an int8 per-token-head cache | none | 0.32.0 | rides with spec-decode-attn |
| vision-tower-cpu-offload | local | Qwen3 vision tower bulk weights in host RAM | none | 0.32.0 | stays |
| int4-kv-per-token-head | feature | int4 per-token-head KV cache with the DFlash2 drafter | none | 0.32.0 | upstreamed |
| marlin-repack-staged-sm80 | local | one grow-only staging buffer for the sm80 Marlin repack (fork #27), opt-in with `VLLM_MARLIN_REPACK_STAGED=1` | none | 0.32.0 | stays |
| offload-dflash-eagle-groups | fix | OffloadingConnector under dflash flagged every KV group as draft attention (fork #33) | none yet | 0.32.0 | upstream PR |
| dflash2-ngram-chains | feature | quantized candidate chains for the drafter; `propose` override | none | 0.32.0: propose() takes dp_sync_state (#60754 renamed dp_sync) and forwards num_speculative_tokens, which the 0.32 runner passes by keyword on every step | upstreamed |
| offload-wsl2-devptr | local | CPU offload tier device pointers on WSL2 | none | 0.32.0 | stays |
| spec-decode-int4-kv-mq3d | feature | multi-query 3D int4 verify path | none | 0.32.0 | rides with int4-kv-per-token-head |
| dflash2-prewarm | fix | compile every DFlash2 rung at boot instead of at first request | none yet | 0.32.0 | upstream PR |
| marlin-tune-table | local | wiring for a locally built tunable Marlin extension, off by default | none | 0.32.0 | stays |
| prefill-attn-int8 | feature | int8-QK Triton prefill attention for head_dim 256 | none | 0.32.0: the int8 prefill branch re-indented after #61126 removed cascade attention | upstreamed |
| spec-sampler-prewarm | fix | compile the rejection sampler's Triton kernels at boot (fork #48) | none yet | 0.32.0 | upstream PR |
| mamba-align-checkpoint-order | fix | keep reachable Mamba state snapshots alive until request end (fork #52) | vllm #45238 (not merged) | 0.32.0 | check against upstream #52789 (internal prefill checkpoints, in 0.29) at each pin |
| spec-decode-scratch-token-units | own | mq3d scratch sized in tokens, not sequences (fork #46, #57) | none | 0.32.0: carries #58482's SM120 64-segment sizing in token units; spec-decode-scratch-within-budget then turns that mode off | rides with mq3d |
| mamba-chunked-prefill-align | fix | state loss and NaN during chunked prefill on Mamba/GDN | none yet | 0.32.0: the chunk_o.py half is retired (#60502 loads g masked with other=0.0, the lanes it zeroed); the mamba_utils.py half stays | upstream PR |
| spec-decode-scratch-within-budget | own | mq3d scratch allocated inside the memory budget (fork #57) | none | 0.32.0: the pool owns the scratch at 16 segments, so #58482's SM120 64-segment mode is off in TRITON_ATTN (max_seqs_64_segments = 0): RTX 50 cards keep 0.31's 16 segments and lose #58482's occupancy gain; correctness unchanged | rides with mq3d |
| dflash2-z-adaptive-emitted | fix | adaptive z counts emitted tokens, not sampling slots | none yet | 0.32.0 | upstream PR |
| dspark-draft-quant-config | fix | bf16 DSpark drafter beside a quantized target (callable `hf_overrides`) | none yet | 0.32.0 | upstream PR |
| engine-completion-log | feature | one log line per completed engine step, so a stalled core is visible without scraping stats gaps | upstream PR (syv-ai #94/#110) | 0.32.0 | upstreamed |
| engine-stall-sentinel | feature | daemon thread warns once per episode when no step completes for `VLLM_ENGINE_STALL_SENTINEL_S` while requests are live | upstream PR (syv-ai #94/#110) | 0.32.0: beside #50300's VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT; #59781 removed VLLM_ENGINE_ITERATION_TIMEOUT_S, the old anchor | upstreamed |
| triton-spec-attn-fp8-kv | feature | split-KV verify attention on the per-tensor fp8 KV cache (TRITON_ATTN, sm89+); registers `VLLM_SPEC_ATTN_DEBUG` | none | 0.32.0 | upstreamed |
| topk-honour-flashinfer-sampler-switch | fix | `VLLM_USE_FLASHINFER_SAMPLER=0` also covers the drafter's candidate top-k, which `_flashinfer_topk()` did not gate | none yet (syv-ai #106 B1) | 0.32.0 | upstream takes it |
| memory-profile-after-warmup | fix | run `profile_run` once before the memory-profiling window, synchronize and empty the allocator cache, so a cold compile cache's scratch is not counted as transient peak and the KV cache the warm boot grants is not refused | none yet | 0.32.0 | upstream profiles after warmup |
| cudagraph-memory-from-allocator | fix | measure captured CUDA-graph memory by the allocator's reserved bytes and log the driver's free-memory delta beside it; under WSL2's driver that delta reads zero once the KV cache fills the budget and collapses by 5.44 GiB during a cold compile, which the graph estimate subtracted from the KV budget and refused the CTX=huge first boot | none yet | 0.32.0: the reserved-memory sample sits ahead of #59160's capture_pool block | upstream measures by the allocator |
| bench-probe-errors | fix | `vllm bench serve`'s /tokenize alignment probe sends the API key (Bearer from OPENAI_API_KEY, --header wins) and classifies its failure (404 route-or-name vs 401 vs unreachable vs timeout) instead of one "endpoint unavailable" line for every cause; the /metrics scrapes (`fetch_spec_decode_metrics`, `fetch_diffusion_metrics`) send the benchmark's headers too, so a keyed server no longer reports the spec-decode block as absent | vllm #59888 | 0.32.0 | upstream PR |
| serve-model-path-match | fix | a model name equal to a served model's root path or its basename is accepted (exact matches only): /v1/models publishes the root, and echoing it back used to 404 | vllm #59890 | 0.32.0 | upstream PR |
| tokenize-v1-route | feature | /tokenize and /detokenize also served under /v1 for OpenAI-SDK base_urls; operation ids stay unique (name+path+method) | none (vllm #59891 was closed: upstream keeps `/v1` for the official OpenAI endpoints) | 0.32.0 | stays |
| marlin-int8-asym-zp | fix | the Marlin int8-activation path (`INT8_ACT=int8`) accepts zero-point `uint4` weights, so asymmetric AWQ exports (compressed-tensors `symmetric: false`) run W4A8 like the symmetric ones; the `kS8 x kU4` kernel is already compiled, only two asserts refused it | none yet | 0.32.0 | upstream PR |
| compile-key-runtime-knobs | fix | keeps this repo's runtime-only env knobs (`VLLM_ENGINE_STALL_SENTINEL_S`, `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS`, `VLLM_DFLASH2_CHAIN_LOG_SEC`, `VLLM_MARLIN_TUNE_DIR`) and the deprecated `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` out of `compile_factors()`, so changing one no longer forces a cold torch.compile (#183) | none | 0.32.0 | stays while the knobs exist |
| auth-deny-default | fix | --api-key guards every path except /health, /ping, /load and /version (deny by default). The old prefix list left /tokenize, /detokenize, /metrics and the docs open without the key. /metrics now needs the key: a scraper that cannot send it must use a separate listener, and the in-tree scrapes send it (bench-probe-errors). The allowlist ignores a trailing slash, so a probe on /health/ does not get a 401. Only a real CORS preflight (OPTIONS with Origin and Access-Control-Request-Method) skips the token; a bare OPTIONS needs it, because /metrics answers any method. The --api-key help text describes the allowlist | vllm #59892 | 0.32.0 | upstream PR |
| spec-attn-smem-fit | fix | the split-KV verify attention sizes its KV tile to the device's shared memory: halve the KV tile when Triton reports OutOfResources, so it launches on Turing (sm75, 64 KB per block) | none | 0.32.0 | upstream with spec-decode-attn |
| bench-sse-keepalive | fix | `vllm bench serve` no longer fails a request whose server sends an SSE keep-alive before the first token: the request functions stripped each network chunk, which deleted the blank line between SSE messages and glued the `: keep-alive` comment to every message after it. With the launchers' `--sse-keep-alive-interval 30`, that failed every prompt whose prefill took over 30 s (`run_benchmarks.sh --long` read zeros; #216 lost its 48k+ rows). The audio request function also skips SSE comments now | none yet | 0.32.0 | upstream PR |
| sampler-warmup-cuda | backport | registers the V2 runner's top-k/top-p sampler JIT warmups on CUDA too (vllm #58092 without #58465's ROCm-only gate): 0 in-request compiles, ~71-78 s of warmup once per cold cache volume, ~0.2 s warm | vllm #58092 (merged after 0.30.0) | 0.32.0: #58092 is in the pin, still gated to ROCm by #58465; the patch removes the gate | the pin that carries #58092 with CUDA registration (upstream gates it to ROCm; an opt-in is the ask) |
| pinned-kv-empty-cache | fix | with `kv_cache_memory_bytes` pinned (the launchers' `KV_MEM`), synchronize and empty the allocator cache after the profile run, so a cold compile cache's scratch (1.2 GiB on a 24 GiB card) does not stay under the KV cache and over-commit the card, which under WSL2's driver moves GPU memory to system RAM and slows the KVarN kernels; `memory-profile-after-warmup` does the same on the measured path only | vllm #59893 | 0.32.0: the synchronize and empty_cache follow upstream's set_current_vllm_config block around profile_run | upstream empties the cache on the pinned path |
| api-root-health | feature | GET/HEAD /, /api/status, /api/experimental/model-recommendations, and /api/show for `ollama launch codex` compatibility. Matches served models dynamically, returns 404 for unserved models (the body names the requested model only, never a served name or path), reads context length from model_config.max_model_len, and adds the routes to UNGUARDED_PATHS | none yet | 0.32.0 | upstream PR |
| qwen3_5-embed-uva | feature | `VLLM_EMBED_UVA=1` keeps the quantized token embedding (main model and MTP module) in pinned host RAM behind a CUDA UVA view, freeing about 1.25 GiB of VRAM for the KV pool (+15.7% tokens on a 3090) at no measured decode or prefill cost, for about 2.1 GiB of host RAM; default off | none | 0.32.0: adjacency with #60623, which removed the vendored qwen3_5 configs (transformers >= 5.16.1 ships them) | stays while the knob exists |
| vllm-pr40371-prompt-progress | feature | opt-in return_progress on streaming chat/completions; SSE chunks carry prompt_progress (total/cache/processed/time_ms) while the prompt prefills | vllm #40371 (open) | 0.32.0: num_computed_tokens follows upstream's new error parameter, and every call site passes it by keyword (a positional call would bind the count to error) | the pin that carries #40371 |
| vllm-pr61050-lru-running-prefix | fix | the CPU KV tier keeps a running request's head chunks; an eviction before the request finishes takes its tail first | vllm #61050 (open); the regression is from vllm #51787 | 0.32.0 | the pin that carries #61050 |
| kv-cache-seat-budget | feature | one INFO line after "GPU KV cache size": resident requests at ~1k, 32k and max_model_len, from each device group's admission cap at that length and the pool's usable blocks | none; vllm #54840 (open) logs a per-group block budget at the same point | 0.32.0 | upstream logs residency by context length |
| gdn-ba-pad-small-m | feature | pad 2-4 token batches of the bf16 GDN `in_proj_ba` GEMM to 5 rows (cuBLAS cliff at that shape; MTP-3 verify) | none | 0.32.0 | upstreamed, or a cuBLAS that picks a sane kernel at M = 2..4 |
| vllm-pr61612-flashinfer-async-barrier | fix | with async scheduling on, FlashInfer's metadata builder takes the blocking seq_lens.cpu() again, the host barrier between steps that vllm #57214 dropped | vllm #61612 (open); the regression is from vllm #57214. vllm #61622 (open, for #61621) makes the blocking read wait on an event instead of holding the CUDA context lock, but leaves the shortcut as it is, so without #61612 it does not fix this | 0.32.0 | the pin that carries #61612, or another change that keeps the barrier under async scheduling |
| quant-scale-view-preserve-pages | backport | create per-token-head scale views without filling shared KV pages, preserving live hybrid state and existing scales | vllm #64141 (open), fixing #63944; runtime change credited to yueyinmochen | 0.32.0 | the pin carries vllm #64141 or an equivalent non-destructive scale-view implementation |
| dflash-context-capacity | fix | reject DFlash contexts exceeding the drafter capacity and fixed lookahead | HyperQwen issue #405 | 0.32.0 | upstream validates DFlash positional capacity before inference |
| async-mamba-kv-sizing | fix | account for async Mamba states and usable grouped KV capacity | none yet | 0.32.0 | upstream accounts for async states and usable grouped capacity |
| continuous-stream-usage | fix | preserve continuous usage while tool parsing buffers deltas | none yet | 0.32.0 | upstream preserves continuous usage during parser buffering |
| gdn-prefill-exact-checkpoints | feature | opt-in exact interior GDN checkpoint export using upstream 0.32 metadata | follow-up to HyperQwen PR #392 | 0.32.0 | upstream GDN exports exact interior recurrent and convolution states |
| kvarn/kvarn-0.32.0 | feature | KVarN cache dtypes, quant mode, backend registration, page size | none (KVarN is Huawei CSL's, Apache-2.0) | 0.32.0: KVQuantMode.KVARN moves from 11 to 100, because #57057's ULTRAQUANT_4BIT took 11 (a duplicate IntEnum value is a silent alias); both dtypes kept in every registry | upstreamed |
| kvarn/kvarn-v2-runner-0.32.0 | own | KVarN with the V2 runner and DFlash2 (SW groups, Mamba block index, selector guards) | none | 0.32.0: the SlidingWindowManager.cache_blocks override stays after #61126 removed get_num_common_prefix_blocks; the mamba_hybrid.py hunk is retired (#55601 seeds the state index with mamba_block_size, the same fix) | rides with KVarN |
| kvarn/kvarn-recycled-pages-0.32.0 | own | both runners hand KVarN each step's block ids, so it drops (never flushes) what it still holds for a page another KV-cache group has taken: a late flush of a finished request's last block, or of an evicted retired sink, into another request's mamba state was the "!!!!" output (#208); the KVarN half is in `kvarn/files` | none | 0.32.0 | rides with KVarN |
| kvarn/kvarn-fp16-dequant-0.32.0 | own | registers `KVARN_FP16_DEQUANT` (#240's fp16 dequant in the fused KVarN decode kernels, default off) in `envs.py`, so `kvarn/files` reads it through `vllm.envs` and it is in the torch.compile cache key | none | 0.32.0 | rides with KVarN |
<!-- table:end -->

`pthfa/` is not in the table because it is not a patch: it is the source of a separate extension, the pth-flash-attn
int8 per-token-head attention backend (`PTH_FA=1`, the default on a compute capability 8.6 GPU for `SPEC=mtp CTX=long` and `SPEC=dflash2 CTX=long`; `docs/long-context.md`), which `pthfa/install.sh` builds and installs
into the venv beside vLLM as a plugin. No vLLM file changes, so the series and `patches/check_vllm_series.sh` are the
same with or without it. `pthfa/PIN` names the commit it is vendored from (a release tag of https://github.com/noon-at-cgn/pth-flash-attn); `pthfa/check_vendor.sh` checks the files.

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
