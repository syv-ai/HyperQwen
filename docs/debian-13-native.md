# RTX 3090 on native Debian 13: bare-metal install

[← back to the main README](../README.md)

Install the stack without Docker using managed Python, the pip CUDA compiler,
and a Debian-specific xgrammar build flag. Follow the [Dockerfile](../Dockerfile)
at **`3d6c954`** and use its requirements, plugin pins and patch series together.
For the general native path, see [install.md](install.md).

The installation, plugin builds, CPU checks, base/fast model preparation and a
**two-card TP=2 DFlash2 service** were verified on Debian 13 in November 2026.
The one-card MTP command below has **not been verified on Debian 13**. These
checks do not establish Debian benchmark results.

| component | verified stack |
|---|---|
| OS | Debian 13 (trixie), amd64 |
| GPU service | Two RTX 3090s, 24 GiB each, TP=2 with DFlash2 |
| NVIDIA driver | 610.57.04 |
| Python | uv-managed CPython **3.12.12**, isolated venv |
| runtime | torch **2.14.0+cu130**, vLLM **0.33.0** + the patch series at `3d6c954` and KVarN |
| CUDA compiler | **13.0.88**, from pip's `nvidia/cu13` tree |

## Interpreter: uv-managed CPython 3.12.12

Leave Debian 13's system Python 3.13 alone. Install the exact 3.12.12 interpreter
managed by `uv`, including its headers. Do not copy a venv from another machine.

Start with the NVIDIA driver working and `uv` available on `PATH`. Install
these system packages as root; no APT CUDA toolkit is needed:

```bash
apt-get update
apt-get install -y --no-install-recommends \
  build-essential patch curl ca-certificates git python3 util-linux
```

After APT, switch to an unprivileged build user who can write the chosen install
and model directories. Run the remaining commands in the same Bash shell as
that user. Replace the two absolute path placeholders first. Keep models
outside the version directory, use a fresh version directory, and do not run
preparation against a serving model. Stop and resolve any failed step before
continuing.

```bash
export PREFIX='<absolute-install-prefix>'
export MODELS='<absolute-model-directory>'
export RELEASE="$PREFIX/3d6c9543292e"

mkdir -p "$PREFIX"
git init "$RELEASE"
git -C "$RELEASE" fetch --depth 1 https://github.com/syv-ai/HyperQwen.git \
  3d6c9543292efc17372b824071ef8d9f33366b21
git -C "$RELEASE" checkout --detach FETCH_HEAD
cd "$RELEASE"
mkdir -p "$MODELS" .build-cache
ln -sfn "$MODELS" models

unset PYTHONHOME PYTHONPATH PYTHONUSERBASE PYTHONOPTIMIZE PYTHONINSPECT PYTHONSTARTUP
export PYTHONNOUSERSITE=1
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export UV_PYTHON_PREFERENCE=only-managed
uv python install 3.12.12
uv venv --python 3.12.12 --seed "$RELEASE/venv"
export PY="$RELEASE/venv/bin/python"
export PATH="$RELEASE/venv/bin:$PATH"
"$PY" -c 'import sys; print(sys.version, sys.base_prefix, sys.prefix)'
cat "$RELEASE/venv/pyvenv.cfg"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r docker/requirements.txt
"$PY" -m pip check
```

The interpreter must report 3.12.12, its `sys.base_prefix` must be under
`$PREFIX/python`, and `sys.prefix` must be `$RELEASE/venv`. In `pyvenv.cfg`,
`include-system-site-packages` must be false. All package installation and
build steps below use this venv.

The requirements file pins vLLM and the compiler set; it is not a complete
transitive lock. Keep the resolved freeze from a successful installation for
replay; the commands here do not promise identical future dependency resolution.

## CUDA_HOME: assemble it from the venv

The compiler wheels install under `site-packages/nvidia/cu13`. Upstream build
steps expect a toolkit layout with `lib64/libcudart.so`; the wheels supply
`lib/libcudart.so.13`. Assemble a symlink tree from the venv's compiler wheels:

```bash
export CUDA_HOME="$RELEASE/cuda13-home"
export PATH="$CUDA_HOME/bin:$PATH"
export FA2_CACHE="$RELEASE/.build-cache/fa2"
export MAX_JOBS=4
(
  set -euo pipefail
  CU=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')/nvidia/cu13
  "$CU/bin/nvcc" --version                 # must report V13.0.88
  mkdir -p "$CUDA_HOME/lib64"
  for d in bin include nvvm cccl; do
    if [ -e "$CU/$d" ]; then
      ln -sfn "$CU/$d" "$CUDA_HOME/$d"
    fi
  done
  for lib in "$CU/lib/"*; do
    if [ -f "$lib" ]; then
      ln -sfn "$lib" "$CUDA_HOME/lib64/$(basename "$lib")"
    fi
  done
  test -f "$CU/lib/libcudart.so.13"
  ln -sfn "$CU/lib/libcudart.so.13" "$CUDA_HOME/lib64/libcudart.so"
)
```

CUDA setup and model preparation stop on failure in subshells; `CUDA_HOME`, `PATH`, `FA2_CACHE` and `MAX_JOBS` stay exported in the parent shell.

This is the **compiler** version, not a claim that every CUDA runtime package
has version 13.0.88. Preserve `CUDA_HOME` and the venv's `PATH` in the service
environment too: a successful build shell does not configure systemd.
Keep `$CUDA_HOME` and its venv targets for the lifetime of the release; they
are runtime dependencies, not disposable build caches.

## Plugins and patches: keep the image's order

xgrammar's pinned source enables LTO and treats warnings as errors. With
Debian 13's GCC 14, linking fails on ODR violations that the image's GCC 13
does not report. **`CXXFLAGS=-fno-lto`** disables cross-unit optimization for
that build; it does not select a different xgrammar source revision.

From `$RELEASE`, with the environment above, build the plugins and run the
Dockerfile's CPU checks. Strict mode (`set -euo pipefail`) applies only inside
this build subshell, so a failure stops the block without closing your
interactive shell:

```bash
(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=
  PY="$PY" bash pthfa/install.sh
  PY="$PY" bash fa2/install.sh
  PY="$PY" CXXFLAGS=-fno-lto bash xgrammar/install.sh
  PY="$PY" bash lsa/install.sh --no-map
  SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1)
  bash patches/apply.sh "$SP"
  bash kvarn/install.sh
  bash verify.sh --install
  for check in \
    bench/test_draft_vocab_stats.py \
    bench/test_gdn_checkpoint_safety.py \
    bench/test_gdn_multi_checkpoint.py \
    bench/test_gdn_checkpoint_export.py \
    bench/test_offload_mamba_swa_boundaries.py \
    bench/test_dflash_context_capacity.py \
    bench/mq3d_scratch_pool_test.py \
    bench/test_quant_scale_views.py \
    pthfa/tests/gate_test.py \
    fa2/tests/gate_test.py \
    bench/test_async_mamba_kv_sizing.py \
    bench/test_kv_cache_seat_budget.py \
    bench/test_continuous_stream_usage.py \
    bench/test_gdn_ba_lora.py \
    lsa/tests/gate_test.py; do
    "$PY" "$check"
  done
  "$PY" fa2/tests/check_mixed_prefill.py --device cpu
)
```

Use `patches/apply.sh`, which reads `patches/series`; glob order is not the
patch order. `verify.sh --install` checks installation state, not live GPU
behavior. Continue only after the entire block exits successfully.

There is one expected metadata conflict after the xgrammar overlay: vLLM
0.33.0 requires `xgrammar==0.2.8`, while the 3d6c954 checkout builds the commit in
`xgrammar/PIN` (`fa893f0cc2d117e51218e55ea5b0d88579741bf7`). Inspect the pin and
installed version, run the grammar check and import the vLLM backend:

```bash
cat xgrammar/PIN
"$PY" -c 'from importlib.metadata import version; print("xgrammar", version("xgrammar"))'
CUDA_VISIBLE_DEVICES= "$PY" xgrammar/check.py
CUDA_VISIBLE_DEVICES= "$PY" -c 'import vllm.v1.structured_output.backend_xgrammar'
"$PY" -m pip check
"$PY" -m pip freeze --all > "$RELEASE/pip-freeze.txt"
```

Expect the post-overlay `pip check` to exit nonzero with only the conflict
between vLLM 0.33.0's `xgrammar==0.2.8` requirement and the installed source
build's version. Match that version to the printed package version; reject any
additional conflict or failed grammar/backend check. The installer checks the
fetched source commit against `xgrammar/PIN` before building. Do not reinstall
dependencies over the patched venv to remove the warning; use a new venv when
changing pins. Keep the complete freeze for comparison. Local build paths in
it are not portable requirements: rebuild plugins from their repository PINs.

## Model preparation: move the two /app assumptions

`docker/prepare.sh` does more than download weights: it checks shards,
requantizes the head, embeddings and MTP module, builds the reduced draft
vocabulary, fetches the fast variant, and hardens/translates chat templates.
Keep those steps. Copy the script under the build directory, replace only its
fixed `cd /app` and venv `PATH`, and supply `BASE_MODEL_DIR` explicitly:

```bash
(
  set -euo pipefail
  grep -Fxq 'cd /app' docker/prepare.sh
  grep -Fxq 'export PATH=/app/venv/bin:$PATH' docker/prepare.sh
  sed 's|^cd /app$|cd "${HYPERQWEN_ROOT:?}"|; s|^export PATH=/app/venv/bin:\$PATH$|export PATH="$HYPERQWEN_ROOT/venv/bin:$PATH"|' \
    docker/prepare.sh > "$RELEASE/.build-cache/prepare-native.sh"

  export HF_HOME="$RELEASE/.build-cache/huggingface"
  export HF_HUB_CACHE="$MODELS/.hf-hub"
  export HF_XET_HIGH_PERFORMANCE=1
  env CUDA_VISIBLE_DEVICES= HYPERQWEN_ROOT="$RELEASE" \
    BASE_MODEL_DIR="$MODELS/Qwen3.8-27B-W4A16-AutoRound" \
    HF_REPO=dbirks/Qwen3.8-27B-W4A16-AutoRound \
    FAST_VARIANT=1 DFLASH2=0 LSA=0 HARDEN_TEMPLATES=1 TRANSLATE_EFFORT=1 \
    bash "$RELEASE/.build-cache/prepare-native.sh"

  # Required for the two-card DFlash2 launch below; stop if this download fails.
  CUDA_VISIBLE_DEVICES= "$PY" prepare/fetch_dflash2.py "$MODELS/Qwen3.8-27B-DFlash2-W4A16"

  for model in Qwen3.8-27B-W4A16-AutoRound Qwen3.8-27B-W4A16-AutoRound-fast; do
    MODEL="$MODELS/$model" bash verify.sh --no-server
  done
)
```

The `models` symlink created earlier matters: the template hardener's default
scan is relative to the checkout. Base and fast preparation needs no GPU;
the download is about 19.5 GB plus roughly 1 GB for the fast variant. Run the
model checks after fetching the additional DFlash2 drafter (about 1.2 GB).
`DFLASH2=0` skips preparation's optional download; the explicit fetch makes
its failure visible before choosing DFlash2. Keep `FAST_VARIANT=1`: both launch
examples below select that prepared target model explicitly.

Prepare as the build user. Ensure the service user can traverse and read the
Python installation, venv, CUDA tree and all model files; builds or preparation
previously run as root can leave files inaccessible. Preserve the fast
variant's hardlinks when copying models.

## One card, or two cards in one engine

For one selected RTX 3090, the standard MTP invocation is below. **This command
has not been verified on Debian 13.** Run from `$RELEASE`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='<gpu-uuid>' \
  MODEL="$MODELS/Qwen3.8-27B-W4A16-AutoRound-fast" \
  SPEC=mtp CTX=long PTH_FA=1 PREFIX_CACHE=1 PREFIX_RETENTION=0 \
  bash single-user/start_qwen.sh
```

For two RTX 3090s in one TP=2 DFlash2 engine, use the drafter fetched and checked
above. This is the service configuration verified on Debian 13:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='<gpu-uuid-0>,<gpu-uuid-1>' \
  MODEL="$MODELS/Qwen3.8-27B-W4A16-AutoRound-fast" \
  SPEC=dflash2 DFLASH_TOKENS=7 CTX=long PTH_FA=1 MAX_SEQS=8 \
  PREFIX_CACHE=1 PREFIX_RETENTION=0 EXTRA_ARGS='--tensor-parallel-size 2' \
  bash single-user/start_qwen.sh
```

These are alternative launches; stop the first engine before starting another
on the same cards. See [multi-gpu.md](multi-gpu.md) for TP sizing and KV-pool
measurement. The single-card pool pin is not a two-card capacity measurement.

**For `CTX=long` workloads that cycle long documents, set
`PREFIX_RETENTION=0`**. Both examples assume this workload. This keeps boundary
checkpoints rather than dense retention; it does not turn prefix caching off.
See [#482](https://github.com/syv-ai/HyperQwen/issues/482). This is workload
advice, not a new default for every profile.

## Run as a systemd service

- Use a system service with an unprivileged user, the checkout as its working
  directory, and writable state/cache directories. Keep checkout and prepared
  weights read-only to the service. Do not enable `PrivateDevices`, which
  hides NVIDIA devices.
- Set runtime paths explicitly; do not inherit the build shell's `HF_HOME`
  or `HF_HUB_CACHE`. Point `HOME`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_HUB_CACHE`,
  `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
  `TORCH_EXTENSIONS_DIR` and `CUDA_CACHE_PATH` at directories writable by the
  service user. For example, adapt this fragment using absolute paths:

  ```ini
  [Service]
  StateDirectory=hyperqwen
  CacheDirectory=hyperqwen
  Environment=HOME=/var/lib/hyperqwen
  Environment=XDG_CACHE_HOME=/var/cache/hyperqwen
  Environment=HF_HOME=/var/cache/hyperqwen/huggingface
  Environment=HF_HUB_CACHE=/var/cache/hyperqwen/huggingface/hub
  Environment=TRITON_CACHE_DIR=/var/cache/hyperqwen/triton
  Environment=VLLM_CACHE_ROOT=/var/cache/hyperqwen/vllm
  Environment=TORCHINDUCTOR_CACHE_DIR=/var/cache/hyperqwen/torchinductor
  Environment=TORCH_EXTENSIONS_DIR=/var/cache/hyperqwen/torch-extensions
  Environment=CUDA_CACHE_PATH=/var/cache/hyperqwen/cuda
  Environment=CUDA_HOME=<absolute-release-directory>/cuda13-home
  Environment=PATH=<absolute-release-directory>/venv/bin:<absolute-release-directory>/cuda13-home/bin:/usr/bin:/bin
  Environment=HOST=127.0.0.1
  Environment=VLLM_NO_USAGE_STATS=1
  Environment=DO_NOT_TRACK=1
  ```

  `StateDirectory` and `CacheDirectory` create the parent directories for the
  configured service user; allow the libraries to create their cache children.
  FlashInfer's default cache follows the writable `HOME`. If you override any
  other library's cache path, keep it writable too. Serve the prepared local
  model paths; do not run model preparation as part of a read-only service.
- Deliver the API key with
  `LoadCredential=<credential-id>:<absolute-key-file>`. In a startup script,
  read `$CREDENTIALS_DIRECTORY/<credential-id>`, trim one optional trailing
  newline, reject an empty, still-multiline or non-ASCII token, export it as
  `VLLM_API_KEY`, then exec the launcher. These validations belong in that
  script; the repository's launcher does not perform them. Keep the source
  file root-owned, mode 0600; keep the value out of the unit, logs and
  `EXTRA_ARGS`.
- At `3d6c954`, `single-user/start_qwen.sh` calls `resolve_vllm_key` in
  `resolve_api_key.sh`: a nonempty `VLLM_API_KEY` takes precedence, otherwise
  it exports the contents of `$REPO/api_key.txt` if that file exists. It does
  **not** add `--api-key` to argv; it leaves the key in the environment for
  vLLM. This does not hide the environment from privileged process inspection.
  `resolve_bind_host` in `launcher_common.sh` uses a nonempty `HOST` first;
  otherwise a nonempty key selects `0.0.0.0`. With neither on a native host
  (no `/.dockerenv`), it selects `127.0.0.1`. Set `HOST=127.0.0.1` explicitly
  to keep a keyed service on loopback.
- Put restart-rate limits in `[Unit]`, and process/restart settings in
  `[Service]`, for example:

  ```ini
  [Unit]
  StartLimitIntervalSec=3600
  StartLimitBurst=3

  [Service]
  LimitCORE=0
  UMask=0077
  Restart=on-failure
  RestartSec=20
  ```
- Before starting a two-card engine, check **both selected UUIDs**, not one
  unnamed VRAM reading. A generic inventory command is:

  ```bash
  nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits
  ```

  Match each selected UUID to its own memory reading and check available VRAM
  on both before launch. If you automate this check, reject invalid queries
  and use a bounded wait with a memory threshold appropriate for your profile.
- Order the engine after the GPU prerequisites your host actually provides
  (persistence and any power-limit service). A systemd `active` state is not
  readiness: wait for `/health` to return 200, then check inference. The
  launcher calls `qwen_exec`, which ends in `exec` unless `PRINT_ARGV=1`.
- Expand `$PREFIX`, `$RELEASE` and `$MODELS` to absolute paths when writing
  service configuration; systemd does not perform shell expansion in path
  directives. Preserve the previous version and its environment for rollback,
  and restart only after the old engine has stopped.
