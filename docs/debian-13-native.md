# RTX 3090 on native Debian 13: installation notes

The native Debian path for one RTX 3090, or two in one TP=2 engine. Companion
to [ubuntu-3090.md](ubuntu-3090.md) (the earlier native Ubuntu campaign) and
[wsl2-4090.md](wsl2-4090.md) (the Windows / WSL2 campaign). No Docker: this
follows the [Dockerfile](../Dockerfile) at **`3d6c954`**, with a managed Python,
the pip CUDA compiler, and one Debian-specific xgrammar build flag.

Stack verified by the native recipe and subsequent service checks on real
hardware in November 2026:

| component | verified stack |
|---|---|
| OS | Debian 13 (trixie), amd64 |
| GPU | RTX 3090, 24 GiB; also two RTX 3090s with TP=2 and DFlash2 |
| NVIDIA driver | 610.57.04 |
| Python | uv-managed CPython **3.12.12**, isolated venv |
| runtime | torch **2.14.0**, vLLM **0.33.0** + this checkout's patch series and KVarN |
| CUDA compiler | **13.0.88**, from pip's `nvidia/cu13` tree |

Installation, plugin builds, CPU checks, base/fast model preparation and its
unchanged second pass were verified; the recipe was installed on real hardware
on 19 November. The later TP=2 systemd service passed health, inference and
recovery checks, including automatic startup after a cold boot on 21 November.
These are installation and service receipts, not a Debian benchmark table or
a claim that the older Ubuntu/WSL2 numbers transfer.

## Interpreter: uv-managed CPython 3.12.12

Debian 13 ships Python 3.13. This recipe leaves it alone and uses the exact
3.12.12 interpreter managed by `uv`, including its headers. Do not copy a venv
from another machine.

Start with the NVIDIA driver working and `uv` available on `PATH`. The system
packages used by the recipe are below (install these as root); no APT CUDA
toolkit is needed:

```bash
apt-get update
apt-get install -y --no-install-recommends \
  build-essential patch curl ca-certificates git python3 util-linux
```

The remaining commands are Bash, in the same shell. Replace the two absolute
path placeholders first. Keep models outside the version directory, use a
fresh version directory, and do not run preparation against a serving model.

```bash
set -euo pipefail
export PREFIX='<absolute-install-prefix>'
export MODELS='<absolute-model-directory>'
export RELEASE="$PREFIX/3d6c9543292e"

mkdir -p "$PREFIX"
git init "$RELEASE"
git -C "$RELEASE" fetch --depth 1 https://github.com/syv-ai/HyperQwen.git \
  3d6c9543292efc17372b824071ef8d9f33366b21
git -C "$RELEASE" checkout --detach FETCH_HEAD
cd "$RELEASE"
mkdir -p "$MODELS" receipts .build-cache
ln -s "$MODELS" models

unset PYTHONHOME PYTHONPATH PYTHONUSERBASE PYTHONOPTIMIZE PYTHONINSPECT PYTHONSTARTUP
export PYTHONNOUSERSITE=1
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export UV_PYTHON_PREFERENCE=only-managed
uv python install 3.12.12
uv venv --python 3.12.12 --seed "$RELEASE/venv"
export PY="$RELEASE/venv/bin/python"
export PATH="$RELEASE/venv/bin:$PATH"
"$PY" -c 'import sys; print(sys.version, sys.base_prefix, sys.prefix)'

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
replay. The verified recipe froze that resolution separately; the commands here
do not promise identical future dependency resolution.

## CUDA_HOME: assemble it from the venv

The compiler wheels install under `site-packages/nvidia/cu13`. Upstream build
steps expect a toolkit layout with `lib64/libcudart.so`; the wheels supply
`lib/libcudart.so.13`. The native recipe builds a small symlink tree rather
than mixing in an APT toolkit or an unrelated host `nvcc`:

```bash
CU=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13
"$CU/bin/nvcc" --version                 # must report V13.0.88
export CUDA_HOME="$RELEASE/.build-cache/cuda13-home"
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
export PATH="$CUDA_HOME/bin:$PATH"
export FA2_CACHE="$RELEASE/.build-cache/fa2"
export MAX_JOBS=4
```

This is the **compiler** version, not a claim that every CUDA runtime package
has version 13.0.88. Preserve `CUDA_HOME` and the venv's `PATH` in the service
environment too: a successful build shell does not configure systemd.

## Plugins and patches: keep the image's order

xgrammar's pinned source enables LTO and treats warnings as errors. With
Debian 13's GCC 14, linking fails on ODR violations that the image's GCC 13
does not report. **`CXXFLAGS=-fno-lto`** disables cross-unit optimization for
that build; it does not select a different xgrammar source revision.

From `$RELEASE`, with the environment above:

```bash
(
  export CUDA_VISIBLE_DEVICES=
  PY="$PY" bash pthfa/install.sh
  PY="$PY" bash fa2/install.sh
  PY="$PY" CXXFLAGS=-fno-lto bash xgrammar/install.sh
  PY="$PY" bash lsa/install.sh --no-map
  SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1)
  bash patches/apply.sh "$SP"
  bash kvarn/install.sh
  bash verify.sh --install
)
```

Use `patches/apply.sh`, which reads `patches/series`; glob order is not the
patch order. The native recipe also ran the CPU checks listed in the
[Dockerfile](../Dockerfile), with the GPU hidden, after applying the overlays.
`verify.sh --install` checks installation state, not live GPU behavior.

There is one expected metadata conflict after the xgrammar overlay: vLLM
0.33.0 requires `xgrammar==0.2.8`, while this checkout builds the commit in
`xgrammar/PIN` (`fa893f0cc2d117e51218e55ea5b0d88579741bf7`). A post-overlay
`pip check` therefore is **not clean**. The recipe checked that source pin,
the installed version, `xgrammar/check.py` and the vLLM backend import, then
accepted only that exact dependency conflict. Any other conflict is an
installation failure. Do not reinstall dependencies over the patched venv to
make the warning disappear; use a new venv when changing pins.

```bash
"$PY" -m pip freeze --all > receipts/pip-freeze.txt
```

Retain the complete freeze as a receipt. Local build paths in it are not a
portable requirements file: rebuild the plugins from their repository PINs.

## Model preparation: move the two /app assumptions

`docker/prepare.sh` does more than download weights: it checks shards,
requantizes the head, embeddings and MTP module, builds the reduced draft
vocabulary, fetches the fast variant, and hardens/translates chat templates.
Keep those steps. The native recipe changes only its fixed `cd /app` and
venv `PATH`, in a copy, and supplies `BASE_MODEL_DIR` explicitly:

```bash
grep -Fxq 'cd /app' docker/prepare.sh
grep -Fxq 'export PATH=/app/venv/bin:$PATH' docker/prepare.sh
sed 's|^cd /app$|cd "${HYPERQWEN_ROOT:?}"|; s|^export PATH=/app/venv/bin:$PATH$|export PATH="$HYPERQWEN_ROOT/venv/bin:$PATH"|' \
  docker/prepare.sh > receipts/prepare-native.sh

export HF_HOME="$RELEASE/.build-cache/huggingface"
export HF_HUB_CACHE="$MODELS/.hf-hub"
export HF_XET_HIGH_PERFORMANCE=1
env CUDA_VISIBLE_DEVICES= HYPERQWEN_ROOT="$RELEASE" \
  BASE_MODEL_DIR="$MODELS/Qwen3.8-27B-W4A16-AutoRound" \
  HF_REPO=dbirks/Qwen3.8-27B-W4A16-AutoRound \
  FAST_VARIANT=1 DFLASH2=0 LSA=0 HARDEN_TEMPLATES=1 TRANSLATE_EFFORT=1 \
  bash receipts/prepare-native.sh

for model in Qwen3.8-27B-W4A16-AutoRound Qwen3.8-27B-W4A16-AutoRound-fast; do
  MODEL="$MODELS/$model" bash verify.sh --no-server
done
```

The `models` symlink created earlier matters: the template hardener's default
scan is relative to the checkout. Base and fast preparation needs no GPU;
the download is about 19.5 GB plus roughly 1 GB for the fast variant. Run the
same preparation command a second time: the verified recipe had no remaining
steps and no changed model bytes. If preparation ran as root, ensure the
unprivileged service can traverse the model directories and read the weights;
requantization can leave root-only files.

Keep the downloaded HF revisions and a SHA-256 manifest of prepared model
files. Verify an existing manifest before preparing again. The recipe recorded
base/fast provenance from download metadata and the HF cache, and checked the
manifest after both passes; a future download from a moving HF branch is not
proof of the same bytes. The fast variant uses hardlinks, so preserve those
when copying it.

## One card, or two cards in one engine

For one selected RTX 3090, from `$RELEASE`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='<gpu-uuid>' \
  SPEC=mtp CTX=long PTH_FA=1 PREFIX_CACHE=1 PREFIX_RETENTION=0 \
  bash single-user/start_qwen.sh
```

The verified two-card service uses DFlash2. The base/fast preparation above
deliberately skips that optional drafter; fetch it before selecting DFlash2:

```bash
"$PY" prepare/fetch_dflash2.py "$MODELS/Qwen3.8-27B-DFlash2-W4A16"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='<gpu-uuid-0>,<gpu-uuid-1>' \
  SPEC=dflash2 DFLASH_TOKENS=7 CTX=long PTH_FA=1 MAX_SEQS=8 \
  PREFIX_CACHE=1 PREFIX_RETENTION=0 EXTRA_ARGS='--tensor-parallel-size 2' \
  bash single-user/start_qwen.sh
```

These are alternative launches; stop the first engine before starting another
on the same cards. See [multi-gpu.md](multi-gpu.md) for TP sizing and KV-pool
measurement. The single-card pool pin is not a two-card capacity measurement.

**For `CTX=long` workloads that cycle long documents, set
`PREFIX_RETENTION=0`**, as in the verified service. This keeps boundary
checkpoints rather than dense retention; it does not turn prefix caching off.
See [#482](https://github.com/syv-ai/HyperQwen/issues/482). This is workload
advice, not a new default for every profile.

## Operator notes from the native service

- Use a system service with an unprivileged user, the checkout as its working
  directory, and writable state/cache directories. Keep checkout and prepared
  weights read-only to the service. Do not enable `PrivateDevices`, which
  hides NVIDIA devices.
- Deliver the API key with
  `LoadCredential=<credential-id>:<absolute-key-file>`. A credential wrapper
  reads `$CREDENTIALS_DIRECTORY/<credential-id>`, rejects an empty, multiline
  or non-ASCII token, exports `VLLM_API_KEY`, then execs the launcher. The
  verified wrapper also rejected tokens longer than 4,096 bytes. Keep the
  source key file root-owned, mode 0600; do not put its value in arguments,
  a unit file, logs or an environment file.
- Set `HOST` explicitly to the intended loopback bind address in the service
  environment: providing an API key otherwise changes the launcher's default
  bind. Set `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1`. The verified service
  used `LimitCORE=0`, `UMask=0077`, `Restart=on-failure`, `RestartSec=20`,
  `StartLimitIntervalSec=3600` and `StartLimitBurst=3`.
- Before starting a two-card engine, check **both selected UUIDs**, not one
  unnamed VRAM reading. A generic inventory command is:

  ```bash
  nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits
  ```

  Match each selected UUID to its own memory reading. The verified service
  queried each selected UUID, waited up to 120 seconds for both to fall below
  1,000 MiB, and refused startup on an invalid query or timeout. It never
  stopped another GPU consumer to make room.
- Order the engine after the GPU prerequisites your host actually provides
  (persistence and any power-limit service). A systemd `active` state is not
  readiness: wait for `/health` to return 200, then check inference. The
  verified launcher ends in `exec vllm`; it does not contain the older image's
  warmup supervisor.
- Expand `$PREFIX`, `$RELEASE` and `$MODELS` to absolute paths when writing
  service configuration; systemd does not perform shell expansion in path
  directives. Preserve the previous version and its environment for rollback,
  and restart only after the old engine has stopped.
