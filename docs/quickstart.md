# Quickstart (Docker)

The one canonical path to a running server. Bare metal instead?
See [docs/install.md](install.md).

[← back to the main README](../README.md)

```bash
git clone https://github.com/syv-ai/HyperQwen && cd HyperQwen
cp .env.example .env                      # PowerShell: Copy-Item .env.example .env
make keygen                               # or: echo "VLLM_API_KEY=$(openssl rand -hex 24)" >> .env
docker compose --profile single up -d     # chatting; `--profile batch` for an API backend
```

**On Windows** this runs through Docker Desktop with its WSL2 backend (turn on
"Use the WSL 2 based engine"); there is no native-Windows path, because the
container and the patched vLLM are Linux. Run the commands above from a WSL2
distro, or from PowerShell if Docker Desktop is set up as above, and read
[docs/wsl2-4090.md](wsl2-4090.md) once. On Linux you only need Docker, the
NVIDIA driver and the NVIDIA Container Toolkit.

One GPU runs one mode at a time (`docker compose --profile single down`
before switching). The shipped `.env` is the single-user DFlash2 profile
(`SPEC=dflash2`, `PREFIX_CACHE=1`); every knob is documented in
[single-user/README.md](../single-user/README.md) and
[batch/README.md](../batch/README.md).

First boot usually takes **2–15 min** once the image is pulled (the entrypoint warns it can pass 30 min on a slow link or disk): it pulls the image (~9.5 GB), downloads and
requantizes the model (~20 GB, once, into `./models`), then compiles
(torch.compile / CUDA graphs / FlashInfer JIT, cached in `qwen-cache`
afterwards). Watch it with:

```bash
make logs                                 # follow the server log; Ctrl-C anytime
```

It is up when `/health` answers. Then verify the install (read-only, prints
no secrets):

```bash
make doctor
```

## If it does not come up (WSL2 + RTX 3090)

- **Refuses to boot / OOM at startup:** the WSL2 fallback is `GPU_UTIL=0.93`
  in `.env` (keep the shipped default on native Linux).
- **Exit 137 during the CPU-only prepare step** with little VRAM in use is
  host-memory pressure, not a GPU OOM. Give WSL room in
  `%USERPROFILE%\.wslconfig`, then apply it:
  ```ini
  [wsl2]
  memory=20GB
  swap=8GB
  ```
  ```powershell
  wsl --shutdown
  ```
  then restart Docker Desktop and `up` again.
- **`RuntimeError: UVA is not available`:** keep
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1` in `.env`.

Still stuck? The failure signatures live in
[docs/gotchas.md](gotchas.md), the full container reference in
[docs/docker.md](docker.md).
