"""The FA2_FP8KV gate and the KV-layout narrowing of the fa2_fp8kv_vllm plugin, with no GPU and no
compiled library: register() is a no-op unless FA2_FP8KV=1, importing the backend never loads a
library on a machine without a matching GPU, paths.check() exits non-zero with a message when a
library is missing, and the backend narrows the layout to LBNHC only for an fp8 cache dtype.

Runs inside the image:  /app/venv/bin/python fa2/tests/gate_test.py
From a checkout: PYTHONPATH=fa2/python venv/bin/python fa2/tests/gate_test.py
Exit 0 = every assertion held."""
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

for name in ("FA2_FP8KV", "FA2_FP8KV_LIBRARY", "FA2_FP8KV_PREFILL_LIBRARY"):
    os.environ.pop(name, None)
logging.getLogger("vllm").setLevel(logging.ERROR)

try:
    import vllm.vllm_flash_attn  # noqa: F401
except ImportError:  # no libcuda.so.1 (the image build): the backend only needs the name
    sys.modules["vllm.vllm_flash_attn"] = SimpleNamespace(flash_attn_varlen_func=None)

import fa2_fp8kv_vllm
from fa2_fp8kv_vllm import paths
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_layout import KVCacheLayout

FLASH = AttentionBackendEnum.FLASH_ATTN
BACKEND_PATH = "fa2_fp8kv_vllm.backend.Fp8FlashAttentionBackend"
checks = 0


def check(condition, message):
    global checks
    assert condition, message
    checks += 1


native_path = FLASH.get_path()
check(not FLASH.is_overridden() and native_path != BACKEND_PATH, "FLASH_ATTN starts native")

for value in (None, "", "0", "true", "2"):
    with patch.dict(os.environ, {} if value is None else {"FA2_FP8KV": value}):
        fa2_fp8kv_vllm.register()
    check(not FLASH.is_overridden(), f"FA2_FP8KV={value!r}: register() leaves FLASH_ATTN alone")

with patch.dict(os.environ, {"FA2_FP8KV": "1"}):
    fa2_fp8kv_vllm.register()
check(FLASH.get_path() == BACKEND_PATH, "FA2_FP8KV=1: register() replaces FLASH_ATTN")
backend = FLASH.get_class()
check(backend.__name__ == "Fp8FlashAttentionBackend", "the replacement imports without a GPU or a library")
FLASH.clear_override()
check(FLASH.get_path() == native_path, "clear_override restores the native path")


def layouts(dtype, capability=(8, 6), config=True):
    cfg = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=dtype)) if config else None
    with patch("fa2_fp8kv_vllm.backend.get_current_vllm_config_or_none", return_value=cfg), \
         patch("fa2_fp8kv_vllm.backend.current_platform.get_device_capability", return_value=capability):
        return backend.supported_kv_cache_layouts()


for dtype in ("fp8", "fp8_e4m3"):
    check(layouts(dtype) == (KVCacheLayout.LBNHC,), f"{dtype} on sm86 narrows to LBNHC")
for dtype in ("auto", "bfloat16", "float16"):
    check(layouts(dtype) is None, f"{dtype} keeps the native any-layout answer")
check(layouts("fp8", capability=(8, 0)) is None, "fp8 on sm80 is not this backend's")
check(layouts("fp8", capability=(12, 0)) is None, "fp8 on sm120: not built for it")
check(layouts("fp8", capability=None) is None, "no GPU: not selected")
check(layouts(None, config=False) == (KVCacheLayout.LBNHC,), "no config (engine RPC): the gate decides, LBNHC")

with tempfile.TemporaryDirectory() as tmp:
    main, prefill = Path(tmp, "a.so"), Path(tmp, "b.so")
    env = {"FA2_FP8KV_LIBRARY": str(main), "FA2_FP8KV_PREFILL_LIBRARY": str(prefill)}
    with patch.dict(os.environ, env):
        check(paths.library() == main and paths.prefill_library() == prefill, "env overrides the library paths")
        for present in ((), (main,), (prefill,)):
            for p in present:
                p.touch()
            try:
                paths.check()
            except SystemExit as e:
                check(e.code and str(main if main not in present else prefill) in str(e.code),
                      "check() exits non-zero and names the missing library")
            else:
                check(False, "check() passed with a library missing")
            for p in present:
                p.unlink()
        main.touch()
        prefill.touch()
        paths.check()
        check(True, "check() passes with both libraries")
check(paths.library() == paths.PACKAGE / "fa2_fp8kv.so", "default: libraries sit in the package directory")

print(f"FA2 GATE TEST OK: {checks} assertions")
