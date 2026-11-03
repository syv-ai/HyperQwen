# SPDX-License-Identifier: Apache-2.0
"""The LSA gate and knobs, all from the environment the launcher exports (lsa/README.md)."""
import os
from pathlib import Path

from hyperqwen_lsa import core

REPO_MAP = Path(__file__).resolve().parents[3] / "models" / "lsa" / "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"


def enabled() -> bool:
    return os.environ.get("LSA") == "1"


def _int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value else default


def map_path() -> Path:
    return Path(os.environ.get("LSA_MAP") or REPO_MAP)


def design() -> core.Design:
    return core.Design(top_k=_int("LSA_TOP_K", core.TOP_K), top_p=_int("LSA_TOP_P", core.TOP_P))


def dense_below(d: core.Design) -> int:
    """Longest sequence an eager batch reads densely: below it LSA reads every key anyway."""
    return _int("LSA_DENSE_BELOW", d.sink + d.top_k + d.window)


def decode_splits() -> int:
    return _int("LSA_DECODE_SPLITS", 16)


def prefill_chunk() -> int:
    return _int("LSA_PREFILL_CHUNK", 256)


def base() -> str:
    """LSA_BASE: pthfa (pth_flash_attn, required), triton (vLLM's own), or unset (pthfa if installed)."""
    return os.environ.get("LSA_BASE", "")
