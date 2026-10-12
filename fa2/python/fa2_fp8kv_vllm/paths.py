# SPDX-License-Identifier: Apache-2.0
"""The FA2_FP8KV gate and where the two compiled libraries live."""
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def enabled():
    return os.environ.get("FA2_FP8KV") == "1"


def library():
    return Path(os.environ.get("FA2_FP8KV_LIBRARY") or PACKAGE / "fa2_fp8kv.so")


def prefill_library():
    return Path(os.environ.get("FA2_FP8KV_PREFILL_LIBRARY") or PACKAGE / "fa2_fp8kv_prefill.so")


def check():
    missing = [str(p) for p in (library(), prefill_library()) if not p.is_file()]
    if missing:
        sys.exit("fa2_fp8kv: missing " + ", ".join(missing) + " (bash fa2/install.sh builds and installs them)")
