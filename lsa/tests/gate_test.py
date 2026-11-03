# SPDX-License-Identifier: Apache-2.0
"""The LSA plugin's gate, on CPU: inert unless LSA=1, and the CUSTOM slot is its own when it is.

    python lsa/tests/gate_test.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "python")

PROBE = r"""
import os, sys
sys.path.insert(0, %r)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
before = AttentionBackendEnum.CUSTOM.get_path() if AttentionBackendEnum.CUSTOM.is_overridden() else None
import hyperqwen_lsa
hyperqwen_lsa.register()
after = AttentionBackendEnum.CUSTOM.get_path() if AttentionBackendEnum.CUSTOM.is_overridden() else None
print("BEFORE", before)
print("AFTER", after)
""" % PY


def run(env_extra):
    env = {k: v for k, v in os.environ.items() if k != "LSA"}
    env.update(env_extra)
    env["VLLM_PLUGINS"] = ""  # no other plugin registers anything in the probe
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)
    if out.returncode:
        sys.exit(out.stderr)
    lines = dict(line.split(" ", 1) for line in out.stdout.splitlines() if line[:5] in ("BEFOR", "AFTER"))
    return lines["BEFORE"], lines["AFTER"]


failures = 0
before, after = run({})
ok = after == before
print(("PASS" if ok else "FAIL"), "LSA unset: register() leaves CUSTOM as it was", f"({after})")
failures += not ok
before, after = run({"LSA": "1"})
ok = after == "hyperqwen_lsa.backend.LsaAttentionBackend"
print(("PASS" if ok else "FAIL"), "LSA=1: register() takes CUSTOM", f"({after})")
failures += not ok

sys.path.insert(0, PY)
from hyperqwen_lsa import core, settings  # noqa: E402

d = settings.design()
ok = (d.top_k, d.top_p, d.window, d.sink) == (4096, 1024, 512, 4) and settings.dense_below(d) == 4612
print(("PASS" if ok else "FAIL"), "defaults are the paper's flat design (top_k 4096, top_p 1024, window 512, sink 4)")
failures += not ok
ok = (core.POOL_SIZE, core.BUCKET_SIZE, core.BUCKET_CAP, core.INDEX_DIM) == (4, 8, 16, 64)
print(("PASS" if ok else "FAIL"), "pools of 4, buckets of 8 capped at 16, index dim 64")
failures += not ok
print("gate_test:", "OK" if not failures else f"{failures} FAILED")
sys.exit(1 if failures else 0)
