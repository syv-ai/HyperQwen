"""replay_offload_serve.py: does the CPU offload tier ever SERVE a stored request back? (syv-ai #95 oracle)

Set DEPTH_CORPUS_GLOB; the key comes from bench/harness.py. Run against a server booted with --kv-offloading-size and a GPU pool
SMALLER than the tier, so a request is evicted from GPU but fits the tier; then it can only come back by a load.

Sequence against a running server:
  1. request X (a fixed, unsalted prompt of DEPTH chars): cold prefill; record TTFT and the CPU->GPU transfer counter.
  2. requests Y1..Yn (fresh prompts, same depth): push X out of the GPU prefix cache into the CPU tier (or out entirely).
  3. request X again: if the tier serves, TTFT is a fraction of (1) and the CPU->GPU counter grows; if the tier stores
     but never serves (the #52735 defect), TTFT is a full re-prefill and the counter stays put.

    TIER_GIB=<--kv-offloading-size> python bench/replay_offload_serve.py TAG PORT DEPTH_CHARS N_EVICTORS
Prints one JSON line per request and a final verdict line. Exits 0 on SERVED, 1 on NOT-SERVED, 2 on
invalid measurements (missing corpus/metrics, counter reset, tier overflow, failed requests or X never evicted).

Sizing rule (the verdict is meaningless otherwise): the GPU pool must be smaller than the tier IN TOKENS, and the
evictor traffic must exceed the GPU pool while staying under the tier. A token costs several times more in the tier
than on the GPU here (37 KB vs 124 KB on a 4090 with DFlash2 k=7 fp8; 6.7 KB vs 80 KB on a 3090 with MTP; these are
per-configuration ratios, and the GPU pool does not scale linearly with KV_MEM on this model, so read both from the
boot log and the metrics rather than computing them), so a tier that is 3x the pool in bytes can be smaller than it in tokens. If the stores overflow the tier, X is evicted from the tier
before X-again and the run reads exactly like the #52735 veto (ratio ~1, zero loads, zero external hits). With
TIER_GIB set the script refuses that verdict and prints INVALID-TIER-OVERFLOW instead.
The CPU tier is allocated in /dev/shm: check its free space (df -h /dev/shm) before boot
and leave room below it; a tier that exceeds free shared memory will not start.
"""
import glob
import http.client
import json
import os
import re
import sys
import time
from pathlib import Path

import harness

sys.stdout.reconfigure(encoding="utf-8")
TAG, PORT, DEPTH, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# The PORT argument names the server, even when VLLM_API is set.
os.environ["VLLM_API"] = f"http://127.0.0.1:{PORT}"
CORPUS = os.environ.get("DEPTH_CORPUS_GLOB", "*.txt")  # any glob of plain-text files to build fresh prompts from
TEXT = "\n\n".join(Path(f).read_text(encoding="utf-8") for f in sorted(glob.glob(CORPUS)))
if not TEXT.strip():
    print(f"INVALID: no nonempty corpus matched {CORPUS!r}", file=sys.stderr)
    sys.exit(2)
while len(TEXT) < (N + 2) * DEPTH + 100000:
    TEXT += "\n\n" + TEXT


def metrics():
    # The prefixes have no _total, so they also take each counter's <name>_created line: drop those.
    return {name + labels: value for name, labels, value in harness.samples()
            if name.startswith(("vllm:kv_offload", "vllm:prefix_cache", "vllm:external_prefix_cache"))
            and "_bucket" not in name and "_created" not in name}


def ask(label, prompt, max_tokens=32):
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    first = None
    usage = None
    for j in harness.stream("/v1/chat/completions", payload, timeout=None):
        if first is None and any((c.get("delta") or {}).get("content") for c in j.get("choices", [])):
            first = time.perf_counter()
        if j.get("usage"):
            usage = j["usage"]
    if first is None:
        raise ValueError(f"{label}: stream produced no content; TTFT is unavailable")
    ttft = round(first - t0, 2)
    m = metrics()
    served = {re.sub(r'\{engine="0",model_name="[^"]*",', "{", k): v for k, v in m.items() if "total_bytes" in k or "load" in k or "store" in k or "usage_perc" in k or "skipped" in k or "lookup" in k or "prefix_cache" in k}
    rec = {"tag": TAG, "req": label, "ttft_s": ttft, "total_s": round(time.perf_counter() - t0, 1),
           "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
           "offload": served}
    print(json.dumps(rec), flush=True)
    return rec


def fixed(i):
    start = i * DEPTH
    return (f"[replay {TAG} #{i}] Read the following document, then answer.\n\n" + TEXT[start:start + DEPTH]
            + "\n\nIn one sentence, what is the main theme of the document above?")


def store_gb(m):
    # One counter: vllm:kv_offload_store_bytes_total and the GPU_to_CPU leg of vllm:kv_offload_total_bytes_total
    # report the same bytes under two names, and summing both doubled the stores and falsely invalidated
    # SERVED runs against the tier size (found by the maintainer on #95).
    stores = [v for k, v in m.items() if k.startswith("vllm:kv_offload_store_bytes_total")]
    if not stores:
        stores = [v for k, v in m.items() if k.startswith("vllm:kv_offload_total_bytes_total") and "GPU_to_CPU" in k]
    return sum(stores) / 1e9


def load_delta(before, after):
    """Choose one counter for both scrapes; a newly appearing flat counter starts at zero."""
    def values(m, name, direction=None):
        return [v for k, v in m.items() if k.split("{", 1)[0] == name
                and (direction is None or direction in k)]

    name = "vllm:kv_offload_load_bytes_total"
    old, new = values(before, name), values(after, name)
    if old or new:
        return sum(new) - sum(old) if new else None
    name = "vllm:kv_offload_total_bytes_total"
    old, new = values(before, name, '"CPU_to_GPU"'), values(after, name, '"CPU_to_GPU"')
    return sum(new) - sum(old) if old and new else None


def verdict(ratio, stores_gb, tier_gib, loaded_bytes):
    """SERVED needs a fast replay and actual loaded bytes; invalid measurements have no verdict."""
    if tier_gib and stores_gb > tier_gib * 1.073741824:
        return "INVALID-TIER-OVERFLOW"
    if loaded_bytes is None:
        return "INVALID-NO-LOAD-METRICS"
    if loaded_bytes < 0:
        return "INVALID-COUNTER-RESET"
    if loaded_bytes == 0 and ratio < 0.5:
        return "INVALID-NOT-EVICTED"
    return "SERVED" if ratio < 0.5 and loaded_bytes > 0 else "NOT-SERVED"


def run():
    m0 = metrics()  # cumulative stores: the overflow guard needs only THIS run
    x1 = ask("X-cold", fixed(0))
    for i in range(1, N + 1):
        ask(f"Y{i}", fixed(i))
    before_replay = metrics()
    x2 = ask("X-again", fixed(0))
    loaded_bytes = load_delta(before_replay, x2["offload"])
    ratio = x2["ttft_s"] / max(x1["ttft_s"], 0.01)
    loads = {k: v for k, v in x2["offload"].items() if "CPU_to_GPU" in k or "load" in k.lower()}
    stores_gb = store_gb(metrics()) - store_gb(m0)
    tier_gib = float(os.environ.get("TIER_GIB", "0") or 0)
    v = verdict(ratio, stores_gb, tier_gib, loaded_bytes)
    advice = {
        "INVALID-TIER-OVERFLOW": f"{stores_gb:.2f} GB stored into a {tier_gib:g} GiB tier; X may have left the tier. Shrink the GPU pool or evictor count, or grow the tier.",
        "INVALID-NOT-EVICTED": "X-again was fast with no CPU load; X may still be on the GPU. Shrink the GPU pool or increase evictors within the tier capacity.",
        "INVALID-NO-LOAD-METRICS": "No comparable load-byte counter pair. Check the server's /metrics and offload configuration.",
        "INVALID-COUNTER-RESET": "The load-byte counter decreased. Check for a server restart and rerun without other traffic.",
    }
    if v in advice:
        print(f"{v}: {advice[v]}", file=sys.stderr)
    print(json.dumps({"tag": TAG, "verdict": v, "stores_gb": round(stores_gb, 2), "tier_gib": tier_gib, "cold_ttft_s": x1["ttft_s"], "again_ttft_s": x2["ttft_s"], "ratio": round(ratio, 2), "cached_again": x2["cached"], "loads": loads, "loaded_bytes": loaded_bytes}), flush=True)
    return {"SERVED": 0, "NOT-SERVED": 1}.get(v, 2)


try:
    sys.exit(run())
except (OSError, ValueError, http.client.HTTPException) as exc:
    print(f"INVALID-REQUEST: {exc}; restore the server connection and rerun.", file=sys.stderr)
    print(json.dumps({"tag": TAG, "verdict": "INVALID-REQUEST", "error": str(exc)}), flush=True)
    sys.exit(2)
