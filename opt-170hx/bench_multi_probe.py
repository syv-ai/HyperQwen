"""Concurrent exact-token-ID streaming probe.

Fires CONC requests at once (same start barrier) and reports per-stream TTFT
and verify-step gaps, plus batch-aggregate decode throughput. Prompts are
generated from SEEDBASE+i so prefix caching cannot hide prefill work across
configs (use a different SEEDBASE per config).
"""
import json, os, random, statistics, sys, threading, time, urllib.request

PORT = int(os.environ.get("PORT", "18020"))
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
API_KEY = os.environ.get("API_KEY", "")
IN = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 128
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SEEDBASE = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

if min(IN, OUT, CONC) < 1:
    print("input/output counts and concurrency must be positive", file=sys.stderr)
    sys.exit(2)

prompts = []
for i in range(CONC):
    rng = random.Random(SEEDBASE + i)
    prompts.append([rng.randrange(1000, 200000) for _ in range(IN)])

barrier = threading.Barrier(CONC + 1)
results = [None] * CONC


def run_one(i):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompts[i],
        "max_tokens": OUT,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body, headers=headers)

    barrier.wait()
    t0 = time.perf_counter()
    first = None
    chunks = []
    usage = None
    done = False
    err = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                d = json.loads(payload)
                if "error" in d:
                    raise RuntimeError(f"stream error: {d['error']}")
                now = time.perf_counter()
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch and ch[0].get("text"):
                    if first is None:
                        first = now
                    chunks.append(now)
        if not done:
            raise RuntimeError("incomplete stream: missing [DONE]")
        if not chunks or not usage or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 1:
            raise RuntimeError("stream has no generated text or valid completion usage")
    except Exception as e:  # report, do not hang the whole run
        err = repr(e)
    t1 = time.perf_counter()
    gaps = [(chunks[j] - chunks[j - 1]) * 1000 for j in range(1, len(chunks))]
    results[i] = {
        "idx": i,
        "ttft_ms": (first - t0) * 1000 if first else float("nan"),
        "wall_ms": (t1 - t0) * 1000,
        "chunks": len(chunks),
        "gen": usage["completion_tokens"] if usage else None,
        "decode_ms": (chunks[-1] - chunks[0]) * 1000 if len(chunks) > 1 else 0.0,
        "step_med": statistics.median(gaps) if gaps else float("nan"),
        "step_mean": sum(gaps) / len(gaps) if gaps else float("nan"),
        "err": err,
        "_chunks": chunks,
        "_gaps": gaps,
        "_first": first,
        "_t0": t0,
        "_t1": t1,
    }


threads = [threading.Thread(target=run_one, args=(i,)) for i in range(CONC)]
for t in threads:
    t.start()
barrier.wait()
for t in threads:
    t.join()

ok = [r for r in results if r and not r["err"]]
if not ok:
    print("ALL FAILED:", [r and r["err"] for r in results])
    sys.exit(1)

all_gaps = [g for r in ok for g in r["_gaps"]]
t0 = min(r["_t0"] for r in ok)
tend = max(r["_t1"] for r in ok)
gen_total = sum(r["gen"] or 0 for r in ok)
decode_wall = (
    max(r["_chunks"][-1] for r in ok) - min(r["_chunks"][0] for r in ok)
    if all(r["_chunks"] for r in ok) else 0.0
)
ttfts = [r["ttft_ms"] for r in ok]
out = {
    "in": IN, "out": OUT, "conc": CONC, "ok": len(ok),
    "ttft_med_ms": round(statistics.median(ttfts), 1),
    "ttft_min_ms": round(min(ttfts), 1),
    "ttft_max_ms": round(max(ttfts), 1),
    "step_med_ms": round(statistics.median(all_gaps), 2) if all_gaps else None,
    "step_mean_ms": round(sum(all_gaps) / len(all_gaps), 2) if all_gaps else None,
    "chunks_per_stream": round(statistics.median(r["chunks"] for r in ok), 1),
    "tok_per_chunk": round(gen_total / sum(r["chunks"] for r in ok), 2),
    "batch_tok_s": round(gen_total / decode_wall, 1) if decode_wall > 0 else None,
    "wall_s": round(tend - t0, 1),
    "errs": [r["err"] for r in results if r and r["err"]],
}
print(json.dumps(out, ensure_ascii=False))

if len(ok) != CONC:
    sys.exit(1)
