"""Is the eight-seat cell's ~1.8x TTFT at N=1 the seat count, or is it warmup?

@mjungnickel18 measured 21.8 s against 12.2 s for the same single 4k prompt at N=1 with
only MAX_SEQS differing, fresh boot each -- a cell where the seat count cannot matter,
because there is one request. He declined to draw a conclusion and asked for a rerun
with the cells interleaved. This is the client half of that.

  venv/bin/python bench/seat_ttft.py <tag> [ctx_tokens] [reps]

Two things his run could not separate, because it measured one request per boot:

  cold   the FIRST request a server ever sees, which pays whatever lazy init survived
         graph capture
  warm   the same shape afterwards, which is the steady state a user actually meets

Each prompt carries its own salt, so no request is ever served out of the prefix cache
and "warm" means warm engine, not warm cache.

A measurement, so it exits 0 whatever the numbers say. It exits 2 if every request failed.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _key():  # a key is optional; keyless servers ignore the header
    try:
        return open(os.path.join(REPO, "api_key.txt")).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key()
BASE = "http://127.0.0.1:" + os.environ.get("PORT", "18020")
TAG = sys.argv[1]
CTXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

# ~1.4 tokens per word for this filler; overshoot slightly and let the count speak
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
          "Meanwhile the engineer reviews the throughput numbers one more time. ")
WORDS_PER_REP = 26


def prompt(salt):
    reps = max(1, int(CTXTOK / (WORDS_PER_REP * 1.15)))
    return f"[seat probe {salt}] " + FILLER * reps + "\nSummarise the passage in one sentence."


def ttft(salt):
    """Time to the first streamed content chunk, plus the prompt length the server saw."""
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt(salt)}],
        "temperature": 0, "max_tokens": 32, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter()
    first = None
    ptok = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                ptok = ev["usage"]["prompt_tokens"]
            ch = ev.get("choices") or []
            if first is None and ch and (ch[0].get("delta") or {}).get("content"):
                first = time.perf_counter() - t0
    return first, time.perf_counter() - t0, ptok


rows = []
for i in range(REPS):
    kind = "cold" if i == 0 else "warm"
    # A stream that yields no content chunk is a data point, not a crash: it is what a
    # dead engine looks like from the client side, and formatting None as a float ends
    # the run with a TypeError that hides it. Same failure @mjungnickel18 fixed in
    # bugb_sweep.py with `content or ""`.
    try:
        f, tot, ptok = ttft(f"{TAG}-{i}")
    except Exception as e:
        print(f"[{TAG}] req {i} ({kind:4s}): FAILED {type(e).__name__}: {e}", flush=True)
        rows.append((kind, None))
        continue
    if f is None:
        print(f"[{TAG}] req {i} ({kind:4s}): NO CONTENT after {tot:.2f} s "
              f"(prompt={ptok} tok) -- engine may be dead, check /health", flush=True)
        rows.append((kind, None))
        continue
    rows.append((kind, f))
    print(f"[{TAG}] req {i} ({kind:4s}): TTFT={f*1000:8.1f} ms  total={tot:6.2f} s  prompt={ptok} tok",
          flush=True)

warm = [f for k, f in rows if k == "warm" and f is not None]
cold = rows[0][1]
if not warm:
    print(f"[{TAG}] SUMMARY cold={'n/a' if cold is None else f'{cold*1000:.1f} ms'}  "
          f"no usable warm samples of {max(0, REPS - 1)}")
else:
    print(f"[{TAG}] SUMMARY cold={'n/a' if cold is None else f'{cold*1000:.1f} ms'}  "
          f"warm_median={sorted(warm)[len(warm)//2]*1000:.1f} ms  "
          f"warm_min={min(warm)*1000:.1f} ms  n_warm={len(warm)}")
sys.exit(2 if all(f is None for _, f in rows) else 0)
