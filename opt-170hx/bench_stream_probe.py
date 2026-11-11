import json, os, random, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "8101"))
MODEL = os.environ.get("MODEL", "qwen38-27b-170hx")
API_KEY = os.environ.get("API_KEY", "")
IN = int(sys.argv[1]) if len(sys.argv) > 1 else 262000
OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 64
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 9111

if min(IN, OUT) < 1:
    print("input/output counts and concurrency must be positive", file=sys.stderr)
    sys.exit(2)

rng = random.Random(SEED)
prompt = [rng.randrange(1000, 200000) for _ in range(IN)]
body = json.dumps({
    "model": MODEL,
    "prompt": prompt,
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
    f"http://127.0.0.1:{PORT}/v1/completions",
    data=body, headers=headers)

t0 = time.perf_counter()
first = None
chunks = []
usage = None
done = False
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
        if ch and (ch[0].get("text") or ""):
            if first is None:
                first = now
            chunks.append(now)
if not done:
    raise RuntimeError("incomplete stream: missing [DONE]")
if not chunks or not usage or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 1:
    raise RuntimeError("stream has no generated text or valid completion usage")
t1 = time.perf_counter()

ttft = (first - t0) * 1000 if first else float("nan")
lat = [ (chunks[i] - chunks[i-1]) * 1000 for i in range(1, len(chunks)) ]
gen = usage["completion_tokens"]
steps = len(chunks)
print(f"in={IN} out={OUT} step_chunks={steps} completion_tokens={gen} "
      f"TTFT_ms={ttft:.1f} wall_ms={(t1-t0)*1000:.1f} "
      f"decode_ms={(chunks[-1]-chunks[0])*1000 if len(chunks)>1 else 0:.1f} "
      f"step_ms_mean={sum(lat)/len(lat) if lat else 0:.1f} "
      f"step_ms={[round(x,1) for x in lat]}")
