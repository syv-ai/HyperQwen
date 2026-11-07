import json, os, random, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "8101"))
MODEL = os.environ.get("MODEL", "qwen38-27b-170hx")
API_KEY = os.environ.get("API_KEY", "")
IN = int(sys.argv[1]) if len(sys.argv) > 1 else 262000
OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 64
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 9111

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

t0 = time.time()
first = None
chunks = []
usage = None
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        now = time.time()
        if d.get("usage"):
            usage = d["usage"]
        ch = d.get("choices") or []
        if ch and (ch[0].get("text") or ""):
            if first is None:
                first = now
            chunks.append(now)
t1 = time.time()

ttft = (first - t0) * 1000 if first else float("nan")
lat = [ (chunks[i] - chunks[i-1]) * 1000 for i in range(1, len(chunks)) ]
gen = usage["completion_tokens"] if usage else OUT
steps = len(chunks)
print(f"in={IN} out={OUT} step_chunks={steps} completion_tokens={gen} "
      f"TTFT_ms={ttft:.1f} wall_ms={(t1-t0)*1000:.1f} "
      f"decode_ms={(chunks[-1]-chunks[0])*1000 if len(chunks)>1 else 0:.1f} "
      f"step_ms_mean={sum(lat)/len(lat) if lat else 0:.1f} "
      f"step_ms={[round(x,1) for x in lat]}")
