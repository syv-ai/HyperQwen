#!/usr/bin/env python3
"""Stress a running server with a long-context, vision and burst mix: the load docs/vllm-0.34.md's extensible KV
cache table was measured with (alternative.sh at 160k, VISION=1, MAX_SEQS=4).

Run on the box that serves (it samples nvidia-smi locally). Stdlib only.

  VLLM_API_KEY=... python3 ekv_stress.py TAG PORT [GPU_INDEX]

Phases:
  1 spike, 3 rounds: four requests at once, each round over a fresh nonce so nothing hits the prefix cache:
      two long prompts (~58k tokens, 256 out), one 8-image prompt (8 PNGs of 1024x1024, 256 out),
      one short prompt with a long answer (1024 out).
  2 reuse: turn 2 of round 3's two long prompts (the prompt, its answer, a follow-up); prints cached_tokens.
  3 burst: 16 requests at concurrency 4, alternating 4-image prompts and ~6k-token text prompts (128 out).
Prints one line per request, then a summary: errors, HTTP status counts, peak card MiB, peak
kv_cache_usage_perc, and turn-2 cached share. Exit 1 if any request failed, or 2 if requests
passed but either telemetry source produced no usable samples.
"""
import base64, http.client, json, math, os, random, struct, subprocess, sys, threading, time, urllib.error, urllib.request, zlib

sys.stdout.reconfigure(encoding="utf-8")
TAG, PORT = sys.argv[1], int(sys.argv[2])
GPU = sys.argv[3] if len(sys.argv) > 3 else "0"
B = f"http://127.0.0.1:{PORT}"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
MODEL = "qwen3.8-27b"
rng = random.Random(20261119)
WORDS = ("river stone ledger harbor lantern copper meadow signal winter orchard engine canyon thread "
         "marble pilot archive cedar valley quartz beacon summit market falcon prism anchor willow "
         "circuit glacier ember compass atlas violet tunnel harvest saddle ferry granite").split()


def text(nwords, seed):
    r = random.Random(seed)
    out, line = [], []
    for i in range(nwords):
        line.append(r.choice(WORDS))
        if len(line) == 14:
            out.append(" ".join(line) + ".")
            line = []
    return " ".join(out)


def png(w, h, seed):
    r = random.Random(seed)
    cols = [bytes(r.randrange(256) for _ in range(3)) for _ in range(4)]
    rows = []
    for y in range(h):
        c = cols[(y * 4) // h]
        rows.append(b"\0" + c * w)
    raw = zlib.compress(b"".join(rows), 6)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", raw) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


samples = {"mib": None, "kv": None}
stop = threading.Event()


def record_peak(name, value):
    if math.isfinite(value) and (samples[name] is None or value > samples[name]):
        samples[name] = value


def sample_kv(text):
    metric = "vllm:kv_cache_usage_perc"
    for line in text.splitlines():
        if line.startswith(metric + "{"):
            fields = line.rpartition("}")[2].split()
        else:
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or fields[0] != metric:
                continue
            fields = fields[1].split()
        if fields:
            try:
                record_peak("kv", float(fields[0]))
            except ValueError:
                continue


def sampler():
    while not stop.is_set():
        try:
            m = int(subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10, check=True).stdout.strip())
            record_peak("mib", m)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            with urllib.request.urlopen(urllib.request.Request(B + "/metrics", headers=H), timeout=10) as response:
                sample_kv(response.read().decode())
        except (OSError, UnicodeError, http.client.HTTPException):
            pass
        stop.wait(0.5)


results = []
lock = threading.Lock()


def send(name, messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    rec = {"name": name}
    try:
        o = json.load(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions",
                                             json.dumps(body).encode(), H), timeout=1800))
        u = o.get("usage") or {}
        rec.update(status=200, prompt=u.get("prompt_tokens"), out=u.get("completion_tokens"),
                   cached=((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                   reply=o["choices"][0]["message"].get("content") or "")
    except urllib.error.HTTPError as e:
        rec.update(status=e.code, error=e.read().decode(errors="replace")[:300])
    except Exception as e:
        rec.update(status=-1, error=f"{type(e).__name__}: {e}"[:300])
    rec["s"] = round(time.time() - t0, 1)
    with lock:
        results.append(rec)
        print(f"[{TAG}] {name}: status {rec['status']} {rec['s']}s prompt {rec.get('prompt')} out {rec.get('out')} "
              f"cached {rec.get('cached')} {rec.get('error', '')}", flush=True)
    return rec


def run_parallel(jobs):
    ts = [threading.Thread(target=lambda j=j: j[3].update(send(j[0], j[1], j[2]))) for j in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def image_msg(n, seed, question):
    parts = [{"type": "image_url", "image_url": {"url": png(1024, 1024, seed * 100 + i)}} for i in range(n)]
    return [{"role": "user", "content": parts + [{"type": "text", "text": question}]}]


monitor = threading.Thread(target=sampler, daemon=True)
monitor.start()
LONG_WORDS = int(os.environ.get("LONG_WORDS", "49000"))  # about 58k tokens with Qwen3.8's tokenizer (40000 words = 47,255)
long_last = []
for rnd in (1, 2, 3):
    nonce = f"{TAG}-{rnd}-{rng.randrange(10**9)}"
    jobs = []
    for k in (1, 2):
        msgs = [{"role": "user", "content": f"[{nonce}-{k}] Summarize this log in three sentences.\n\n" +
                 text(LONG_WORDS, f"{nonce}-{k}")}]
        holder = {}
        jobs.append((f"r{rnd}-long{k}", msgs, 256, holder))
        if rnd == 3:
            long_last.append((msgs, holder))
    jobs.append((f"r{rnd}-img8", image_msg(8, rnd, f"[{nonce}] For each image, name its four band colors top to bottom."), 256, {}))
    jobs.append((f"r{rnd}-decode", [{"role": "user", "content": f"[{nonce}] Write a long, detailed story about a lighthouse keeper."}], 1024, {}))
    run_parallel(jobs)

turn2 = []
for k, (msgs, holder) in enumerate(long_last, 1):
    m2 = msgs + [{"role": "assistant", "content": holder.get("reply", "")},
                 {"role": "user", "content": "Now list the five most frequent words in it."}]
    turn2.append((f"reuse-long{k}", m2, 128, {}))
run_parallel(turn2)

burst = []
for i in range(16):
    if i % 2:
        burst.append((f"burst{i}", image_msg(4, 50 + i, "Name each image's colors."), 128, {}))
    else:
        burst.append((f"burst{i}", [{"role": "user", "content": f"[{TAG}-b{i}] Summarize:\n\n" + text(4500, f"b{i}")}], 128, {}))
for i in range(0, 16, 4):
    run_parallel(burst[i:i + 4])
stop.set()
monitor.join()

bad = [r for r in results if r["status"] != 200]
codes = {}
for r in results:
    codes[r["status"]] = codes.get(r["status"], 0) + 1
shares = [
    f"{r['cached']}/{r['prompt']}" if r["status"] == 200 else "unavailable"
    for r in results if r["name"].startswith("reuse")
]
peak_mib = "unavailable" if samples["mib"] is None else samples["mib"]
peak_kv = "unavailable" if samples["kv"] is None else f"{samples['kv']:.3f}"
print(f"[{TAG}] SUMMARY requests {len(results)} failed {len(bad)} status {codes} peak_card_MiB {peak_mib} "
      f"peak_kv_usage {peak_kv} turn2_cached {shares}", flush=True)
sys.exit(1 if bad else 2 if None in samples.values() else 0)
