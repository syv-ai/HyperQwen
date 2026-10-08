"""The "!!!!" collapse from #208: two requests prefilling in the same steps.

Each trial sends a short request and a long one (~9k tokens) at the same moment, both over a
prefix no earlier trial has used (a fresh nonce in the first block), and then sends the long
one again alone, as a full prefix hit. A trial collapses when any of the three answers carries
a run of 8 or more "!" (token 0). That's the symptom #208 traced to a late KVarN flush landing
on a page that had become another request's mamba state. The failure is probabilistic (4 of 30
on a single 3090 on 0.28 in #208), and KVarN's fp16 pool has to fill with finished requests'
sinks before sinks are evicted, so run all the trials on one boot, one after the other.
Exits 1 if any trial collapsed or a request errored.

  venv/bin/python bench/concurrent_collapse.py <label> [trials=30] [long_tokens=9000]

On CTX=huge SPEC=dflash2 PREFIX_CACHE=1. With kvarn/kvarn-recycled-pages-0.30.0.patch applied the
engine log says "KVarN: dropped the pending tiles of N block(s)" the first time the fix fires.
"""
import json, os, re, secrets, sys, threading, urllib.request

LABEL = sys.argv[1]
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
LONG = int(sys.argv[3]) if len(sys.argv) > 3 else 9000
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
BASE = "http://127.0.0.1:" + os.environ.get("PORT", "18020")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
# Filler from the repo's own docs, so the script needs nothing but a checkout.
DOCS = "\n\n".join(open(os.path.join(REPO, f)).read() for f in sorted(os.listdir(REPO)) if f.endswith(".md"))
DOCS += "\n\n" + "\n\n".join(open(os.path.join(REPO, "docs", f)).read()
                            for f in sorted(os.listdir(os.path.join(REPO, "docs"))) if f.endswith(".md"))
BANG = re.compile(r"!{8,}")


def post(messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}),
        timeout=1800).read().decode())
    return r["choices"][0]["message"].get("content") or "", r["usage"]["prompt_tokens"]


def corrupted():  # counts only with VLLM_COMPUTE_NANS_IN_LOGITS=1; None when the metric is absent
    try:
        txt = urllib.request.urlopen(urllib.request.Request(
            BASE + "/metrics", headers={"Authorization": "Bearer " + KEY}), timeout=30).read().decode()
    except OSError:
        return None
    v = re.findall(r"^vllm:corrupted_requests_total(?:\{[^}]*\})? ([0-9.e+]+)$", txt, re.M)
    return sum(float(x) for x in v) if v else None


def trial(i):
    nonce = secrets.token_hex(16)            # first block differs from every earlier trial
    start = (i * 7919) % max(1, len(DOCS) - LONG * 4)
    long_msgs = [{"role": "system", "content": f"Session {nonce}. You are a careful technical assistant."},
                 {"role": "user", "content": DOCS[start:start + LONG * 4] +
                  "\n\nSummarise the text above in eight bullet points."}]
    short_msgs = [{"role": "system", "content": f"Session {nonce}-s. Answer with one JSON object."},
                  {"role": "user", "content": 'Give {"city": ..., "country": ...} for the capital of Denmark.'}]
    out = {}

    def run(name, msgs, n):
        try:
            out[name] = post(msgs, n)
        except Exception as e:           # a 400 on NaN logits is a finding, not a crash
            out[name] = ("ERROR " + repr(e), 0)

    ts = [threading.Thread(target=run, args=("long", long_msgs, 300)),
          threading.Thread(target=run, args=("short", short_msgs, 40))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    run("rehit", long_msgs, 300)              # the same prompt alone, a full prefix hit
    bad = [k for k in ("long", "short", "rehit") if BANG.search(out[k][0]) or out[k][0].startswith("ERROR")]
    return out, bad


c0 = corrupted()
collapsed = 0
for i in range(TRIALS):
    out, bad = trial(i)
    collapsed += bool(bad)
    print(f"[{LABEL}] trial {i + 1:2d}/{TRIALS}  prompt {out['long'][1]:5d} tok  "
          f"{'COLLAPSED ' + ','.join(bad) if bad else 'ok'}  {out['long'][0][:60]!r}", flush=True)
    for k in bad:
        print(f"    {k}: {out[k][0][:200]!r}", flush=True)
c1 = corrupted()
nan = "" if c0 is None or c1 is None else f", corrupted_requests_total +{c1 - c0:.0f}"
print(f"[{LABEL}] DONE: {collapsed} of {TRIALS} trials collapsed{nan}")
sys.exit(1 if collapsed else 0)
