"""Concurrency ladder: per-stream decode tok/s at N=1..8, with the numbers that say
WHY it falls off -- forward passes, preemptions, KV pool occupancy, and the spread
across streams.

  venv/bin/python bench/conc_ladder.py [--max-n 8 | --n 1,2,4,8] [--out 256] [--reps 2]
                                       [--ctx-tokens 4096] [--shared] [--json f.json]

Three throughput columns, because they answer different questions and disagree:
per-stream is each stream's own first-token-to-last-token rate, averaged -- what one
user feels, but inflated at high N because a stream that outlives the others finishes
alone. `decode agg` is the server's own generated-token counter over the decode-only
window (opened once EVERY stream has its first token), which is the steady-state
aggregate rate and the honest scaling number. `e2e agg` is all output tokens over the
whole wall clock including TTFT, so it is dominated by prefill once several long
prompts arrive at once. Prompts are distinct per stream, per repetition and per N (each carries
its own salt), so nothing here is ever served out of the prefix cache -- which is the
condition that matters, since a shared prefix hides exactly the failure this measures.
--shared makes them identical instead, which is what a cache-friendly client looks like.

Four columns diagnose a collapse, and they separate three different causes:

  tok/pass    a batched decode step at N streams and k drafts runs N*(k+1) query
              tokens. If this stays flat as N rises, the streams are taking turns
              instead of batching.
  ms/pass     wall time per forward pass over the decode-only window (from the last
              stream's first token to the last stream's last token). Flat ms/pass with
              rising tok/pass is healthy batching; ms/pass rising in proportion to N
              means the step is being paid N times.
  preempt     scheduler preemptions in the window. Nonzero means the KV pool could not
              hold the resident set and requests are being evicted and recomputed --
              a capacity problem, not a speculator problem.
  kv%         peak KV pool occupancy, sampled every 250 ms. Read it with preempt.

and the per-stream spread (min/med/max) separates "everyone is evenly slow" from "one
stream is served and the rest starve", which are different bugs.

One thing per-stream tok/s is NOT, once the prompts are long: a property of the decode
step. Chunked prefill puts somebody else's 16k prompt into the same forward pass as your
decode, so at N=4 with 16k-token prompts this reads 15 / 17 / 107 across three streams
while `decode agg` and `ms/pass` barely move. If you are trying to decide whether a
speculator batches, read those two columns and `run`; if you are trying to predict what
a user feels, read per-stream and TTFT, and know that most of what you are seeing is
queueing rather than the verify step.

Written for issue #16 (a reported per-stream dip at exactly N=2, which did not
reproduce here: 105 / 98 / 86 tok/s at N=1/2/3, SPEC=mtp k=3 PREFIX_CACHE=1) and
extended for issue #25 (dflash2 reported to collapse from N=2 up on 4-8 independent
long streams).
"""
import json, sys, time, threading

import harness


def _arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


MAXN = _arg("--max-n", 8)
# --n 1,2,4,8 runs those stream counts only. Long prompts make the full ladder mostly
# prefill, and the interesting N are the powers of two.
NLIST = ([int(x) for x in _arg("--n", "", str).split(",")] if "--n" in sys.argv
         else list(range(1, MAXN + 1)))
NOUT = _arg("--out", 256)
REPS = _arg("--reps", 2)
CTXTOK = _arg("--ctx-tokens", 4096)
JSONOUT = _arg("--json", "", str)
LABEL = _arg("--label", "", str)
SHARED = "--shared" in sys.argv

FILLER = ("The RTX 3090 has 24 GB of GDDR6X and 82 streaming multiprocessors. "
          "Memory bandwidth is 936 GB/s, which is what decode is bound by. ")
# 44 tokens per repetition of FILLER, counted with this model's tokenizer -- not the
# ~15 an earlier version of this file assumed, which made its "4k-token prompts"
# 11.4k-token ones and its N>1 rows a long-context measurement without saying so.
TOK_PER_REPEAT = 44
REPEATS = max(1, round(CTXTOK / TOK_PER_REPEAT))


def make_prompt(i, salt):
    # The salt goes FIRST: a distinct prefix is what defeats the prefix cache. A
    # distinct suffix would still let every stream share the body's blocks.
    head = "" if SHARED else f"Document {i} (revision {salt}). "
    return (head + FILLER * REPEATS +
            "\n\nWrite a detailed technical explanation of why memory bandwidth, "
            "not compute, limits single-stream decoding on this hardware. Be thorough.")


COUNTERS = ("vllm:generation_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_drafts_total",
            "vllm:iteration_tokens_total_sum",
            "vllm:iteration_tokens_total_count",
            "vllm:num_preemptions_total")
GAUGES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running",
          "vllm:num_requests_waiting")


def _short(name):
    return name.split("vllm:")[-1].replace("spec_decode_", "")


def metrics():
    m = harness.metrics(*COUNTERS, *GAUGES, timeout=20)
    return {_short(name): m.get(name, 0.0) for name in COUNTERS + GAUGES}


def stream(i, salt, res, first_tok):
    payload = {
        "messages": [{"role": "user", "content": make_prompt(i, salt)}],
        "max_tokens": NOUT, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter(); first = None; last = None; ntok = 0
    for d in harness.stream("/v1/chat/completions", payload, timeout=3600):
        if d.get("usage"):
            ntok = d["usage"]["completion_tokens"]
        ch = d.get("choices") or []
        if ch and ch[0].get("delta", {}).get("content"):
            now = time.perf_counter()
            if first is None:
                first = now
                first_tok[i] = now
            last = now
    res[i] = dict(ttft=(first - t0) if first else None,
                  decode_s=(last - first) if first and last else None, ntok=ntok,
                  rate=((ntok - 1) / (last - first)) if first and last and last > first else None)


def sampler(stop, peak):
    """Gauges are instantaneous; a run that preempts spends most of it below the peak."""
    while not stop.is_set():
        try:
            m = metrics()
            peak["kv"] = max(peak["kv"], m["kv_cache_usage_perc"])
            peak["running"] = max(peak["running"], m["num_requests_running"])
            peak["waiting"] = max(peak["waiting"], m["num_requests_waiting"])
        except Exception:
            pass
        stop.wait(0.25)


def run_one(n, salt):
    res, first_tok = {}, {}
    peak = {"kv": 0.0, "running": 0.0, "waiting": 0.0}
    stop = threading.Event()
    samp = threading.Thread(target=sampler, args=(stop, peak), daemon=True); samp.start()
    m0 = metrics()
    ts = [threading.Thread(target=stream, args=(i, salt, res, first_tok)) for i in range(n)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    # The measurement window is exactly the interval in which ALL N streams are
    # decoding: it opens when the last of them produces its first token (so no prefill
    # is inside it) and closes when the FIRST of them finishes (so no thinning tail is
    # either). Both ends matter -- a window that ran to the last completion would spend
    # its second half at N=1 and report a collapse that is really just stragglers.
    md, td, me, te = None, None, None, None
    while True:
        # Both snapshots are taken BEFORE the liveness check, or the last stream to
        # finish races the loop and the window silently loses its right-hand end.
        if md is None and len(first_tok) == n:
            md, td = metrics(), time.perf_counter()
        if me is None and len(res) >= 1:
            me, te = metrics(), time.perf_counter()
        if not any(t.is_alive() for t in ts):
            break
        time.sleep(0.02)
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    m1 = metrics(); t1 = time.perf_counter()
    stop.set()
    # When the pool cannot hold N requests there is no instant at which all N are
    # decoding, so the window is undefined and the row falls back to "from the last
    # first-token to the end" and is marked with a * -- which is itself the finding.
    full = md is not None and me is not None and te > td
    if md is None:
        md, td = m0, t0
    if not full:
        me, te = m1, t1

    good = [v for v in res.values() if v["decode_s"] and v["ntok"] > 1]
    if len(good) != n:
        return None
    rates = sorted(v["rate"] for v in good)
    per = sum(rates) / n
    agg = sum(v["ntok"] for v in good) / wall
    dpass = me["iteration_tokens_total_count"] - md["iteration_tokens_total_count"]
    dtok = me["iteration_tokens_total_sum"] - md["iteration_tokens_total_sum"]
    toks_per_pass = dtok / dpass if dpass else float("nan")
    win = te - td
    ms_per_pass = win / dpass * 1000 if dpass else float("nan")
    dgen = me["generation_tokens_total"] - md["generation_tokens_total"]
    dn = m1["num_drafts_total"] - m0["num_drafts_total"]
    da = m1["num_accepted_tokens_total"] - m0["num_accepted_tokens_total"]
    tps = (da / dn + 1) if dn else float("nan")
    return dict(n=n, per=per, agg=agg, full=full,
                decode_agg=(dgen / win if win > 0 else float("nan")),
                tok_step=tps, tok_pass=toks_per_pass,
                ms_pass=ms_per_pass, ttft=sum(v["ttft"] for v in good) / n * 1000,
                preempt=m1["num_preemptions_total"] - m0["num_preemptions_total"],
                kv=peak["kv"] * 100, running=peak["running"], waiting=peak["waiting"],
                rates=[round(r, 1) for r in rates])


print(f"prompts: {'shared' if SHARED else 'distinct'}, ~{CTXTOK} ctx tokens, "
      f"out={NOUT}, reps={REPS}, N={','.join(str(n) for n in NLIST)}"
      + (f", label={LABEL}" if LABEL else ""))
print(f"{'N':>2} {'per-stream':>11} {'decode agg':>11} {'e2e agg':>8} {'tok/step':>9} "
      f"{'tok/pass':>9} {'ms/pass':>8} {'preempt':>8} {'kv%':>6} {'run':>4} {'TTFT ms':>8}  spread")
rows = []
for n in NLIST:
    best = None
    for rep in range(REPS):
        r = run_one(n, salt=f"{LABEL}-n{n}-r{rep}-{int(time.time())}")
        if r is None:
            print(f"{n:>2}  incomplete"); continue
        # Keep the rep with the best STEADY-STATE aggregate: a collapse that survives
        # the best of two runs is not a transient, and a healthy config is not penalised
        # by one slow rep. Selecting on per-stream instead would prefer the rep where the
        # streams overlapped least, which is the opposite of what is being measured.
        if best is None or r["decode_agg"] > best["decode_agg"]:
            best = r
    if best:
        rows.append(best)
        s = best["rates"]
        mark = "" if best["full"] else "*"
        print(f"{n:>2} {best['per']:>11.1f} {best['decode_agg']:>10.1f}{mark:1} {best['agg']:>8.1f} "
              f"{best['tok_step']:>9.2f} {best['tok_pass']:>9.1f} {best['ms_pass']:>8.1f} "
              f"{best['preempt']:>8.0f} {best['kv']:>6.1f} {best['running']:>4.0f} "
              f"{best['ttft']:>8.0f}  {s[0]:.0f}/{s[len(s)//2]:.0f}/{s[-1]:.0f}")
print()
if any(not r["full"] for r in rows):
    print("* no instant at which all N streams were decoding: the pool could not hold "
          "them, so that row is a tail measurement, not a steady state.")
print("| N | " + " | ".join(str(r["n"]) for r in rows) + " |")
print("|---|" + "---|" * len(rows))
print("| per-stream tok/s | " + " | ".join(f"{r['per']:.1f}" for r in rows) + " |")
print("| decode aggregate tok/s | " + " | ".join(f"{r['decode_agg']:.0f}" for r in rows) + " |")
print("| end-to-end aggregate tok/s | " + " | ".join(f"{r['agg']:.0f}" for r in rows) + " |")
print("| tok/step | " + " | ".join(f"{r['tok_step']:.2f}" for r in rows) + " |")
print("| tok/pass | " + " | ".join(f"{r['tok_pass']:.1f}" for r in rows) + " |")
print("| ms/pass | " + " | ".join(f"{r['ms_pass']:.1f}" for r in rows) + " |")
print("| preemptions | " + " | ".join(f"{r['preempt']:.0f}" for r in rows) + " |")
if JSONOUT:
    json.dump({"label": LABEL, "ctx_tokens": CTXTOK, "out": NOUT, "shared": SHARED,
               "rows": rows}, open(JSONOUT, "w"), indent=1)
    print(f"\nwrote {JSONOUT}")
