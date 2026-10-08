#!/usr/bin/env python3
"""Prefix reuse under alternating conversations — reproducer and A/B arm.

Two independent long conversations (A and B), advanced in strict alternation.
For A, every one of B's turns is interleaved traffic, and vice versa. Each turn
is its own request, which is what an agentic loop does.

The pass criterion is not "a high hit rate" but "this turn reused the whole
prefix the previous turn established". A hybrid model never serves back the block
holding the previous request's end (#102), so a healthy turn reuses
``max(0, prev_prompt // B - 1) * B`` tokens, B being the engine's hybrid attention
block, read from ``vllm:cache_config_info`` on /metrics. A turn below that is
PREFIX-LOST. If the block cannot be read, the fallback is 90% of the previous
prompt, which misreads short conversations: with B = 2176 a healthy turn under
~44K tokens is below 90%.

Regimes are sharp, so read the size table in the issue rather than one run:

    one conversation, no second one     ~99.7% reused
    two conversations, sequential       ~96.6-99.7%
    two conversations, alternating      0% every turn  (above a size knee)

Exits 1 when the arm reproduces the defect (a PREFIX-LOST turn), 0 when it does not. It
exits 2 when it checked no turn. Round 1 is cold, so a run needs at least two rounds inside
the budget. A clean arm is not a fix on its own: run the dense arm too. If that is also
clean, the test failed to reproduce rather than the fix working.

Usage:
    python bench/prefix_alternation.py --target-tokens 60000 --rounds 3
    python bench/prefix_alternation.py --target-tokens 96000 --noise 2 --rounds 4
    HQ_UNIT=qwen38-hq-vllm python bench/prefix_alternation.py    # engine-env label
"""
import argparse
import re
import datetime as dt
import json
import os
import subprocess
import sys
import time

import harness

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b")
UNIT = os.environ.get("HQ_UNIT", "")

LOG = []


def engine_block():
    """The engine's resolved hybrid attention block (what its boot line calls the
    attention block size), from /metrics; None if it cannot be read. When the engine
    was launched with an explicit --block-size (CTX=huge passes 128, KVarN's tile),
    cache_config_info reports that value rather than the resolved hybrid block, so it
    is not trusted: pass --block from the boot line instead."""
    try:
        argv = [x.decode(errors="replace")
                for x in open(f"/proc/{_engine_pid()}/cmdline", "rb").read().split(b"\0")]
        if any(x == "--block-size" or x.startswith("--block-size=") for x in argv):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        for name, labels, _ in harness.samples():
            if name == "vllm:cache_config_info":
                m = re.search(r'\bblock_size="(\d+)"', labels)
                return int(m.group(1)) if m else None
        return None
    except Exception:  # noqa: BLE001
        return None


def healthy_floor(prev_prompt, block):
    """Least cached_tokens a turn that reused the whole previous prefix can show."""
    if not block:
        return 0.9 * prev_prompt
    return max(0, prev_prompt // block - 1) * block


def call(messages, max_tokens=24):
    payload = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    d = harness.post("/v1/chat/completions", payload, timeout=1800)
    el = time.time() - t0
    u = d["usage"]
    det = u.get("prompt_tokens_details") or {}
    return {"prompt": u["prompt_tokens"], "cached": det.get("cached_tokens", 0),
            "gen": u["completion_tokens"], "secs": el,
            "text": (d["choices"][0]["message"].get("content") or "").strip()}


FILLERS = {
    "A": ("The scheduler interleaves prefill chunks with decode steps. "
          "A Mamba state snapshot is materialised at the last prefill chunk boundary. "
          "Attention prefix caching is the intersection of per-group hit sets. "),
    "B": ("Quantisation geometry decides how many tokens fit in the pool. "
          "Speculative decoding acceptance is what turns steps into throughput. "
          "Chunked prefill bounds how long a co-tenant can be starved. "),
    "N": ("Unrelated interleaved traffic allocates blocks between turns. "
          "This text exists only to consume KV blocks and force eviction order. "),
}


RUN = f"{time.time_ns():x}"   # per-run salt: round 1 must be cold even if a previous run's prefix is cached


def make_doc(side, n_chars):
    filler = FILLERS[side]
    body = filler * (n_chars // len(filler) + 1)
    lines = [body[i:i + 900] for i in range(0, n_chars, 900)]
    return f"[run {RUN}]\n" + "\n".join(
        f"[{side}{i:05d}] {ln}" for i, ln in enumerate(lines))[:n_chars]


def log(rec):
    LOG.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def _engine_pid():
    """PID of the serving engine: HQ_UNIT's MainPID if set (user unit first, then
    system), else the `vllm serve` process on this API's port, found from /proc.
    This repo's launcher execs `vllm serve`, so the port is on its command line; the
    port scan also works inside a container, where there is no unit."""
    if UNIT:
        for scope in (["--user"], []):
            try:
                pid = subprocess.check_output(
                    ["systemctl", *scope, "show", UNIT, "-p", "MainPID", "--value"],
                    text=True, timeout=10, stderr=subprocess.DEVNULL).strip()
                if pid and pid != "0":
                    return pid
            except Exception:  # noqa: BLE001
                pass
    port = harness.base_url().split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[-1]
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        args = [x.decode(errors="replace") for x in argv]
        if "serve" in args and any("vllm" in x for x in args[:3]) and \
                any(x == "--port" and nxt == port for x, nxt in zip(args, args[1:])):
            return pid
    raise RuntimeError(f"no vllm serve on port {port} (set HQ_UNIT to name its unit)")


def engine_env():
    """Read the ENGINE process's settings, not this shell's.

    Otherwise an arm label is whatever the caller happened to export, which is
    how a run gets reported against the wrong arm. The launcher computes both
    values and passes them as flags, so the command line is read first and the
    environment is only the fallback (the deprecated env spelling of the
    retention interval). Returns placeholders if the engine cannot be read.
    """
    try:
        pid = _engine_pid()
        argv = [x.decode(errors="replace")
                for x in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")]
        raw = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
        env = dict(x.split("=", 1) for x in raw.split("\0") if "=" in x)

        def flag(name):
            for i, tok in enumerate(argv):
                if tok == name and i + 1 < len(argv):
                    return argv[i + 1]
                if tok.startswith(name + "="):
                    return tok.split("=", 1)[1]
            return None

        ret = flag("--prefix-cache-retention-interval") \
            or env.get("VLLM_PREFIX_CACHE_RETENTION_INTERVAL") or "dense"
        return ret, flag("--max-model-len") or env.get("MAX_LEN", "?")
    except Exception as exc:  # noqa: BLE001
        return f"<err:{exc}>", "?"


def main():
    ap = argparse.ArgumentParser(
        description="Prefix reuse under alternating conversations (A/B arm).")
    ap.add_argument("--target-tokens", type=int, default=100000,
                    help="base document size per conversation, in tokens")
    ap.add_argument("--rounds", type=int, default=8, help="turns per conversation")
    ap.add_argument("--noise", type=int, default=2,
                    help="unrelated requests injected between turns")
    ap.add_argument("--noise-tokens", type=int, default=4000)
    ap.add_argument("--budget-min", type=float, default=25.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--block", type=int, default=0,
                    help="the engine's hybrid attention block, from its boot line "
                         "('Setting attention block size to N tokens'); default: read "
                         "from /metrics, which is only right without an explicit --block-size")
    args = ap.parse_args()

    retention, maxlen = engine_env()
    block = args.block or engine_block()
    tag = f"RET{retention}_ML{maxlen}"
    t_start = time.time()
    deadline = t_start + args.budget_min * 60
    out = args.out or os.path.join(
        HERE, f"prefix-alternation-{tag}-{dt.datetime.now():%Y%m%d-%H%M%S}.json")
    print(f"══ alternating-conversation prefix reuse ══  engine: retention={retention} "
          f"MAX_LEN={maxlen}  (read from /proc/<engine>, not this shell)")
    print(f"   engine attention block: {block if block else 'unreadable, falling back to 90% (pass --block <the boot line attention block size>)'}")
    print(f"   {args.target_tokens:,} tok/conversation  rounds={args.rounds}  "
          f"noise={args.noise}x{args.noise_tokens} tok  budget={args.budget_min:.0f} min")
    print(f"   output -> {out}\n")

    # Calibrate tokens/char against THIS stack's tokenizer; a hardcoded ratio
    # silently changes the arm you think you ran by tens of percent.
    cal = call([{"role": "user", "content": "CALIB\n" + make_doc("A", 20000)}], max_tokens=1)
    ratio = cal["prompt"] / 20000.0
    n_chars = int(args.target_tokens / ratio)
    print(f"calibration: 20,000 chars -> {cal['prompt']:,} tok ({ratio:.4f} tok/char)")
    print(f"per-conversation document: {n_chars:,} chars ≈ {args.target_tokens:,} tok\n")
    log({"event": "calib", "ratio": ratio, "n_chars": n_chars, "arm": tag})

    docs = {s: make_doc(s, n_chars) for s in ("A", "B")}
    conv = {s: [{"role": "user", "content": docs[s]}] for s in ("A", "B")}
    prev_prompt = {s: 0 for s in ("A", "B")}
    stats = {"reqs": 0, "lost": 0, "healthy": 0, "cold": 0}
    fail_rows = []

    for rnd in range(1, args.rounds + 1):
        for side in ("A", "B"):
            if time.time() > deadline:
                print(f"\nbudget exhausted before round {rnd} {side}")
                rnd = args.rounds + 1
                break
            q = (f"This is {side}'s question for round {rnd}. Answer in one sentence, "
                 f"do not restate the document.")
            conv[side].append({"role": "user", "content": q})
            res = call(conv[side], max_tokens=24)
            conv[side].append({"role": "assistant", "content": res["text"] or "OK"})

            stats["reqs"] += 1
            if rnd == 1:
                stats["cold"] += 1
                verdict = "COLD"
            else:
                need = healthy_floor(prev_prompt[side], block)
                if res["cached"] < need:
                    stats["lost"] += 1
                    verdict = "PREFIX-LOST"
                    fail_rows.append((rnd, side, prev_prompt[side], res["cached"], res["secs"]))
                else:
                    stats["healthy"] += 1
                    verdict = "ok"
            pct = (res["cached"] / res["prompt"] * 100) if res["prompt"] else 0
            print(f"  r{rnd:>2} {side}  {verdict:<11} "
                  f"prompt={res['prompt']:>7,}  cached={res['cached']:>7,} ({pct:5.1f}%)  "
                  f"{res['secs']:6.2f}s")
            log({"event": "turn", "round": rnd, "side": side, "verdict": verdict,
                 "prompt": res["prompt"], "cached": res["cached"], "pct": round(pct, 1),
                 "secs": round(res["secs"], 2), "prev_prompt": prev_prompt[side]})
            prev_prompt[side] = res["prompt"]

            for k in range(args.noise):
                if time.time() > deadline:
                    break
                nd = f"NOISE-{side}-{rnd}-{k}-{time.time():.6f}\n" + make_doc(
                    "N", int(args.noise_tokens / ratio))
                nr = call([{"role": "user", "content": nd}], max_tokens=4)
                log({"event": "noise", "round": rnd, "side": side, "k": k,
                     "prompt": nr["prompt"], "cached": nr["cached"],
                     "secs": round(nr["secs"], 2)})
        if time.time() > deadline:
            break

    el = time.time() - t_start
    print("\n" + "=" * 84)
    print(f"arm={tag}  {el:.0f}s  conversation requests {stats['reqs']} "
          f"(cold {stats['cold']} / healthy {stats['healthy']} / **lost {stats['lost']}**)")
    if fail_rows:
        print("\nprefix reuse lost:")
        for r, s, pp, c, sec in fail_rows:
            print(f"  r{r} {s}  previous prefix {pp:,} tok, reused {c:,} tok "
                  f"(short by {pp - c:,}, {sec:.1f}s)")
        print("\nthis arm REPRODUCES the defect.")
    elif not stats["healthy"]:
        print("\nINVALID: no turn was checked. Round 1 is cold, so run at least two")
        print("   rounds inside --budget-min.")
    else:
        print("\n✅ no prefix loss: every turn reused the previous turn's whole prefix.")
        print("   ⚠️ This only says so for this arm. To claim a fix, run the dense")
        print("      arm too — if that is also clean, the test failed to reproduce")
        print("      rather than the fix working.")
    with open(out, "w") as f:
        json.dump({"arm": tag, "args": vars(args), "stats": stats,
                   "elapsed_s": el, "log": LOG}, f, ensure_ascii=False, indent=1)
    print(f"\nraw records -> {out}")
    return 1 if fail_rows else 0 if stats["healthy"] else 2


if __name__ == "__main__":
    sys.exit(main())
