#!/usr/bin/env python3
"""Interleaved-traffic dose-response, and the single-conversation control.

Both controls exist because the obvious experiment is misleading.

1. Single conversation, advancing N turns with nothing else on the box. If this
   is clean, the cache is working and the defect is about interleaving, not about
   caching being broken. Without this arm, "conversation X lost its prefix" is
   uninterpretable.

2. One conversation with a second, independent request injected between two of
   its turns, sweeping the injected size. The knee this finds is the quantity to
   compare across arms: it moves with pool size roughly linearly, which is how
   capacity is separated from a fixed structural trigger.

The injected document is sized AFTER calibrating tokens/char against the running
tokenizer. A hardcoded ratio is a silent arm change — the first version of this
script asked for 55,000 tokens, produced 48,000, and landed below the knee it was
trying to find.

Usage:
    python bench/interleave_dose.py --mode single --tokens 130000 --turns 3
    python bench/interleave_dose.py --mode dose --tokens 60000 --doses 0,5000,30000,60000
"""
import argparse
import json
import sys
import time

import harness

WORDS = ("the scheduler interleaves prefill chunks with decode steps while a mamba "
         "state snapshot is materialised at the last prefill chunk boundary so the "
         "attention prefix cache is the intersection of per-group hit sets").split()

RATIO = None  # calibrated at runtime


def make_doc_chars(n_chars, salt):
    out, cur, i = [], 0, 0
    while cur < n_chars:
        w = WORDS[i % len(WORDS)]
        out.append(w)
        cur += len(w) + 1
        i += 1
    return f"[{salt}] " + " ".join(out)


def make_doc(tokens, salt):
    return make_doc_chars(int(tokens / RATIO), salt)


def ask(messages, max_tokens=24):
    payload = {
        "messages": messages, "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    d = harness.post("/v1/chat/completions", payload, timeout=1800)
    el = time.time() - t0
    u = d["usage"]
    det = u.get("prompt_tokens_details") or {}
    return {"prompt": u["prompt_tokens"], "cached": det.get("cached_tokens", 0),
            "secs": el, "text": (d["choices"][0]["message"].get("content") or "").strip()}


def calibrate():
    global RATIO
    n_chars = 20000
    doc = make_doc_chars(n_chars, "calib")
    r = ask([{"role": "user", "content": "CALIB\n" + doc}], max_tokens=1)
    RATIO = r["prompt"] / float(len(doc))
    print(f"calibration: {len(doc):,} chars -> {r['prompt']:,} tok ({RATIO:.4f} tok/char)")


def mode_single(args):
    doc = make_doc(args.tokens, f"single-{int(time.time())}")
    conv = [{"role": "user", "content": doc}]
    print(f"\nsingle conversation, {args.turns} turns, no other traffic")
    prev = 0
    for t in range(1, args.turns + 1):
        if t > 1:
            conv.append({"role": "user", "content": f"round {t}: one short sentence."})
        r = ask(conv)
        if t == 1:
            verdict = "COLD"
        else:
            verdict = "ok" if r["cached"] > 0.5 * prev else "PREFIX-LOST"
        print(f"  turn {t}  {verdict:<12} prompt={r['prompt']:>8,} "
              f"cached={r['cached']:>8,} ({r['cached'] / max(r['prompt'], 1) * 100:5.1f}%) "
              f"{r['secs']:6.2f}s")
        sys.stdout.flush()
        conv.append({"role": "assistant", "content": r["text"] or "ok"})
        prev = r["prompt"]
    print("\nclean here means the cache works; interleaving is what breaks it.")


def mode_dose(args):
    doses = [int(x) for x in args.doses.split(",")]
    print(f"\nbig conversation {args.tokens:,} tok; injected sizes {doses}")
    print(f"  {'injected':>10} | {'turn2 prompt':>13} {'cached':>10} {'reused':>7} {'secs':>7} | verdict")
    print("  " + "-" * 68)
    results = []
    for dose in doses:
        salt = f"dose{dose}-{int(time.time())}"
        conv = [{"role": "user", "content": make_doc(args.tokens, salt + "-A")}]
        r1 = ask(conv)
        conv.append({"role": "assistant", "content": r1["text"] or "ok"})
        if dose > 0:
            ask([{"role": "user", "content": make_doc(dose, salt + "-B")}])
        conv.append({"role": "user", "content": "round 2: one short sentence."})
        r2 = ask(conv)
        pct = r2["cached"] / max(r2["prompt"], 1) * 100
        verdict = "ok" if pct > 50 else ("partial" if pct > 1 else "WIPED")
        print(f"  {dose:>10,} | {r2['prompt']:>13,} {r2['cached']:>10,} {pct:>6.1f}% "
              f"{r2['secs']:>6.2f}s | {verdict}")
        sys.stdout.flush()
        results.append({"dose": dose, "prompt": r2["prompt"],
                        "cached": r2["cached"], "pct": round(pct, 1),
                        "secs": round(r2["secs"], 2), "verdict": verdict})
    print("\ncompare the knee across arms; it should move with the KV pool size.")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("single", "dose"), default="dose")
    ap.add_argument("--tokens", type=int, default=60000,
                    help="the long conversation's size, in tokens")
    ap.add_argument("--turns", type=int, default=3, help="--mode single only")
    ap.add_argument("--doses", default="0,5000,30000,60000",
                    help="--mode dose only; injected sizes in tokens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    calibrate()
    if args.mode == "single":
        mode_single(args)
        return 0
    results = mode_dose(args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "ratio": RATIO, "results": results},
                      f, ensure_ascii=False, indent=1)
        print(f"raw records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
