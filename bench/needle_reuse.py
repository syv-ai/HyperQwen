#!/usr/bin/env python3
"""Reuse correctness: the answer lives inside the prefix that was reused.

Why this is separate from needle_test.py. That probe sends cold prompts, so it
never exercises the prefix-reuse path. Anything that changes *which* KV is
restored — retention intervals, checkpoint ordering, KV dtype, block promotion —
is invisible to it, and the failure it misses is the worst kind: a high hit rate
with a stale or wrong restored state. A hit-rate-only test cannot see that.

So this test does both, and requires both:

  turn 1   long document with a passcode buried at a given depth, stored in the
           conversation, establishing the cache
  turn 2   same conversation, asking only for the passcode -> must reuse turn 1's
           prefix
  pass     cached/prompt over 50%  AND  the passcode is correct

An optional interleaved second conversation can be advanced between the turns,
which is the case where reuse actually matters. A needle at depth 0.95 sits right
at the edge of the reused region and is the most sensitive position.

Usage:
    python bench/needle_reuse.py --tokens 100000 --depths 0.05,0.5,0.95
    python bench/needle_reuse.py --tokens 100000 --depths 0.5 --interleave 60000
"""
import argparse
import json
import sys
import time

import harness

# ~46 chars, ~11 tokens of filler per unit
UNIT = "All work and no play makes Jack a dull boy. "


def ask(messages, max_tokens=48):
    payload = {
        "messages": messages, "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    d = harness.post("/v1/chat/completions", payload, timeout=1800)
    u = d["usage"]
    det = u.get("prompt_tokens_details") or {}
    return {"prompt": u["prompt_tokens"], "cached": det.get("cached_tokens", 0),
            "secs": time.time() - t0,
            "ans": (d["choices"][0]["message"].get("content") or "").strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--depths", default="0.05,0.5,0.95")
    ap.add_argument("--interleave", type=int, default=0,
                    help="advance a second conversation of this size between turns")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    filler = UNIT * int(args.tokens / 11)
    print(f"  ~{args.tokens:,} tok per conversation; needle depths {args.depths}")
    results, allok = [], True

    for ds in args.depths.split(","):
        dep = float(ds)
        # a distinct passcode per depth, so one case cannot pass on another's cache
        needle = f"PASS{dep:.2f}".replace(".", "") + "XYZ"
        cut = int(len(filler) * dep)
        ctx = (filler[:cut] + f"\n\nThe secret passcode is {needle}. Remember it exactly.\n\n"
               + filler[cut:])

        conv = [{"role": "user", "content": ctx +
                 "\n\nPlease acknowledge with the single word OK."}]
        r1 = ask(conv, max_tokens=8)
        conv.append({"role": "assistant", "content": r1["ans"] or "OK"})

        if args.interleave > 0:
            ask([{"role": "user", "content":
                  f"UNRELATED-{time.time():.6f}\n" + UNIT * int(args.interleave / 11)}],
                max_tokens=4)

        conv.append({"role": "user",
                     "content": "What is the secret passcode? Reply with the passcode only."})
        r2 = ask(conv, max_tokens=48)

        pct = r2["cached"] / max(r2["prompt"], 1) * 100
        correct = needle in r2["ans"]
        # BOTH conditions. A hit without the right answer is the silent failure.
        if pct > 50 and correct:
            verdict = "OK"
        elif pct > 50:
            verdict = "WRONG-ANSWER"
        else:
            verdict = "NO-REUSE"
        allok = allok and verdict == "OK"
        print(f"    depth {dep:>5.0%}  cache {pct:5.1f}%  {r2['secs']:5.2f}s  "
              f"{verdict:<13} answer={r2['ans'][:32]!r}")
        sys.stdout.flush()
        results.append({"depth": dep, "needle": needle, "cached": r2["cached"],
                        "prompt": r2["prompt"], "pct": round(pct, 1),
                        "secs": round(r2["secs"], 2), "correct": correct,
                        "verdict": verdict})

    print(f"  -> {'all passed' if allok else 'FAILURES PRESENT'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f,
                      ensure_ascii=False, indent=1)
        print(f"  raw records -> {args.out}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
