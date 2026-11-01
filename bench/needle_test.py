#!/usr/bin/env python3
"""Needle-in-a-haystack probe against the running server.

Builds a ~TARGET_TOKENS filler context, hides a secret passcode at a given
fractional depth, asks the model for it, and reports whether the answer
contains the passcode (exit 0) or not (exit 1). Complements quality_battery.py's GSM8K lane for the
"quality at depth" question on long-context KV configs.

Thinking is off. The chat template turns it on when the flag is absent, and the
model then spends the 32-token reply on reasoning and returns no content: every
depth reads MISSED, whatever the cache holds.

Usage:
    python bench/needle_test.py [target_tokens] [depth]
    # default: 100000 tokens, needle at 90% depth
"""
import sys
import time

import harness

TARGET_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
DEPTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9

NEEDLE = "ZXCVBNM12345"
needle_line = f"The secret passcode is {NEEDLE}. Remember it exactly."

# "All work and no play makes Jack a dull boy. " is ~46 chars, ~11 tokens.
unit = "All work and no play makes Jack a dull boy. "
filler = unit * int(TARGET_TOKENS / 11)

depth = int(len(filler) * DEPTH)
context = filler[:depth] + "\n\n" + needle_line + "\n\n" + filler[depth:]

prompt = context + "\n\nQuestion: what is the secret passcode? Reply with the passcode only."


t0 = time.perf_counter()
resp = harness.post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": False},
}, timeout=1800)
elapsed = time.perf_counter() - t0

answer = (resp["choices"][0]["message"].get("content") or "")
print(f"context ~{TARGET_TOKENS} tokens, needle at {DEPTH:.0%} depth")
print(f"elapsed {elapsed:.1f}s, answer: {answer[:120]!r}")
print("RETRIEVED" if NEEDLE in answer else "MISSED")
sys.exit(0 if NEEDLE in answer else 1)
