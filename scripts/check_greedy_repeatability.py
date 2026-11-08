#!/usr/bin/env python3
"""Prompt-for-prompt greedy repeatability: the same prompt at temperature 0 must give the same text.

Sends a fixed set of prompts (short, code, long-context) ROUNDS times each, one request at a time, and fails if any
prompt's output differs between rounds. Round 1 is a cold prefill; later rounds hit the prefix cache, so a
difference between them also catches cached-vs-cold drift. With --save FILE the outputs are written; with
--compare FILE they are checked against an earlier run, which is how to compare two server configurations
(e.g. VLLM_FLASHINFER_NATIVE_SPEC_GRAPHS=0 vs 1, or CUDA GDN on vs off). Concurrent requests are not exercised:
batch composition can change greedy output on any vLLM setup.

    python3 scripts/check_greedy_repeatability.py [--rounds 3] [--max-tokens 256] [--save out.json | --compare out.json]
Env: HYPERQWEN_URL (default http://127.0.0.1:18020), VLLM_API_KEY. Exit 0: identical; 1: a difference.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import urllib.request

URL = os.environ.get("HYPERQWEN_URL", "http://127.0.0.1:18020").rstrip("/")
KEY = os.environ.get("VLLM_API_KEY", "")

LONG_DOC = "\n".join(
    f"Section {i}: the {w} subsystem keeps {i * 7 % 13 + 2} queues and reports status code {i * 31 % 97}."
    for i, w in enumerate(("ingest", "routing", "billing", "audit", "cache", "scheduler", "telemetry") * 90))

PROMPTS = {
    "short-fact": "Name the planets of the solar system in order from the sun, one per line.",
    "code": "Write a Python function that merges overlapping intervals, with a docstring and three assert tests.",
    "reasoning": "A train leaves at 09:40 and travels 212 km at 88 km/h. When does it arrive? Show the steps.",
    "story": "Write a 150-word story about a lighthouse keeper who finds a message in a bottle.",
    "long-context": LONG_DOC + "\n\nUsing only the text above, which subsystem has status code 0, and how many queues "
                    "does the 'billing' subsystem keep in Section 2? Answer in two sentences.",
}


def ask(model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1,
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  **({"Authorization": "Bearer " + KEY} if KEY else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    choice = out["choices"][0]
    msg = choice["message"]
    text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "␞" + (msg.get("content") or "")
    return {"text": text, "finish": choice.get("finish_reason"), "tokens": out["usage"]["completion_tokens"]}


def diff(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    d = "\n".join(list(difflib.unified_diff(a.splitlines(), b.splitlines(), "A", "B", lineterm="", n=0))[:8])
    return f"first difference at character {i} of {len(a)}/{len(b)}\n{d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()
    if a.rounds < 2 and not a.compare:
        ap.error("--rounds must be at least 2 (or use --compare)")

    req = urllib.request.Request(URL + "/v1/models", headers={"Authorization": "Bearer " + KEY} if KEY else {})
    model = json.load(urllib.request.urlopen(req, timeout=10))["data"][0]["id"]
    print(f"{URL}  model {model}  {len(PROMPTS)} prompts x {a.rounds} rounds, max {a.max_tokens} tokens, temperature 0")

    runs: dict[str, list[dict]] = {name: [] for name in PROMPTS}
    for rnd in range(a.rounds):
        for name, prompt in PROMPTS.items():
            runs[name].append(ask(model, prompt, a.max_tokens))
        print(f"  round {rnd + 1}/{a.rounds} done", flush=True)

    base = json.load(open(a.compare)) if a.compare else {}
    bad = 0
    for name, rs in runs.items():
        texts = [r["text"] for r in rs]
        same = all(t == texts[0] for t in texts)
        line = f"{'OK  ' if same else 'DIFF'} {name:<13} {rs[0]['tokens']:>4} tokens, finish={rs[0]['finish']}"
        if not same:
            bad += 1
            k = next(i for i, t in enumerate(texts) if t != texts[0])
            line += f"\n     round 1 vs round {k + 1}: {diff(texts[0], texts[k])}"
        if a.compare and name in base:
            if base[name]["text"] == texts[0]:
                line += "  (matches saved run)"
            else:
                bad += 1
                line += f"\n     DIFFERS from saved run: {diff(base[name]['text'], texts[0])}"
        print(line)
    if a.save:
        json.dump({n: rs[0] for n, rs in runs.items()}, open(a.save, "w"), indent=1)
        print("saved", a.save)
    print("PASS: every prompt repeated byte-identically" if not bad else f"FAIL: {bad} difference(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
