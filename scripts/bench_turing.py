#!/usr/bin/env python3
"""Single-stream decode and prefill speed against a running HyperQwen/vLLM server.

For each prompt length it builds a deterministic document of about that many tokens (sized with the server's
/tokenize), asks for a long completion at temperature 0 with ignore_eos, and streams it. Per length it reports:
  cold prefill   time to the first token of a never-seen prompt, and its rate over the uncached prompt tokens
  cached prefill time to the first token when the same prompt is sent again (prefix cache hit)
  decode         tokens per second after the first token, per run and the median

    python3 scripts/bench_turing.py [--lengths 1000 4000 7000] [--gen 512] [--runs 3] [--json out.json]
Env: HYPERQWEN_URL (default http://127.0.0.1:18020), VLLM_API_KEY. One request at a time; no batching.
Greedy decoding with ignore_eos can fall into repetition, which speculative decoding accepts easily, so read
decode numbers as "this completion", not as a universal rate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request

URL = os.environ.get("HYPERQWEN_URL", "http://127.0.0.1:18020").rstrip("/")
KEY = os.environ.get("VLLM_API_KEY", "")
HEAD = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}
SUBSYSTEMS = ("ingest", "routing", "billing", "audit", "cache", "scheduler", "telemetry")


def post(path: str, body: dict, timeout: int = 900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), HEAD)
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(model: str, text: str) -> int:
    with post("/tokenize", {"model": model, "prompt": text}) as r:
        return json.load(r)["count"]


def make_prompt(model: str, target: int, nonce: str) -> str:
    """About `target` prompt tokens. The nonce opens the prompt so each cold run misses the prefix cache."""
    def build(n_lines: int) -> str:
        lines = [f"Section {i}: the {SUBSYSTEMS[i % 7]} subsystem keeps {i * 7 % 13 + 2} queues and reports "
                 f"status code {i * 31 % 97}." for i in range(n_lines)]
        return (f"[document {nonce}]\n" + "\n".join(lines) +
                "\n\nUsing the text above, write a long, detailed essay about how such a system could be "
                "operated, monitored and scaled. Be thorough.")
    lo, hi = 1, max(8, target // 8)
    while count_tokens(model, build(hi)) < target:
        hi *= 2
    while lo < hi:                                           # smallest line count that reaches the target
        mid = (lo + hi) // 2
        if count_tokens(model, build(mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    return build(lo)


def stream_once(model: str, prompt: str, gen: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1,
            "max_tokens": gen, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first = last = None
    usage = {}
    done = False
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(f"stream error: {chunk['error']}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
    if not done:
        raise RuntimeError("incomplete stream: missing [DONE]")
    if first is None or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 1:
        raise RuntimeError("stream has no generated text or valid completion usage")
    if type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] < 1:
        raise RuntimeError("stream has no valid prompt usage")
    out = usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    ttft = (first - t0) if first else None
    decode = (out - 1) / (last - first) if first and last and last > first and out > 1 else None
    return {"prompt_tokens": usage.get("prompt_tokens", 0), "cached_tokens": cached, "output_tokens": out,
            "ttft_s": ttft, "decode_tok_s": decode}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1000, 4000, 7000])
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3, help="requests per length; the first is cold, the rest cached")
    ap.add_argument("--json")
    a = ap.parse_args()
    if min(*a.lengths, a.gen, a.runs) < 1:
        ap.error("lengths, --gen and --runs must be positive")
    req = urllib.request.Request(URL + "/v1/models", headers=HEAD)
    info = json.load(urllib.request.urlopen(req, timeout=10))["data"][0]
    model = info["id"]
    print(f"{URL} model {model}, max_model_len {info.get('max_model_len')}, {a.gen} tokens out, {a.runs} runs per length")
    post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Say hi."}],
                                  "max_tokens": 8, "temperature": 0}).read()   # warm connection and graphs
    results = []
    print(f"{'prompt':>8} {'cold prefill':>16} {'cached prefill':>15} {'decode tok/s (runs)':>34} {'median':>8}")
    for n in a.lengths:
        prompt = make_prompt(model, n, f"{time.time_ns():x}")
        runs = [stream_once(model, prompt, a.gen) for _ in range(a.runs)]
        cold, warm = runs[0], runs[1:]
        new = cold["prompt_tokens"] - cold["cached_tokens"]
        rate = new / cold["ttft_s"] if cold["ttft_s"] else 0
        dec = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
        cached_ttft = statistics.median([r["ttft_s"] for r in warm if r["ttft_s"]]) if warm else None
        print(f"{cold['prompt_tokens']:>8} {cold['ttft_s']:>7.2f} s {rate:>6,.0f}/s "
              f"{(f'{cached_ttft:.2f} s' if cached_ttft else '-'):>15} {' / '.join(f'{d:.1f}' for d in dec):>34} "
              f"{statistics.median(dec) if dec else 0:>8.1f}")
        results.append({"target": n, "prompt_tokens": cold["prompt_tokens"], "cold_ttft_s": cold["ttft_s"],
                        "cold_prefill_tok_s": rate, "cached_ttft_s": cached_ttft, "decode_tok_s": dec,
                        "output_tokens": cold["output_tokens"]})
    if a.json:
        json.dump({"url": URL, "model": model, "max_model_len": info.get("max_model_len"), "gen": a.gen,
                   "results": results}, open(a.json, "w"), indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
