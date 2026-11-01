#!/usr/bin/env python3
"""Quick quality battery against the running server. Catches "benchmarks great,
outputs garbage" in a minute — run it after every kernel/quant change.
  1. perplexity over ~300-token windows: wikitext-2 test (en), fineweb-2 dan test (da),
     vLLM's own python source (code)
  2. GSM8K exact-match, 200 test questions, thinking off, greedy
Data (once):
  hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir bench/quality-data/wikitext
  hf download openai/gsm8k --repo-type dataset --include "main/test-*" --local-dir bench/quality-data/gsm8k
  hf download HuggingFaceFW/fineweb-2 --repo-type dataset --include "data/dan_Latn/test/000_00000.parquet" --local-dir bench/quality-data/fineweb2
Perplexity needs prompt_logprobs, which needs memory headroom: run the server with
GPU_UTIL=0.93 for this (gotcha 10 in the README).

On vLLM 0.28 and earlier, CTX=huge (KVarN) with SPEC=mtp and PREFIX_CACHE=1 corrupted
prompt_logprobs: some requests 400 with "Out of range float values are not JSON
compliant: nan", and the batteries that did complete read ~23% high on English
(12.6-13.7 against 10.76) and drifted run to run. On 0.27.1, PREFIX_CACHE=0 (caching
off) made the same server exact and stable to four decimals. The 0.29 pin fixed it
(#64, gotcha 46), and on 0.31 caching off, PREFIX_CACHE unset and PREFIX_CACHE=1 all
read 10.76. If it comes back, compare against caching off, which is PREFIX_CACHE=0
again.
SPEC=off, SPEC=dflash2 and CTX=fast were never affected, so the published quality
tables are not implicated.
Usage: venv/bin/python bench/quality_battery.py <tag> [--ppl-only] [--gsm-only] [--gsm-n 200]
Set CODE_CORPUS_DIR to one fixed vllm package directory for all compared arms.
PPL requires a source tree (explicit, checkout venv, or active install fallback).
--gsm-only does not read the code corpus or require a local vLLM install.
"""
import argparse
import json, os, sys, glob, re, math, time, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


parser = argparse.ArgumentParser(description="Measure quality against the running server.", allow_abbrev=False)
parser.add_argument("tag", help="label for result_<tag>.json, for example mycard; not a model name")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--ppl-only", action="store_true", help="measure perplexity only")
mode.add_argument("--gsm-only", action="store_true", help="measure GSM8K only (no local vLLM source needed)")
parser.add_argument("--gsm-n", type=positive_int, default=200, help="maximum GSM8K questions (default: 200)")
args = parser.parse_args()

# Parse help and invalid arguments before importing optional data dependencies.
import pyarrow.parquet as pq

import harness
from code_corpus import source_files

HERE = os.path.dirname(os.path.abspath(__file__))
# data dir: wikitext-2 test parquet, fineweb-2 dan_Latn test parquet, gsm8k test parquet (see README)
Q = os.environ.get("QUALITY_DATA", os.path.join(HERE, "quality-data"))
tag = args.tag
ppl_only = args.ppl_only; gsm_only = args.gsm_only
gsm_n = args.gsm_n

def docs():
    # Resolve before reading parquet: never silently omit the code cohort.
    src, files = source_files("v1/core")
    out = []
    t = pq.read_table(f"{Q}/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet").column("text").to_pylist()
    txt = "".join(t); 
    for i in range(0, min(len(txt), 40*1200), 1200): out.append(("en", txt[i:i+1200]))
    tb = pq.read_table(f"{Q}/fineweb2/data/dan_Latn/test/000_00000.parquet", columns=["text"]).column("text").to_pylist()
    random.Random(0).shuffle(tb)
    n=0
    for d in tb:
        if len(d) > 1500:
            out.append(("da", d[:1200])); n+=1
        if n>=40: break
    selected = 0
    for f in files:
        s = Path(f).read_text(encoding="utf-8")
        selected += 1
        for i in range(0, min(len(s), 4*1200), 1200):
            if len(s[i:i+1200]) > 800: out.append(("code", s[i:i+1200]))
        if sum(1 for l,_ in out if l=="code") >= 40: break
    windows = sum(1 for lang, _ in out if lang == "code")
    if not windows:
        raise RuntimeError(f"No eligible code windows under {src}; set CODE_CORPUS_DIR to a fixed vllm package directory")
    print(f"code corpus: {src}; {selected} selected / {len(files)} source files, {windows} windows", flush=True)
    return out

def ppl_one(item):
    lang, text = item
    r = harness.post("/v1/completions", {"prompt":text,"max_tokens":1,"temperature":0,
                              "prompt_logprobs":0,"echo":False})
    pl = r["choices"][0]["prompt_logprobs"]  # list; first is None
    lps = []
    for e in pl[1:]:
        if e is None: continue
        # entry: dict token_id -> {logprob, rank, decoded_token}; the actual token has rank... take the entry with 'rank' == 1? no: prompt token's own logprob is included; pick max? Use the one that is the actual token: vLLM includes actual token always; with prompt_logprobs=0 dict has only the actual token
        v = list(e.values())[0]
        lps.append(v["logprob"] if isinstance(v, dict) else v)
    return lang, sum(lps), len(lps)

def run_ppl():
    items = docs()
    with ThreadPoolExecutor(2) as ex: res = list(ex.map(ppl_one, items))
    agg = {}
    for lang, s, n in res:
        a = agg.setdefault(lang, [0.0, 0]); a[0]+=s; a[1]+=n
    out = {lang: (math.exp(-s/n), n) for lang,(s,n) in agg.items()}
    tot_s = sum(s for s,n in agg.values()); tot_n = sum(n for s,n in agg.values())
    out["all"] = (math.exp(-tot_s/tot_n), tot_n)
    return out

def extract_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$",""))
    return m[-1].replace(",","") if m else None

def gsm_one(row):
    q, a = row
    gold = a.split("####")[-1].strip().replace(",","")
    r = harness.post("/v1/chat/completions", {"messages":[{"role":"user","content":q+"\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."}],
        "max_tokens":768,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}})
    txt = r["choices"][0]["message"]["content"] or ""
    m = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", txt)
    pred = (m.group(1).replace(",","") if m else extract_num(txt))
    try: ok = abs(float(pred) - float(gold)) < 1e-6
    except Exception: ok = False
    return ok, r["usage"]["completion_tokens"]

def run_gsm():
    t = pq.read_table(f"{Q}/gsm8k/main/test-00000-of-00001.parquet")
    rows = list(zip(t.column("question").to_pylist(), t.column("answer").to_pylist()))[:gsm_n]
    if not rows:
        print("No GSM8K questions found; invalid quality run", file=sys.stderr)
        sys.exit(2)
    with ThreadPoolExecutor(32) as ex: res = list(ex.map(gsm_one, rows))
    acc = sum(1 for ok,_ in res if ok)/len(res); toks = sum(n for _,n in res)/len(res)
    return acc, toks, len(res)

t0=time.perf_counter(); result={"tag":tag}
if not gsm_only:
    result["ppl"] = run_ppl(); print(tag, "PPL", {k:(round(v[0],4),v[1]) for k,v in result["ppl"].items()}, f"{time.perf_counter()-t0:.0f}s", flush=True)
if not ppl_only:
    t1=time.perf_counter(); acc, toks, n = run_gsm(); result["gsm8k"]={"n":n,"acc":acc,"mean_tokens":toks}
    print(tag, f"GSM8K n={n} acc={acc:.3f} mean_tokens={toks:.0f} {time.perf_counter()-t1:.0f}s", flush=True)
with open(os.path.join(Q, f"result_{tag}.json"),"w") as f: json.dump(result,f)
