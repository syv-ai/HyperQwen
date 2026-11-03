# SPDX-License-Identifier: Apache-2.0
"""RULER-style multi-key needle retrieval against the running server (niah_multikey_2's shape).

    python lsa/tests/ruler_mk.py <target_tokens> <trials> <label>

The haystack is real text, vLLM's own Python sources (as the quality battery's code lane), so the
keys sit among natural tokens. Each trial hides KEYS needles "One of the special magic numbers for
<word> is: <7 digits>." at random depths and asks for one of them. Seeds are fixed, so two arms are
asked the same questions over the same haystacks. Thinking is off; greedy.
"""
import glob
import json
import os
import random
import sys
import time
import urllib.request

import vllm

TARGET = int(sys.argv[1])
TRIALS = int(sys.argv[2])
LABEL = sys.argv[3] if len(sys.argv) > 3 else ""
KEYS = 8
CHARS_PER_TOKEN = 3.2
WORDS = ("amber basil cobalt dune ember fjord garnet harbor indigo juniper kestrel lumen maple nectar onyx "
         "pewter quartz russet saffron tundra umber velvet willow xenon yarrow zephyr").split()

files = sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "**", "*.py"), recursive=True))
corpus = "\n".join(open(f, errors="ignore").read() for f in files)


def ask(prompt):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 32,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:18020/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"] or "", d["usage"]["prompt_tokens"]


hits = 0
for trial in range(TRIALS):
    rng = random.Random(1000 * TARGET + trial)
    n_chars = int(TARGET * CHARS_PER_TOKEN)
    start = rng.randrange(0, len(corpus) - n_chars)
    hay = corpus[start:start + n_chars]
    keys = rng.sample(WORDS, KEYS)
    values = {k: str(rng.randrange(1_000_000, 9_999_999)) for k in keys}
    cuts = sorted(rng.randrange(0, len(hay)) for _ in keys)
    pieces, last = [], 0
    for cut, key in zip(cuts, keys):
        nl = hay.rfind("\n", last, cut)
        cut = nl + 1 if nl >= last else cut
        pieces += [hay[last:cut], f"One of the special magic numbers for {key} is: {values[key]}.\n"]
        last = cut
    pieces.append(hay[last:])
    target = rng.choice(keys)
    prompt = ("Some special magic numbers are hidden within the following text. Make sure to memorize them. "
              "I will quiz you about the numbers afterwards.\n" + "".join(pieces) +
              f"\nWhat is the special magic number for {target} mentioned in the provided text? "
              "Answer with the number only.")
    t0 = time.perf_counter()
    answer, tokens = ask(prompt)
    ok = values[target] in answer
    hits += ok
    depth = cuts[keys.index(target)] / len(hay)
    print(f"{LABEL} trial {trial}: {tokens} tokens, key at {depth:.0%}: {'HIT ' if ok else 'MISS'} "
          f"({answer.strip()[:24]!r} want {values[target]}) {time.perf_counter() - t0:.0f}s", flush=True)
print(f"{LABEL} ruler_mk {TARGET}: {hits}/{TRIALS}")
