"""Build ~/bench/labd_corpus_long.txt, the document the long-context LABD runs use.

The frozen corpus (~/bench/labd_corpus.txt, built by labd_bench.py on first run) is this
repo's own docs repeated until it reaches 200k characters -- about 84k tokens. Two problems
past ~65k of context: labd_bench.py slices `corpus[:ctx * 3.6]` and would silently measure a
shorter prompt than asked for, and the repetition hands the suffix lookup free matches from
an earlier copy of the same text, which flatters exactly the number the lookup is judged on.

This keeps the frozen corpus as the head -- byte-identical, so every number taken at
--ctx 20000 stays comparable -- and appends vLLM's own source. That is varied (no repeats)
and it is the realistic shape for this mode anyway: a coding assistant with a large real
codebase in context.

  venv/bin/python bench/make_long_corpus.py
  venv/bin/python bench/labd_bench.py <tag> --ctx 100000 --corpus ~/bench/labd_corpus_long.txt

Note the corpus runs ~2.9 characters per token, not the 3.6 labd_bench.py assumes, so
--ctx N produces roughly 1.24 N tokens. --ctx 100000 gave 112,655-token prompts.
Set CODE_CORPUS_DIR to a fixed vllm package tree for comparisons across installs.
Without it, prefer the checkout's historical venv source, then the active Python's install.
"""
import os
import sys
from pathlib import Path

from code_corpus import source_files

BASE = os.path.expanduser("~/bench/labd_corpus.txt")
OUT = os.path.expanduser("~/bench/labd_corpus_long.txt")
TARGET = 900_000  # characters of filler, on top of the frozen head

if not os.path.exists(BASE):
    sys.exit(f"{BASE} not found -- run bench/labd_bench.py once to build the frozen corpus")

with open(BASE, encoding="utf-8", newline="") as f:
    base = f.read()
if not base[:200000].strip():
    sys.exit(f"{BASE} is blank in its frozen head -- delete it and run bench/labd_bench.py once to rebuild it")
try:
    src, files = source_files("v1", recursive=True)
except RuntimeError as exc:
    sys.exit(str(exc))

parts, seen = [], 0
for f in files:
    try:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if len(text) < 2000:
        continue
    parts.append("\n\n### " + os.path.basename(f) + "\n\n" + text)
    seen += len(text)
    if seen > TARGET:
        break

if not parts:
    sys.exit(f"No eligible source files under {src}; set CODE_CORPUS_DIR to a fixed vllm package directory")
long = base + "".join(parts)
assert long[: len(base)] == base, "the frozen head must stay byte-identical"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="") as f:
    f.write(long)
print(f"{OUT}: {len(long)} chars ({len(base)} frozen head + {len(parts)} source files)")
print(f"code corpus: {src}; {len(parts)} selected / {len(files)} source files, {seen} source chars")
