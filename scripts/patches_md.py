#!/usr/bin/env python3
"""Write the patch table in PATCHES.md from the headers in each patch file.

Each file in patches/ and kvarn/ ends its preamble with one paragraph of headers:

    Kind: fix
    What: the text of the table's "what" cell; a long value continues on lines
      that start with whitespace
    Upstream: vllm #59892
    Cut-against: 0.30.0
    Retires-when: upstream PR

The headers come from the fork commit body, like the rest of the preamble, so change
them there and re-export (scripts/export-patch.sh). The row order is the apply order,
from `bash patches/apply.sh --list` and `--list --kvarn`. This script replaces the lines
between the table markers in PATCHES.md. CI runs it, then `git diff --exit-code PATCHES.md`.

Exit codes: 0 written; 1 a file has a bad or missing header (each one named);
2 apply.sh found a list and its directory in disagreement.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "PATCHES.md"
KEYS = ("Kind", "What", "Upstream", "Cut-against", "Retires-when")
KINDS = ("backport", "fix", "feature", "local", "own")
BEGIN = "<!-- table:begin. scripts/patches_md.py writes this table from the patch headers; do not edit it -->"
END = "<!-- table:end -->"
COLUMNS = ("patch", "kind", "what", "upstream", "cut against", "retires when")


def apply_order():
    for args, folder in ((["--list"], "patches"), (["--list", "--kvarn"], "kvarn")):
        run = subprocess.run(["bash", str(ROOT / "patches/apply.sh"), *args],
                             stdout=subprocess.PIPE, text=True)
        if run.returncode:
            sys.exit(run.returncode)
        yield from (f"{folder}/{name}" for name in run.stdout.split())


def headers(rel):
    """The header paragraph of one patch file as {key: value}, or a list of errors."""
    lines = (ROOT / rel).read_text().split("\n")
    end = next((i for i, l in enumerate(lines) if l.startswith("--- exported from ")), None)
    if end is None:
        return [f"{rel}: no '--- exported from' line"]
    block = []
    for line in reversed(lines[:end]):
        if not line.strip():
            if block:
                break
            continue
        block.insert(0, line)
    found, errors, key = {}, [], None
    for line in block:
        if line[:1].isspace() and key:
            found[key] += " " + line.strip()
            continue
        key, sep, value = line.partition(": ")
        if not sep or key not in KEYS:
            return [f"{rel}: the last preamble paragraph is not the header block ({', '.join(KEYS)})"]
        if key in found:
            errors.append(f"{rel}: {key} given twice")
        found[key] = value.strip()
    errors += [f"{rel}: no {k}: header" for k in KEYS if not found.get(k)]
    if found.get("Kind") and found["Kind"] not in KINDS:
        errors.append(f"{rel}: Kind: {found['Kind']} is not one of {', '.join(KINDS)}")
    errors += [f"{rel}: {k} contains '|'" for k, v in found.items() if "|" in v]
    return errors or found


def main():
    rows, errors = [], []
    for rel in apply_order():
        got = headers(rel)
        if isinstance(got, list):
            errors += got
            continue
        name = rel.removeprefix("patches/").removesuffix(".patch")
        rows.append("| " + " | ".join((name, *(got[k] for k in KEYS))) + " |")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    text = INDEX.read_text()
    head, begin, rest = text.partition(BEGIN + "\n")
    _, end, tail = rest.partition(END + "\n")
    if not begin or not end:
        print(f"PATCHES.md: the table markers are missing ({BEGIN} ... {END})", file=sys.stderr)
        return 1
    table = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS), *rows]
    INDEX.write_text(head + begin + "\n".join(table) + "\n" + end + tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
