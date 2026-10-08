#!/usr/bin/env python3
"""Find names our series still uses that upstream removed between two pins.

    python scripts/port-removed-names.py --fork ../vllm --old v0.29.0 --new v0.30.0 --branch qwen38/0.30

A name is REMOVED when it occurs somewhere in the old tag's Python tree and nowhere in the new tag's. The net diff
new tag..branch is read for the lines the series adds; any removed name on such a line is reported with its topic
(by blame), file and line. Every hit is a runtime AttributeError/TypeError/NameError waiting for the
path that reaches it, and none of them shows in a replay, a clean apply or port-drift: the name is in OUR added line,
never in upstream's hunk context. 2026-09-23: the 0.30 port kept `c.has_g_idx` and `perm=layer.g_idx_sort_indices`
after vllm #54809 removed both, and the first INT8_ACT=int8 boot (batch mode's default) died at load.

A name the series defines itself (assignment, def, parameter) is skipped. A use reviewed and kept is passed as
--accept NAME=REASON and listed as accepted. Exit 1 when anything unaccepted is reported. For 0.29.0 -> 0.30.0:
  --accept 'is_k_full=the tuned standalone build keeps its 0.27.1 schema'
  --accept '_seq_lens_cpu=getattr fallback to seq_lens.tolist(), exact; upstream #55353 removed the cached copy'
  --accept 'VLLM_PREFIX_CACHE_RETENTION_INTERVAL=a hash-exclusion list entry; harmless'
The falsifier, 2026-09-23: at a710337ed (before the fixups) it reported has_g_idx, g_idx_sort_indices and both
is_k_full call sites, the first two confirmed by boots that died with exactly those AttributeErrors.
"""
import argparse, re, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")


def git(fork, *a):
    return subprocess.run(["git", "-C", fork, *a], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def names_at(fork, tag):
    out = git(fork, "grep", "-ohE", r"[A-Za-z_][A-Za-z0-9_]{3,}", tag, "--", "vllm/*.py", "vllm/**/*.py")
    return {ln.split(":", 1)[-1] for ln in out.splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--accept", action="append", default=[], metavar="NAME=REASON",
                    help="a reviewed use of a removed name (repeatable); listed as accepted, not failed")
    a = ap.parse_args()
    a.accept = dict(x.split("=", 1) for x in a.accept)
    removed = names_at(a.fork, a.old) - names_at(a.fork, a.new)
    # The NET diff (new tag -> branch tip), so a fixup that removes a line is counted; blame names the topic.
    added = []  # (file, line number at the tip, code) for every line the series adds
    f = None
    for ln in git(a.fork, "diff", "-U0", a.new, a.branch, "--", "*.py").splitlines():
        if ln.startswith("+++ "):
            f = ln[6:]
        elif ln.startswith("@@"):
            n = int(re.search(r"\+(\d+)", ln).group(1))
        elif ln.startswith("+") and f:
            added.append((f, n, ln[1:].split("#", 1)[0]))  # a comment naming an old symbol is not a use
            n += 1

    def topic(f, n):
        out = git(a.fork, "blame", "--porcelain", "-L", f"{n},{n}", a.branch, "--", f)
        return next((l[8:] for l in out.splitlines() if l.startswith("summary ")), "?")
    # A name the series itself defines is ours, whatever upstream did with the same spelling: an assignment
    # (`x = ...`, `self.x = ...`, not a `kw=...,` argument line), a def/class, or a bare parameter line (`x,`).
    ours = set()
    for _, _, code in added:
        m = re.match(r"\s*(?:self\.)?([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)(.*)$", code)
        if m and not m.group(2).rstrip().endswith(","):
            ours.add(m.group(1))
        ours.update(re.findall(r"\b(?:def|class)\s+([A-Za-z_]\w*)", code))
        m = re.match(r"\s*([A-Za-z_]\w*)\s*(?::[^,=]+)?(?:=[^,]+)?,?\s*\)?\s*(?:->.*)?:?\s*$", code)
        if m:
            ours.add(m.group(1))
    hits = 0
    for f, n, code in added:
        gone = sorted({w for w in IDENT.findall(code) if w in removed and w not in ours})
        if not gone:
            continue
        why = [a.accept[w] for w in gone if w in a.accept]
        if len(why) == len(gone):
            print(f"accepted {f}:{n}: {', '.join(gone)} ({'; '.join(why)})")
            continue
        hits += 1
        print(f"REMOVED  {topic(f, n)[:60]:60}  {f}:{n}: {', '.join(gone)}  |  {code.strip()[:100]}")
    print(f"port-removed-names {a.old} -> {a.new}: {len(removed)} names removed upstream, {len(removed & ours)} of them "
          f"redefined by the series; {hits} added line(s) of ours use one")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
