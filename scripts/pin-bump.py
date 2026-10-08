#!/usr/bin/env python3
"""Move every mechanical vLLM pin in this repo from one release to the next, in one pass.

    python scripts/pin-bump.py --from 0.29.0 --to 0.30.0 --fork ../vllm [--repo .] [--dry-run]

What it changes (and only this; narrative docs that describe a past pin are left for a person):
  - docker/requirements.txt  vllm==<old>          -> vllm==<new>
  - verify.sh                the version check, the kvarn patch names
  - patches/apply.sh         the KVARN array's kvarn-*-<old>.patch names (the list lives here since syv-ai #242)
  - Dockerfile               the header comment's "vLLM <old>"
  - kvarn/install.sh         the "vLLM <old> venv" comment
  - docs/install.md          the `pip install vllm==<old>` line, the patches' "written against", and
                             flashinfer-cubin==<x>, read from vLLM's own requirements/cuda.txt at the NEW tag
    (the cubin package is not on PyPI and must match flashinfer-python exactly; 0.30.0 moved it to 0.6.18.post1)
Every file it touches is listed with its line count changed; anything it expected and did not find is an error.
"""
import argparse, re, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def flashinfer_at(fork, tag):
    txt = subprocess.run(["git", "-C", fork, "show", f"{tag}:requirements/cuda.txt"], capture_output=True, text=True,
                         encoding="utf-8").stdout
    m = re.search(r"^flashinfer-cubin==(\S+)", txt, re.M)
    if not m:
        sys.exit(f"no flashinfer-cubin pin in {tag}:requirements/cuda.txt")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--fork", required=True, help="vLLM fork repo (bare or checkout) with the release tags")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    old, new, root = a.old, a.new, Path(a.repo)
    fi_old, fi_new = flashinfer_at(a.fork, f"v{old}"), flashinfer_at(a.fork, f"v{new}")
    o = re.escape(old)
    # Every KVarN patch is named kvarn-[<part>-]<pin>.patch (four at 0.30.0: kvarn, kvarn-v2-runner,
    # kvarn-recycled-pages, kvarn-fp16-dequant), so one pattern follows the list as it grows.
    kv = (rf"\bkvarn-((?:[a-z0-9]+-)*){o}\.patch", rf"kvarn-\g<1>{new}.patch")
    edits = {
        "docker/requirements.txt": [(rf"^vllm=={o}(?=\r?$)", f"vllm=={new}"), (rf"# vllm=={o} ", f"# vllm=={new} ")],
        "verify.sh": [(rf'\[ "\$VER" = "{o}" \]', f'[ "$VER" = "{new}" ]'), (rf"written against {o}\)", f"written against {new})"),
                      kv],
        "patches/apply.sh": [kv],
        "Dockerfile": [(rf"# vLLM {o} ", f"# vLLM {new} ")],
        "kvarn/install.sh": [(rf"vLLM {o} venv", f"vLLM {new} venv")],
        # (?![.\w]), not \b: "0.6.18\b" also matches inside "0.6.18.post1", so a second run would append ".post1" again
        "README.md": [(rf'alt="vLLM {o}"', f'alt="vLLM {new}"'), (rf"badge/vLLM-{o}-", f"badge/vLLM-{new}-")],
        "docs/docker.md": [(rf"vLLM {o} pinned", f"vLLM {new} pinned")],
        "docs/install.md": [(rf"pip install vllm=={o} ", f"pip install vllm=={new} "),
                            (rf"written against {o};", f"written against {new};"),
                            (rf"flashinfer-cubin=={re.escape(fi_old)}(?![.\w])", f"flashinfer-cubin=={fi_new}")],
    }
    errors = 0
    for rel, subs in edits.items():
        p = root / rel
        s = p.read_bytes().decode("utf-8")  # bytes, not read_text: keep each file's own line endings
        before, changed = s, 0
        for pat, rep in subs:
            s, n = re.subn(pat, rep, s, flags=re.M)
            if n == 0:
                print(f"  MISSING  {rel}: /{pat}/"); errors += 1
            changed += n
        if s != before and not a.dry_run:
            p.write_bytes(s.encode("utf-8"))
        print(f"  {'would edit' if a.dry_run else 'edited'} {rel}: {changed} replacement(s)")
        # What is left of the old version in a file this touched is for a person to read: narrative ("on 0.29.0,
        # ==0.6.18") may be right, a missed pin is not. 2026-09-23: the cubin pin was bumped and the `pip install
        # vllm==0.29.0` two lines above it was not, and only a reader of the doc caught it.
        for n, ln in enumerate(s.splitlines(), 1):
            if re.search(rf"(?<![\d.]){o}(?![\d])", ln):
                print(f"    REVIEW {rel}:{n}: {ln.strip()[:140]}")
    # The same question for every tracked doc and script this did NOT edit, as a count per file: the README badge and
    # the long-context, docker and mode READMEs' "the <old> this repo runs" lines all survived the 0.30 port's first
    # pass because nothing listed them. docs/vllm-<old>.md and PATCHES.md are history and are skipped.
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "*.md", "*.sh", "*.txt", "Dockerfile*"],
                             capture_output=True, text=True, encoding="utf-8").stdout.split()
    skip = set(edits) | {f"docs/vllm-{old.rsplit('.', 1)[0]}.md", "PATCHES.md"}
    for rel in tracked:
        if rel in skip or rel.startswith(("patches/", "kvarn/files/")):
            continue
        try:
            n = len(re.findall(rf"(?<![\d.]){o}(?![\d])", (root / rel).read_bytes().decode("utf-8", "replace")))
        except OSError:
            continue
        if n:
            print(f"  REVIEW-FILE {rel}: {n} mention(s) of {old}")
    # Co-pins: every exact pin in docker/requirements.txt must satisfy the NEW vLLM's own floors. 0.30.0 raised
    # huggingface_hub to >=1.31.0 and our 1.28.0 pin made the image unbuildable (ResolutionImpossible); this reports
    # such a pin before a 20-minute build does.
    reqs = {}
    for f in ("requirements/common.txt", "requirements/cuda.txt"):
        txt = subprocess.run(["git", "-C", a.fork, "show", f"v{new}:{f}"], capture_output=True, text=True, encoding="utf-8").stdout
        for m in re.finditer(r"^([A-Za-z0-9_.-]+)\s*(>=|==|~=)\s*([0-9][^\s#;,]*)", txt, re.M):
            reqs[m.group(1).lower().replace("_", "-")] = (m.group(2), m.group(3))
    def ver(v):
        return tuple(int(x) if x.isdigit() else x for x in re.split(r"[.+-]", v))
    ours = (root / "docker/requirements.txt").read_bytes().decode("utf-8")
    for m in re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", ours, re.M):
        name, pinned = m.group(1).lower().replace("_", "-"), m.group(2)
        if name == "vllm" or name not in reqs:
            continue
        op, want = reqs[name]
        bad = (op == ">=" and ver(pinned) < ver(want)) or (op == "==" and pinned != want)
        if bad:
            print(f"  CO-PIN   docker/requirements.txt {name}=={pinned} violates vLLM {new}'s {name}{op}{want}"); errors += 1
    print(f"pin-bump {old} -> {new}; flashinfer-cubin {fi_old} -> {fi_new}; {errors} problem(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
