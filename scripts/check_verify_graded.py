#!/usr/bin/env python3
"""Check that each `graded PATCH FILE MARKER ...` call in verify.sh still points at its patch.

graded() FAILs a live check when MARKER is in the installed FILE and WARNs otherwise. If the patch
leaves the series, stops touching FILE, or no longer adds MARKER, the grep never matches and a real
regression reads as WARN forever. So for each call: PATCH is in `bash patches/apply.sh --list`, the
patch has `+++ b/FILE`, and MARKER is in a line the patch adds to FILE.

Exit codes: 0 every call checks out; 1 a call is stale (each one named) or no call was found.
"""
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def added_lines(patch_text, file):
    lines, inside = [], False
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            inside = line == f"+++ b/{file}"
        elif line.startswith("diff --git "):
            inside = False
        elif inside and line.startswith("+"):
            lines.append(line[1:])
    return lines


def main():
    series = subprocess.run(["bash", "patches/apply.sh", "--list"], cwd=ROOT,
                            capture_output=True, text=True).stdout.split()
    calls = [line for line in (ROOT / "verify.sh").read_text().splitlines()
             if re.search(r"(^|[|&;]\s*)graded\s", line.strip())]
    errors = []
    for call in calls:
        args = shlex.split(re.sub(r"^.*?\bgraded\s", "", call).rstrip(" \\"))
        if len(args) < 3:
            errors.append(f"cannot read PATCH FILE MARKER from: {call.strip()}")
            continue
        name, file, marker = args[:3]
        if f"{name}.patch" not in series:
            errors.append(f"{name}: not in patches/series")
            continue
        added = added_lines((ROOT / "patches" / f"{name}.patch").read_text(), file)
        if not added:
            errors.append(f"{name}: adds no line to {file}")
        elif not any(marker in line for line in added):
            errors.append(f"{name}: no line it adds to {file} contains {marker!r}")
    if not calls:
        errors.append("no graded call found in verify.sh")
    for e in errors:
        print(f"ERROR: verify.sh graded(): {e}", file=sys.stderr)
    if errors:
        return 1
    print(f"verify.sh graded(): {len(calls)} calls point at their patches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
