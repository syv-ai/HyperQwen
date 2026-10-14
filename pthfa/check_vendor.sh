#!/bin/bash
# Check the vendored pth-flash-attn sources in pthfa/. No GPU, no network.
#
#   bash pthfa/check_vendor.sh               every vendored file matches pthfa/SHA256SUMS, and nothing
#                                            under pthfa/ is unlisted except this repo's own files
#   bash pthfa/check_vendor.sh --source DIR  also re-vendor from DIR, a pth-flash-attn git checkout that
#                                            has PTH_COMMIT (pthfa/PIN): `git archive PTH_COMMIT`, then
#                                            tools/export_upstream.py --out, then diff -r against pthfa/
#
# The vendored files are never edited here: change PTH_COMMIT, re-vendor, regenerate SHA256SUMS
# (pthfa/README.md, "Re-vendoring"). Exit 0 = OK, 1 = a mismatch (named), 2 = usage.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/PIN"
SRC=
case "${1:-}" in
  "") ;;
  --source) SRC=${2:?--source needs a pth-flash-attn checkout} ;;
  *) echo "usage: bash pthfa/check_vendor.sh [--source DIR]" >&2; exit 2 ;;
esac
# This repo's own files in pthfa/; everything else is vendored and must be in SHA256SUMS.
OWN="README.md PIN SHA256SUMS check_vendor.sh install.sh tests"

cd "$HERE"
echo "== pthfa: vendored files against SHA256SUMS (PTH_COMMIT ${PTH_COMMIT:0:12})"
sha256sum --quiet -c SHA256SUMS || { echo "FAILED: a vendored file differs from SHA256SUMS" >&2; exit 1; }
listed=$(awk '{print $2}' SHA256SUMS | sed 's|^\./||' | sort)
prune=(); for o in $OWN; do prune+=(-path "./$o" -prune -o); done
on_disk=$(find . "${prune[@]}" -path ./build -prune -o -name '*.egg-info' -prune -o -name __pycache__ -prune \
  -o -type f -print | sed 's|^\./||' | sort)
extra=$(comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$on_disk"))
[ -z "$extra" ] || { echo "FAILED: files under pthfa/ that SHA256SUMS does not list:" >&2; printf '    %s\n' $extra >&2; exit 1; }
echo "   $(printf '%s\n' "$listed" | wc -l) files, all match, none unlisted"

if [ -n "$SRC" ]; then
  echo "== pthfa: re-vendor from $SRC at $PTH_COMMIT and diff"
  git -C "$SRC" cat-file -e "$PTH_COMMIT^{commit}" 2>/dev/null \
    || { echo "FAILED: $SRC has no commit $PTH_COMMIT" >&2; exit 1; }
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  mkdir "$T/tree" "$T/vendor"
  git -C "$SRC" archive "$PTH_COMMIT" | tar -x -C "$T/tree"
  [ -f "$T/tree/tools/export_upstream.py" ] \
    || { echo "FAILED: $PTH_COMMIT has no tools/export_upstream.py, so it is not a commit this copy can come from" >&2; exit 1; }
  python3 "$T/tree/tools/export_upstream.py" --out "$T/vendor" >/dev/null \
    || { echo "FAILED: tools/export_upstream.py at $PTH_COMMIT did not run" >&2; exit 1; }
  mkdir -p "$T/vendor/tools"
  for f in setup.py pyproject.toml tools/build_flags.py LICENSE NOTICE; do cp "$T/tree/$f" "$T/vendor/$f"; done
  diff -r -x README.md -x PIN -x SHA256SUMS -x check_vendor.sh -x install.sh -x tests -x build \
       -x '*.egg-info' -x __pycache__ "$T/vendor" "$HERE" \
    || { echo "FAILED: pthfa/ is not the vendored copy of $PTH_COMMIT" >&2; exit 1; }
  echo "   identical to tools/export_upstream.py --out at $PTH_COMMIT plus the five build files"
fi
echo "pthfa vendor check: OK"
