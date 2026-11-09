#!/bin/bash
# CPU test for #445: patches/apply.sh changes the package directory only when every patch applies, and its
# stamp (.hyperqwen-patches) tells an applied series, a different series and a stale stamp apart.
# Builds a small fake package and a three-patch series (the third edits a file the first creates, as the real
# series does), plus four KVarN patches under the names apply.sh lists, in a temp copy of patches/apply.sh.
# No vLLM, no GPU, no network.
#
#   bash bench/test_apply_staged.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mapfile -t KV < <(bash "$REPO/patches/apply.sh" --list --kvarn)
FAILS=0
ok() { echo "  ok    $1"; }
bad() { echo "  FAIL  $1"; FAILS=$((FAILS + 1)); }
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }
treehash() { (cd "$1" && find . -type f ! -name .hyperqwen-patches -print0 | sort -z | xargs -0 cat | sha256sum | cut -c1-16); }
strays() { find "$1" \( -name '*.rej' -o -name '*.orig' -o -name '*.hyperqwen-new' \) | wc -l; }

# mkpatch OUT FILE OLD NEW: a git-style patch of one file (OLD empty = a new file), appended to OUT.
mkpatch() {
  local out=$1 f=$2 old=$3 new=$4
  printf 'diff --git a/%s b/%s\n' "$f" "$f" >> "$out"
  if [ -z "$old" ]; then diff -u --label /dev/null --label "b/$f" /dev/null "$new" >> "$out"
  else diff -u --label "a/$f" --label "b/$f" "$old" "$new" >> "$out"; fi
}
lines() { seq 1 30 | sed "s/^/$1 line /"; }

R=$T/repo; mkdir -p "$R/patches" "$R/kvarn" "$T/src"
cp "$REPO/patches/apply.sh" "$R/patches/"
P=$T/pristine; mkdir -p "$P/sub"
lines a > "$P/a.py"; lines b > "$P/sub/b.py"
# p1 creates new.py; p2 edits a.py and sub/b.py (two hunks in b.py); p3 edits new.py and a.py again.
lines n > "$T/src/new1"; mkpatch "$R/patches/p1.patch" new.py "" "$T/src/new1"
sed 's/^a line 5$/a line 5 p2/' "$P/a.py" > "$T/src/a2"; mkpatch "$R/patches/p2.patch" a.py "$P/a.py" "$T/src/a2"
sed -e 's/^b line 4$/b line 4 p2/' -e 's/^b line 25$/b line 25 p2/' "$P/sub/b.py" > "$T/src/b2"
mkpatch "$R/patches/p2.patch" sub/b.py "$P/sub/b.py" "$T/src/b2"
sed 's/^n line 10$/n line 10 p3/' "$T/src/new1" > "$T/src/new3"; mkpatch "$R/patches/p3.patch" new.py "$T/src/new1" "$T/src/new3"
sed 's/^a line 20$/a line 20 p3/' "$T/src/a2" > "$T/src/a3"; mkpatch "$R/patches/p3.patch" a.py "$T/src/a2" "$T/src/a3"
printf 'p1.patch\np2.patch\np3.patch\n' > "$R/patches/series"
# The KVarN patches: the first edits a line p3 wrote (so it needs the series), the rest edit sub/b.py.
prev=$T/src/a3; cp "$T/src/b2" "$T/src/bk"
sed 's/^a line 20 p3$/a line 20 p3 kv/' "$prev" > "$T/src/ak"; mkpatch "$R/kvarn/${KV[0]}" a.py "$prev" "$T/src/ak"
for i in 1 2 3; do
  sed "s/^b line $((i * 7))\$/b line $((i * 7)) kv/" "$T/src/bk" > "$T/src/bk2"
  mkpatch "$R/kvarn/${KV[i]}" sub/b.py "$T/src/bk" "$T/src/bk2"; mv "$T/src/bk2" "$T/src/bk"
done
APPLY="bash $R/patches/apply.sh"
fresh() { rm -rf "$T/D"; cp -a "$P" "$T/D"; }
D=$T/D

echo "== a clean apply, then a rerun"
fresh; $APPLY "$D" >/dev/null 2>&1; check "series applies" "$?" 0
check "a.py has p2 and p3" "$(grep -c ' p[23]$' "$D/a.py")" 2
check "new.py is created and edited by a later patch" "$(grep -c 'p3$' "$D/new.py" 2>/dev/null)" 1
check "a stamp is written" "$([ -f "$D/.hyperqwen-patches" ] && echo yes)" yes
h=$(treehash "$D"); out=$($APPLY "$D" 2>&1); rc=$?
check "rerun exits 0" "$rc" 0
check "rerun says already applied" "$(printf '%s\n' "$out" | grep -c '^== the series is already applied')" 1
check "rerun changes nothing" "$(treehash "$D")" "$h"

echo "== a drifted line under p2's second file (the failure #437 hit)"
fresh; sed -i 's/^b line 24$/b line 24 drift/' "$D/sub/b.py"; h=$(treehash "$D")
out=$($APPLY "$D" 2>&1); rc=$?
check "exit 1" "$rc" 1
check "names p2" "$(printf '%s\n' "$out" | grep -c '^FAILED: p2.patch')" 1
check "the tree is exactly as before (p1's new.py and p2's a.py hunk not written)" "$(treehash "$D")" "$h"
check "no .rej, .orig or temporary file left" "$(strays "$D")" 0
check "no stamp" "$([ -f "$D/.hyperqwen-patches" ] && echo yes || echo no)" no

echo "== a different series on a tree the stamp describes"
fresh; $APPLY "$D" >/dev/null 2>&1; h=$(treehash "$D")
cp -a "$R" "$T/repo2"; sed -i '1i # regenerated' "$T/repo2/patches/p1.patch"
out=$(bash "$T/repo2/patches/apply.sh" "$D" 2>&1); rc=$?
check "exit 1" "$rc" 1
check "says a different series" "$(printf '%s\n' "$out" | grep -c 'carries a different patch series')" 1
check "the tree is unchanged" "$(treehash "$D")" "$h"

echo "== a stale stamp (the files reinstalled, the stamp left behind)"
fresh; $APPLY "$D" >/dev/null 2>&1; cp "$D/.hyperqwen-patches" "$T/stamp"; h=$(treehash "$D")
fresh; cp "$T/stamp" "$D/.hyperqwen-patches"; rm -f "$D/new.py"
out=$($APPLY "$D" 2>&1); rc=$?
check "exit 0" "$rc" 0
check "notes the stale stamp" "$(printf '%s\n' "$out" | grep -c 'does not match the tree')" 1
check "the series lands as on a fresh tree" "$(treehash "$D")" "$h"

echo "== KVarN"
fresh; out=$($APPLY --kvarn "$D" 2>&1); rc=$?
check "--kvarn before the series: exit 1" "$rc" 1
check "--kvarn before the series: the tree is unchanged" "$(treehash "$D")" "$(treehash "$P")"
check "--kvarn before the series: no stray files" "$(strays "$D")" 0
fresh; $APPLY "$D" >/dev/null 2>&1; $APPLY --kvarn "$D" >/dev/null 2>&1; check "--kvarn after the series applies" "$?" 0
check "the stamp gains a kvarn line" "$(grep -c '^kvarn ' "$D/.hyperqwen-patches")" 1
h=$(treehash "$D")
check "series rerun after KVarN: already applied" "$($APPLY "$D" 2>&1 | grep -c '^== the series is already applied')" 1
check "--kvarn rerun: all ${#KV[@]} already applied" "$($APPLY --kvarn "$D" 2>&1 | grep -c '(already applied)$')" "${#KV[@]}"
check "neither rerun changes the tree" "$(treehash "$D")" "$h"
fresh; $APPLY "$D" >/dev/null 2>&1; sed -i 's/^b line 14$/b line 14 drift/' "$D/sub/b.py"; h=$(treehash "$D")
$APPLY --kvarn "$D" >/dev/null 2>&1; check "a later KVarN patch fails: exit 1" "$?" 1
check "a later KVarN patch fails: the earlier ones are not written either" "$(treehash "$D")" "$h"

echo
[ "$FAILS" = 0 ] && echo "test_apply_staged: all checks passed" || { echo "test_apply_staged: $FAILS FAILED"; exit 1; }
