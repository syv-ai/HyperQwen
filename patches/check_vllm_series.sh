#!/usr/bin/env bash
set -euo pipefail

# Validate the patch series against a pristine checkout of the pinned vLLM (docker/requirements.txt).
#
# Two passes, because two tools are in play and they answer different questions:
#
#   1. Every patch, in the order of patches/series, then the four KVarN patches in
#      kvarn/, applied by patches/apply.sh (which the Dockerfile, docs/install.md and
#      kvarn/install.sh call) with GNU `patch` -- the tool that actually
#      installs this stack. This is the pass that says "a clone of this repo
#      still builds". It was missing entirely until 2026-09-07; the job checked
#      five of thirty files. The KVarN patches were checked only by the image
#      build until 2026-09-30; they apply at exactly the point the series ends.
#   2. The five DFlash patches whose order and hunk metadata are part of the
#      0.28.0 contract, applied with `git apply`, which is strict about offsets
#      and catches a hand-edited hunk header immediately.
#
# Five patches are accepted by GNU patch and rejected by git apply once the
# patches before them have moved their context. That is why pass 1 uses GNU
# patch: the installed tree is produced by GNU patch, so GNU patch defines
# "applies", and git apply stays on the five where hunk metadata is contractual.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE=${1:?usage: bash patches/check_vllm_series.sh /path/to/vllm-<pinned tag>/vllm  (the package directory inside the checkout, not its root)}
VLLM_SOURCE=$(cd -- "$VLLM_SOURCE" && pwd)

git -C "$VLLM_SOURCE" rev-parse --is-inside-work-tree >/dev/null

# The patches address files relative to the installed vllm package, not to the
# checkout root. `git apply` resolves a patch path against the REPOSITORY root
# and silently skips ("Skipped patch '...'") anything outside the subdirectory
# it runs in -- so running it from the package directory checked nothing at all
# and reported OK. Run from the repository root and name the prefix explicitly.
GIT_ROOT=$(git -C "$VLLM_SOURCE" rev-parse --show-toplevel)
PREFIX=${VLLM_SOURCE#"$GIT_ROOT"/}
if [ "$PREFIX" = "$VLLM_SOURCE" ]; then PREFIX=.; fi

# patches/apply.sh owns the series: it reads the apply order, checks that it and the
# patches/ directory agree exactly (exit 2 and each offender named, if not), and applies it.
LIST=$(bash "$HERE/patches/apply.sh" --list)
mapfile -t SERIES <<<"$LIST"

echo "== pass 1: the whole series, GNU patch, patches/series order, then the KVarN patches"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
# The apply policy (--fuzz 0: an offset is benign and reported, fuzz fails by name) is written
# down in patches/apply.sh, next to the code it describes. The Dockerfile and the install pages
# call the same file, so the image cannot carry what this check would refuse.
out=$(bash "$HERE/patches/apply.sh" "$VLLM_SOURCE" 2>&1) || {
  printf '%s\n' "$out" | sed 's/^/   /'; exit 1
}
printf '%s\n' "$out" | grep 'at an offset' | sed 's/^== /   /' || true
count=$(printf '%s\n' "$out" | grep -c '^== ' || true)
offset=$(printf '%s\n' "$out" | grep -c 'at an offset' || true)
if git -C "$GIT_ROOT" diff --quiet; then
  echo "ERROR: the series applied but changed nothing -- the paths did not resolve." >&2
  exit 1
fi
echo "   $count patches applied with exact context, $offset of them at an offset, 0 with fuzz"
# The KVarN patches are exported from commits that sit after the whole series, so they go on this tree.
# On a pristine checkout every one must apply now; "already applied" here means the checkout is not pristine.
kout=$(bash "$HERE/patches/apply.sh" --kvarn "$VLLM_SOURCE" 2>&1) || {
  printf '%s\n' "$kout" | sed 's/^/   /'; exit 1
}
printf '%s\n' "$kout" | sed 's/^== /   kvarn: /'
if printf '%s\n' "$kout" | grep -q 'already applied'; then
  echo "ERROR: a KVarN patch reads as applied on a pristine checkout -- the checkout is not pristine." >&2
  exit 1
fi
kcount=$(printf '%s\n' "$kout" | grep -c '^== ' || true)
echo "   $kcount KVarN patches applied after the series; $((count + kcount)) in total"
# A rerun of kvarn/install.sh must be a no-op: on the fully installed tree, every KVarN patch must
# reverse-apply on its own (apply.sh --kvarn's idempotence check). A later KVarN patch whose hunk
# lands inside an earlier one's context breaks that, and the second install fails by name.
rout=$(bash "$HERE/patches/apply.sh" --kvarn "$VLLM_SOURCE" 2>&1) || {
  echo "ERROR: a second apply.sh --kvarn run failed -- a KVarN patch no longer reverse-applies on its own:" >&2
  printf '%s\n' "$rout" | sed 's/^/   /' >&2; exit 1
}
if [ "$(printf '%s\n' "$rout" | grep -c 'already applied')" -ne "$kcount" ]; then
  echo "ERROR: a second apply.sh --kvarn run applied something again:" >&2
  printf '%s\n' "$rout" | sed 's/^/   /' >&2; exit 1
fi
echo "   a second apply.sh --kvarn run: all $kcount already applied (kvarn/install.sh rerun is a no-op)"

echo "== pass 2: the ordered DFlash patches, git apply --check"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
# These five patches' order and hunk metadata are part of the 0.28.0 contract.
# The set is fixed here; their relative order is taken from patches/series so
# the two can no longer drift.
CONTRACTUAL=(
  vllm-pr50021-gdn-spec-bounds.patch
  dflash2-lookup-drafting.patch
  dflash2-ngram-chains.patch
  dflash2-prewarm.patch
  dflash2-z-adaptive-emitted.patch
)
PATCHES=()
for name in "${SERIES[@]}"; do
  for c in "${CONTRACTUAL[@]}"; do [ "$name" = "$c" ] && PATCHES+=("$name"); done
done
[ "${#PATCHES[@]}" -eq "${#CONTRACTUAL[@]}" ] || {
  echo "ERROR: a contractual DFlash patch is missing from patches/series" >&2
  exit 1
}
for name in "${PATCHES[@]}"; do
  patch="$HERE/patches/$name"
  echo "   git apply --check $name"
  git -C "$GIT_ROOT" apply --check --whitespace=error -p1 --directory="$PREFIX" < "$patch"
  git -C "$GIT_ROOT" apply --whitespace=error -p1 --directory="$PREFIX" < "$patch"
done
if git -C "$GIT_ROOT" diff --quiet; then
  echo "ERROR: pass 2 applied but changed nothing." >&2
  exit 1
fi

git -C "$GIT_ROOT" diff --check
echo "patch integrity: OK"
