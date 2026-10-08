#!/usr/bin/env bash
# Export one topic commit from a vLLM checkout as a patch file in this repo's convention
# (paths relative to the vllm package, applied with `patch -p1 -d site-packages/vllm`; prose above the first hunk).
# The file is the source; this is how a commit becomes one, and the file is never edited by hand (PATCHES.md,
# the "Cut against" paragraph). The marker names no repo: the hash belongs to whoever exported it.
#
#   bash scripts/export-patch.sh <vllm checkout> <commit> [patches/<name>.patch]
set -eu
FORK="$1"; COMMIT="$2"; OUT="${3:-}"
G="git -C $FORK"
SUBJ=$($G log -1 --format='%s' "$COMMIT"); TOPIC=$(echo "$SUBJ" | sed -nE 's/^\[qwen38\] ([A-Za-z0-9._-]+).*/\1/p')
[ -n "$TOPIC" ] || { echo "not a topic commit: $SUBJ"; exit 1; }
[ -n "$OUT" ] || OUT="patches/$TOPIC.patch"
# The body is prose. A commit imported from an old patch file can carry a stray diff line (a new-file
# patch's "--- /dev/null" and "@@" preamble boundary); GNU patch would read that as the start of a hunk.
BODY=$($G log -1 --format='%b' "$COMMIT" | sed -e '/^Source: /,$d' | grep -vE '^(@@ |--- |\+\+\+ |diff --git )' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}')
SHORT=$($G rev-parse --short "$COMMIT")
{
  printf '%s\n\n' "$BODY"
  printf -- '--- exported from %s (%s); regenerate with scripts/export-patch.sh, do not edit ---\n\n' "$SHORT" "$TOPIC"
  # strip the vllm/ prefix from the paths and the function context git appends to hunk headers
  $G diff "$COMMIT^" "$COMMIT" -- vllm \
    | sed -E 's#^(--- |\+\+\+ )([ab])/vllm/#\1\2/#' \
    | sed -E 's#^diff --git a/vllm/(.*) b/vllm/#diff --git a/\1 b/#' \
    | sed -E 's#^(@@ [^@]* @@).*#\1#'
} > "$OUT"
echo "wrote $OUT ($(grep -c '^@@' "$OUT") hunks, $(grep -c '^--- a/' "$OUT") files)"
