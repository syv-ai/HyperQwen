#!/usr/bin/env bash
# The one door into patches/series: this file reads the apply order and applies it.
# Every other consumer (the Dockerfile, verify.sh, patches/check_vllm_series.sh,
# kvarn/install.sh and the install pages in docs/) calls this file instead of parsing
# patches/series itself.
#
#   bash patches/apply.sh --list      print the series in apply order, one basename per line
#   bash patches/apply.sh DIR         apply the series to DIR, the vllm PACKAGE directory
#                                     (site-packages/vllm, not site-packages)
#   bash patches/apply.sh --kvarn DIR apply the four KVarN patches in kvarn/ to DIR, after
#                                     the series (kvarn/install.sh calls this; so does CI)
#
# Exit codes: 0 success; 1 a patch did not apply (the message names it); 2 a usage error,
# or patches/series and the patches/ directory disagree.
#
# --list always prints the series. When the series and the directory disagree, it also
# names each offender on stderr and exits 2: a patch that is not in the series is never
# applied, and a name in the series with no file is a typo. The apply mode refuses to start
# on that disagreement.
#
# Apply policy: GNU `patch -p1 --forward --fuzz 0 --no-backup-if-mismatch`, in order, stop
# at the first patch that fails. An offset means the context matched exactly and the
# file only grew around it: that is benign, and the output reports it. Fuzz means the
# context did NOT match and GNU patch accepted an approximate anchor: that is a patch cut
# against a tree that no longer exists, so it fails here by name instead of landing by guess.
# --forward makes a patch that is already applied fail by name, instead of a question on
# the terminal.
#
# --kvarn is idempotent, so a second kvarn/install.sh run is a no-op. Before it applies a
# KVarN patch, it checks the patch with an exact reverse dry-run (`patch -R --dry-run
# --fuzz 0`). That passes only when EVERY hunk of the patch is in the tree; then the patch
# is skipped. Each KVarN patch reverses cleanly on its own in a fully installed tree, so the
# check is exact. A partly applied patch fails both the reverse check and the forward apply,
# so it fails by name. (`patch -N` is not a safe idempotence test: when a file's first hunk
# is present, it skips the whole file and reports no failure, so a missing later hunk in that
# file stays missing.)
#
# Output contract (patches/check_vllm_series.sh counts these lines), one line per patch:
#   == <name>
#   == <name> (<N> hunk(s) at an offset)
#   == <name> (already applied)            (--kvarn only)
# On a failure, the patch output goes to stderr, indented, after a FAILED: line.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KVARN_DIR=$(cd -- "$HERE/../kvarn" && pwd)
# The KVarN patches are exported from the fork branch at a point after the whole series, in
# this order: kvarn-0.30.0 and kvarn-v2-runner carry context that the series adds,
# kvarn-v2-runner also carries context that kvarn-0.30.0 adds, and kvarn-fp16-dequant's
# envs.py hunk is cut against the tree with the other three in place.
KVARN=(kvarn-0.30.0.patch kvarn-v2-runner-0.30.0.patch kvarn-recycled-pages-0.30.0.patch
       kvarn-fp16-dequant-0.30.0.patch)

usage() {
  echo "usage: bash patches/apply.sh --list | [--kvarn] DIR  (DIR = the installed vllm package directory)" >&2
  exit 2
}

series() {
  sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series"
}

list() {
  local names on_disk in_series extra missing
  names=$(series)
  printf '%s\n' "$names"
  on_disk=$(for f in "$HERE"/*.patch; do [ -e "$f" ] && basename "$f"; done | sort)
  in_series=$(printf '%s\n' "$names" | sort)
  [ "$on_disk" = "$in_series" ] && return 0
  echo "ERROR: patches/series and the patches/ directory disagree:" >&2
  extra=$(comm -23 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$in_series"))
  missing=$(comm -13 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$in_series"))
  [ -z "$extra" ] || printf '    not in patches/series (never applied): %s\n' $extra >&2
  [ -z "$missing" ] || printf '    in patches/series, no such file (or listed twice): %s\n' $missing >&2
  return 2
}

# apply_one DIR PATCH_FILE NAME HINT: apply one patch by the policy above, or exit 1 by name.
apply_one() {
  local dir=$1 file=$2 name=$3 hint=$4 out n
  if ! out=$(patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$dir" -i "$file" 2>&1); then
    {
      echo "FAILED: $name does not apply to $dir with exact context. Possible causes:"
      printf '%s\n' "$hint"
      printf '%s\n' "$out" | sed 's/^/    /'
    } >&2
    exit 1
  fi
  n=$(printf '%s\n' "$out" | grep -c 'offset' || true)
  if [ "$n" -gt 0 ]; then echo "== $name ($n hunk(s) at an offset)"; else echo "== $name"; fi
}

SERIES_HINT="    the patch is already applied (bash verify.sh --no-server says which patches are),
    DIR is not the vllm package directory, or the vLLM version is not the pin in
    docker/requirements.txt (then regenerate the patch against the pin)."
KVARN_HINT="    the series is not applied yet (bash patches/apply.sh DIR first), the patch is
    partly applied (reinstall vllm, then apply the series and KVarN again), DIR is not
    the vllm package directory, or the vLLM version is not the pin in docker/requirements.txt."

apply_series() {
  local dir=$1 names name
  [ -d "$dir" ] || { echo "ERROR: $dir is not a directory" >&2; usage; }
  names=$(list) || exit 2
  while IFS= read -r name; do
    apply_one "$dir" "$HERE/$name" "$name" "$SERIES_HINT"
  done <<<"$names"
}

apply_kvarn() {
  local dir=$1 name
  [ -d "$dir" ] || { echo "ERROR: $dir is not a directory" >&2; usage; }
  for name in "${KVARN[@]}"; do
    if patch -p1 -R --dry-run -s --fuzz 0 -d "$dir" -i "$KVARN_DIR/$name" >/dev/null 2>&1; then
      echo "== $name (already applied)"
      continue
    fi
    apply_one "$dir" "$KVARN_DIR/$name" "$name" "$KVARN_HINT"
  done
}

case "${1:-}" in
  --list) [ $# -eq 1 ] || usage; list ;;
  --kvarn) [ $# -eq 2 ] || usage; apply_kvarn "$2" ;;
  ""|-*) usage ;;
  *) [ $# -eq 1 ] || usage; apply_series "$1" ;;
esac
