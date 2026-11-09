#!/usr/bin/env bash
# The one door into patches/series: this file reads the apply order and applies it.
# Every other consumer (the Dockerfile, verify.sh, patches/check_vllm_series.sh,
# kvarn/install.sh and the install pages in docs/) calls this file instead of parsing
# patches/series itself.
#
#   bash patches/apply.sh --list      print the series in apply order, one basename per line
#   bash patches/apply.sh --list --kvarn
#                                     print the four KVarN patches in apply order, the same way
#   bash patches/apply.sh DIR         apply the series to DIR, the vllm PACKAGE directory
#                                     (site-packages/vllm, not site-packages)
#   bash patches/apply.sh --kvarn DIR apply the four KVarN patches in kvarn/ to DIR, after
#                                     the series (kvarn/install.sh calls this; so does CI)
#
# Exit codes: 0 success; 1 a patch did not apply (the message names it); 2 a usage error,
# or a list and its directory disagree (patches/series and patches/, or KVARN and kvarn/).
#
# --list always prints the series. When the series and the directory disagree, it also
# names each offender on stderr and exits 2: a patch that is not in the series is never
# applied, and a name in the series with no file is a typo. The apply mode refuses to start
# on that disagreement. --list --kvarn and --kvarn do the same for the KVARN list below and
# kvarn/*.patch.
#
# Apply policy: GNU `patch -p1 --forward --fuzz 0 --no-backup-if-mismatch`, in order, stop
# at the first patch that fails. An offset means the context matched exactly and the
# file only grew around it: that is benign, and the output reports it. Fuzz means the
# context did NOT match and GNU patch accepted an approximate anchor: that is a patch cut
# against a tree that no longer exists, so it fails here by name instead of landing by guess.
# --forward makes a patch that is already applied fail by name, instead of a question on
# the terminal.
#
# Staged: GNU patch writes every hunk that matches, in every file of a patch, and saves only
# the failing hunks to a .rej, so a patch applied in place and failing leaves DIR partly
# patched, and every later run fails on it (#437, #445). So both apply modes copy the files
# their patches touch (the `diff --git` paths) into a temporary directory, apply there, and
# put the results into DIR only when every patch has applied. Each result is first copied
# next to its target under a temporary name and then renamed over it, so even a full disk
# fails before DIR changes. A failed run leaves DIR exactly as it was. A per-patch --dry-run
# cannot replace this: later patches edit files that earlier ones create.
#
# Stamp: a successful series apply writes DIR/.hyperqwen-patches, a hash of the series and
# the sha256 of every file it touched; --kvarn adds a hash of the KVarN patches and their
# files, when the stamp matched the tree before it ran. On the next series apply, a stamp
# whose files all match says either "this series is already applied" (exit 0, nothing done)
# or "a different series is applied" (exit 1, nothing done: install vLLM into a new venv and
# apply there). A stamp whose files do not match (vLLM reinstalled, or a file edited since)
# is ignored, and the staged apply runs as if it were absent.
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
# and in place of all of them, when the stamp says so:
#   == the series is already applied (...)
# On a failure, the patch output goes to stderr, indented, after a FAILED: line.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KVARN_DIR=$(cd -- "$HERE/../kvarn" && pwd)
# The KVarN patches are exported from commits that sit after the whole series, in
# this order: kvarn-0.32.0 and kvarn-v2-runner carry context that the series adds,
# kvarn-v2-runner also carries context that kvarn-0.32.0 adds, and kvarn-fp16-dequant's
# envs.py hunk is cut against the tree with the other three in place.
KVARN=(kvarn-0.33.0.patch kvarn-v2-runner-0.33.0.patch kvarn-recycled-pages-0.33.0.patch
       kvarn-fp16-dequant-0.33.0.patch)

usage() {
  echo "usage: bash patches/apply.sh --list [--kvarn] | [--kvarn] DIR  (DIR = the installed vllm package directory)" >&2
  exit 2
}

series() {
  sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series"
}

# agree WHAT NAMES DIR: 0 when NAMES (one per line) and DIR/*.patch are the same set; else
# name each offender on stderr and return 2.
agree() {
  local what=$1 names=$2 dir=$3 on_disk listed x
  on_disk=$(for f in "$dir"/*.patch; do [ -e "$f" ] && basename "$f"; done | sort)
  listed=$(printf '%s\n' "$names" | sort)
  [ "$on_disk" = "$listed" ] && return 0
  echo "ERROR: $what and the $(basename "$dir")/ directory disagree:" >&2
  for x in $(comm -23 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$listed")); do
    echo "    not in $what (never applied): $x" >&2
  done
  for x in $(comm -13 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$listed")); do
    echo "    in $what, no such file (or listed twice): $x" >&2
  done
  return 2
}

list() {
  local names
  names=$(series)
  printf '%s\n' "$names"
  agree patches/series "$names" "$HERE"
}

list_kvarn() {
  printf '%s\n' "${KVARN[@]}"
  agree "the KVARN list in patches/apply.sh" "$(printf '%s\n' "${KVARN[@]}")" "$KVARN_DIR"
}

STAMP=.hyperqwen-patches
STAGE=
trap '[ -z "$STAGE" ] || rm -rf "$STAGE"' EXIT

# targets PATCH_FILE...: every package-relative path the patches touch, sorted, once each. A
# stray extra path is harmless (a file that is not there is not staged), so both headers count.
targets() {
  sed -n -e 's#^diff --git a/.* b/##p' -e 's#^+++ b/##p' "$@" | sed 's/\t.*//' | sort -u
}

# hash_of NAME...: one hash over the names and contents of the patches, in this order.
hash_of() {
  local dir=$1 name; shift
  for name in "$@"; do printf '%s\n' "$name"; cat "$dir/$name"; done | sha256sum | cut -d' ' -f1
}

# stage DIR PATH...: copy each PATH that exists in DIR into a new temporary directory, $STAGE.
stage() {
  local dir=$1 f; shift
  STAGE=$(mktemp -d "${TMPDIR:-/tmp}/hyperqwen-apply.XXXXXX")
  for f in "$@"; do
    [ -e "$dir/$f" ] || continue
    mkdir -p "$STAGE/$(dirname "$f")"
    cp -p "$dir/$f" "$STAGE/$f"
  done
}

# commit DIR PATH...: put each staged PATH that differs from DIR's copy into DIR. Every file is
# copied next to its target under a temporary name first, and only then renamed over it.
commit() {
  local dir=$1 f g new=(); shift
  for f in "$@"; do
    [ -f "$STAGE/$f" ] || continue
    ! cmp -s "$STAGE/$f" "$dir/$f" || continue
    if ! { mkdir -p "$dir/$(dirname "$f")" && cp -p "$STAGE/$f" "$dir/$f.hyperqwen-new"; }; then
      for g in "${new[@]}"; do rm -f "$dir/$g.hyperqwen-new"; done
      rm -f "$dir/$f.hyperqwen-new"
      echo "FAILED: could not write $dir/$f.hyperqwen-new; nothing in $dir was changed" >&2
      exit 1
    fi
    new+=("$f")
  done
  for f in "${new[@]}"; do mv -f "$dir/$f.hyperqwen-new" "$dir/$f"; done
}

# stamp_matches DIR: 0 when DIR has a stamp and every file in it hashes as the stamp says.
stamp_matches() {
  [ -f "$1/$STAMP" ] || return 1
  (cd "$1" && grep -E '^[0-9a-f]{64}  ' "$STAMP" | sha256sum -c --status --strict -) 2>/dev/null
}

# write_stamp DIR HEADER_LINES PATH...: the stamp, with each PATH's hash as DIR now has it.
write_stamp() {
  local dir=$1 head=$2 f; shift 2
  {
    echo "# Written by HyperQwen's patches/apply.sh: what it applied here, and each file's sha256 after."
    printf '%s\n' "$head"
    for f in "$@"; do [ -f "$dir/$f" ] && (cd "$dir" && sha256sum -- "$f"); done
  } > "$dir/$STAMP.hyperqwen-new"
  mv -f "$dir/$STAMP.hyperqwen-new" "$dir/$STAMP"
}

# apply_one DIR SHOWN PATCH_FILE NAME HINT: apply one patch to DIR by the policy above, or exit 1
# by name. DIR is the staged copy; SHOWN is the directory the messages name.
apply_one() {
  local dir=$1 shown=$2 file=$3 name=$4 hint=$5 out n
  if ! out=$(patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$dir" -i "$file" 2>&1); then
    {
      echo "FAILED: $name does not apply to $shown with exact context; nothing in $shown was changed."
      echo "Possible causes:"
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
  local dir=$1 names name want have n
  local -a files=() paths=()
  [ -d "$dir" ] || { echo "ERROR: $dir is not a directory" >&2; usage; }
  names=$(list) || exit 2
  while IFS= read -r name; do files+=("$HERE/$name"); done <<<"$names"
  mapfile -t paths < <(targets "${files[@]}")
  # shellcheck disable=SC2086  # the names are basenames without spaces (agree() checked them)
  want=$(hash_of "$HERE" $names)
  if stamp_matches "$dir"; then
    have=$(sed -n 's/^series //p' "$dir/$STAMP")
    n=$(grep -cE '^[0-9a-f]{64}  ' "$dir/$STAMP")
    if [ "$have" = "$want" ]; then
      echo "== the series is already applied ($dir/$STAMP: all $n files as apply.sh left them)"
      exit 0
    fi
    {
      echo "FAILED: $dir carries a different patch series (an earlier checkout of this repo, or"
      echo "another vLLM pin; $STAMP and all $n of its files agree), so this one cannot apply on"
      echo "top. Nothing in $dir was changed. Install vLLM into a new venv and apply there."
    } >&2
    exit 1
  elif [ -f "$dir/$STAMP" ]; then
    echo "note: $dir/$STAMP does not match the tree (vLLM reinstalled, or a file edited since);" \
         "applying as if it were absent" >&2
  fi
  stage "$dir" "${paths[@]}"
  while IFS= read -r name; do
    apply_one "$STAGE" "$dir" "$HERE/$name" "$name" "$SERIES_HINT"
  done <<<"$names"
  commit "$dir" "${paths[@]}"
  write_stamp "$dir" "series $want" "${paths[@]}"
}

apply_kvarn() {
  local dir=$1 name matched=0 head
  local -a files=() paths=()
  [ -d "$dir" ] || { echo "ERROR: $dir is not a directory" >&2; usage; }
  agree "the KVARN list in patches/apply.sh" "$(printf '%s\n' "${KVARN[@]}")" "$KVARN_DIR" || exit 2
  for name in "${KVARN[@]}"; do files+=("$KVARN_DIR/$name"); done
  mapfile -t paths < <(targets "${files[@]}")
  ! stamp_matches "$dir" || matched=1
  stage "$dir" "${paths[@]}"
  for name in "${KVARN[@]}"; do
    if patch -p1 -R --dry-run -s --fuzz 0 -d "$STAGE" -i "$KVARN_DIR/$name" >/dev/null 2>&1; then
      echo "== $name (already applied)"
      continue
    fi
    apply_one "$STAGE" "$dir" "$KVARN_DIR/$name" "$name" "$KVARN_HINT"
  done
  commit "$dir" "${paths[@]}"
  # Only a stamp that described the tree before this run is extended; a stale one stays stale.
  if [ "$matched" = 1 ]; then
    head="$(grep '^series ' "$dir/$STAMP")"$'\n'"kvarn $(hash_of "$KVARN_DIR" "${KVARN[@]}")"
    mapfile -t paths < <({ grep -E '^[0-9a-f]{64}  ' "$dir/$STAMP" | cut -c67-; printf '%s\n' "${paths[@]}"; } | sort -u)
    write_stamp "$dir" "$head" "${paths[@]}"
  fi
}

case "${1:-}" in
  --list)
    if [ $# -eq 1 ]; then list
    elif [ $# -eq 2 ] && [ "$2" = --kvarn ]; then list_kvarn
    else usage; fi ;;
  --kvarn) [ $# -eq 2 ] || usage; apply_kvarn "$2" ;;
  ""|-*) usage ;;
  *) [ $# -eq 1 ] || usage; apply_series "$1" ;;
esac
