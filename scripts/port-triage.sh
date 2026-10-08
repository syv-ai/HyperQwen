#!/usr/bin/env bash
# Port triage: replay every topic commit of a fork branch onto a new vLLM tag, one at a time, and report which apply
# cleanly, which conflict (and on which files), and which come out EMPTY because the new tag already carries them.
# Nothing is pushed and no branch is moved: it works in a throwaway worktree and removes it. The table is the port's
# to-do list (docs/vllm-0.29.md "Porting the next pin", step 1, made mechanical).
#
#   bash scripts/port-triage.sh <fork bare repo or checkout> <old tag> <new tag> <fork branch or commit> [out.md]
#   e.g. bash scripts/port-triage.sh ../vllm v0.29.0 v0.30.0 qwen38/0.29-hq2
#
# Each topic is tried on top of the new tag plus every topic that applied before it, the order a real rebase takes.
# A conflicting topic is recorded and skipped, so later topics that depend on it may report conflicts that go away
# once it is resolved: read the table top-down. rerere is on, so resolutions recorded in this repo are reused.
set -uo pipefail
FORK=$1; OLD=$2; NEW=$3; BR=$4; OUT=${5:-}
G="git -C $FORK"
$G rev-parse -q --verify "$NEW^{commit}" >/dev/null || { echo "no $NEW in $FORK (fetch upstream --tags?)"; exit 2; }
$G config rerere.enabled true; $G config rerere.autoupdate true
T=$(mktemp -d); W="$T/w"; $G worktree add -q --detach "$W" "$NEW" || exit 2
trap '$G worktree remove --force "$W" >/dev/null 2>&1; rm -rf "$T"' EXIT
clean=0; conf=0; empty=0; rows=()
for c in $($G rev-list --reverse "$OLD..$BR"); do
  subj=$($G log -1 --format=%s "$c"); topic=$(echo "$subj" | sed -nE 's/^\[qwen38\] ([A-Za-z0-9._-]+).*/\1/p'); topic=${topic:-$subj}
  if git -C "$W" -c user.name=triage -c user.email=t@t cherry-pick --allow-empty -x "$c" >/dev/null 2>&1; then
    if [ -z "$(git -C "$W" diff --name-only HEAD^ HEAD)" ]; then empty=$((empty+1)); rows+=("| $topic | EMPTY (the new tag carries it) | |")
    else clean=$((clean+1)); rows+=("| $topic | clean | |"); fi
  else
    files=$(git -C "$W" diff --name-only --diff-filter=U | sed 's#^vllm/##' | tr '\n' ' ')
    if [ -z "$files" ] && git -C "$W" -c user.name=triage -c user.email=t@t cherry-pick --continue >/dev/null 2>&1; then
      clean=$((clean+1)); rows+=("| $topic | clean (rerere resolved) | |")
    else
      git -C "$W" cherry-pick --abort >/dev/null 2>&1; conf=$((conf+1)); rows+=("| $topic | CONFLICT | $files |")
    fi
  fi
done
# A backport rarely comes out EMPTY: upstream's final code differs from what we carried, so it CONFLICTS instead.
# The PATCHES.md "upstream" and "retires when" columns name the vLLM PRs behind each row; a row whose every named PR
# has its merge commit in the new tag is a retirement candidate whatever the cherry-pick said. Needs `gh`.
PM="$(dirname "$0")/../PATCHES.md"
if [ -f "$PM" ] && command -v gh >/dev/null; then
  for i in "${!rows[@]}"; do
    topic=$(echo "${rows[$i]}" | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    line=$(grep -E "^\| ${topic} \|" "$PM" | head -1); [ -n "$line" ] || continue
    prs=$(echo "$line" | awk -F'|' '{print $5" "$7}' | grep -oE '#[0-9]{5}' | tr -d '#' | sort -u); [ -n "$prs" ] || continue
    all=1; seen=""
    for pr in $prs; do
      oid=$(gh pr view "$pr" -R vllm-project/vllm --json mergeCommit --jq '.mergeCommit.oid // ""' 2>/dev/null)
      if [ -n "$oid" ] && $G merge-base --is-ancestor "$oid" "$NEW" 2>/dev/null; then seen="$seen #$pr"; else all=0; fi
    done
    [ "$all" = 1 ] && rows[$i]="${rows[$i]%|} RETIRE? every named PR is in $NEW:$seen |"
  done
fi
# Release-branch commits the new tag does not carry. A release tag is not a superset of the one before it: fixes can
# be cherry-picked onto a release branch and never reach main. 0.29.0 -> 0.30.0 dropped vllm #55760, #55861 and
# #55863 that way (the dense prefix-retention default for hybrid + EAGLE), and 0.30 changed behaviour silently. Match
# by PR number, because a cherry-pick has a different hash from the commit on main.
SUBJ="$T/new-subjects"; $G log --format=%s "$NEW" > "$SUBJ"
dropped=()
for c in $($G rev-list "$NEW..$OLD"); do
  s=$($G log -1 --format=%s "$c"); pr=$(echo "$s" | grep -oE '\(#[0-9]{4,6}\)' | tail -1)
  if [ -z "$pr" ]; then dropped+=("| (no PR number) | ${s:0:100} |")
  elif ! grep -qF "$pr" "$SUBJ"; then dropped+=("| ${pr//[()]/} | ${s:0:100} |"); fi
done
{
  echo "# Port triage: $BR ($OLD..) onto $NEW, $(date -u +%Y-%m-%dT%H:%MZ)"
  echo
  echo "$((clean+conf+empty)) topics: $clean clean, $conf conflict, $empty empty (carried upstream)."
  echo
  echo "| topic | result | conflicting files |"; echo "|---|---|---|"
  printf '%s\n' "${rows[@]}"
  echo
  echo "## In $OLD, not in $NEW (${#dropped[@]}): read each for behaviour the port loses"
  echo
  if [ ${#dropped[@]} -gt 0 ]; then echo "| PR | subject |"; echo "|---|---|"; printf '%s\n' "${dropped[@]}"; else echo "(none)"; fi
} | if [ -n "$OUT" ]; then tee "$OUT"; else cat; fi
