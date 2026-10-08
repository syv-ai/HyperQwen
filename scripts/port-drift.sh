#!/usr/bin/env bash
# Port drift: compare every topic on a new integration branch with the same topic on the old one, by content (the lines
# each commit adds, per file), so a port that took a resolution from somewhere else (a rebase onto upstream main, a hand merge)
# cannot silently lose work the old line gained later. The replay oracle proves the patch files match the new fork
# branch; this proves the new fork branch still carries what the old one meant.
#
#   bash scripts/port-drift.sh <fork repo> <new tag> <new branch> <old tag> <old branch> [--show]
#   e.g. bash scripts/port-drift.sh ../vllm v0.30.0 qwen38/0.30 v0.29.0 qwen38/0.29-hq2
#
# Every line it prints is a difference to explain: an upstream retirement, a resolution against moved upstream code,
# or a line order change. --show prints the lines themselves under each DRIFT row ("LOST" = the old topic added it and
# the new one does not; "NEW" = the reverse). A count is not a review: 2026-09-23 the cut2 report read
# "kvarn-v2-runner-0.30.0 -> attention.py:18" and was filed as a documented adaptation; four of those lines were the
# divisor rule for the drafter's sliding-window block, and without it CTX=huge + dflash2 + prefix caching (#179) was
# refused at boot. 2026-09-23: the first 0.30 cut took seven topics from main-track (cut 09-13) and this report
# showed the knob sweep, the KVARN_* registration and the #86 cast missing, all landed on 0.29 after main-track was cut.
set -uo pipefail
FORK=$1; NEWTAG=$2; NEWBR=$3; OLDTAG=$4; OLDBR=$5; SHOW=${6:-}
G="git -C $FORK"
topic_of(){ $G log -1 --format=%s "$1" | sed -nE 's/^\[qwen38\] ([A-Za-z0-9._-]+).*/\1/p'; }
# exact topic-name match (a --grep with "\[" was unreliable here, and an unanchored one matched other topics' bodies)
find_old(){ $G log --format='%H %s' "$OLDTAG..$OLDBR" | awk -v want="$1" '{h=$1; s=$0; sub(/^[^ ]+ /,"",s); if (s ~ /^\[qwen38\] /) { t=s; sub(/^\[qwen38\] /,"",t); sub(/[^A-Za-z0-9._-].*$/,"",t); if (t==want) {print h; exit} }}'; }
added(){ $G diff "$1^" "$1" -- "$2" | grep -E '^\+[^+]' | sed 's/[[:space:]]*$//'; }
clean=0; drift=0
for c in $($G rev-list --reverse "$NEWTAG..$NEWBR"); do
  t=$(topic_of "$c"); [ -n "$t" ] || continue
  old_t=$(echo "$t" | sed -E "s/${NEWTAG#v}/${OLDTAG#v}/")        # kvarn-0.30.0 <-> kvarn-0.29.0
  o=$(find_old "$old_t"); [ -n "$o" ] || { echo "NEW   $t (no counterpart on $OLDBR)"; continue; }
  line=""; body=""
  for f in $( ($G diff --name-only "$c^" "$c"; $G diff --name-only "$o^" "$o") | sort -u); do
    d=$(diff <(added "$o" "$f") <(added "$c" "$f") | grep -E '^[<>]')
    n=$(printf '%s' "$d" | grep -cE '^[<>]')
    [ "$n" -gt 0 ] || continue
    line="$line $(basename "$f"):$n"
    [ "$SHOW" = "--show" ] && body="$body    == $f"$'
'"$(printf '%s
' "$d" | sed -E 's/^< \+/    LOST  /; s/^> \+/    NEW   /')"$'
'
  done
  if [ -n "$line" ]; then drift=$((drift+1)); echo "DRIFT $t ->$line"; [ -n "$body" ] && printf '%s' "$body"; else clean=$((clean+1)); fi
done
echo "port-drift: $clean topics identical in content, $drift with differences to explain"
