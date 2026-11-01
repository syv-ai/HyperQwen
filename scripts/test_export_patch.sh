#!/usr/bin/env bash
# export-patch.sh on a throwaway vllm repo: a new file takes the commit body, and a re-export
# into that file keeps its edited preamble while the hunks follow the new commit.
set -eu
EXPORT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/export-patch.sh"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
g() { git -C "$T/repo" -c user.name=t -c user.email=t@t "$@"; }
mkdir -p "$T/repo/vllm"; g init -q
printf 'a\n' > "$T/repo/vllm/x.py"; g add -A; g commit -q -m base
printf 'a\nb\n' > "$T/repo/vllm/x.py"; g commit -q -am '[qwen38] demo' -m 'Commit body.'
OUT="$T/demo.patch"
fail() { echo "FAIL: $1"; cat "$OUT"; exit 1; }

bash "$EXPORT" "$T/repo" HEAD "$OUT" >/dev/null
grep -qx 'Commit body.' "$OUT" || fail "a new file takes the commit body"

sed -i 's/^Commit body\.$/Kind: fix/' "$OUT"
printf 'a\nc\n' > "$T/repo/vllm/x.py"; g commit -q -a --amend -m '[qwen38] demo' -m 'Another body.'
bash "$EXPORT" "$T/repo" HEAD "$OUT" >/dev/null
grep -qx 'Kind: fix' "$OUT" || fail "a re-export keeps the edited preamble"
! grep -q 'Another body' "$OUT" || fail "a re-export does not take the commit body"
grep -qx '+c' "$OUT" && ! grep -qx '+b' "$OUT" || fail "the hunks follow the new commit"
grep -q "^--- exported from $(g rev-parse --short HEAD) (demo)" "$OUT" || fail "the marker names the new commit"
echo "export-patch round trip: OK"
