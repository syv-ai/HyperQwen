#!/bin/bash
# CPU test for #204: which --host each launcher passes to `vllm serve`, with and without a key, and that
# each launcher refuses to boot when it cannot source resolve_api_key.sh. Also which INT8 variables each
# launcher exports (qwen_int8_exports): the include regex only with the input dtype.
# Copies the checkout to a temp dir, so the api_key.txt rows never touch the real one, and runs the three
# launchers with PRINT_ARGV=1 (launcher_common.sh), which prints the argv instead of starting vLLM. No GPU,
# no model, no network. In a container (/.dockerenv) the default differs, so this is skipped there.
#
#   bash bench/test_no_key_bind.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
[ -f /.dockerenv ] && { echo "skip: /.dockerenv exists, the container default is 0.0.0.0 by design"; exit 0; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
# The stub vllm prints the two INT8 exports for int8_of, and no real vllm starts. A launcher that ignores
# PRINT_ARGV=1 reaches it too, prints no --host line, and fails its host_of rows.
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\necho "${VLLM_MARLIN_INPUT_DTYPE-unset} ${VLLM_MARLIN_INT8_INCLUDE_RE-unset}"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
# host_of <launcher> [@keyfile] <env assignments...>: the argument after --host in the PRINT_ARGV=1 argv.
# @keyfile puts the key only in api_key.txt.
host_of() { local sc=$1; shift
  rm -f "$T/api_key.txt"; [ "${1:-}" = @keyfile ] && { echo filekey > "$T/api_key.txt"; shift; }
  (cd "$T" && env -u VLLM_API_KEY -u HOST PRINT_ARGV=1 "$@" bash "$sc" 2>/dev/null </dev/null) | sed -n '/^--host$/{n;p;}'; }
# int8_of <launcher> <env assignments...>: "<VLLM_MARLIN_INPUT_DTYPE> <VLLM_MARLIN_INT8_INCLUDE_RE>" as the
# stub vllm sees them, "unset" for one the launcher did not export.
int8_of() { local sc=$1; shift
  rm -f "$T/api_key.txt"
  (cd "$T" && env -u VLLM_API_KEY -u HOST -u PRINT_ARGV -u VLLM_MARLIN_INPUT_DTYPE -u VLLM_MARLIN_INT8_INCLUDE_RE "$@" bash "$sc" 2>/dev/null </dev/null) | tail -n 1; }
# refused <launcher>: resolve_api_key.sh missing, the key only in api_key.txt. Prints "exit 1, no --host" when
# the launcher stops before it prints an argv.
refused() { local out rc
  mv "$T/resolve_api_key.sh" "$T/resolve_api_key.sh.off"; echo filekey > "$T/api_key.txt"
  out=$(cd "$T" && env -u VLLM_API_KEY -u HOST PRINT_ARGV=1 bash "$1" 2>/dev/null </dev/null); rc=$?
  mv "$T/resolve_api_key.sh.off" "$T/resolve_api_key.sh"
  out=$(printf '%s\n' "$out" | sed -n '/^--host$/{n;p;}')
  [ $rc = 1 ] && [ -z "$out" ] && echo "exit 1, no --host" || echo "exit $rc, --host ${out:-<none>}"; }
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$what" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
for sc in single-user/start_qwen.sh batch/start_qwen.sh single-user/alternative.sh; do
  echo "== $sc"
  check 127.0.0.1 "$(host_of $sc)"                              "no key, no HOST"
  check 0.0.0.0   "$(host_of $sc VLLM_API_KEY=k)"               "key in the environment"
  check 0.0.0.0   "$(host_of $sc @keyfile)"                     "key only in api_key.txt"
  check 0.0.0.0   "$(host_of $sc HOST=0.0.0.0)"                 "no key, HOST=0.0.0.0 (explicit)"
  check 127.0.0.1 "$(host_of $sc HOST=127.0.0.1 VLLM_API_KEY=k)" "key, HOST=127.0.0.1"
  check 10.1.2.3  "$(host_of $sc HOST=10.1.2.3 VLLM_API_KEY=k)"  "key, HOST=10.1.2.3"
  check "exit 1, no --host" "$(refused $sc)"                    "resolve_api_key.sh missing, key only in api_key.txt"
  check "int8 mlp"    "$(int8_of $sc INT8_ACT=int8 INT8_LAYERS=mlp)" "INT8_ACT=int8 INT8_LAYERS=mlp"
  check "unset unset" "$(int8_of $sc INT8_ACT= INT8_LAYERS=mlp)"     "INT8_ACT= INT8_LAYERS=mlp: no regex without the dtype"
  check "int8 unset"  "$(int8_of $sc INT8_ACT=int8 INT8_LAYERS=)"    "INT8_ACT=int8 INT8_LAYERS=: empty is unset (#20)"
done
echo "== verify.sh key check (the section only, as the launcher env would set it)"
vk() { (cd "$T" && rm -f api_key.txt; env -u VLLM_API_KEY -u HOST "$@" bash -c '
  ok(){ echo "PASS $1"; }; warn(){ echo "WARN $1"; }; fail(){ echo "FAIL $1"; }
  eval "$(sed -n "/^if \[ -s api_key.txt \] || \[ -n/,/^fi\$/p" verify.sh)"' 2>&1 | cut -c1-4); }
[ "$(vk)" = WARN ] && echo "  PASS  no key, no HOST -> WARN" || { echo "  FAIL  no key, no HOST"; FAILS=$((FAILS+1)); }
[ "$(vk HOST=127.0.0.1)" = WARN ] && echo "  PASS  no key, HOST=127.0.0.1 -> WARN" || { echo "  FAIL  no key, HOST=127.0.0.1"; FAILS=$((FAILS+1)); }
[ "$(vk HOST=0.0.0.0)" = FAIL ] && echo "  PASS  no key, HOST=0.0.0.0 -> FAIL" || { echo "  FAIL  no key, HOST=0.0.0.0"; FAILS=$((FAILS+1)); }
[ "$(vk VLLM_API_KEY=k HOST=0.0.0.0)" = PASS ] && echo "  PASS  key, HOST=0.0.0.0 -> PASS" || { echo "  FAIL  key, HOST=0.0.0.0"; FAILS=$((FAILS+1)); }
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
