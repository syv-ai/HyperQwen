#!/bin/bash
# CPU test for the resident-request note in single-user/start_qwen.sh (SPEC=dflash2). The note
# used to divide KV_MEM by a flat 104,988,089 B per (k+2): no context term and one page size for
# every k and KV dtype. It now computes the engine's block and pool page from the model's shape,
# k and the KV dtype, and charges 6 x (2 + k) state pages per request (align mode, which vLLM
# 0.31.0 runs by default for this model) plus 2 x ceil(ctx / block) attention pages.
#
# Each arm below is a measured boot on the reference 3090 (RTX 3090, vLLM 0.31.0), run here with
# the KV_MEM, KV dtype and prefix setting it booted with. The checks are the engine's own
# numbers from those boots:
#   num_gpu_blocks and the attention block  -> must match exactly;
#   resident requests at ~1k-token prompts  -> must lie inside the printed range;
#   resident requests at ~61.5k             -> must equal the MAX_LEN=65536 figure.
#
# Copies the checkout to a temp dir, stubs vllm, and dry-runs the launcher with PRINT_ARGV=1 --
# no GPU, no model, no network.
#
#   bash bench/test_resident_print.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
# Stands for the FA2 plugin's install probe, so FA2_FP8KV=1 reaches the fp8 arm.
printf '#!/bin/sh\nexit 0\n' > "$T/venv/bin/python"; chmod +x "$T/venv/bin/python"
# The dflash2 branch refuses to start without a drafter directory; a marker file is enough for a dry run.
mkdir -p "$T/models/Qwen3.8-27B-DFlash2-W4A16" && : > "$T/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"

INT8='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head'

# note <env...>: the launcher's note line, or nothing. Every knob the note reads is cleared first,
# so the caller's environment (a CTX or DFLASH_MAX_LEN in the shell) cannot change an arm.
# FA2_FP8KV=0: the H3 CTX=long row is the int8 arm, and on a compute capability 8.6 host an unset
# knob resolves to fp8 + FA2 (the #153 row sets FA2_FP8KV=1 itself).
note() {
  (cd "$T" && env -u PRINT_ARGV -u DFLASH_TOKENS -u KV_MEM -u MAX_SEQS -u MAX_LEN -u EXTRA_ARGS -u PREFIX_CACHE \
     -u CTX -u DFLASH_MAX_LEN -u FA2_FP8KV \
     SPEC=dflash2 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" FA2_FP8KV=0 "$@" \
     bash single-user/start_qwen.sh 2>/dev/null </dev/null) | grep -o '\[start_qwen\] note: the pool holds.*' || true
}
# fields <env...>: "lo:hi:len:pool:block" from the note (a single 1k figure prints as "n:n").
fields() {
  note "$@" | sed -En 's/.*about ([0-9]+)( to ([0-9]+))? resident requests on ~1k-token prompts and ([0-9]+) at MAX_LEN=[0-9]+ -- ([0-9]+) pages of ([0-9]+) tokens.*/\1:\3:\4:\5:\6/p' \
    | awk -F: '{ if ($2 == "") $2 = $1; print $1 ":" $2 ":" $3 ":" $4 ":" $5 }'
}
# inside <lo:hi:...> <observed>: 1 if lo <= observed <= hi.
inside() { awk -F: -v o="$2" '{ print ($1 <= o && o <= $2) ? 1 : 0 }' <<< "$1"; }

echo "== measured arms: pool and block exact, observed residency inside the printed figures"
# arm                          KV_MEM      k  dtype  num_gpu_blocks  block  seen ~1k  seen ~32.8k  seen ~61.5k
# batch 5 Q2-a                 5583457484  7  int8   382             864    7         2            1
# batch 5 Q2-b                 5583457484  3  int8   404             816    13        3            2
# batch 5 Q2-c                 4509715660  7  int8   308             864    5         2            1
# batch 5 Q2-d (shipped KV)    5583457484  7  bf16   380             448    6         1            1
# batch 4 Q1 seat gauge        5315022028  7  int8   364             864    6         -            -
# batch 4 Q2 k=3               5315022028  3  int8   385             816    -         -            -
# batch 4 Q4 (24 seats)        5315022028  1  int8   393             800    21        -            -
# 2026-10-16 H1 (PREFIX_CACHE=1) 5583457484 7 bf16 380           448    6         1            -
# 2026-10-16 H2 (PREFIX_CACHE=0) 5583457484 7 bf16 380           448    6         -            -
# 2026-10-16 H3 (CTX=long, int8, MAX_LEN 131072) 5583457484 7 382 864  7         2            -
# #153 FA2 arm (CTX=long, fp8, MAX_LEN 131072)    5583457484 7 387 880  -         -            -
# The #153 boot logged its pool as 140,902 tokens, not num_gpu_blocks. vLLM 0.31.0 logs
# num_gpu_blocks / blocks per request x MAX_LEN, and 387 / 360 x 131072 is the only fit at 2,048 B
# a token: 360 = 54 state + 2 x 149 context + 8 drafter-window blocks at block 880. The same sum at
# int8's block 864 is 367, the 382 / 367 the H3 engine reported, and gives its logged 136,429.
# Each charged 6 x (2 + k) state pages per request (vllm:kv_cache_usage_perc x (num_gpu_blocks - 1)
# / Running: 54.00 at k=7, 30.00 at k=3, 18.0 at k=1), including H2: vLLM 0.31.0 enables prefix
# caching, and so align mode, by default for this model, with or without PREFIX_CACHE=1.
f=$(fields KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")
check "6:7:1:382:864" "$f" "Q2-a int8 k=7: range 6-7, 1 at 64k, pool 382, block 864"
check 1 "$(inside "$f" 7)" "  ... the observed 7 at ~1k is inside"
f=$(fields KV_MEM=5583457484 DFLASH_TOKENS=3 MAX_SEQS=16 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")
check "11:13:2:404:816" "$f" "Q2-b int8 k=3: range 11-13, 2 at 64k, pool 404, block 816"
check 1 "$(inside "$f" 13)" "  ... the observed 13 at ~1k is inside"
f=$(fields KV_MEM=4509715660 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")
check "5:5:1:308:864" "$f" "Q2-c int8 k=7, KV_MEM -1 GiB: 5, 1 at 64k, pool 308"
f=$(fields KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 PREFIX_CACHE=1)
check "6:7:1:380:448" "$f" "Q2-d bf16 k=7 (shipped KV): range 6-7, 1 at 64k, pool 380, block 448"
check 1 "$(inside "$f" 6)" "  ... the observed 6 at ~1k is inside"
f=$(fields KV_MEM=5315022028 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")
check "6:6:1:364:864" "$f" "batch 4 int8 k=7: 6, pool 364"
f=$(fields KV_MEM=5315022028 DFLASH_TOKENS=1 MAX_SEQS=24 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")
check "17:21:2:393:800" "$f" "batch 4 Q4 int8 k=1: range 17-21, pool 393, block 800"
check 1 "$(inside "$f" 21)" "  ... the observed 21 at ~1k is inside"
f=$(fields KV_MEM=5315022028 DFLASH_TOKENS=3 MAX_SEQS=16 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")
check "385:816" "$(cut -d: -f4,5 <<< "$f")" "batch 4 int8 k=3: pool 385, block 816"
f=$(fields KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)
check "6:7:1:380:448" "$f" "H2 bf16 k=7 with PREFIX_CACHE unset: the same 54 state pages, range 6-7"
check 1 "$(inside "$f" 6)" "  ... the observed 6 at ~1k is inside"
f=$(fields DFLASH_TOKENS=7 MAX_SEQS=16 CTX=long PREFIX_CACHE=1)
check "6:7:1:382:864" "$f" "H3 CTX=long (int8, MAX_LEN 131072): range 6-7, pool 382, block 864"
check 1 "$(inside "$f" 7)" "  ... the observed 7 at ~1k is inside"
f=$(fields DFLASH_TOKENS=7 MAX_SEQS=16 CTX=long PREFIX_CACHE=1 FA2_FP8KV=1)
check "6:7:1:387:880" "$f" "#153 CTX=long FA2 arm (fp8, MAX_LEN 131072): range 6-7, pool 387, block 880"

echo "== prefix caching switched off explicitly is unmeasured"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 EXTRA_ARGS="--no-enable-prefix-caching")" \
  "--no-enable-prefix-caching (1 + k state pages, never booted): no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 EXTRA_ARGS="--mamba-cache-mode none")" \
  "--mamba-cache-mode none: no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="--mamba-cache-mode=all")" \
  "--mamba-cache-mode=all (not align, never booted): no note"
check "6:7:1:380:448" "$(fields KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 EXTRA_ARGS="--mamba-cache-mode=align")" \
  "--mamba-cache-mode=align still prints"

echo "== MAX_LEN moves the long-context figure, not the ~1k range"
check "6:7:2:382:864" "$(fields KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=32768 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")" \
  "int8 k=7 at MAX_LEN=32768: 2 (Q2-a held 2 at ~32.8k)"
check "6:7:1:380:448" "$(fields KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=32768 PREFIX_CACHE=1)" \
  "bf16 k=7 at MAX_LEN=32768: 1 (H1 held 1 at ~32k)"

echo "== silent where it should be"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=7 MAX_LEN=65536 PREFIX_CACHE=1 EXTRA_ARGS="$INT8")" "MAX_SEQS equal to the upper bound: no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=4 MAX_LEN=65536 PREFIX_CACHE=1)" "MAX_SEQS=4: no note"
check "" "$(note KV_MEM=4900000000 DFLASH_TOKENS=3 MAX_SEQS=16 CTX=huge)" "CTX=huge (KVarN): no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 EXTRA_ARGS="--kv-cache-dtype fp8_e5m2")" "fp8_e5m2 KV (unmeasured): no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 EXTRA_ARGS="--block-size 1024")" "explicit --block-size: no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 EXTRA_ARGS="--mamba-ssm-cache-dtype float32")" "float32 SSM state (unmeasured): no note"
check "" "$(note KV_MEM= DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" "no KV_MEM pin (the pool is profiled, not known): no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=64k)" "MAX_LEN=64k (not a plain token count): no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=auto)" "MAX_LEN=auto: no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 EXTRA_ARGS="--tensor-parallel-size 2")" "TP=2 with an explicit KV_MEM (per-rank heads): no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 EXTRA_ARGS="--pipeline-parallel-size 2")" "PP=2 (per-rank layers): no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 EXTRA_ARGS="-pp=2")" "-pp=2: no note"
check "" "$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=065536)" "zero-padded MAX_LEN (bash would read octal): no note"
check "" "$(note DFLASH_TOKENS=7 CTX=long PREFIX_CACHE=1)" "CTX=long at its default 4 seats: no note"
[ -n "$(note DFLASH_TOKENS=7 MAX_SEQS=16 CTX=long PREFIX_CACHE=1)" ] && check 1 1 "CTX=long at 16 seats prints" || check 1 0 "CTX=long at 16 seats prints"

echo "== the launcher still dry-runs"
(cd "$T" && env -u PRINT_ARGV SPEC=dflash2 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" \
   bash single-user/start_qwen.sh >/dev/null 2>&1 </dev/null); check 0 "$?" "dry run still exits 0"
(cd "$T" && env -u PRINT_ARGV SPEC=dflash2 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" MAX_SEQS=16 MAX_LEN=64k \
   bash single-user/start_qwen.sh >/dev/null 2>&1 </dev/null); check 0 "$?" "dry run with MAX_LEN=64k still exits 0"

echo; [ "$FAILS" = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
