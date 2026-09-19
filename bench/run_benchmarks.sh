#!/bin/bash
# Reproduce the README numbers against the server you have running on this
# box. Prints one ROW line per measurement; ~10 min for a mode, longer with
# --prefill / --long.
#
#   bash bench/run_benchmarks.sh batch            # 64-concurrent + real-prompt cohorts (batch mode)
#   bash bench/run_benchmarks.sh single           # real-prompt cohorts, T=default and greedy, MTP acceptance
#   bash bench/run_benchmarks.sh batch --prefill  # + the prefill matrix (1k..100k)
#   bash bench/run_benchmarks.sh batch --long     # + 1x100k and 4x60k long-context requests (needs the context)
#
# Run it twice after a restart and keep the second numbers: the first run
# after start includes JIT warmup and reads 30-50% low. Then run
# bench/quality_battery.py — a fast server that emits garbage is worth nothing.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
MODE=${1:-batch}; shift || true
DO_PREFILL=0; DO_LONG=0
for a in "$@"; do case $a in --prefill) DO_PREFILL=1;; --long) DO_LONG=1;; esac; done
export PATH="$REPO/venv/bin:$PATH"
export OPENAI_API_KEY=${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-18020}
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
B="venv/bin/vllm bench serve --host $HOST --port $PORT --model $MODEL --served-model-name qwen3.8-27b"
OUT=${OUT:-$HERE/results}; mkdir -p "$OUT"

curl -sf -o /dev/null http://$HOST:$PORT/health || { echo "no server on $HOST:$PORT"; exit 1; }
metrics() { curl -s http://$HOST:$PORT/metrics -H "Authorization: Bearer $OPENAI_API_KEY"; }
# tok/step needs the spec-decode counters, which vLLM registers only when
# speculation is configured. Batch mode has no speculative decoding
# (speculative_config=None), so the counters are absent, grep exits 1, and under
# `set -e` + `pipefail` the assignment `S0=$(spec)` aborted the whole run at the
# first cohort - silently, with no FAIL/INVALID line and no cohort log. tokstep()
# already renders missing values as '-', so treat absence as an empty sample.
# $NF, not $2: the series carry labels, so field 2 is `{engine="0",...}` and only
# the last field is the sample.
spec() { metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | awk '{print $NF}' | tr "\n" " "; return 0; }
num() { awk "/$1/ {print \$$2}" "$3"; }
row() { # label logfile conc
  local L=$1 F=$2 C=$3
  local E2E=$(num "Output token throughput" 5 $F) TPOT=$(num "Mean TPOT" 4 $F) MTPOT=$(num "Median TPOT" 4 $F) TTFT=$(num "Mean TTFT" 4 $F) DUR=$(num "Benchmark duration" 4 $F)
  local DEC=$(python3 -c "print(f'{$C*1000/$MTPOT:.0f}')" 2>/dev/null)
  echo "ROW $L | e2e=$E2E tok/s | decode(C/medTPOT)=$DEC | medTPOT=$MTPOT ms | meanTTFT=$TTFT ms | dur=${DUR}s"
}
tokstep() { python3 -c "
a='$1'.split(); b='$2'.split()
try:
    d=float(b[0])-float(a[0]); acc=float(b[1])-float(a[1]); print(f'{1+acc/d:.2f}' if d>0 else '-')
except Exception: print('-')"; }

echo "# $(date) mode=$MODE server=$HOST:$PORT"
# Every random-dataset call gets its own --seed. The bench default (0) hands
# every call the SAME prompts, and with --enable-prefix-caching that turns
# later calls into silent partial prefix-cache hits whose size depends on the
# server's pool geometry: measured on the single-user stack, the old protocol
# read prefill 15-20% LOW on one config and 3x HIGH on another (a 16k prefill
# "measured" at 4.0 s that cold costs 11.2 s). Distinct seeds per call make
# every request a cold miss; SEED_BASE pins them for reproducibility.
SEED=${SEED_BASE:-1000}
$B --dataset-name random --seed $((SEED+=1)) --random-input-len 256 --random-output-len 256 --num-prompts 16 --max-concurrency 8 > /dev/null 2>&1   # warmup

if [ "$MODE" = "batch" ]; then
  for W in "128 512" "256 256"; do set -- $W
    $B --dataset-name random --seed $((SEED+=1)) --ignore-eos --random-input-len $1 --random-output-len $2 --num-prompts 256 --max-concurrency 64 > $OUT/batch_${1}_${2}.log 2>&1
    row "64conc $1in/$2out" $OUT/batch_${1}_${2}.log 64
  done
fi
# real-prompt cohorts (both modes)
for T in default 0; do
  [ "$MODE" = "batch" ] && [ "$T" = "0" ] && break
  TARG=""; [ "$T" = "0" ] && TARG="--temperature 0"
  for C in 1 2 4 8; do
    S0=$(spec)
    $B --dataset-name custom --dataset-path $HERE/prompts_real.jsonl --custom-output-len 1024 --num-prompts 8 --max-concurrency $C $TARG > $OUT/cohort_T${T}_c$C.log 2>&1
    S1=$(spec)
    L="cohort C$C real prompts T=$T"; F=$OUT/cohort_T${T}_c$C.log
    echo "ROW $L | e2e=$(num "Output token throughput" 5 $F) tok/s | decode(C/meanTPOT)=$(python3 -c "print(f'{$C*1000/$(num "Mean TPOT" 4 $F):.1f}')") | tok/step=$(tokstep "$S0" "$S1") | meanTTFT=$(num "Mean TTFT" 4 $F) ms"
  done
done
if [ $DO_PREFILL = 1 ]; then
  pf() { LEN=$1; C=$2; N=$3
    $B --dataset-name random --seed $((SEED+=1)) --random-output-len 1 --random-input-len $LEN --num-prompts $N --max-concurrency $C > $OUT/prefill_${LEN}_c$C.log 2>&1
    IN=$(num "Total input tokens" 4 $OUT/prefill_${LEN}_c$C.log); DUR=$(num "Benchmark duration" 4 $OUT/prefill_${LEN}_c$C.log)
    echo "ROW prefill len=$LEN conc=$C | $(python3 -c "print(f'{$IN/$DUR:.0f}')") tok/s | meanTTFT=$(num "Mean TTFT" 4 $OUT/prefill_${LEN}_c$C.log) ms"; }
  pf 1024 1 16; pf 1024 4 32; pf 1024 16 64
  pf 4096 1 8;  pf 4096 4 16;  pf 4096 16 32
  pf 16384 1 4; pf 16384 4 8;  pf 16384 8 16
  pf 65536 1 2; pf 65536 2 4
  pf 102400 1 2
fi
if [ $DO_LONG = 1 ]; then
  $B --dataset-name random --seed $((SEED+=1)) --ignore-eos --random-input-len 100000 --random-output-len 256 --num-prompts 1 --max-concurrency 1 > $OUT/long_100k.log 2>&1
  echo "ROW 1x100k/256 | meanTTFT=$(num "Mean TTFT" 4 $OUT/long_100k.log) ms | TPOT=$(num "Mean TPOT" 4 $OUT/long_100k.log) ms"
  $B --dataset-name random --seed $((SEED+=1)) --ignore-eos --random-input-len 60000 --random-output-len 1024 --num-prompts 4 --max-concurrency 4 > $OUT/long_4x60k.log 2>&1
  echo "ROW 4x60k/1024 conc4 | e2e=$(num "Output token throughput" 5 $OUT/long_4x60k.log) tok/s | medITL=$(num "Median ITL" 4 $OUT/long_4x60k.log) ms | dur=$(num "Benchmark duration" 4 $OUT/long_4x60k.log)s"
fi
echo "# raw logs in $OUT"
