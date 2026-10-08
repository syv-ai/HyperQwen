#!/bin/bash
# The C1 row of bench/run_benchmarks.sh, repeated: 8 realistic prompts, 1,024 tokens each,
# one at a time. Prints tokens/step and ms/step per repeat, which is what to compare when
# two configurations look different (greedy e2e moves between sessions; tokens/step does not).
#
#   [PORT=18020] bash bench/real_rep.sh <tag> [reps] [temperature]     # temperature 0 = greedy
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
export PATH="$REPO/venv/bin:$PATH"
# One precedence chain for every caller (#113): an explicit OPENAI_API_KEY
# wins, else the server key, else api_key.txt, else the EMPTY placeholder --
# never a bare "Bearer " against a server that bound a key.
source "$REPO/resolve_api_key.sh"
resolve_client_key
M=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}; TAG=$1; N=${2:-3}; T=${3:-}
# --model is the served name (the /tokenize alignment probe posts it as the
# request's model; a checkpoint path 404s there); --tokenizer loads locally.
PORT=${PORT:-18020}
# An array, as in warmup.sh: an unquoted string re-splits and re-globs a path with a space or glob character.
B=(venv/bin/vllm bench serve --host 127.0.0.1 --port "$PORT" --model qwen3.8-27b --tokenizer "$M" --served-model-name qwen3.8-27b)
# Drafts and accepted tokens by name, summed over engines (bench/harness.py).
snap() { VLLM_API="http://127.0.0.1:$PORT" python3 "$HERE/harness.py" spec; }
for i in $(seq 1 $N); do
  S0=$(snap)
  [ -n "$T" ] && TA="--temperature $T" || TA=""
  "${B[@]}" --dataset-name custom --dataset-path "$HERE/prompts_real.jsonl" --custom-output-len 1024 --num-prompts 8 --max-concurrency 1 $TA > /tmp/rr_${TAG}_$i.log 2>&1
  S1=$(snap)
  OUT=$(awk '/Total generated tokens/ {print $4}' /tmp/rr_${TAG}_$i.log); DUR=$(awk '/Benchmark duration/ {print $4}' /tmp/rr_${TAG}_$i.log); E2E=$(awk '/Output token throughput/ {print $5}' /tmp/rr_${TAG}_$i.log); TPOT=$(awk '/Mean TPOT/ {print $4}' /tmp/rr_${TAG}_$i.log)
  python3 - "$S0" "$S1" "$TAG" "$i" "$OUT" "$DUR" "$E2E" "$TPOT" <<PY
import sys
a=[float(x) for x in sys.argv[1].split()]; b=[float(x) for x in sys.argv[2].split()]
d=[y-x for x,y in zip(a,b)]  # drafts, accepted
steps=d[0]; acc=d[1]
print(f"REP {sys.argv[3]} #{sys.argv[4]} out={sys.argv[5]} dur={sys.argv[6]}s e2e={sys.argv[7]} tok/s decode={1000/float(sys.argv[8]):.1f} tok/s | steps={steps:.0f} tok/step={1+acc/steps:.2f} ms/step={1000*float(sys.argv[6])/steps:.1f}")
PY
done
