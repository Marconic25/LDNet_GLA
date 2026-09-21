#!/usr/bin/env bash
# Launch the ablation sweep as parallel per-cell jobs.
#
# One process per gust cell, each writing its own JSON shard, so a failure or
# a stall costs one cell rather than the whole sweep and partial results can
# be read while the rest run. Shards are merged by collect.py.
#
# Usage:  ./launch.sh [n_parallel]

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../tfvenv/bin/python"
OUT="$HERE/shards"
mkdir -p "$OUT" "$HERE/logs"

NPAR="${1:-6}"
ARMS="${ARMS:-ldnet,linear}"
RLADDER="${RLADDER:-1e-4,3e-4,1e-3,3e-3,1e-2}"

CELLS=()
for w in 10 20 30; do
  for t in 0.3 0.4 0.5 0.7 1.0 1.2; do
    CELLS+=("$w:$t")
  done
done

echo "[launch] ${#CELLS[@]} cells, ${NPAR} at a time, arms=${ARMS}"

running=0
for c in "${CELLS[@]}"; do
  tag="${c/:/_}"
  shard="$OUT/cell_${tag}.json"
  log="$HERE/logs/cell_${tag}.log"
  if [ -f "$shard" ] && grep -q '"ldnet"' "$shard" 2>/dev/null; then
    echo "[launch] skip $c (shard exists)"
    continue
  fi
  echo "[launch] start $c -> $log"
  (
    cd "$HERE" && \
    DAMULT=3 CELLS="$c" ARMS="$ARMS" RLADDER="$RLADDER" RESUME=0 \
    OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
    "$PY" -u sweep.py "$shard" > "$log" 2>&1
  ) &
  running=$((running+1))
  if [ "$running" -ge "$NPAR" ]; then
    wait -n 2>/dev/null || wait
    running=$((running-1))
  fi
done

wait
echo "[launch] all cells finished"
ls -la "$OUT"
