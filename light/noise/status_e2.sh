#!/bin/bash
# One-shot status of the E2 jobs (run ON the cluster; called via ssh).
#   ./status_e2.sh          -> full-run logs E2?.log
#   ./status_e2.sh .smoke   -> smoke logs E2?.smoke.log
cd "$(dirname "$0")" || exit 1
for p in M:e2_mpc; do
  L=${p%%:*}; S=${p##*:}
  LOG="E2$L$1.log"
  if grep -q "^# DONE" "$LOG" 2>/dev/null; then st="DONE"
  elif grep -qE "Traceback|MemoryError|Killed" "$LOG" 2>/dev/null; then st="ERROR"
  elif pgrep -f "python3 -s -u $S.py" >/dev/null 2>&1; then st="RUNNING"
  else st="DEAD"
  fi
  last=$(grep -E "^  |^===|^#" "$LOG" 2>/dev/null | tail -1)
  echo "$L $st | $last"
done
echo "procs:$(pgrep -f 'python3 -s -u e2_' | wc -l)"
echo "results: $(ls results/E2_* 2>/dev/null | tr '\n' ' ')"
