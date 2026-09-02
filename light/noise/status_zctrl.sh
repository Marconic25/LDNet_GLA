#!/bin/bash
# One-shot status of the E2-zctrl jobs (run ON the cluster; called via ssh).
#   ./status_zctrl.sh          -> full-run logs E2Z{H,C}.log
#   ./status_zctrl.sh .smoke   -> smoke logs E2Z?.smoke.log
cd "$(dirname "$0")" || exit 1
for p in H:home C:cells; do
  L=${p%%:*}; PART=${p##*:}
  LOG="E2Z$L$1.log"
  if grep -q "^# DONE" "$LOG" 2>/dev/null; then st="DONE"
  elif grep -qE "Traceback|MemoryError|Killed|AssertionError" "$LOG" 2>/dev/null; then st="ERROR"
  elif pgrep -f "python3 -s -u e2_zctrl.py $PART" >/dev/null 2>&1; then st="RUNNING"
  else st="DEAD"
  fi
  last=$(grep -E "^  |^===|^#" "$LOG" 2>/dev/null | tail -1)
  echo "$L $st | $last"
done
echo "procs:$(pgrep -f 'python3 -s -u e2_zctrl.py' | wc -l)"
echo "results: $(ls results/E2_zctrl_* 2>/dev/null | tr '\n' ' ')"
