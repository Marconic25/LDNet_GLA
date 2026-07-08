#!/bin/bash
# One-shot status of the E2-combo jobs (run ON the cluster; called via ssh).
cd "$(dirname "$0")" || exit 1
for pair in "E2C.smoke|e2_combo.py --smoke" "E2Cflat|e2_combo.py flat" "E2Cdlr|e2_combo.py dlr"; do
  L=${pair%%|*}; P=${pair##*|}
  [ -f "$L.log" ] || continue
  if grep -q "^# DONE" "$L.log"; then st="DONE"
  elif grep -qE "Traceback|MemoryError|Killed" "$L.log"; then st="ERROR"
  elif pgrep -f "python3 -s -u $P" >/dev/null 2>&1; then st="RUNNING"
  else st="DEAD"
  fi
  last=$(grep -E "^  |^===|^#" "$L.log" 2>/dev/null | tail -1)
  echo "$L $st | $last"
done
echo "procs:$(pgrep -f 'python3 -s -u e2_combo' | wc -l)"
