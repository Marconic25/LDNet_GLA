#!/bin/bash
# One-shot status of the five axis jobs (run ON the cluster; called via ssh).
cd "$(dirname "$0")" || exit 1
for p in A:noise_white B:noise_bandlim C:noise_struct D:noise_cells E:noise_mitigation; do
  L=${p%%:*}; S=${p##*:}
  if grep -q "^# DONE" "$L.log" 2>/dev/null; then st="DONE"
  elif grep -qE "Traceback|MemoryError|Killed" "$L.log" 2>/dev/null; then st="ERROR"
  elif pgrep -f "python3 -s -u $S.py" >/dev/null 2>&1; then st="RUNNING"
  else st="DEAD"
  fi
  last=$(grep -E "^  |^===|^#" "$L.log" 2>/dev/null | tail -1)
  echo "$L $st | $last"
done
echo "procs:$(pgrep -f 'python3 -s -u noise_' | wc -l)"
echo "results: $(ls results/ 2>/dev/null | tr '\n' ' ')"
