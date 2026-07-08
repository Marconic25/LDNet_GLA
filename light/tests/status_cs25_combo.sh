#!/bin/bash
# Status of CS-25 combo jobs (run ON the cluster; called via ssh -n).
cd /work/u10677113/LDNet_GLA/light/tests 2>/dev/null || exit 1
for pair in \
    "cs25combo.smoke:cs25_combo_study.py" \
    "cs25combo_c10:cs25_combo_study.py" \
    "cs25combo_c20:cs25_combo_study.py" \
    "cs25combo_c30:cs25_combo_study.py" \
    "cs25combo_o10:cs25_combo_study.py" \
    "cs25combo_o20:cs25_combo_study.py" \
    "cs25combo_o30:cs25_combo_study.py"; do
  L=${pair%%:*}; P=${pair##*:}
  LOG="${L}.log"
  [ -f "$LOG" ] || continue
  if grep -q "^# DONE" "$LOG"; then st="DONE"
  elif grep -qE "Traceback|Error|Killed" "$LOG"; then st="ERROR"
  elif pgrep -f "python3 -s -u $P" >/dev/null 2>&1; then st="RUNNING"
  else st="DEAD"; fi
  last=$(grep -E "^  BEST|^# ROW|^# DONE|^#" "$LOG" 2>/dev/null | tail -1)
  echo "$L: $st | $last"
done
