#!/bin/bash
cd /work/u10677113/LDNet_GLA/light/noise 2>/dev/null || exit 1
for pair in "Wco.smoke:noise_white_combo.py --smoke" "Wco:noise_white_combo.py"; do
    L=${pair%%:*}; P=${pair##*:}
    LOG="${L}.log"
    [ -f "$LOG" ] || continue
    if grep -q "^# DONE" "$LOG"; then st="DONE"
    elif grep -qE "Traceback|Error|Killed" "$LOG"; then st="ERROR"
    elif pgrep -f "python3 -s -u $P" >/dev/null 2>&1; then st="RUNNING"
    else st="DEAD"; fi
    last=$(grep -E "^  combo|^# DONE|^#" "$LOG" 2>/dev/null | tail -1)
    echo "$L: $st | $last"
done
