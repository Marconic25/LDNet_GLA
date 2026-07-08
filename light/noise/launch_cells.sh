#!/bin/bash
# Launch the E2CC generality-check jobs (run ON the cluster; called via ssh).
#   ./launch_cells.sh smoke   -> e2_combo_cells.py --smoke   (E2CC.smoke.log)
#   ./launch_cells.sh full    -> cells W10T07 + W30T07 in parallel
cd "$(dirname "$0")" || exit 1
if [ "$1" = "smoke" ]; then
  nohup ./run_axis.sh "e2_combo_cells.py --smoke" > E2CC.smoke.log 2>&1 &
  echo "launched cells smoke pid $!"
else
  nohup ./run_axis.sh "e2_combo_cells.py W10T07" > E2CCw10.log 2>&1 &
  echo "launched cells W10T07 pid $!"
  nohup ./run_axis.sh "e2_combo_cells.py W30T07" > E2CCw30.log 2>&1 &
  echo "launched cells W30T07 pid $!"
fi
