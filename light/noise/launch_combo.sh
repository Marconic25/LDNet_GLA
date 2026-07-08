#!/bin/bash
# Launch the E2-combo jobs (run ON the cluster; called via ssh by path).
#   ./launch_combo.sh smoke   -> e2_combo.py --smoke        (E2C.smoke.log)
#   ./launch_combo.sh full    -> parts flat + dlr in parallel (E2Cflat/E2Cdlr.log)
cd "$(dirname "$0")" || exit 1
if [ "$1" = "smoke" ]; then
  nohup ./run_axis.sh "e2_combo.py --smoke" > E2C.smoke.log 2>&1 &
  echo "launched combo smoke pid $!"
else
  nohup ./run_axis.sh "e2_combo.py flat" > E2Cflat.log 2>&1 &
  echo "launched combo flat pid $!"
  nohup ./run_axis.sh "e2_combo.py dlr" > E2Cdlr.log 2>&1 &
  echo "launched combo dlr pid $!"
fi
