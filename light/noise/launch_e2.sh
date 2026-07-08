#!/bin/bash
# Launch E2 axis jobs (run ON the cluster; called via ssh by path).
#   ./launch_e2.sh smoke [G K M S]   -> e2_*.py --smoke, logs E2?.smoke.log
#   ./launch_e2.sh full  [G K M S]   -> full runs, logs E2?.log
# With no letters, launches all four.
cd "$(dirname "$0")" || exit 1
sfx=""; arg=""
if [ "$1" = "smoke" ]; then sfx=".smoke"; arg=" --smoke"; fi
shift
sel="$*"; [ -z "$sel" ] && sel="G K M S"
for p in G:e2_gate K:e2_kalman M:e2_mpc S:e2_sensor; do
  L=${p%%:*}; S=${p##*:}
  case " $sel " in *" $L "*) ;; *) continue;; esac
  nohup ./run_axis.sh "$S.py$arg" > "E2$L$sfx.log" 2>&1 &
  echo "launched $S$arg -> E2$L$sfx.log pid $!"
done
