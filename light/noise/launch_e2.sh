#!/bin/bash
# Launch E2 axis jobs (run ON the cluster; called via ssh by path).
#   ./launch_e2.sh smoke [M]   -> e2_*.py --smoke, logs E2?.smoke.log
#   ./launch_e2.sh full  [M]   -> full runs, logs E2?.log
# With no letters, launches all.
cd "$(dirname "$0")" || exit 1
sfx=""; arg=""
if [ "$1" = "smoke" ]; then sfx=".smoke"; arg=" --smoke"; fi
shift
sel="$*"; [ -z "$sel" ] && sel="M"
for p in M:e2_mpc; do
  L=${p%%:*}; S=${p##*:}
  case " $sel " in *" $L "*) ;; *) continue;; esac
  nohup ./run_axis.sh "$S.py$arg" > "E2$L$sfx.log" 2>&1 &
  echo "launched $S$arg -> E2$L$sfx.log pid $!"
done
