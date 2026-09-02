#!/bin/bash
# Launch E2-zctrl axis jobs (run ON the cluster; called via ssh by path).
#   ./launch_zctrl.sh smoke [H C]   -> e2_zctrl.py --smoke, log E2ZH.smoke.log
#   ./launch_zctrl.sh full  [H C]   -> full parts, logs E2Z{H,C}.log
# Letters: H=home cell (clean+white sweep+dlr), C=cells (W10/Tg0.7, W30/Tg0.7).
# With no letters, launches both. Each part is one e2_zctrl.py invocation.
cd "$(dirname "$0")" || exit 1
sfx=""; arg=""
if [ "$1" = "smoke" ]; then sfx=".smoke"; arg=" --smoke"; fi
shift
sel="$*"; [ -z "$sel" ] && sel="H C"
for p in H:home C:cells; do
  L=${p%%:*}; PART=${p##*:}
  case " $sel " in *" $L "*) ;; *) continue;; esac
  nohup ./run_axis.sh "e2_zctrl.py $PART$arg" > "E2Z$L$sfx.log" 2>&1 &
  echo "launched e2_zctrl.py $PART$arg -> E2Z$L$sfx.log pid $!"
done
