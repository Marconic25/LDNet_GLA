#!/bin/bash
# Launch noise_mismatch_combo.py (cluster only).
#   ./launch_noise_mismatch_combo.sh smoke
#   ./launch_noise_mismatch_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_mismatch_combo.py --smoke" > D2.smoke.log 2>&1 &
    echo "launched noise_mismatch_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_mismatch_combo.py" > D2.log 2>&1 &
    echo "launched noise_mismatch_combo full pid $!"
fi
