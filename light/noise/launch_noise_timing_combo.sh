#!/bin/bash
# Launch noise_timing_combo.py (cluster only).
#   ./launch_noise_timing_combo.sh smoke
#   ./launch_noise_timing_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_timing_combo.py --smoke" > B2.smoke.log 2>&1 &
    echo "launched noise_timing_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_timing_combo.py" > B2.log 2>&1 &
    echo "launched noise_timing_combo full pid $!"
fi
