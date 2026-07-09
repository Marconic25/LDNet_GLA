#!/bin/bash
# Launch noise_calib_combo.py (cluster only).
#   ./launch_noise_calib_combo.sh smoke
#   ./launch_noise_calib_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_calib_combo.py --smoke" > A2.smoke.log 2>&1 &
    echo "launched noise_calib_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_calib_combo.py" > A2.log 2>&1 &
    echo "launched noise_calib_combo full pid $!"
fi
