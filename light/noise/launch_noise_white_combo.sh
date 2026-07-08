#!/bin/bash
# Launch noise_white_combo.py (cluster only).
#   ./launch_noise_white_combo.sh smoke
#   ./launch_noise_white_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_white_combo.py --smoke" > Wco.smoke.log 2>&1 &
    echo "launched noise_white_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_white_combo.py" > Wco.log 2>&1 &
    echo "launched noise_white_combo full pid $!"
fi
