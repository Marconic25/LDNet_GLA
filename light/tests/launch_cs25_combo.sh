#!/bin/bash
# Launch CS-25 combo + optimal (dp45) rows in parallel (cluster only).
# Usage:
#   ./launch_cs25_combo.sh smoke   -> W30 combo, 1 config, quick smoke
#   ./launch_cs25_combo.sh full    -> 3 combo rows + 3 optimal rows
cd /work/u10677113/LDNet_GLA/light/tests || exit 1
APP="apptainer exec --writable-tmpfs --env PYTHONNOUSERSITE=1 --env DAMULT=3 \
  --env OMP_NUM_THREADS=3 --env TF_NUM_INTRAOP_THREADS=3 --env TF_NUM_INTEROP_THREADS=1 \
  --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"

# Wrap in apptainer properly
run_row_apptainer() {
    local MODE=$1 W0=$2 LOG=$3
    nohup $APP bash -c \
        "pip install -q scipy h5py matplotlib; \
         cd /work/u10677113/LDNet_GLA/light/tests && \
         MODE=$MODE W0=$W0 python3 -s -u cs25_combo_study.py" > "$LOG" 2>&1 &
    echo "launched MODE=$MODE W0=$W0 -> $LOG pid $!"
}

if [ "$1" = "smoke" ]; then
    nohup $APP bash -c \
        "pip install -q scipy h5py matplotlib; \
         cd /work/u10677113/LDNet_GLA/light/tests && \
         MODE=combo W0=30 TG_SMOKE=1 python3 -s -u cs25_combo_study.py" \
        > cs25combo.smoke.log 2>&1 &
    echo "smoke pid $!"
else
    for W0 in 10 20 30; do
        run_row_apptainer combo   $W0 "cs25combo_c${W0}.log"
        run_row_apptainer optimal $W0 "cs25combo_o${W0}.log"
    done
fi
