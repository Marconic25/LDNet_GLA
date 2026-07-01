#!/bin/bash
set -e
cd /work/u10677113/LDNet_GLA/clean
mkdir -p results
apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif bash -c 'pip install matplotlib -q && DAMULT=3 python3 -u compare_grid.py' 2>&1 | tee /work/u10677113/LDNet_GLA/compare_grid_run.log
echo 'COMPARE_GRID_DONE'
