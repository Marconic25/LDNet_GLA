#!/bin/bash
# Smoke regression: open + optimal (R=3e-4, R=1e-4) + combo oracle clean
# at W30/Tg0.4 DAMULT=3, dp45 tree. Run from cluster light/ dir.
cd /work/u10677113/LDNet_GLA/light
apptainer exec --writable-tmpfs \
  --env PYTHONNOUSERSITE=1 --env DAMULT=3 \
  --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif \
  bash -c "pip install -q scipy h5py matplotlib; cd /work/u10677113/LDNet_GLA/light && python3 -s -u smoke_dp45_baseline.py"
