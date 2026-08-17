#!/bin/bash
# Smoke test the depth-enabled loads trainer: (1) default flags must reproduce the
# original 2x7/4x24 architecture (param counts), (2) deep flags must build & train.
set -e
cd /work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  echo "=== SMOKE 1: default depth (expect NNdyn 2x7, NNrec 4x24) ==="
  LATENT_SWEEP=1 ADAM=2 BFGS=3 RESULTS_OVERRIDE=models/loads_depth_smoke/default \
    python3 -u loads_depth/train_loads_depth.py 2>&1 | grep -iE "num_latent|params|Trainable|NNdyn|NNrec|NRMSE" | head -20
  echo "=== SMOKE 2: deep (DYN_LAYERS=4 REC_LAYERS=8) ==="
  LATENT_SWEEP=1 ADAM=2 BFGS=3 DYN_LAYERS=4 REC_LAYERS=8 SEED_BASE=100 \
    RESULTS_OVERRIDE=models/loads_depth_smoke/deep \
    python3 -u loads_depth/train_loads_depth.py 2>&1 | grep -iE "num_latent|params|Trainable|NNdyn|NNrec|NRMSE" | head -20
  echo SMOKE_LOADS_DONE'
