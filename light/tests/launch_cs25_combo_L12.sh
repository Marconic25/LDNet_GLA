#!/bin/bash
# CS-25 combo closed-loop analysis with the L12 rollout model (clean/models_rollout_L12),
# for comparison against the L6 production results in light/results_cs25_combo/summary.md.
# Writes to light/results_cs25_combo_L12/ (does not touch the production results).
# Requires train_l12_damped_full.sh + train_l12_rollout.sh to have completed first.
cd /work/u10677113/LDNet_GLA/light/tests || exit 1
APP="apptainer exec --writable-tmpfs --env PYTHONNOUSERSITE=1 --env DAMULT=3 \
  --env OMP_NUM_THREADS=3 --env TF_NUM_INTRAOP_THREADS=3 --env TF_NUM_INTEROP_THREADS=1 \
  --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"

echo "=== CS25 combo L12 start @ $(date) ==="
$APP bash -c \
    "pip install -q scipy h5py matplotlib; \
     cd /work/u10677113/LDNet_GLA/light/tests && \
     MD_OVERRIDE=/work/u10677113/LDNet_GLA/clean/models_rollout_L12/latent_10 \
     RESULTS_DIR_OVERRIDE=/work/u10677113/LDNet_GLA/light/results_cs25_combo_L12 \
     python3 -s -u cs25_combo_study.py"
echo "=== CS25 combo L12 done @ $(date) exit=$? ==="
