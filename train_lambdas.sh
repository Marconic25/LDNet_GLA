#!/bin/bash
cd /work/u10677113/LDNet_GLA
APP="apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
for LAM in 0.005 0.003; do
  TAG=$(echo $LAM | sed 's/0\.//')
  echo "=== TRAIN lambda=$LAM -> models_damped_l$TAG @ $(date) ==="
  $APP bash -c "pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA && OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 LAMBDA_DAMP=$LAM NADAM=400 NBFGS=150 OUTDIR=/work/u10677113/LDNet_GLA/clean/models_damped_l$TAG python3 -u src/sensitivity_latent_damped.py"
  echo "=== DONE lambda=$LAM @ $(date) ==="
done
echo "ALL TRAINING DONE @ $(date)"
