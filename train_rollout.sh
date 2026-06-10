#!/bin/bash
cd /work/u10677113/LDNet_GLA
echo "=== ROLLOUT train start @ $(date) ==="
apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif bash -c "pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA && OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 WARMSTART=/work/u10677113/LDNet_GLA/clean/models_damped_l003_full/latent_10 LAMBDA_DAMP=0.003 ROLLOUT_LEN=800 NADAM=0 NBFGS=500 W_LOAD=1.0 OUTDIR=/work/u10677113/LDNet_GLA/clean/models_rollout python3 -u src/sensitivity_latent_rollout.py"
echo "=== ROLLOUT train done @ $(date) ==="
