#!/bin/bash
# Stage 2 of the L12 closed-loop pipeline: closed-loop rollout fine-tune at
# decoder depth L=12, warm-started from train_l12_damped_full.sh's output.
# Same recipe as the production rollout (LAMBDA_DAMP=0.003, ROLLOUT_LEN=800,
# NBFGS=500) — see train_rollout.sh. Output is the model light/run.py loads
# via MD_OVERRIDE for the CS-25 closed-loop comparison.
cd /work/u10677113/LDNet_GLA
echo "=== L12 ROLLOUT train start @ $(date) ==="
apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif bash -c "pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA && OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 DYN_LAYERS=4 REC_LAYERS=8 WARMSTART=/work/u10677113/LDNet_GLA/clean/models_damped_l003_full_L12/latent_10 LAMBDA_DAMP=0.003 ROLLOUT_LEN=800 NADAM=0 NBFGS=500 W_LOAD=1.0 OUTDIR=/work/u10677113/LDNet_GLA/clean/models_rollout_L12 python3 -u src/sensitivity_latent_rollout.py"
echo "=== L12 ROLLOUT train done @ $(date) exit=$? ==="
