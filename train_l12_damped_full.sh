#!/bin/bash
# Stage 1 of the L12 closed-loop pipeline: damped teacher-forced pretrain at
# decoder depth L=12 (DYN_LAYERS=4, REC_LAYERS=8; baseline production is L=6,
# DYN_LAYERS=2, REC_LAYERS=4 — see train_l003_full.sh). Same recipe as the L6
# warmstart (LAMBDA_DAMP=0.003, NADAM=400, NBFGS=4000) so the two are
# comparable. Output is the WARMSTART for train_l12_rollout.sh.
cd /work/u10677113/LDNet_GLA
echo "=== L12 DAMPED-full start @ $(date) ==="
apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif bash -c "pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA && OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 DYN_LAYERS=4 REC_LAYERS=8 LAMBDA_DAMP=0.003 NADAM=400 NBFGS=4000 OUTDIR=/work/u10677113/LDNet_GLA/clean/models_damped_l003_full_L12 python3 -u src/sensitivity_latent_damped_ckpt.py"
echo "=== L12 DAMPED-full done @ $(date) exit=$? ==="
