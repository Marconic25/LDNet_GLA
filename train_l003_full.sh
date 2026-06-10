#!/bin/bash
cd /work/u10677113/LDNet_GLA
echo "=== FULL-CONV l003 start @ $(date) ==="
apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif bash -c "pip install scipy matplotlib h5py -q 2>/dev/null; cd /work/u10677113/LDNet_GLA && OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 LAMBDA_DAMP=0.003 NADAM=400 NBFGS=4000 OUTDIR=/work/u10677113/LDNet_GLA/clean/models_damped_l003_full python3 -u src/sensitivity_latent_damped_ckpt.py"
echo "=== FULL-CONV l003 done @ $(date) ==="
