#!/bin/bash
# Smoke test for the M-SPLIT follow-up arms (login node, light):
# (1) --mean-ref t0; (2) --alpha-reg 3e-4 (train loss includes reg, finite);
# (3) --wall-feats (+2 decoder inputs, param count grows, recon path with
# features + mean add-back works). Cleans up after itself.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/ms2_smoke
  mkdir -p models/ms2_smoke
  echo "=== SMOKE 1: t0 mean ref ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/ms2_smoke/t0 --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --output-nl linear --subsample 128 --mean-split --mean-ref t0 \
    | grep -E "mean-split ON|NRMSE|arch:"
  echo "=== SMOKE 2: Tikhonov 3e-4 ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/ms2_smoke/tik --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --output-nl linear --subsample 128 --mean-split --alpha-reg 3e-4 \
    | grep -E "NRMSE|arch:|loss"  | tail -6
  echo "=== SMOKE 3: wall features ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/ms2_smoke/wall --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --output-nl linear --subsample 128 --mean-split \
    --wall-feats --airfoil-nodes analysis_hmetric/airfoil_nodes.npy \
    | grep -E "wall-feats ON|arch:|NRMSE"
  echo "=== SMOKE 4: reconstruct wall-feats model (features + mean add-back) ==="
  python3 -u reconstruct_fields.py --model models/ms2_smoke/wall/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/ms2_smoke/rom
  python3 - <<EOF
import numpy as np
rom = np.load("models/ms2_smoke/rom/rom_smoke.npy")
assert np.isfinite(rom).all()
assert rom[...,0].max() > 30.0, "ROM not in total units"
print("OK: wall-feats recon in total-field units, vx max", rom[...,0].max())
EOF
'
echo SMOKE_MS2_DONE
