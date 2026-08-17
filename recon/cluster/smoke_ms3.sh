#!/bin/bash
# Smoke for D-RES arm A (login node, light): mean-split + Fourier features trains
# (param count grows with the widened decoder spatial input), and reconstruct_fields
# reloads fourier_B + FF-encodes at inference (ROM finite, total units). Self-cleans.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/ms3_smoke
  mkdir -p models/ms3_smoke
  echo "=== SMOKE: mean-split + Fourier features (scales 1,5, m=16 -> spatial dim 64) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/ms3_smoke/ff --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --output-nl linear --subsample 128 --mean-split \
    --fourier-scales 1,5 --fourier-m 16 | grep -E "fourier-feats ON|arch:|NRMSE|mean-split"
  test -f models/ms3_smoke/ff/fourier_B.npy && echo "OK: fourier_B.npy saved"
  grep -o "\"fourier\": {[^}]*}" models/ms3_smoke/ff/latent_1/config.json && echo "OK: config"
  echo "=== SMOKE: reconstruct with FF + mean add-back ==="
  python3 -u reconstruct_fields.py --model models/ms3_smoke/ff/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/ms3_smoke/rom
  python3 - <<EOF
import numpy as np
rom = np.load("models/ms3_smoke/rom/rom_smoke.npy")
assert np.isfinite(rom).all(), "non-finite ROM"
assert rom[...,0].max() > 30.0, "ROM not in total units (mean add-back missing)"
print("OK: FF recon finite, total-field units, vx max", round(float(rom[...,0].max()),1))
EOF
'
echo SMOKE_MS3_DONE
