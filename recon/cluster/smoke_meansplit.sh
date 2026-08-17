#!/bin/bash
# M-SPLIT smoke test (login node, light — same pattern as smoke_depth.sh):
# (1) --mean-split trains, saves mean_fields.npy + config flag, finite total-field
#     metrics; (2) reconstruct_fields.py adds the mean back (ROM in physical total
#     units, combined NRMSE finite); (3) baseline path untouched (no mean_fields.npy).
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/ms_smoke
  mkdir -p models/ms_smoke/base models/ms_smoke/ms
  echo "=== SMOKE 1: baseline path (must run exactly as before, no mean_fields.npy) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/ms_smoke/base --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --output-nl linear --log-every 1 --subsample 128
  test ! -f models/ms_smoke/base/mean_fields.npy && echo "OK: baseline has no mean file"
  echo "=== SMOKE 2: --mean-split ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/ms_smoke/ms --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --output-nl linear --log-every 1 --subsample 128 --mean-split
  test -f models/ms_smoke/ms/latent_1/mean_fields.npy && echo "OK: mean_fields.npy saved"
  grep -o "\"mean_split\": true" models/ms_smoke/ms/latent_1/config.json \
    && echo "OK: config flag set"
  echo "=== SMOKE 3: reconstruct with mean add-back ==="
  python3 -u reconstruct_fields.py --model models/ms_smoke/ms/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/ms_smoke/rom
  python3 - <<EOF
import numpy as np
rom = np.load("models/ms_smoke/rom/rom_smoke.npy")
fom = np.load("models/ms_smoke/rom/fom_smoke.npy")
assert np.isfinite(rom).all() and np.isfinite(fom).all()
# mean add-back check: ROM must be in TOTAL units (vx ~ O(80 m/s)), not fluct O(0.2)
print("rom vx range", rom[...,0].min(), rom[...,0].max())
print("fom vx range", fom[...,0].min(), fom[...,0].max())
assert rom[...,0].max() > 30.0, "ROM looks like fluctuations - mean add-back missing!"
print("OK: ROM in total-field units")
EOF
'
echo SMOKE_MEANSPLIT_DONE
