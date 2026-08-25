#!/bin/bash
# Smoke for the residual-curriculum lever (2026-08-22): --loss-weight-mode residual,
# the #1-ranked candidate from STALL_LITERATURE_NOTES.md section 5 (curriculum-PINN
# region-adaptive reweighting by local residual magnitude), on top of the champion
# (mean-split + CORAL o10). Follows the exact pattern of smoke_stall.sh.
# Self-cleans into models/stall_smoke_residual.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/stall_smoke_residual; mkdir -p models/stall_smoke_residual

  echo "=== SMOKE 1: --loss-weight-mode residual power=1.0 (moderate) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_residual/p1 --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    --loss-weight-mode residual --loss-weight-residual-power 1.0 \
    | grep -E "loss-weight|arch:|NRMSE"
  python3 - <<EOF
import json
c=json.load(open("models/stall_smoke_residual/p1/latent_1/config.json"))
assert c["loss_weight"]["mode"] == "residual", c.get("loss_weight")
assert c["loss_weight"]["residual_power"] == 1.0, c["loss_weight"]
assert c["loss_weight"]["n_weight_cols"] == 0, c["loss_weight"]
print("OK: config loss_weight mode=residual power=1.0 n_weight_cols=0")
EOF
  python3 -u reconstruct_fields.py --model models/stall_smoke_residual/p1/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke_residual/rom_p1
  python3 -c "import numpy as np; r=np.load(\"models/stall_smoke_residual/rom_p1/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: residual p1 recon finite, total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE 2: --loss-weight-mode residual power=2.0 (strong) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_residual/p2 --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    --loss-weight-mode residual --loss-weight-residual-power 2.0 \
    | grep -E "loss-weight|arch:|NRMSE"
  python3 - <<EOF
import json
c=json.load(open("models/stall_smoke_residual/p2/latent_1/config.json"))
assert c["loss_weight"]["mode"] == "residual", c.get("loss_weight")
assert c["loss_weight"]["residual_power"] == 2.0, c["loss_weight"]
print("OK: config loss_weight mode=residual power=2.0")
EOF
  python3 -u reconstruct_fields.py --model models/stall_smoke_residual/p2/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke_residual/rom_p2
  python3 -c "import numpy as np; r=np.load(\"models/stall_smoke_residual/rom_p2/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: residual p2 recon finite, total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE 3: power=0.0 must be a numerically-verified no-op vs mode=none (byte-identical-when-off, done via actual training run not just forward pass) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_residual/off --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 --seed-base 0 \
    | grep -E "arch:|NRMSE" > models/stall_smoke_residual/off.log
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_residual/p0 --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 --seed-base 0 \
    --loss-weight-mode residual --loss-weight-residual-power 0.0 \
    | grep -E "loss-weight|arch:|NRMSE" > models/stall_smoke_residual/p0.log
  cat models/stall_smoke_residual/off.log
  cat models/stall_smoke_residual/p0.log
  python3 - <<EOF
import json
off = json.load(open("models/stall_smoke_residual/off/latent_1/metrics.json"))
p0 = json.load(open("models/stall_smoke_residual/p0/latent_1/metrics.json"))
assert off["NRMSE"] == p0["NRMSE"], (off["NRMSE"], p0["NRMSE"])
print("OK: power=0.0 test NRMSE bit-identical to mode=none:", off["NRMSE"])
EOF

  echo "STALL_SMOKE_RESIDUAL_ALL_DONE"
'
