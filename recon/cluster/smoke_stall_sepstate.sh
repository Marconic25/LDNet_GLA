#!/bin/bash
# Smoke for the Goman-Khrabrov separation-lag-state lever (--dyn-sep-state),
# stall investigation. Follows the exact pattern of smoke_stall.sh.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/stall_smoke_sepstate; mkdir -p models/stall_smoke_sepstate

  echo "=== SMOKE 1: --dyn-sep-state ON ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_sepstate/on --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 --dyn-sep-state \
    | grep -E "sep-state|arch:|NRMSE"
  python3 - <<EOF
import json
c=json.load(open("models/stall_smoke_sepstate/on/latent_1/config.json"))
assert c["dyn_sep_state"] is True, c.get("dyn_sep_state")
assert c["sep_state"]["tau1_learned"] is not None, c.get("sep_state")
print("OK: config dyn_sep_state=True, tau1_learned=", c["sep_state"]["tau1_learned"])
EOF
  python3 -u reconstruct_fields.py --model models/stall_smoke_sepstate/on/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke_sepstate/rom_on
  python3 -c "import numpy as np; r=np.load(\"models/stall_smoke_sepstate/rom_on/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: sep-state recon finite, total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE 2: baseline (--dyn-sep-state OFF) still byte-behaves (regression guard) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_sepstate/off --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    | grep -E "arch:|NRMSE"

  echo STALL_SMOKE_SEPSTATE_ALL_DONE
'
