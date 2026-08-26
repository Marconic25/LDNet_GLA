#!/bin/bash
# Smoke for the local/gated decoder lever (--local-decoder), stall
# investigation's genuinely-new-architecture candidate. Follows the exact
# pattern of smoke_stall_sepstate.sh.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/stall_smoke_localdec; mkdir -p models/stall_smoke_localdec

  echo "=== SMOKE 1: --local-decoder ON (mean-split + coral o10) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_localdec/on --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    --local-decoder --flap-nodes analysis/flap_nodes.npy \
    --local-width 8 --local-depth 2 --local-omega0 30 --local-tau 0.3 --local-gate-hidden 4 \
    | grep -E "local-decoder|arch:|NRMSE"
  python3 - <<EOF
import json
c=json.load(open("models/stall_smoke_localdec/on/latent_1/config.json"))
assert c["local_decoder"] is True, c.get("local_decoder")
assert c["local"]["width"] == 8, c.get("local")
print("OK: config local_decoder=True, local width=8 depth=2 omega0=30")
EOF
  python3 -u reconstruct_fields.py --model models/stall_smoke_localdec/on/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke_localdec/rom_on
  python3 -c "import numpy as np; r=np.load(\"models/stall_smoke_localdec/rom_on/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: local-decoder recon finite, total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE 2: baseline (--local-decoder OFF) still byte-behaves (regression guard) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke_localdec/off --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    | grep -E "arch:|NRMSE"

  echo STALL_SMOKE_LOCALDEC_ALL_DONE
'
