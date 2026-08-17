#!/bin/bash
# Smoke for the CORAL decoder (login node, light): mean-split + shift-modulated
# SIREN trains (finite loss, sine base + (z,u)->shift modulation), config records
# decoder/siren, and reconstruct_fields reloads and rebuilds the coral net for a
# finite ROM in total-field units. Self-cleans.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/coral3_smoke
  mkdir -p models/coral3_smoke
  echo "=== SMOKE: mean-split + CORAL (shift-modulated SIREN, omega0=30) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/coral3_smoke/coral --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --subsample 128 --mean-split --decoder coral --siren-omega0 30 \
    | grep -E "coral|arch:|NRMSE|mean-split|forcing"
  python3 - <<EOF
import json
c = json.load(open("models/coral3_smoke/coral/latent_1/config.json"))
assert c.get("decoder") == "coral", c.get("decoder")
assert isinstance(c.get("siren"), dict) and "omega0" in c["siren"], c.get("siren")
assert c.get("output_nl") == "linear", c.get("output_nl")
print("OK: config decoder=coral, siren=%s, output_nl=linear" % c["siren"])
EOF
  echo "=== SMOKE: reconstruct with the coral net + mean add-back ==="
  python3 -u reconstruct_fields.py --model models/coral3_smoke/coral/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/coral3_smoke/rom
  python3 - <<EOF
import numpy as np
rom = np.load("models/coral3_smoke/rom/rom_smoke.npy")
assert np.isfinite(rom).all(), "non-finite ROM"
assert rom[...,0].max() > 30.0, "ROM not in total units (mean add-back missing)"
print("OK: coral recon finite, total-field units, vx max", round(float(rom[...,0].max()),1))
EOF
'
echo SMOKE_CORAL3_DONE
