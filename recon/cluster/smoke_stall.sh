#!/bin/bash
# Smoke for the stall-investigation levers (2026-08-22): --loss-weight-mode flap
# and --add-signal-rates, on top of the champion (mean-split + CORAL o10). Follows
# the exact pattern of smoke_coral_ab.sh. Self-cleans into models/stall_smoke.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/stall_smoke; mkdir -p models/stall_smoke

  echo "=== SMOKE 1: --add-signal-rates (Wdot, deltad, +2 input channels) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke/rates --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 --add-signal-rates \
    | grep -E "signal-rates|arch:|NRMSE"
  python3 - <<EOF
import json
c=json.load(open("models/stall_smoke/rates/latent_1/config.json"))
assert c["add_signal_rates"] is True, c.get("add_signal_rates")
assert len(c["problem"]["input_signals"]) == 8, c["problem"]["input_signals"]
print("OK: config add_signal_rates=True, 8 input_signals")
EOF
  python3 -u reconstruct_fields.py --model models/stall_smoke/rates/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke/rom_rates
  python3 -c "import numpy as np; r=np.load(\"models/stall_smoke/rom_rates/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: rates recon finite, total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE 2: --loss-weight-mode flap (geometric flap-proximity reweight) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke/lw --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    --loss-weight-mode flap --flap-nodes analysis/flap_nodes.npy \
    --loss-weight-tau 0.3 --loss-weight-boost 5.0 \
    | grep -E "loss-weight|arch:|NRMSE"
  python3 - <<EOF
import json
c=json.load(open("models/stall_smoke/lw/latent_1/config.json"))
assert c["loss_weight"]["mode"] == "flap", c.get("loss_weight")
assert c["loss_weight"]["n_weight_cols"] == 1
print("OK: config loss_weight mode=flap n_weight_cols=1")
EOF
  python3 -u reconstruct_fields.py --model models/stall_smoke/lw/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke/rom_lw
  python3 -c "import numpy as np; r=np.load(\"models/stall_smoke/rom_lw/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: loss-weight recon finite, total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE 3: BOTH levers combined ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke/both --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 --add-signal-rates \
    --loss-weight-mode flap --flap-nodes analysis/flap_nodes.npy \
    --loss-weight-tau 0.3 --loss-weight-boost 5.0 \
    | grep -E "signal-rates|loss-weight|arch:|NRMSE"
  python3 -u reconstruct_fields.py --model models/stall_smoke/both/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/stall_smoke/rom_both
  python3 -c "import numpy as np; r=np.load(\"models/stall_smoke/rom_both/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: both recon finite, total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE 4: baseline (both flags OFF) still byte-behaves (regression guard) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/stall_smoke/base --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    | grep -E "arch:|NRMSE"

  echo "STALL_SMOKE_ALL_DONE"
'
