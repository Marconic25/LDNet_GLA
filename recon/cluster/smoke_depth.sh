#!/bin/bash
# Depth-study smoke test (login node, light): verify (1) default flags reproduce
# the original 2x7+4x24 architecture (param counts 127/2115 at d_s=1) and
# (2) the deep path + history logging works. Cleans up after itself.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/depth_smoke
  mkdir -p models/depth_smoke/default models/depth_smoke/deep
  echo "=== SMOKE 1: default flags (expect NNdyn 2x7 = 127, NNrec 4x24 = 2115) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/depth_smoke/default --latents 1 --adam 5 --bfgs 10 \
    --restarts 1 --output-nl linear --log-every 1 --subsample 128
  echo "=== SMOKE 2: deep flags --dyn-layers 4 --rec-layers 8 (expect 239 / 4515) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/depth_smoke/deep --latents 1 --adam 5 --bfgs 10 \
    --dyn-layers 4 --rec-layers 8 --restarts 1 --output-nl linear --log-every 1 --subsample 128
  echo "=== artifacts ==="
  ls models/depth_smoke/default/latent_1 models/depth_smoke/deep/latent_1
  echo "--- loss_history.csv head (default) ---"
  head -10 models/depth_smoke/default/latent_1/loss_history.csv
  echo "--- run_info.json (deep) ---"
  cat models/depth_smoke/deep/latent_1/run_info.json
'
echo SMOKE_SCRIPT_DONE
