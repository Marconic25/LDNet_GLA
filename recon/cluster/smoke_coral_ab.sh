#!/bin/bash
# Smoke for D-RES arms A+B (login node, light): coral FiLM trains + config records
# mod_type=film + recon reloads it; and coral d_s=2 trains (latent_2 dir) + recon
# from latent_2 works. Self-cleans.
set -e
RECON=/work/u10677113/NACA2312/recon
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp

apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c '
  pip install -q h5py scipy matplotlib pandas 2>/dev/null
  cd /work/u10677113/NACA2312/recon
  rm -rf models/coralab_smoke; mkdir -p models/coralab_smoke

  echo "=== SMOKE B: coral FiLM (omega0=10, scale+shift) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/coralab_smoke/film --latents 1 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 --siren-mod-type film \
    | grep -E "coral|arch:|NRMSE|film"
  python3 - <<EOF
import json
c=json.load(open("models/coralab_smoke/film/latent_1/config.json"))
assert c["decoder"]=="coral" and c["siren"]["mod_type"]=="film", c.get("siren")
print("OK: config mod_type=film")
EOF
  python3 -u reconstruct_fields.py --model models/coralab_smoke/film/latent_1 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/coralab_smoke/rom_film
  python3 -c "import numpy as np; r=np.load(\"models/coralab_smoke/rom_film/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: film recon finite total-units vx max\", round(float(r[...,0].max()),1))"

  echo "=== SMOKE A: coral d_s=2 (shift) ==="
  python3 -u train_fields.py \
    --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 --test data/FIELDS_smoke.h5 \
    --out models/coralab_smoke/ds2 --latents 2 --adam 5 --bfgs 10 --restarts 1 \
    --subsample 128 --mean-split --decoder coral --siren-omega0 10 \
    | grep -E "num_latent_states|arch:|NRMSE"
  test -d models/coralab_smoke/ds2/latent_2 && echo "OK: latent_2 dir exists"
  python3 -u reconstruct_fields.py --model models/coralab_smoke/ds2/latent_2 \
    --data data/FIELDS_smoke.h5 --index 0 --name smoke --out models/coralab_smoke/rom_ds2
  python3 -c "import numpy as np; r=np.load(\"models/coralab_smoke/rom_ds2/rom_smoke.npy\"); assert np.isfinite(r).all() and r[...,0].max()>30.0; print(\"OK: d_s=2 recon finite total-units vx max\", round(float(r[...,0].max()),1))"
'
echo SMOKE_CORAL_AB_DONE
