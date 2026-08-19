#!/bin/bash
export APPTAINER_TMPDIR=/work/u10677113/apptainer_tmp
export APPTAINER_CACHEDIR=/work/u10677113/apptainer_tmp
cd /work/u10677113/NACA2312/recon
apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \
    /work/u10677113/tensorflow_gpu.sif \
    bash -c "pip install -q h5py scipy matplotlib pandas 2>/dev/null; cd /work/u10677113/NACA2312/recon; \
      python3 -u train_fields.py --train data/FIELDS_smoke.h5 --valid data/FIELDS_smoke.h5 \
        --test data/FIELDS_smoke.h5 --out /tmp/smoke_shoot --latents 1 --adam 5 --bfgs 5 \
        --restarts 1 --output-nl linear --subsample 64 --mean-split --decoder coral \
        --siren-omega0 10 --shooting-segments 4 --shooting-lambda 1.0"
echo SMOKE_SHOOTING_EXIT_$?
