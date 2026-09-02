#!/bin/bash
# Launch one study axis inside the TF container (cluster only).
#   Usage:  nohup ./run_axis.sh e2_mpc.py > E2M.log 2>&1 &
# Thread caps: several axes run concurrently on the login node (16 threads),
# so each job gets 3 intra-op threads.
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
exec apptainer exec --writable-tmpfs \
  --env PYTHONNOUSERSITE=1 --env DAMULT=3 \
  --env OMP_NUM_THREADS=3 --env TF_NUM_INTRAOP_THREADS=3 --env TF_NUM_INTEROP_THREADS=1 \
  --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif \
  bash -c "pip install -q scipy h5py matplotlib; cd '$HERE' && python3 -s -u $1"
