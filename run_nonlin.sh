#!/bin/bash
cd /work/u10677113/LDNet_GLA/clean
apptainer exec --writable-tmpfs \
  --pwd /work/u10677113/LDNet_GLA/clean \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean \
  --env DAMULT="${DAMULT:-3}" \
  --env CELLS="${CELLS:-decisive}" \
  --env TEND="${TEND:-2.5}" \
  --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif \
  python3 -s -u "${SCRIPT:-nonlin_diag.py}" 2>&1 \
  | grep --line-buffered -v -E "cuda_blas|cpu_feature_guard|rebuild TensorFlow|To enable|cuda_fft|cuda_dnn|register|stream_executor|oneDNN"
