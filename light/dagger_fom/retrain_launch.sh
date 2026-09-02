#!/bin/bash
# Fine-tune LDNet from the CURRENT production checkpoint on the newly
# collected closed-loop FOM trajectories (DAgger iteration N). Invokes
# src/sensitivity_latent_rollout.py UNMODIFIED (see light/dagger_fom/NOTES.md
# for why -- this file is never edited). Never overwrites
# clean/models_rollout/latent_10 -- writes to a new OUTDIR each iteration.
#
# Run FROM the cluster:
#   ITER=1 bash /work/u10677113/LDNet_GLA/light/dagger_fom/retrain_launch.sh
#
# ROLLOUT_LEN must be < T (samples in GLA_{train,valid}.h5, from
# build_dagger_h5.py's --t-common). Our short TEND=Tg+0.5 collection cells
# give T~400-450 @ dt=0.002s; 350 leaves comfortable margin (see
# sensitivity_latent_rollout.py's rollout(): needs sig[:, 0:L+1, ...]).
#
# NADAM=0/NBFGS=500 matches the exact recipe that produced the current
# clean/models_rollout/latent_10 (train_rollout.sh) -- straight to L-BFGS,
# no Adam warmup, since we're fine-tuning from an already-good warm start.

set -e

BASE=/work/u10677113/LDNet_GLA
ITER="${ITER:?set ITER=<n>, e.g. ITER=1}"
WARMSTART="${WARMSTART:-$BASE/clean/models_rollout/latent_10}"
DATA_OVERRIDE="$BASE/light/dagger_fom/data/iter${ITER}"
OUTDIR="$BASE/light/dagger_fom/models/iter${ITER}"
LAMBDA_DAMP="${LAMBDA_DAMP:-0.003}"
ROLLOUT_LEN="${ROLLOUT_LEN:-350}"
NADAM="${NADAM:-0}"
NBFGS="${NBFGS:-500}"
W_LOAD="${W_LOAD:-1.0}"

if [ ! -f "$DATA_OVERRIDE/GLA_train.h5" ]; then
  echo "ERROR: $DATA_OVERRIDE/GLA_train.h5 not found -- run build_dagger_h5.py first" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
echo "=== DAgger retrain iter=$ITER  WARMSTART=$WARMSTART  DATA=$DATA_OVERRIDE  OUT=$OUTDIR  LAMBDA_DAMP=$LAMBDA_DAMP  ROLLOUT_LEN=$ROLLOUT_LEN  NADAM=$NADAM  NBFGS=$NBFGS  W_LOAD=$W_LOAD ==="

apptainer exec --writable-tmpfs --env PYTHONPATH=/work/u10677113/LDNet_GLA/clean \
  --bind /work/u10677113:/work/u10677113 \
  /work/u10677113/tensorflow_gpu.sif bash -c "
    cd $BASE && \
    WARMSTART='$WARMSTART' \
    DATA_OVERRIDE='$DATA_OVERRIDE' \
    OUTDIR='$OUTDIR' \
    LAMBDA_DAMP=$LAMBDA_DAMP \
    ROLLOUT_LEN=$ROLLOUT_LEN \
    NADAM=$NADAM \
    NBFGS=$NBFGS \
    W_LOAD=$W_LOAD \
    python3 -u src/sensitivity_latent_rollout.py
  " 2>&1 | tee "$OUTDIR/retrain.log"

echo "=== DONE. Candidate model: $OUTDIR/latent_10 ==="
