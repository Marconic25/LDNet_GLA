#!/bin/bash
# Arm the N=30 learning-curve pipeline: build job fires when the pending rung-30
# extractions terminate (afterany, computed dynamically from what is still queued),
# then 5 training links chained 1-wide (ds = 1, 10, 3, 5, 20 — anchors first).
cd /work/u10677113/NACA2312/recon/cluster
mkdir -p /work/u10677113/NACA2312/recon/models/lc_N30

PEND=""
for J in 25702 25704 25706 25708 25710 25712 25713; do
    if qstat ${J}.login01 >/dev/null 2>&1; then PEND="$PEND:${J}.login01"; fi
done
if [ -n "$PEND" ]; then DEP="-W depend=afterany$PEND"; else DEP=""; fi

B=$(qsub $DEP lc_build_n30.pbs)
echo "build -> $B (waiting on:${PEND:- none})"

PREV=$B
for K in 1 10 3 5 20; do
    J=$(qsub -W depend=afterany:$PREV \
            -o /work/u10677113/NACA2312/recon/models/lc_N30/pbs_ds$K.log \
            -v DS=$K lc_train_n30.pbs)
    echo "train ds=$K -> $J (after $PREV)"
    PREV=$J
done
