#!/bin/bash
# Arm the N=60 learning-curve pipeline. Build fires when the pending rung-60
# extractions (+ a small buffer of early rung-100 sims, in case a slow one gets
# walltime-killed and the dynamic pool needs substitutes) have terminated.
# Then: standard arm ds={1,10,3,5,20}, then the L12 depth arm ds={1,10} (2x2 cell).
cd /work/u10677113/NACA2312/recon/cluster
mkdir -p /work/u10677113/NACA2312/recon/models/lc_N60 /work/u10677113/NACA2312/recon/models/lc_N60_L12

PEND=""
for J in 25731 25732 25733 25734 25735 25736 25737 25738 25739 25740 25741 25742 \
         25743 25744 25745 25746 25747 25748; do
    if qstat ${J}.login01 >/dev/null 2>&1; then PEND="$PEND:${J}.login01"; fi
done
if [ -n "$PEND" ]; then DEP="-W depend=afterany$PEND"; else DEP=""; fi

B=$(qsub $DEP lc_build_n60.pbs)
echo "build60 -> $B (waiting on:${PEND:- none})"

PREV=$B
for K in 1 10 3 5 20; do
    J=$(qsub -W depend=afterany:$PREV \
            -o /work/u10677113/NACA2312/recon/models/lc_N60/pbs_ds$K.log \
            -v DS=$K lc_train_n60.pbs)
    echo "train N60 ds=$K -> $J (after $PREV)"
    PREV=$J
done
for K in 1 10; do
    J=$(qsub -W depend=afterany:$PREV \
            -o /work/u10677113/NACA2312/recon/models/lc_N60_L12/pbs_ds$K.log \
            -v DS=$K,DYNL=4,RECL=8 lc_train_n60.pbs)
    echo "train N60-L12 ds=$K -> $J (after $PREV)"
    PREV=$J
done
