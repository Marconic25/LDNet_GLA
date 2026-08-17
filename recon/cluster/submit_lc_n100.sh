#!/bin/bash
# Arm the N=100 (final rung) pipeline: build fires when ALL remaining extraction
# jobs (rung-100 leftovers + the three tail retries) have terminated. Then the
# standard arm ds={1,10,3,5,20} and the L12 arm ds={1,10}.
cd /work/u10677113/NACA2312/recon/cluster
mkdir -p /work/u10677113/NACA2312/recon/models/lc_N100 /work/u10677113/NACA2312/recon/models/lc_N100_L12

PEND=""
for J in $(qstat -u u10677113 | grep -E "fldrun" | cut -d. -f1); do
    PEND="$PEND:${J}.login01"
done
if [ -n "$PEND" ]; then DEP="-W depend=afterany$PEND"; else DEP=""; fi

B=$(qsub $DEP lc_build_n100.pbs)
echo "build100 -> $B (waiting on:${PEND:- none})"

PREV=$B
for K in 1 10 3 5 20; do
    J=$(qsub -W depend=afterany:$PREV \
            -o /work/u10677113/NACA2312/recon/models/lc_N100/pbs_ds$K.log \
            -v DS=$K lc_train_n100.pbs)
    echo "train N100 ds=$K -> $J (after $PREV)"
    PREV=$J
done
for K in 1 10; do
    J=$(qsub -W depend=afterany:$PREV \
            -o /work/u10677113/NACA2312/recon/models/lc_N100_L12/pbs_ds$K.log \
            -v DS=$K,DYNL=4,RECL=8 lc_train_n100.pbs)
    echo "train N100-L12 ds=$K -> $J (after $PREV)"
    PREV=$J
done
