#!/bin/bash
# Residual-curriculum lever full launch: n=3 seeds {0,100,200} x 2 strength
# settings (POWER=1.0 moderate, POWER=2.0 strong) on top of the champion
# (mean-split + CORAL o10). 6 jobs total; PBS/max_user_run=4 throttles
# concurrency automatically, no manual dependency chaining needed (matches
# how coral_cde.pbs's independent per-seed jobs were launched).
# Usage: bash submit_coral_residual.sh ["0 100 200"] ["1.0 2.0"]
export PATH=$PATH:/opt/pbs/bin
cd /work/u10677113/NACA2312/recon/cluster
STUDY=/work/u10677113/NACA2312/recon/models/meansplit_study
mkdir -p "$STUDY"
SEEDS=${1:-"0 100 200"}
POWERS=${2:-"1.0 2.0"}

MANIFEST=$STUDY/coral_residual_jobs.tsv
: > "$MANIFEST"
echo "power seed jobid" >> "$MANIFEST"

for P in $POWERS; do
  PTAG=${P%.*}
  for S in $SEEDS; do
    LOG=$STUDY/pbs_coral_res_p${PTAG}_s${S}.log
    J=$(qsub -N cor_res_p${PTAG}s${S} -v SEED=$S,POWER=$P -o "$LOG" coral_residual.pbs)
    echo "power=$P seed=$S -> $J"
    echo -e "$P\t$S\t$J" >> "$MANIFEST"
  done
done
echo "manifest -> $MANIFEST"
