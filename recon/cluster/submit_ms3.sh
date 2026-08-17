#!/bin/bash
# D-RES arm A: mean-split + Fourier-feature (x,y) encoding, scale sweep x seeds.
# Scales guard against a single-sigma false null (coords are min-max normalized to
# ~[0,1], so the useful sigma is not obvious a priori): 1_5 (Aero-Nef multiscale),
# 5_20, 10_40. One lane per seed, scales chained afterany, lanes parallel.
# Control = the existing `ms` arm (5 seeds). Usage: bash submit_ms3.sh ["0 100 200"]
export PATH=$PATH:/opt/pbs/bin
cd /work/u10677113/NACA2312/recon/cluster
STUDY=/work/u10677113/NACA2312/recon/models/meansplit_study
mkdir -p "$STUDY"
SEEDS=${1:-"0 100 200"}
SCALESET="1_5 5_20 10_40"
for S in $SEEDS; do
  DEP=""
  LINE="seed $S:"
  for SC in $SCALESET; do
    LOG=$STUDY/pbs_ff${SC}_s${S}.log
    if [ -z "$DEP" ]; then
      J=$(qsub -v ARM=ms_ff,SEED=$S,SC=$SC -o "$LOG" ms3.pbs)
    else
      J=$(qsub -W depend=afterany:"$DEP" -v ARM=ms_ff,SEED=$S,SC=$SC -o "$LOG" ms3.pbs)
    fi
    LINE="$LINE ff${SC}=$J"
    DEP=$J
  done
  echo "$LINE"
done
