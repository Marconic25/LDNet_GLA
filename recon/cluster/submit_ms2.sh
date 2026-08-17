#!/bin/bash
# M-SPLIT follow-up: 4 arms (ms_t0, ms_tik, ms_wall, ms_tik_d10) x 5 seeds.
# One lane per seed, arms chained with afterany inside the lane, 5 lanes in
# parallel -> peak 5 jobs x 8 cores, ~4 runs x ~1-1.5 h = ~5-6 h per lane.
# Usage: bash submit_ms2.sh ["0 100 200 300 400"]
export PATH=$PATH:/opt/pbs/bin
cd /work/u10677113/NACA2312/recon/cluster
STUDY=/work/u10677113/NACA2312/recon/models/meansplit_study
mkdir -p "$STUDY"
SEEDS=${1:-"0 100 200 300 400"}
ARMS="ms_t0 ms_tik ms_wall ms_tik_d10"
for S in $SEEDS; do
  DEP=""
  LINE="seed $S:"
  for A in $ARMS; do
    LOG=$STUDY/pbs_${A}_s${S}.log
    if [ -z "$DEP" ]; then
      J=$(qsub -v ARM=$A,SEED=$S -o "$LOG" ms2.pbs)
    else
      J=$(qsub -W depend=afterany:"$DEP" -v ARM=$A,SEED=$S -o "$LOG" ms2.pbs)
    fi
    LINE="$LINE $A=$J"
    DEP=$J
  done
  echo "$LINE"
done
