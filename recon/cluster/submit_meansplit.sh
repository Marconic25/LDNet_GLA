#!/bin/bash
# M-SPLIT study: 2 arms (base | ms) x 5 seeds at d_s=1, final_div protocol.
# One lane per seed (base -> ms chained with afterany), lanes in parallel:
# peak 5 jobs x 8 cores — sized for the currently idle cpu queue (qstat empty
# 2026-07-17; the 88-job extraction chain is long done). ~2 h/run -> ~4-5 h total.
# Usage: bash submit_meansplit.sh ["0 100 200 300 400"]
cd /work/u10677113/NACA2312/recon/cluster
mkdir -p /work/u10677113/NACA2312/recon/models/meansplit_study
SEEDS=${1:-"0 100 200 300 400"}
for S in $SEEDS; do
  LOGB=/work/u10677113/NACA2312/recon/models/meansplit_study/pbs_base_s${S}.log
  LOGM=/work/u10677113/NACA2312/recon/models/meansplit_study/pbs_ms_s${S}.log
  JB=$(qsub -v ARM=base,SEED=$S -o "$LOGB" meansplit.pbs)
  JM=$(qsub -W depend=afterany:"$JB" -v ARM=ms,SEED=$S -o "$LOGM" meansplit.pbs)
  echo "seed $S: base=$JB -> ms=$JM"
done
