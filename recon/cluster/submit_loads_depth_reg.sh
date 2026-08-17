#!/bin/bash
# Thesis-grade REGULARIZED loads depth sweep (Tikhonov ALPHA_REG=3e-4, matching the
# finalized production loads model). Kills the leftover unregularized L24-d10 jobs,
# then submits all 6 (depth,seed) cells unchained; each does d_s in {1,10}.
# L24 gets 48h (deep+BFGS20000 slow); L6/L12 12h. Results -> models/loads_depth_reg/.
cd /work/u10677113/NACA2312/recon/cluster
mkdir -p /work/u10677113/NACA2312/recon/models/loads_depth_reg
LOG=/work/u10677113/NACA2312/recon/models/loads_depth_reg

# stop the unregularized in-flight L24-d10 resubmits (27523/27524) if still there
qdel 27523 27524 2>/dev/null

for L in 6 12 24; do
  WT=12:00:00; [ "$L" = "24" ] && WT=48:00:00
  for S in 0 100; do
    J=$(qsub -l walltime=$WT -o "$LOG/pbs_L${L}_s${S}.log" -v RUNL=$L,SEED=$S loads_depth.pbs)
    echo "reg L=$L seed=$S (wt=$WT) -> $J"
  done
done
