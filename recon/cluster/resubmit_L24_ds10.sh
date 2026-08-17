#!/bin/bash
# Resubmit the two L24 d_s=10 loads cells cut short by the walltime/account outage.
# 48h walltime (L24 loads BFGS=20000 is slow); LAT=10 only (latent_1 already done).
cd /work/u10677113/NACA2312/recon/cluster
LOG=/work/u10677113/NACA2312/recon/models/loads_depth
J1=$(qsub -l walltime=48:00:00 -o "$LOG/pbs_L24_s0_ds10.log"   -v RUNL=24,SEED=0,LAT=10   loads_depth.pbs)
echo "L24_s0_ds10  -> $J1"
J2=$(qsub -l walltime=48:00:00 -o "$LOG/pbs_L24_s100_ds10.log" -v RUNL=24,SEED=100,LAT=10 loads_depth.pbs)
echo "L24_s100_ds10 -> $J2"
