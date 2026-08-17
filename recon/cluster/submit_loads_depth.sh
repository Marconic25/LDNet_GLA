#!/bin/bash
# Launch the loads depth sweep TONIGHT with priority: submit all 6 (depth,seed) jobs
# UNCHAINED so they start immediately (not queued behind our chains). Grid:
#   depth L in {6,12,24} x seed in {0,100}, each doing d_s in {1,10}.
# Loads jobs are small (8-core, scalar outputs) and short; they coexist with the
# 4 extraction lanes. If the scheduler E-states any (oversubscription), resubmit the
# E'd ones behind a running one — but try fully-parallel first per the priority ask.
cd /work/u10677113/NACA2312/recon/cluster
mkdir -p /work/u10677113/NACA2312/recon/models/loads_depth
LOG=/work/u10677113/NACA2312/recon/models/loads_depth

for L in 6 12 24; do
  for S in 0 100; do
    J=$(qsub -o "$LOG/pbs_L${L}_s${S}.log" -v RUNL=$L,SEED=$S loads_depth.pbs)
    echo "L=$L seed=$S -> $J"
  done
done
