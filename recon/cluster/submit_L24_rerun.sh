#!/bin/bash
# L24 reruns as per-run 48h PBS jobs (full bfgs=4000 budget): the chain-link design
# killed L24_ds1 twice (11h cap, then 22h cap — actual cost >22h, the 4x-per-doubling
# estimate was optimistic). 48h fits. L48 decision deferred until L24 lands.
cd /work/u10677113/NACA2312/recon/cluster
LOG=/work/u10677113/NACA2312/recon/models/depth_study

J1=$(qsub -o $LOG/pbs_depthrun_L24_ds1.log -v RUNL=24,DS=1 depth_run.pbs)
echo "L24_ds1  -> $J1"
J2=$(qsub -W depend=afterany:$J1 -o $LOG/pbs_depthrun_L24_ds10.log -v RUNL=24,DS=10 depth_run.pbs)
echo "L24_ds10 -> $J2 (after $J1)"
