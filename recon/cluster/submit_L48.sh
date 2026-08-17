#!/bin/bash
# Chain the two L48 depth runs (own 48h PBS jobs) after the L24 links.
cd /work/u10677113/NACA2312/recon/cluster
LOG=/work/u10677113/NACA2312/recon/models/depth_study

J1=$(qsub -W depend=afterany:26213.login01 -o $LOG/pbs_depthrun_L48_ds1.log \
        -v RUNL=48,DS=1 depth_run.pbs)
echo "L48_ds1  -> $J1 (after 26213)"
J2=$(qsub -W depend=afterany:$J1 -o $LOG/pbs_depthrun_L48_ds10.log \
        -v RUNL=48,DS=10 depth_run.pbs)
echo "L48_ds10 -> $J2 (after $J1)"
