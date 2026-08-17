#!/bin/bash
# Depth-first program (user decision 2026-07-14): two parallel 1-wide chains.
# Chain A — complete the d_s sweep at L24 (ds 3, 5, 20; driver --only, bfgs=4000).
# Chain B — seed replication of the L24 flip (ds 10 then 1, seed-base 100).
# Chain B's head is submitted UNCHAINED as a probe of the 4th concurrent job
# (2 extraction + A + B); if the scheduler errors it, resubmit behind chain A.
cd /work/u10677113/NACA2312/recon/cluster
LOG=/work/u10677113/NACA2312/recon/models/depth_study
mkdir -p /work/u10677113/NACA2312/recon/models/depth_replication

PREV=""
for K in 3 5 20; do
    DEP=""
    [ -n "$PREV" ] && DEP="-W depend=afterany:$PREV"
    J=$(qsub $DEP -o $LOG/pbs_depthrun_L24_ds$K.log -v RUNL=24,DS=$K depth_run.pbs)
    echo "sweep L24 ds=$K -> $J (after ${PREV:-none})"
    PREV=$J
done

R1=$(qsub -o /work/u10677113/NACA2312/recon/models/depth_replication/pbs_repl_ds10.log \
        -v DS=10,SEED=100 depth_repl.pbs)
echo "repl ds=10 s=100 -> $R1 (unchained probe)"
R2=$(qsub -W depend=afterany:$R1 \
        -o /work/u10677113/NACA2312/recon/models/depth_replication/pbs_repl_ds1.log \
        -v DS=1,SEED=100 depth_repl.pbs)
echo "repl ds=1 s=100 -> $R2 (after $R1)"
