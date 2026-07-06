#!/bin/bash
# Submit the remaining B/Cc field re-runs as a 2-wide dependency chain.
# This cluster errors (E state) excess concurrent 16-core jobs instead of queuing them,
# so we cap concurrency at ~2 via afterany dependencies seeded by the two running jobs.
cd /work/u10677113/NACA2312/recon/cluster

# clean up any leftover errored jobs (ignore failures)
qdel 24464 24465 24466 24467 24468 24469 24470 24471 2>/dev/null
sleep 3

A=24448   # currently running (Cc_060)
B=24463   # currently running (Cc_000)
for S in sim_Cc_001_train sim_Cc_002_train sim_Cc_003_train sim_Cc_004_train \
         sim_Cc_005_train sim_B_000_train sim_B_001_train sim_B_002_train; do
    J=$(qsub -W depend=afterany:$A \
             -o /work/u10677113/NACA2312/recon_fields/pbs_${S}.log \
             -v SIM=$S field_run_flap.pbs)
    echo "$S -> $J  (after $A)"
    A=$B
    B=$J
done
