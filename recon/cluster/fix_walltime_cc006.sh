#!/bin/bash
# One-shot recovery after Cc_006 hit the 12h walltime (steady progress, just slow:
# W_g0=27.7 + delta_rate_max=95.5 deg/s -> small CFD steps). Bump every still-held
# extraction job to 24h, wipe the partial output dir (only run.log inside, no fields),
# and resubmit Cc_006 at the chain tail keeping the 2-wide pattern.
cd /work/u10677113/NACA2312/recon/cluster

echo "--- qalter held extraction jobs to walltime=24:00:00 ---"
for J in $(qstat -u u10677113 | grep fldrun | grep " H " | cut -d. -f1); do
    if qalter -l walltime=24:00:00 "${J}.login01" 2>/dev/null; then
        echo "qalter OK   $J"
    else
        echo "qalter FAIL $J"
    fi
done

echo "--- clean partial Cc_006 and resubmit after 25781 (chain tail, 2-wide) ---"
rm -rf /work/u10677113/NACA2312/recon_fields/sim_Cc_006_train
J=$(qsub -W depend=afterany:25781.login01 -l walltime=24:00:00 \
        -o /work/u10677113/NACA2312/recon_fields/pbs_sim_Cc_006_train_retry.log \
        -v SIM=sim_Cc_006_train field_run_flap.pbs)
echo "Cc_006 resubmitted -> $J (after 25781, walltime 24h)"
