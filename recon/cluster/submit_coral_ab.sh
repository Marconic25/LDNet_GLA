#!/bin/bash
# D-RES follow-ups on the winning coral (omega0=10, mean-split):
#   Arm A (coralds.pbs): latent-state sweep d_s in {2,3,5} x seeds -- does the fixed
#     decoder unblock latent scaling? (d_s=1 already = coral_o10). One lane per seed,
#     DS chained afterany, lanes parallel.
#   Arm B (coralfilm.pbs): FiLM (scale+shift) modulation x seeds -- more decoder
#     conditioning capacity than shift-only? 3 independent jobs.
# Dumps ms_coral_ds{DS}_s{seed}_* and ms_coral_film_s{seed}_* -> compare.py-discoverable.
# Usage: bash submit_coral_ab.sh ["0 100 200"]
export PATH=$PATH:/opt/pbs/bin
cd /work/u10677113/NACA2312/recon/cluster
STUDY=/work/u10677113/NACA2312/recon/models/meansplit_study
mkdir -p "$STUDY"
SEEDS=${1:-"0 100 200"}

echo "=== Arm A: coral d_s sweep {2,3,5} ==="
for S in $SEEDS; do
  DEP=""; LINE="seed $S:"
  for DS in 2 3 5; do
    LOG=$STUDY/pbs_coral_ds${DS}_s${S}.log
    if [ -z "$DEP" ]; then
      J=$(qsub -v SEED=$S,DS=$DS -o "$LOG" coralds.pbs)
    else
      J=$(qsub -W depend=afterany:"$DEP" -v SEED=$S,DS=$DS -o "$LOG" coralds.pbs)
    fi
    LINE="$LINE ds${DS}=$J"; DEP=$J
  done
  echo "$LINE"
done

echo "=== Arm B: coral FiLM ==="
for S in $SEEDS; do
  LOG=$STUDY/pbs_coral_film_s${S}.log
  J=$(qsub -v SEED=$S -o "$LOG" coralfilm.pbs)
  echo "seed $S: film=$J"
done
