#!/bin/bash
# Launch closed-loop FOM trajectories for the DAgger training-data set:
# the high-k corner of the grid (W30/Tg 0.30, 0.40, 0.50), using the best
# controller found in Phase A (QADOT below) and the CURRENT production
# model (iteration 0 -- collects the data needed to fine-tune it).
#
# Run FROM the cluster (ssh'd in), from anywhere:
#   bash /work/u10677113/LDNet_GLA/light/dagger_fom/cluster/collect_closedloop_data.sh
#
# Respects max_user_run=4 -- submits all cells; PBS queues the rest.
# Uses TEND=Tg+0.5 (short, ~45min/run -- see light/dagger_fom/NOTES.md for
# why this window is sufficient for training-data purposes: it is the same
# "informative window" convention used throughout light/noise/'s robustness
# metrics, t<=Tg+0.5).

set -e
export PATH=$PATH:/opt/pbs/bin

PBS=/work/u10677113/LDNet_GLA/light/dagger_fom/cluster/mpc_fom_dagger.pbs
DAMULT=3.0
QADOT="${QADOT:-0.0}"   # override once Phase A picks a value, e.g. QADOT=0.01 bash collect...
MPC_MODEL="${MPC_MODEL:-/work/u10677113/LDNet_GLA/clean/models_rollout/latent_10}"
TAG="${TAG:-iter0_qadot${QADOT}}"

# (W0, Tg, R*) -- R* from light/results_cs25_combo/summary.md
CELLS=(
  "30 0.30 0.0001"
  "30 0.40 0.0003"
  "30 0.50 0.0003"
)

for cell in "${CELLS[@]}"; do
  read -r W0 TG MPCR <<< "$cell"
  TEND=$(python3 -c "print(round($TG + 0.5, 2))")
  echo "Submitting W0=$W0 Tg=$TG R*=$MPCR TEND=$TEND QADOT=$QADOT tag=$TAG"
  qsub -l select=1:ncpus=16:mpiprocs=16 \
       -v W0=$W0,TG=$TG,MPCR=$MPCR,TEND=$TEND,QADOT=$QADOT,DAMULT=$DAMULT,MPC_MODEL=$MPC_MODEL,TAG=$TAG \
       "$PBS"
done

echo "--- queue ---"
qstat -u "$USER"
