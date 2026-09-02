#!/bin/bash
# Sync the rollout-sweep trainer + PBS launcher to the cluster and submit 3 jobs.
set -e
KEY=/home/marco/.ssh/id_ed25519
HOST=u10677113@10.78.18.100
WORKDIR=/work/u10677113/LDNet_GLA

scp -i $KEY /home/marco/LDNet_OF/src/sensitivity_rollout_sweep.py $HOST:$WORKDIR/src/
scp -i $KEY /home/marco/LDNet_OF/run_rollout_sweep.pbs $HOST:$WORKDIR/
echo "--- files synced ---"

ssh -i $KEY $HOST /opt/pbs/bin/qsub -v INPUT_SET=2 $WORKDIR/run_rollout_sweep.pbs
ssh -i $KEY $HOST /opt/pbs/bin/qsub -v INPUT_SET=4 $WORKDIR/run_rollout_sweep.pbs
ssh -i $KEY $HOST /opt/pbs/bin/qsub -v INPUT_SET=6 $WORKDIR/run_rollout_sweep.pbs
echo "--- jobs submitted ---"
ssh -i $KEY $HOST "/opt/pbs/bin/qstat -u u10677113 | tail -8"
