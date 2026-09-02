#!/bin/bash
# Full L12 closed-loop comparison pipeline (run on the cluster, detached):
#   1. damped teacher-forced pretrain at L=12   (train_l12_damped_full.sh, ~GPU hours)
#   2. closed-loop rollout fine-tune at L=12    (train_l12_rollout.sh, ~20-30 min)
#   3. CS-25.341 combo closed-loop analysis     (light/tests/launch_cs25_combo_L12.sh)
# Compares against the L=6 production results in light/results_cs25_combo/summary.md.
# Aborts the chain (without deleting anything) if an earlier stage fails.
set -e
cd /work/u10677113/LDNet_GLA
echo "=== L12 PIPELINE start @ $(date) ==="

echo "--- stage 1/3: damped teacher-forced pretrain (L12) ---"
bash train_l12_damped_full.sh

echo "--- stage 2/3: rollout fine-tune (L12) ---"
bash train_l12_rollout.sh

echo "--- stage 3/3: CS-25 combo closed-loop analysis (L12) ---"
bash light/tests/launch_cs25_combo_L12.sh

echo "=== L12 PIPELINE done @ $(date) ==="
