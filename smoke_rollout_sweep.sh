#!/bin/bash
# Local CPU smoke of src/sensitivity_rollout_sweep.py (tiny budget, 1 config).
set -e
cd /home/marco/LDNet_OF/src
source /home/marco/LDNet_OF/tfvenv/bin/activate
export CUDA_VISIBLE_DEVICES=""
INPUT_SET=4 \
NUM_LATENT=3 \
LAMBDA_DAMP=0.003 \
ROLLOUT_LEN=50 \
NADAM=0 \
NBFGS=5 \
PATIENCE=2 \
DATA_OVERRIDE=/home/marco/LDNet_OF/data \
WARMSTART_ROOT=/home/marco/LDNet_OF/results/sensitivity \
RESULTS_OVERRIDE=/tmp/rollout_sweep_smoke \
OMP_NUM_THREADS=4 \
python3 -u sensitivity_rollout_sweep.py
echo SMOKE_OK
ls -la /tmp/rollout_sweep_smoke/in4/latent_3/
python3 - <<'EOF'
import json
m = json.load(open('/tmp/rollout_sweep_smoke/in4/latent_3/metrics.json'))
print(json.dumps(m, indent=2))
EOF
