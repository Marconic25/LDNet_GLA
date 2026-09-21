#!/usr/bin/env bash
# Test the fairer linear-baseline variants on representative cells.
# Each variant gets the full chapter-3 R ladder, so it is judged at its own
# best tuning rather than at a weight chosen for a different model.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../tfvenv/bin/python"
cd "$HERE"

CELLS="${CELLS:-10:0.4,30:0.4}"
VARIANTS="${VARIANTS:-linear_coeffs.json linear_coeffs_ctrlgain.json linear_coeffs_secant.json linear_coeffs_lsqflap.json}"
RLADDER="${RLADDER:-1e-4,3e-4,1e-3,3e-3,1e-2}"

for v in $VARIANTS; do
  echo "########## $v ##########"
  OMP_NUM_THREADS=1 DAMULT=3 COEFFS="$v" CELLS="$CELLS" ARMS=linear \
    RLADDER="$RLADDER" RESUME=0 \
    "$PY" -u sweep.py "variant_${v%.json}.json" 2>&1 \
    | grep -E "\[.*\] linear|done in"
done
echo "########## done ##########"
