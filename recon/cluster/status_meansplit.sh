#!/bin/bash
# M-SPLIT study status: queue lines + per-run progress/result one-liners.
# Run on the login node: bash status_meansplit.sh
export PATH=$PATH:/opt/pbs/bin
STUDY=/work/u10677113/NACA2312/recon/models/meansplit_study
echo "== queue =="
qstat -u u10677113 2>/dev/null | grep -E "msplit|Job ID" || echo "(no msplit jobs in queue)"
echo "== runs =="
for d in "$STUDY"/*_s*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  m="$d/latent_1/metrics.json"
  if [ -f "$m" ]; then
    vx=$(grep -o '"NRMSE_vx": [0-9.e-]*' "$m" | cut -d' ' -f2)
    all=$(grep -o '"NRMSE": [0-9.e-]*' "$m" | cut -d' ' -f2)
    tr=$(grep -o '"final_train_loss": [0-9.e-]*' "$d/latent_1/run_info.json" 2>/dev/null | cut -d' ' -f2)
    echo "$n  DONE   NRMSE_vx=$vx  combined=$all  train=$tr"
  elif [ -f "$d/train.log" ]; then
    ep=$(grep -o 'epoch *[0-9]*' "$d/train.log" | tail -1)
    ph=$(grep -oE 'Adam|BFGS' "$d/train.log" | tail -1)
    echo "$n  RUNNING  phase=$ph $ep"
  else
    echo "$n  (no log yet)"
  fi
done
