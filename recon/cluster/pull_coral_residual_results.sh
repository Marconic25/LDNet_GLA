#!/bin/bash
# Run LOCALLY (WSL) to scp back the residual-curriculum study's FOM/ROM dumps
# and run_info/metrics for the region/static-dynamic decomposition and
# sign-flip readout (decomp_residual.py, stall_signflip_residual.py).
set -e
HOST=u10677113@10.78.18.100
REMOTE=/work/u10677113/NACA2312/recon
LOCAL=/home/marco/LDNet_OF/recon

for P in 1 2; do
  for S in 0 100 200; do
    ARM=coral_o10_res_p${P}_s${S}
    for SIM in cc060 a025; do
      RD=ms_${ARM}_rom_${SIM}
      mkdir -p "$LOCAL/results/$RD"
      echo "pulling $RD ..."
      scp -q "$HOST:$REMOTE/results/$RD/"*.npy "$LOCAL/results/$RD/" 2>/dev/null \
        && echo "  OK" || echo "  MISSING/FAILED (job may not be done yet)"
    done
    mkdir -p "$LOCAL/models/meansplit_study/$ARM"
    scp -q "$HOST:$REMOTE/models/meansplit_study/$ARM/latent_1/run_info.json" \
      "$LOCAL/models/meansplit_study/$ARM/run_info.json" 2>/dev/null || true
    scp -q "$HOST:$REMOTE/models/meansplit_study/$ARM/latent_1/metrics.json" \
      "$LOCAL/models/meansplit_study/$ARM/metrics.json" 2>/dev/null || true
    scp -q "$HOST:$REMOTE/models/meansplit_study/$ARM/latent_1/config.json" \
      "$LOCAL/models/meansplit_study/$ARM/config.json" 2>/dev/null || true
  done
done
echo "done"
