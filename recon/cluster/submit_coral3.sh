#!/bin/bash
# D-RES lever (post arm-A): mean-split + shift-modulated SIREN decoder (CORAL),
# omega0 sweep x seeds. omega0 guards against a single-frequency false null (coords
# are min-max normalized ~[0,1], so the useful SIREN frequency is not obvious a
# priori): 10 (low), 30 (Sitzmann default), 60 (high, for the sharp flap-gap).
# One lane per seed, omega0 chained afterany, lanes parallel. Control = the `ms`
# arm (5 seeds) and the FF arm A. Usage: bash submit_coral3.sh ["0 100 200"]
export PATH=$PATH:/opt/pbs/bin
cd /work/u10677113/NACA2312/recon/cluster
STUDY=/work/u10677113/NACA2312/recon/models/meansplit_study
mkdir -p "$STUDY"
SEEDS=${1:-"0 100 200"}
OMEGASET="10 30 60"
for S in $SEEDS; do
  DEP=""
  LINE="seed $S:"
  for W in $OMEGASET; do
    LOG=$STUDY/pbs_coral_o${W}_s${S}.log
    if [ -z "$DEP" ]; then
      J=$(qsub -v ARM=ms_coral,SEED=$S,OMEGA=$W -o "$LOG" coral3.pbs)
    else
      J=$(qsub -W depend=afterany:"$DEP" -v ARM=ms_coral,SEED=$S,OMEGA=$W -o "$LOG" coral3.pbs)
    fi
    LINE="$LINE o${W}=$J"
    DEP=$J
  done
  echo "$LINE"
done
