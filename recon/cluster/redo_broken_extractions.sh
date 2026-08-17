#!/bin/bash
# Purge and re-run every extraction poisoned by the mutated live checkpoint
# (window_idx=858 -> only 3 field snapshots). Detection = field_times.npy shorter
# than 100 entries (healthy runs have 860). Resubmits 2-wide with the PATCHED PBS
# (pinned to checkpoint_W0_baseline), 24h walltime, 48h for the known-slow trio.
# Finally re-arms the N=100 build+train pipeline (fresh IDs, dynamic deps).
set -u
cd /work/u10677113/NACA2312/recon/cluster
RF=/work/u10677113/NACA2312/recon_fields

echo "--- detecting broken extractions (field_times < 100 entries) ---"
BROKEN=$(apptainer exec --bind /work/u10677113:/work/u10677113 \
    /work/u10677113/tensorflow_gpu.sif python3 - <<'PY'
import numpy as np, pathlib
rf = pathlib.Path("/work/u10677113/NACA2312/recon_fields")
for d in sorted(rf.glob("sim_*")):
    ft = d / "field_times.npy"
    if ft.exists():
        try:
            n = len(np.load(ft))
        except Exception:
            n = -1
        if n < 100:
            print(d.name)
PY
)
echo "$BROKEN"
N_BROKEN=$(echo "$BROKEN" | grep -c "sim_" || true)
echo "--- $N_BROKEN broken dirs: deleting and resubmitting ---"

A=""; B=""; N=0
for S in $BROKEN; do
    rm -rf "$RF/$S"
    case "$S" in
        sim_A_*) SCRIPT=field_run.pbs ;;
        *)       SCRIPT=field_run_flap.pbs ;;
    esac
    WT=24:00:00
    case "$S" in sim_B_004_train|sim_Cc_006_train|sim_Cc_014_train) WT=48:00:00 ;; esac
    DEP=""
    [ -n "$A" ] && DEP="-W depend=afterany:$A"
    J=$(qsub $DEP -l walltime=$WT -o "$RF/pbs_${S}_r3.log" -v SIM="$S" "$SCRIPT")
    if [ -z "$J" ]; then echo "QSUB FAILED at $S" >&2; exit 1; fi
    N=$((N+1))
    echo "$N: $S -> $J (wt=$WT, after ${A:-none})"
    A=$B; B=$J
done
echo "resubmitted $N; lane tails: ${A:-?} ${B:-?}"

echo "--- re-arming N=100 pipeline ---"
bash submit_lc_n100.sh
