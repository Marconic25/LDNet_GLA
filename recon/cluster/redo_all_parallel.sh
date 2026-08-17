#!/bin/bash
# Re-run every 3-snapshot-broken extraction with the FIXED pipeline (pinned driver
# numeric[:0] + controlDict purgeWrite 0). Fix verified on A_015 quick-test: 20 snaps.
# Parallelize across 4 lanes (scheduler tolerates 4+ concurrent jobs) instead of a
# 2-wide chain — 21 sims / 4 lanes ~ 5-6 deep, cuts wall-clock ~2x vs 2-wide.
# 48h walltime for the slow trio (B_004, Cc_006, Cc_014) + generous 30h for the rest
# (A_015 full ran ~18h under contention). Then re-arm the N=100 pipeline.
set -u
cd /work/u10677113/NACA2312/recon/cluster
RF=/work/u10677113/NACA2312/recon_fields

echo "--- detecting broken extractions (field_times < 100) ---"
BROKEN=$(apptainer exec --bind /work/u10677113:/work/u10677113 \
    /work/u10677113/tensorflow_gpu.sif python3 - <<'PY'
import numpy as np, pathlib
rf = pathlib.Path("/work/u10677113/NACA2312/recon_fields")
for d in sorted(rf.glob("sim_*_train")):
    ft = d / "field_times.npy"
    n = len(np.load(ft)) if ft.exists() else 0
    if n < 100:
        print(d.name)
# the two eval sims used by lc pipelines
for extra in ["sim_A_025_test", "sim_B_038_test", "sim_Cc_050_val", "sim_B_030_val"]:
    ft = rf / extra / "field_times.npy"
    if ft.exists() and len(np.load(ft)) < 100:
        print(extra)
PY
)
echo "$BROKEN"

# 4 round-robin lanes of afterany chains
declare -a LANE_TAIL=("" "" "" "")
i=0
for S in $BROKEN; do
    rm -rf "$RF/$S"
    case "$S" in sim_A_*) SCRIPT=field_run.pbs ;; *) SCRIPT=field_run_flap.pbs ;; esac
    WT=30:00:00
    case "$S" in *B_004*|*Cc_006*|*Cc_014*) WT=48:00:00 ;; esac
    lane=$((i % 4))
    DEP=""
    [ -n "${LANE_TAIL[$lane]}" ] && DEP="-W depend=afterany:${LANE_TAIL[$lane]}"
    J=$(qsub $DEP -l walltime=$WT -o "$RF/pbs_${S}_r4.log" -v SIM="$S" "$SCRIPT")
    [ -z "$J" ] && { echo "QSUB FAILED at $S" >&2; exit 1; }
    echo "lane$lane: $S -> $J (wt=$WT, after ${LANE_TAIL[$lane]:-none})"
    LANE_TAIL[$lane]=$J
    i=$((i+1))
done
echo "submitted $i extractions across 4 lanes; tails: ${LANE_TAIL[*]}"

echo "--- re-arming N=100 pipeline (depends on all lane tails) ---"
bash submit_lc_n100.sh
