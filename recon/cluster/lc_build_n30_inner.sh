#!/bin/bash
# Build the N=30 learning-curve datasets (runs INSIDE the apptainer container).
# Canonical rung-30 set = 15 original + 15 new; Cc_006 was walltime-killed and its
# retry sits at the chain tail, so Cc_014 is its decided substitute (2026-07-08).
# Six sims may still be missing if the slow B_004 lane stalled: each has a same-family
# rung-60 fallback (extracted right after in the chain). Fails loudly if neither
# a primary nor its fallback is complete (completeness = fields_<sim>.npy present).
set -u
RF=/work/u10677113/NACA2312/recon_fields
cd /work/u10677113/NACA2312/recon

pick() {  # pick <primary> [fallback] -> echoes the usable sim name
    local p="$1" f="${2:-}"
    if [ -f "$RF/$p/fields_$p.npy" ]; then echo "$p"; return 0; fi
    if [ -n "$f" ] && [ -f "$RF/$f/fields_$f.npy" ]; then
        echo "SWAP: $p -> $f" 1>&2; echo "$f"; return 0
    fi
    echo "FATAL: $p incomplete and fallback '${f:-none}' incomplete" 1>&2
    return 1
}

LIST=""
add() {
    local S
    S=$(pick "$@") || exit 9
    local FAM
    case "$S" in sim_A_*) FAM=A ;; sim_B_*) FAM=B ;; *) FAM=Cc ;; esac
    LIST="$LIST --sim $RF/$S:$FAM"
}

# A x8
add sim_A_000_train; add sim_A_002_train; add sim_A_005_train; add sim_A_010_train
add sim_A_012_train; add sim_A_017_train; add sim_A_001_train; add sim_A_003_train
# B x8
add sim_B_000_train; add sim_B_001_train; add sim_B_002_train; add sim_B_003_train
add sim_B_004_train sim_B_008_train
add sim_B_005_train sim_B_009_train
add sim_B_006_train; add sim_B_007_train
# Cc x14 (Cc_014 replaces the walltime-killed Cc_006)
add sim_Cc_000_train; add sim_Cc_001_train; add sim_Cc_002_train; add sim_Cc_003_train
add sim_Cc_004_train; add sim_Cc_005_train; add sim_Cc_007_train; add sim_Cc_008_train
add sim_Cc_009_train sim_Cc_015_train
add sim_Cc_010_train sim_Cc_016_train
add sim_Cc_011_train sim_Cc_017_train
add sim_Cc_012_train
add sim_Cc_013_train sim_Cc_018_train
add sim_Cc_014_train sim_Cc_015_train

echo "=== building FIELDS_lc_N30_train.h5 (30 sims) ==="
python3 -u build_fields_h5.py --out data/FIELDS_lc_N30_train.h5 --n-times 150 $LIST

echo "=== building extra eval sets ==="
python3 -u build_fields_h5.py --out data/FIELDS_B038.h5 --n-times 150 --sim $RF/sim_B_038_test:B
python3 -u build_fields_h5.py --out data/FIELDS_B030val.h5 --n-times 150 --sim $RF/sim_B_030_val:B
python3 -u build_fields_h5.py --out data/FIELDS_Cc050val.h5 --n-times 150 --sim $RF/sim_Cc_050_val:Cc
echo BUILD_N30_DONE
