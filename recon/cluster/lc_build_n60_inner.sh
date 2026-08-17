#!/bin/bash
# Build the N=60 learning-curve dataset (runs INSIDE the apptainer container).
# Dynamic pool selection (2026-07-11): take the first {12 A, 17 B, 31 Cc} COMPLETE
# train sims in sim-index order (completeness = fields_<sim>.npy present). This is
# robust to walltime-killed sims (their later same-family substitutes fill in) and
# is a superset of the actual N=30 set by construction (lower indices come first).
# Fails loudly if a family pool is insufficient.
set -u
RF=/work/u10677113/NACA2312/recon_fields
cd /work/u10677113/NACA2312/recon

pool() {  # pool <family> <count>  -> newline list of first <count> complete train sims
    local fam="$1" want="$2" got=0
    for d in $(ls -d $RF/sim_${fam}_*_train 2>/dev/null | sort); do
        s=$(basename "$d")
        if [ -f "$d/fields_$s.npy" ]; then
            echo "$s"
            got=$((got+1))
            [ "$got" -eq "$want" ] && return 0
        fi
    done
    echo "FATAL: family $fam has only $got complete sims, need $want" 1>&2
    return 1
}

LIST=""
for S in $(pool A 12) ; do LIST="$LIST --sim $RF/$S:A"  ; done
for S in $(pool B 17) ; do LIST="$LIST --sim $RF/$S:B"  ; done
for S in $(pool Cc 31); do LIST="$LIST --sim $RF/$S:Cc" ; done
N=$(echo $LIST | tr " " "\n" | grep -c ":")
if [ "$N" -ne 60 ]; then
    echo "FATAL: assembled $N sims instead of 60 — aborting" 1>&2
    exit 9
fi
echo "=== N=60 sim list ==="
echo $LIST | tr " " "\n" | grep ":" | sed "s|$RF/||"

echo "=== building FIELDS_lc_N60_train.h5 ==="
python3 -u build_fields_h5.py --out data/FIELDS_lc_N60_train.h5 --n-times 150 $LIST
echo BUILD_N60_DONE
