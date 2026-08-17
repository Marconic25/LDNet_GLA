#!/bin/bash
# Build the N=100 (final rung) learning-curve dataset. Dynamic pool: ALL complete
# train sims per family, target {20 A, 30 B, 50 Cc}. The three tail retries
# (Cc_006, B_004, Cc_014) are the LAST members of their families — no substitutes
# exist — so on shortfall we WARN and build with what is there (min 18/28/46),
# recording the actual composition. The rung is whatever N results (98-100 is fine
# for the curve; the actual list is echoed and stored next to the h5).
set -u
RF=/work/u10677113/NACA2312/recon_fields
cd /work/u10677113/NACA2312/recon

pool() {  # pool <family> <want> <min>
    local fam="$1" want="$2" min="$3" got=0
    for d in $(ls -d $RF/sim_${fam}_*_train 2>/dev/null | sort); do
        s=$(basename "$d")
        if [ -f "$d/fields_$s.npy" ]; then
            echo "$s"; got=$((got+1))
            [ "$got" -eq "$want" ] && break
        fi
    done
    if [ "$got" -lt "$min" ]; then
        echo "FATAL: family $fam has $got complete sims, minimum $min" 1>&2
        return 1
    fi
    if [ "$got" -lt "$want" ]; then
        echo "WARN: family $fam short: $got/$want complete — building anyway" 1>&2
    fi
    return 0
}

LIST=""
for S in $(pool A 20 18)  ; do LIST="$LIST --sim $RF/$S:A"  ; done
for S in $(pool B 30 28)  ; do LIST="$LIST --sim $RF/$S:B"  ; done
for S in $(pool Cc 50 46) ; do LIST="$LIST --sim $RF/$S:Cc" ; done
N=$(echo $LIST | tr " " "\n" | grep -c ":")
echo "=== N=$N sim list ==="
echo $LIST | tr " " "\n" | grep ":" | sed "s|$RF/||" | tee data/lc_N100_simlist.txt

echo "=== building FIELDS_lc_N100_train.h5 (actual N=$N) ==="
python3 -u build_fields_h5.py --out data/FIELDS_lc_N100_train.h5 --n-times 150 $LIST
echo BUILD_N100_DONE
