#!/bin/bash
# submit_ladder.sh — submit the extraction ladder as a 2-wide afterany dependency chain.
# This cluster ERRORS (E state, exit 1) excess concurrent 16-core jobs instead of queuing
# them, so at most 2 jobs may be runnable at any time; the rest must be held (H) behind
# afterany dependencies. Family A sims use field_run.pbs (zero flap), B/Cc use
# field_run_flap.pbs (replays recorded delta(t)).
#
# Usage:  bash submit_ladder.sh [ladder_sims.txt]
set -u
cd /work/u10677113/NACA2312/recon/cluster
LIST="${1:-ladder_sims.txt}"
LOGDIR=/work/u10677113/NACA2312/recon_fields

PREV2=""   # job two positions back (the new job's dependency)
PREV1=""   # job one position back
N=0
while IFS= read -r S; do
    case "$S" in ''|\#*) continue ;; esac
    if [ -d "$LOGDIR/$S" ] && [ -n "$(ls -A "$LOGDIR/$S" 2>/dev/null)" ]; then
        echo "SKIP $S (recon_fields/$S already non-empty)"
        continue
    fi
    case "$S" in
        sim_A_*) SCRIPT=field_run.pbs ;;
        *)       SCRIPT=field_run_flap.pbs ;;
    esac
    DEP=""
    [ -n "$PREV2" ] && DEP="-W depend=afterany:$PREV2"
    J=$(qsub $DEP -o "$LOGDIR/pbs_${S}.log" -v SIM="$S" "$SCRIPT")
    if [ -z "$J" ]; then
        echo "QSUB FAILED at $S — chain stops here; resubmit the remainder later." >&2
        exit 1
    fi
    N=$((N+1))
    echo "$N: $S -> $J ($SCRIPT, after ${PREV2:-none})"
    PREV2=$PREV1
    PREV1=$J
done < "$LIST"
echo "Submitted $N jobs."
