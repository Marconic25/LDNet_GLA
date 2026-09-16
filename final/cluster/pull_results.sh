#!/bin/bash
#
# pull_results.sh — bring the v6 dataset back from the cluster to the local repo.
#
# Run this LOCALLY (WSL), not on the cluster. Follows the pattern of
# recon/cluster/pull_coral_residual_results.sh.
#
# Pulls per-sim: fields_<sim>.npy, field_times.npy, mesh_points.npy,
# mesh_triangles.npy, structural_trajectory.csv, sim_info.txt, run.log.
# Skips the OpenFOAM case itself — it lives on /scratch_local and is deleted by
# the job that created it.
#
# Size: ~120 MB per sim, ~17.5 GB for the full 146-run campaign (see
# final/ESTIMATE.md §5). Check you have the room before pulling everything.
#
# Usage:
#   ./pull_results.sh                        # everything that is DONE
#   ./pull_results.sh --family A             # one family
#   ./pull_results.sh --sims sim_A_001_train,sim_B_003_val
#   ./pull_results.sh --no-fields            # metadata + CSV only (fast, ~1 MB/sim)
#   ./pull_results.sh --dry-run

set -euo pipefail

HOST="${HOST:-u10677113@10.78.18.100}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/u10677113/NACA2312/dataset_v6}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$HERE/../data/dataset_v6}"
MATRIX="${MATRIX:-$HERE/../design/run_matrix.csv}"

FAMILY=""
SIMS=""
NO_FIELDS=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --family)    FAMILY="$2"; shift 2 ;;
        --sims)      SIMS="$2"; shift 2 ;;
        --no-fields) NO_FIELDS=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   sed -n '3,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -f "$MATRIX" ] || { echo "run_matrix.csv not found: $MATRIX" >&2; exit 1; }

rows=$(awk -F, -v fam="$FAMILY" -v sims="$SIMS" '
    NR == 1 { next }
    fam != "" && $2 != fam { next }
    {
        if (sims != "") {
            found = 0
            n = split(sims, want, ",")
            for (i = 1; i <= n; i++) if ($1 == want[i]) found = 1
            if (!found) next
        }
        print $1
    }' "$MATRIX")

[ -n "$rows" ] || { echo "no rows selected" >&2; exit 1; }
total=$(printf '%s\n' "$rows" | wc -l | tr -d ' ')

SSH_OPTS=(-o ConnectTimeout=30 -o ServerAliveInterval=15)
[ -f "$KEY" ] && SSH_OPTS+=(-i "$KEY")

RSYNC_INCLUDE=(
    --include='field_times.npy' --include='mesh_points.npy'
    --include='mesh_triangles.npy' --include='structural_trajectory.csv'
    --include='sim_info.txt' --include='run.log'
)
[ "$NO_FIELDS" -eq 0 ] && RSYNC_INCLUDE+=(--include='fields_*.npy')

echo "# pulling $total sim(s)"
echo "#   from: $HOST:$REMOTE_ROOT"
echo "#   to:   $LOCAL_ROOT"
[ "$NO_FIELDS" -eq 1 ] && echo "#   (--no-fields: skipping fields_*.npy)"
[ "$DRY_RUN" -eq 1 ]   && echo "#   DRY RUN"
echo

ok=0; skipped=0
for sim in $rows; do
    dst="$LOCAL_ROOT/$sim"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would pull $sim -> $dst"
        ok=$((ok+1))
        continue
    fi
    mkdir -p "$dst"
    if rsync -a --partial --info=progress2 \
            -e "ssh ${SSH_OPTS[*]}" \
            "${RSYNC_INCLUDE[@]}" --exclude='*' \
            "$HOST:$REMOTE_ROOT/$sim/" "$dst/"; then
        ok=$((ok+1))
        printf '  %-24s ok\n' "$sim"
    else
        skipped=$((skipped+1))
        printf '  %-24s SKIPPED (not on the cluster yet, or transfer failed)\n' "$sim"
    fi
done

echo
echo "pulled $ok, skipped $skipped, of $total"
echo "next:  python3 final/data/preprocess_GLA_v6.py --root $LOCAL_ROOT --matrix $MATRIX"
echo "       python3 final/data/build_fields_h5_v6.py --root $LOCAL_ROOT --matrix $MATRIX"
