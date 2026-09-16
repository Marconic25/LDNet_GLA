#!/bin/bash
#
# submit_campaign.sh — submit the v6 dataset campaign, one PBS job per row of
# final/design/run_matrix.csv.
#
# Concurrency model — 4 round-robin lanes chained with -W depend=afterany.
# This is NOT a stylistic choice:
#   - max_user_run = 4 on this cluster, and it is SHARED with any other session
#     of yours that submits work.
#   - This scheduler puts EXCESS concurrent jobs into E state (exit 1) instead of
#     queueing them (recon/cluster/submit_ladder.sh documents the incident that
#     established this). So "submit all 146 and let PBS sort it out" loses jobs.
# Round-robin over N lanes, with each job depending on the previous job IN ITS
# OWN LANE, keeps exactly N runnable at any instant with no held-job bookkeeping:
# lane heads start immediately, everything else is gated by a dependency.
# afterany (not afterok) so one failed sim does not strand the rest of its lane —
# status_campaign.sh --relaunch picks the failures back up.
#
# PBS-only rule: this script only ever calls qsub. Nothing here runs OpenFOAM,
# python or any compute over ssh — a documented incident (login01 load average
# 107, jobs administratively removed) makes that absolute.
#
# Usage:
#   ./submit_campaign.sh --dry-run                 # print the qsub commands, submit nothing
#   ./submit_campaign.sh --family A                # just family A
#   ./submit_campaign.sh --family Cc --limit 4     # first 4 Cc rows
#   ./submit_campaign.sh --lanes 2                 # fall back to a 2-wide chain under contention
#   ./submit_campaign.sh --sims sim_A_001_train,sim_B_003_val
#
# Run it ON THE CLUSTER (login01 is fine — qsub is bookkeeping), from
# /work/u10677113/NACA2312/final/cluster.

set -euo pipefail

WORK_BASE="${WORK_BASE:-/work/u10677113/NACA2312}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX="${MATRIX:-$WORK_BASE/final/design/run_matrix.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-$HERE/run_sim.pbs}"
LOGDIR="${LOGDIR:-$WORK_BASE/dataset_v6/_pbs_logs}"
QSUB="${QSUB:-/opt/pbs/bin/qsub}"

LANES=4
FAMILY=""
LIMIT=0
SIMS=""
DRY_RUN=0
WALL_DEFAULT="${WALL_DEFAULT:-12:00:00}"
WALL_MPC="${WALL_MPC:-24:00:00}"
# Lane -> node pinning. MEASURED, not defensive: on 2026-09-03 the scheduler packed
# two campaign jobs onto cpu01 (11 of 112 cores free, 4 other users' jobs) while
# cpu03/04/05 sat completely empty. After 25 minutes those two had completed 3
# coupling windows each; the two on the near-empty cpu02 had completed 180. That is
# a 60x throughput difference — worse than the ~35x already documented in
# light/dagger_fom/NOTES.md — and it is fatal, not merely slow: at 3 windows/25 min
# a full run needs ~208 h and would hit the 12 h walltime and die.
# One lane per node, 16 of 112 cores each, so this does not monopolise the cluster.
NODES="${NODES:-cpu02,cpu03,cpu04,cpu05}"
NO_PIN=0

usage() { sed -n '3,30p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --family)  FAMILY="$2"; shift 2 ;;
        --limit)   LIMIT="$2";  shift 2 ;;
        --lanes)   LANES="$2";  shift 2 ;;
        --sims)    SIMS="$2";   shift 2 ;;
        --nodes)   NODES="$2";  shift 2 ;;
        --no-pin)  NO_PIN=1;    shift ;;
        --dry-run) DRY_RUN=1;   shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[ -f "$MATRIX" ]     || { echo "run_matrix.csv not found: $MATRIX" >&2; exit 1; }
[ -f "$JOB_SCRIPT" ] || { echo "run_sim.pbs not found: $JOB_SCRIPT" >&2; exit 1; }
case "$LANES" in ''|*[!0-9]*) echo "--lanes must be a positive integer" >&2; exit 1 ;; esac
[ "$LANES" -ge 1 ] || { echo "--lanes must be >= 1" >&2; exit 1; }

if [ "$DRY_RUN" -eq 0 ]; then
    command -v "$QSUB" >/dev/null 2>&1 || { echo "qsub not found at $QSUB -- are you on the cluster?" >&2; exit 1; }
    mkdir -p "$LOGDIR"
fi

# ── Select rows ─────────────────────────────────────────────────────────────
# Columns: sim_id,family,split,W0,Tg,r_g,controller,K_ff,t_d,delta_pk,rate_max,R_star,t_end
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
        print $1 "," $2 "," $7
    }' "$MATRIX")

[ -n "$rows" ] || { echo "no rows selected (family='$FAMILY' sims='$SIMS')" >&2; exit 1; }
if [ "$LIMIT" -gt 0 ]; then
    rows=$(printf '%s\n' "$rows" | head -n "$LIMIT")
fi

total=$(printf '%s\n' "$rows" | wc -l | tr -d ' ')

# ── Lane -> node map, with a preflight on how loaded each target node is ──
IFS=',' read -r -a NODE_ARR <<< "$NODES"
if [ "$NO_PIN" -eq 0 ] && [ "${#NODE_ARR[@]}" -gt 0 ]; then
    if [ "${#NODE_ARR[@]}" -lt "$LANES" ]; then
        echo "# note: ${#NODE_ARR[@]} node(s) for $LANES lane(s) -- lanes will share nodes" >&2
    fi
    echo "# lane -> node pinning (see the header of this script for why):"
    for ((i = 0; i < LANES; i++)); do
        n="${NODE_ARR[$((i % ${#NODE_ARR[@]}))]}"
        # Tolerant: pbsnodes is absent when dry-running off-cluster, and a failed
        # preflight must never stop a submission (set -e would otherwise abort here).
        free=$(/opt/pbs/bin/pbsnodes -aSj 2>/dev/null | tr -s ' ' \
               | awk -v n="$n" '$1 == n {print $7}' || true)
        echo "#   lane $i -> $n   free ncpus: ${free:-unknown}"
        case "$free" in
            */*) [ "${free%%/*}" -lt 16 ] 2>/dev/null && \
                 echo "#     WARNING: fewer than the 16 cores a job needs are free on $n" ;;
        esac
    done
    echo
fi

echo "# submitting $total run(s) across $LANES lane(s)"
echo "#   matrix: $MATRIX"
echo "#   job:    $JOB_SCRIPT"
[ "$NO_PIN" -eq 1 ] && echo "#   --no-pin: letting the scheduler place jobs (it packs onto loaded nodes)"
[ "$DRY_RUN" -eq 1 ] && echo "#   DRY RUN — nothing will be submitted"
echo

# Per-lane "previous job id", indexed 0..LANES-1.
declare -a PREV
for ((i = 0; i < LANES; i++)); do PREV[$i]=""; done

idx=0
while IFS=, read -r sim family ctrl; do
    [ -n "$sim" ] || continue
    lane=$((idx % LANES))

    # Cc/mpc rows carry ~2100 mpc_call() round trips through a TF-container
    # subprocess; 12h is not enough headroom. qsub -l overrides the script's
    # own #PBS -l walltime directive.
    if [ "$family" = "Cc" ] && [ "$ctrl" = "mpc" ]; then
        wall="$WALL_MPC"
    else
        wall="$WALL_DEFAULT"
    fi

    dep=""
    [ -n "${PREV[$lane]}" ] && dep="-W depend=afterany:${PREV[$lane]}"

    cmd=("$QSUB" -N "v6_$sim" -l "walltime=$wall" -o "$LOGDIR/${sim}.log" -v "SIM=$sim")
    if [ "$NO_PIN" -eq 0 ] && [ "${#NODE_ARR[@]}" -gt 0 ]; then
        node="${NODE_ARR[$((lane % ${#NODE_ARR[@]}))]}"
        cmd+=(-l "select=1:ncpus=16:mpiprocs=16:host=$node")
    fi
    [ -n "$dep" ] && cmd+=($dep)
    cmd+=("$JOB_SCRIPT")

    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'lane %d  %s\n' "$lane" "${cmd[*]}"
        PREV[$lane]="<job$idx>"
    else
        jid=$("${cmd[@]}")
        printf 'lane %d  %-24s %s  (wall %s)\n' "$lane" "$jid" "$sim" "$wall"
        PREV[$lane]="$jid"
    fi
    idx=$((idx + 1))
done <<< "$rows"

echo
echo "# submitted $idx job(s). Track with:  ./status_campaign.sh"
if [ "$DRY_RUN" -eq 0 ]; then
    echo "# Check the nodes your jobs landed on before assuming they are healthy:"
    echo "#   pbsnodes -aSj     # a job sharing a busy node has run ~35x slower here"
fi
