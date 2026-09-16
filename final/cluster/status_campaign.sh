#!/bin/bash
#
# status_campaign.sh — state of the v6 campaign, one line per run_matrix.csv row.
#
# Deliberately reads NOTHING but small text files (sim_info.txt) and qstat. It
# does not parse .npy, does not start python, does not touch apptainer — so it is
# safe to run on login01, which is the only place you would want to run it.
# run_sim.pbs writes status= and n_snapshots= into sim_info.txt precisely so this
# script never has to open a binary.
#
# States:
#   DONE     sim_info.txt says status=OK
#   BAD      sim_info.txt says status=BAD, or n_snapshots < 100 (the same
#            truncated-extraction criterion as recon/cluster/redo_broken_extractions.sh)
#   RUNNING  a v6_<sim> job is in qstat with state R
#   QUEUED   a v6_<sim> job is in qstat, held or waiting on its lane dependency
#   MISSING  no output directory and no job — never submitted, or lost
#
# Usage:
#   ./status_campaign.sh                 # summary + per-state counts
#   ./status_campaign.sh --list BAD      # list the sim_ids in one state
#   ./status_campaign.sh --relaunch      # resubmit every BAD/MISSING run
#   ./status_campaign.sh --relaunch --dry-run
#
# A job that appears to restart from scratch (SessID/elapsed reset) has NOT been
# evicted — this cluster does that. Do not conclude a job is gone from a single
# failed ssh during a connectivity blip.

set -euo pipefail

WORK_BASE="${WORK_BASE:-/work/u10677113/NACA2312}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX="${MATRIX:-$WORK_BASE/final/design/run_matrix.csv}"
OUT_ROOT="${OUT_ROOT:-$WORK_BASE/dataset_v6}"
QSTAT="${QSTAT:-/opt/pbs/bin/qstat}"
MIN_SNAPS="${MIN_SNAPS:-100}"

LIST_STATE=""
RELAUNCH=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --list)     LIST_STATE="$2"; shift 2 ;;
        --relaunch) RELAUNCH=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '3,27p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -f "$MATRIX" ] || { echo "run_matrix.csv not found: $MATRIX" >&2; exit 1; }

# ── One qstat call for the whole campaign (login01 is loaded; do not loop ssh) ──
QLINES=""
if command -v "$QSTAT" >/dev/null 2>&1; then
    QLINES=$("$QSTAT" -u "$USER" -w 2>/dev/null || true)
    if [ -z "$QLINES" ]; then
        # -w unsupported: fall back, but say so. Plain qstat truncates job names
        # to 9 chars, which cannot distinguish sims within a family, so RUNNING /
        # QUEUED will be unreliable (DONE / BAD still come from sim_info.txt).
        QLINES=$("$QSTAT" -u "$USER" 2>/dev/null || true)
        [ -n "$QLINES" ] && echo "# note: qstat -w unavailable; RUNNING/QUEUED may be imprecise" >&2
    fi
fi

job_state() {  # $1 = sim_id -> R | Q | ""  (Q covers Q, H and W)
    # qstat truncates Jobname to 14 chars plus a "*" even with -w, so
    # "v6_sim_A_002_train" shows up as "v6_sim_A_002_t*" and an exact-name match
    # finds nothing — which silently reported every running job as MISSING.
    # Match by prefix instead: strip the trailing "*" and test whether the full
    # name starts with what qstat kept. 14 chars is enough to keep our names
    # distinct (v6_sim_Cc_041_ vs v6_sim_Cc_042_), but only in -w output; plain
    # qstat truncates to 9 chars, where "v6_sim_A_" would match all of family A.
    local st
    st=$(printf '%s\n' "$QLINES" | tr -s ' ' \
         | awk -v want="v6_$1" '
             { n = $4; sub(/\*$/, "", n)
               if (n != "" && index(want, n) == 1) { print $10; exit } }')
    case "$st" in
        R) echo R ;;
        Q|H|W) echo Q ;;
        *) echo "" ;;
    esac
}

declare -a BAD_LIST=() MISSING_LIST=()
n_done=0; n_bad=0; n_run=0; n_queued=0; n_missing=0
# --list emits bare sim_ids so it can be piped straight into --sims; no header.
[ -z "$LIST_STATE" ] && printf '%-24s %-10s %-8s %-10s %s\n' SIM FAMILY STATE SNAPSHOTS NOTE

while IFS=, read -r sim family _split _w0 _tg _rg ctrl _rest; do
    [ "$sim" = "sim_id" ] && continue
    [ -n "$sim" ] || continue

    info="$OUT_ROOT/$sim/sim_info.txt"
    state=""; snaps="-"; note=""

    if [ -f "$info" ]; then
        st=$(sed -n 's/^status=//p' "$info" | head -1)
        snaps=$(sed -n 's/^n_snapshots=//p' "$info" | head -1)
        snaps="${snaps:-0}"
        if [ "$st" = "OK" ] && [ "$snaps" -ge "$MIN_SNAPS" ] 2>/dev/null; then
            state=DONE
        else
            state=BAD
            rc=$(sed -n 's/^driver_exit_code=//p' "$info" | head -1)
            note="driver_exit=$rc"
            [ "$snaps" -lt "$MIN_SNAPS" ] 2>/dev/null && note="$note truncated(<$MIN_SNAPS)"
        fi
    else
        case "$(job_state "$sim")" in
            R) state=RUNNING ;;
            Q) state=QUEUED  ;;
            *) state=MISSING ;;
        esac
    fi

    case "$state" in
        DONE)    n_done=$((n_done+1)) ;;
        BAD)     n_bad=$((n_bad+1));         BAD_LIST+=("$sim") ;;
        RUNNING) n_run=$((n_run+1)) ;;
        QUEUED)  n_queued=$((n_queued+1)) ;;
        MISSING) n_missing=$((n_missing+1)); MISSING_LIST+=("$sim") ;;
    esac

    if [ -z "$LIST_STATE" ]; then
        printf '%-24s %-10s %-8s %-10s %s\n' "$sim" "$family/$ctrl" "$state" "$snaps" "$note"
    elif [ "$LIST_STATE" = "$state" ]; then
        echo "$sim"
    fi
done < "$MATRIX"

if [ -n "$LIST_STATE" ]; then exit 0; fi

total=$((n_done + n_bad + n_run + n_queued + n_missing))
echo
echo "DONE $n_done   BAD $n_bad   RUNNING $n_run   QUEUED $n_queued   MISSING $n_missing   (of $total)"

if [ "$RELAUNCH" -eq 1 ]; then
    redo=("${BAD_LIST[@]:-}" "${MISSING_LIST[@]:-}")
    redo=($(printf '%s\n' "${redo[@]}" | awk 'NF'))
    if [ "${#redo[@]}" -eq 0 ]; then
        echo; echo "nothing to relaunch."
        exit 0
    fi
    echo
    echo "relaunching ${#redo[@]} run(s): ${redo[*]}"
    # Wipe the stale output first: a leftover sim_info.txt from the failed attempt
    # would otherwise make the rerun look DONE/BAD before it has written anything.
    for s in "${redo[@]}"; do
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  would rm -rf $OUT_ROOT/$s"
        else
            rm -rf "${OUT_ROOT:?}/$s"
        fi
    done
    args=(--sims "$(IFS=,; echo "${redo[*]}")")
    [ "$DRY_RUN" -eq 1 ] && args+=(--dry-run)
    "$HERE/submit_campaign.sh" "${args[@]}"
fi
