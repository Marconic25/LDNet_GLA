#!/bin/bash
# Repair the extraction chain after the 2026-07-14 22:05 user-hold + E-cascade:
# (1) qrls the held survivors — their intact afterany graph re-establishes 2-wide
#     flow with heads 25763 / 25772; retries (25850, 26194, 26305) stay gated.
# (2) chain the two parked -h replacement jobs (26932 Cc_038, 26933 A_016) at the
#     lane tails after the retries, then release them.
# (3) resubmit every ladder sim that has neither complete fields nor a queued job
#     (the E-cascade victims), chained 2-wide behind the replacements, 24h walltime.
# (4) rewire lcbuild100 (26827) to depend on the new lane tails (transitively gates
#     on everything), using qhold/qrls so it cannot fire mid-rewire.
cd /work/u10677113/NACA2312/recon/cluster
RF=/work/u10677113/NACA2312/recon_fields

qhold 26827.login01 2>/dev/null
echo "--- (1) releasing held survivors ---"
for J in 25763 25765 25767 25769 25771 25772 25773 25774 25775 25776 25777 25778 \
         25779 25780 25781 25782 25850 26194 26305; do
    qrls ${J}.login01 2>&1 | head -1
done
echo RELEASED_SURVIVORS

echo "--- (2) chaining parked replacements at lane tails ---"
qalter -W depend=afterany:26305.login01 26932.login01 && echo DEP_26932_OK
qalter -W depend=afterany:26194.login01 26933.login01 && echo DEP_26933_OK
qalter -l walltime=24:00:00 26932.login01 2>/dev/null
qalter -l walltime=24:00:00 26933.login01 2>/dev/null
qrls 26932.login01 26933.login01
echo RELEASED_REPLACEMENTS

echo "--- (3) resubmitting uncovered victims ---"
QUEUED_SIMS=$(for J in $(qstat -u u10677113 | grep fldrun | cut -d. -f1); do
    qstat -f ${J}.login01 2>/dev/null | tr -d "\n\t " | grep -oE "SIM=sim_[A-Za-z0-9_]+"
done | cut -d= -f2 | sort -u)
A=26932.login01
B=26933.login01
N=0
while IFS= read -r S; do
    case "$S" in ''|\#*) continue ;; esac
    [ -f "$RF/$S/fields_$S.npy" ] && continue
    if echo "$QUEUED_SIMS" | grep -qx "$S"; then echo "COVERED $S"; continue; fi
    case "$S" in sim_A_*) SCRIPT=field_run.pbs ;; *) SCRIPT=field_run_flap.pbs ;; esac
    J=$(qsub -W depend=afterany:$A -l walltime=24:00:00 \
            -o $RF/pbs_${S}_r2.log -v SIM=$S $SCRIPT)
    if [ -z "$J" ]; then echo "QSUB FAILED at $S" >&2; exit 1; fi
    echo "RESUB $S -> $J (after $A)"
    A=$B; B=$J; N=$((N+1))
done < ladder_sims.txt
echo "RESUBMITTED $N victims; lane tails: $A $B"

echo "--- (4) rewiring lcbuild100 deps to lane tails ---"
qalter -W depend=afterany:${A}:${B} 26827.login01 && echo BUILD_DEPS_OK
qrls 26827.login01 2>/dev/null
echo DONE
