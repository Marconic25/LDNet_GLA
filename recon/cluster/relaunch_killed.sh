#!/bin/bash
# Identify extraction sims uncovered (no fields npy AND no queued/held job) — these
# are the fldrunf jobs killed to prioritise arm-A — and, with --go, resubmit each
# chained (afterany) behind the current extraction-lane tails so they run as part of
# the held campaign when it is released (i.e. AFTER arm-A), 2-wide safe. Also rewires
# lcbuild100 to depend on the new tails so the N100 build waits for the re-extraction.
# Dry-run by default. Usage: bash relaunch_killed.sh [--go]
export PATH=$PATH:/opt/pbs/bin
RF=/work/u10677113/NACA2312/recon_fields
cd /work/u10677113/NACA2312/recon/cluster

# sims currently represented by a queued/held/running extraction job (SIM= var)
QUEUED_SIMS=$(for J in $(qstat -u u10677113 | grep -iE 'fldrun' | cut -d. -f1); do
    qstat -f ${J}.login01 2>/dev/null | tr -d '\n\t ' | grep -oE 'SIM=sim_[A-Za-z0-9_]+'
done | cut -d= -f2 | sort -u)

UNCOV=()
while IFS= read -r S; do
    case "$S" in ''|\#*) continue ;; esac
    [ -f "$RF/$S/fields_$S.npy" ] && continue
    echo "$QUEUED_SIMS" | grep -qx "$S" && continue
    UNCOV+=("$S")
done < ladder_sims.txt
echo "QUEUED/HELD extraction sims: $(echo "$QUEUED_SIMS" | tr '\n' ' ')"
echo "UNCOVERED (${#UNCOV[@]}): ${UNCOV[*]}"

# current extraction-lane tails = highest two held fldrunf job ids
TAILS=$(qstat -u u10677113 | grep -iE 'fldrunf' | grep ' H ' | cut -d. -f1 | sort -n | tail -2)
echo "extraction held-tails: $TAILS"
BUILD=$(qstat -u u10677113 | grep -iE 'lcbuild' | cut -d. -f1 | head -1)
echo "lcbuild100 job: ${BUILD:-none}"

[ "$1" = "--go" ] || { echo "(dry-run; pass --go to resubmit)"; exit 0; }

A=$(echo "$TAILS" | sed -n 1p).login01
B=$(echo "$TAILS" | sed -n 2p).login01
NEWTAILS=""
for S in "${UNCOV[@]}"; do
    # The jobs killed to prioritise arm-A were all `fldrunf` (flap = non-A sims).
    # sim_A_* uncovered entries are PRE-EXISTING campaign gaps, not our kills -> flag,
    # do not relaunch (stay in scope: "relaunch what you killed").
    case "$S" in
      sim_A_*) echo "SKIP (pre-existing gap, not a killed job): $S"; continue ;;
      *) SCRIPT=field_run_flap.pbs ;;
    esac
    J=$(qsub -h -W depend=afterany:$A -l walltime=24:00:00 \
            -o $RF/pbs_${S}_relaunch.log -v SIM=$S $SCRIPT)
    echo "RESUB(held) $S -> $J (after $A, script $SCRIPT)"
    NEWTAILS="$J"; A=$B; B=$J
done
if [ -n "$BUILD" ] && [ -n "$NEWTAILS" ]; then
    OLD=$(qstat -f ${BUILD}.login01 2>/dev/null | tr -d '\n\t ' | grep -oE 'depend=[^,]*afterany:[^ ]*' | head -1)
    qalter -W depend=afterany:$A:$B ${BUILD}.login01 && \
        echo "rewired lcbuild100 $BUILD to depend on new tails $A:$B (was: $OLD)"
fi
echo "RELAUNCH DONE (all held; qrls the campaign to run after arm-A)"
