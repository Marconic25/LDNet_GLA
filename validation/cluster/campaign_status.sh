#!/bin/bash
# campaign_status.sh — one status snapshot of the NACA2312 verification campaign.
# Emits STATE/COURANT/TEMP lines; performs staged release of held campaign jobs
# and, only when the campaign has nothing left in R/Q/H, releases the mid-tier
# (lcbuild/lcN100/depth) holds. fld* jobs are never touched.
. /etc/profile.d/pbs.sh 2>/dev/null
cd /work/u10677113/NACA2312 || exit 1

CAMPAIGN_HELD_ORDER="26923 26920 26921 26915 26922"
MIDTIER="26827 26828 26829 26830 26831 26832 26833 26834 26839 26840 26842"

QS=$(qstat -u u10677113 2>/dev/null | tail -n +6)
ST=$(echo "$QS" | grep -E "TEMP|CO_")
R=$(echo "$ST" | awk '$10=="R"' | grep -c .)
Q=$(echo "$ST" | awk '$10=="Q"' | grep -c .)
H=$(echo "$ST" | awk '$10=="H"' | grep -c .)

# staged campaign release: one job per free slot
if [ $((R + Q)) -lt 4 ] && [ "$H" -gt 0 ]; then
  for j in $CAMPAIGN_HELD_ORDER; do
    s=$(echo "$ST" | grep "^$j" | awk '{print $10}')
    if [ "$s" = "H" ]; then
      qrls -h u "$j" 2>/dev/null && echo "RELEASED campaign $j"
      break
    fi
  done
fi

# campaign fully drained -> release mid-tier (idempotent)
if [ $((R + Q + H)) -eq 0 ]; then
  echo "CAMPAIGN_QUEUE_EMPTY -> releasing midtier (fld still held)"
  for j in $MIDTIER; do qrls -h u "$j" 2>/dev/null; done
fi

echo "$ST" | awk 'NF {print "STATE", $1, $4, $10}'

for c in 0.5 1 2 4 fixed; do
  f=courant_sweep/log_Co_$c.txt
  [ -f "$f" ] && echo "COURANT_$c $(grep -E 'probe OK|PROBE FAILED|FULL RUN FAILED|\] done' "$f" | tail -1)"
done

for L in DT1 DT2 DT3 DT4; do
  [ -f "temporal_study/${L}_result.json" ] && echo "TEMP_$L RESULT_DONE"
  f=temporal_study/log_$L.txt
  [ -f "$f" ] && grep -m1 -E "mapFields failed|topoSet failed|Traceback|FATAL|quota" "$f" | sed "s/^/TEMP_${L}_ERR /"
done
exit 0
