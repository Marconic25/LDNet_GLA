#!/bin/bash
# Submit N depth-ladder chain links, strictly 1-wide via afterany dependencies
# (queue politeness: never more than one depth-study job running; the 88-job
# extraction chain 25695-25782 must not be crowded — hopt_ds10 proved one extra
# 8-core job is tolerated). Each link resumes the ladder where the previous
# left off; links that find the STOP file exit in seconds.
# Usage: bash submit_depth_chain.sh [N_links]  (default 3)
cd /work/u10677113/NACA2312/recon/cluster
mkdir -p /work/u10677113/NACA2312/recon/models/depth_study
N=${1:-3}
DEP=""
for i in $(seq 1 $N); do
  LOG=/work/u10677113/NACA2312/recon/models/depth_study/pbs_depthlad_link${i}_$(date +%Y%m%d%H%M).log
  if [ -z "$DEP" ]; then
    J=$(qsub -o "$LOG" depth_ladder.pbs)
  else
    J=$(qsub -W depend=afterany:"$DEP" -o "$LOG" depth_ladder.pbs)
  fi
  echo "link $i -> $J (after ${DEP:-none})  log=$LOG"
  DEP=$J
done
