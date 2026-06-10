#!/bin/bash
cd /work/u10677113/LDNet_GLA/clean
MD=/work/u10677113/LDNet_GLA/clean/models_damped/latent_10
for G in 10 40 100; do
  echo "##### PROPORTIONAL gain=$G #####"
  MPLBACKEND=Agg python3 -u run.py --model-dir $MD --controller proportional --gain $G --settle 6000 2>/dev/null | grep -E 'C_L|h_ddot|flap|excurs|alpha'
done
echo '##### MPC horizon=6 #####'
MPLBACKEND=Agg python3 -u run.py --model-dir $MD --controller mpc --mpc-horizon 6 --settle 6000 2>/dev/null | grep -E 'C_L|h_ddot|flap|excurs|alpha'
echo '##### OPTIMAL (1-step) #####'
MPLBACKEND=Agg python3 -u run.py --model-dir $MD --controller optimal --settle 6000 2>/dev/null | grep -E 'C_L|h_ddot|flap|excurs|alpha'
echo ALLDONE
