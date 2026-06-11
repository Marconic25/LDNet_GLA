import numpy as np, csv, json, os
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; S=0.05; C=1.0; DT=0.002; q=0.5*RHO*U**2*S
DS='/work/u10677113/NACA2312/dataset_v5'
MD=os.environ.get('MD','models_rollout/latent_10')
SIM=os.environ.get('SIM','sim_A_025_test')
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(f'{DS}/{SIM}/structural_trajectory.csv')))[1:]])
# cols t,h,hd,a,ad,Fy,Mz,W,delta (0..8). Subsample to dt=0.002 (every 7 of 0.00028)
idx=range(0,len(d),7); d=d[list(idx)]
t=d[:,0]; W=d[:,7]; delta=d[:,8]
h_t=d[:,1]; a_t=d[:,3]; Fy_t=d[:,5]; CL_t=Fy_t/q
a=LDNetAero(MD); a.reset(dt=DT)             # z=0 start (matches rollout training)
x=np.array([d[0,1],d[0,2],d[0,3],d[0,4]])   # structural IC from data
# FULL loads, no trim subtraction (matches rollout training; data trim state is the
# equilibrium under full loads: K_H*h_trim balances Fy_trim).
hh=[];aa=[];CL=[]
for i in range(len(d)):
    cl,cm=a.predict(x,delta[i],W[i],U); CL.append(float(cl))
    Fy=q*cl; Mz=q*cm*C
    a.advance(x,delta[i],W[i],U,DT); x=structure.step_rk4(x,Fy,Mz,DT)
    hh.append(x[0]);aa.append(x[2])
hh=np.array(hh);aa=np.array(aa);CL=np.array(CL)
def nrmse(r,f): return np.sqrt(np.mean((r-f)**2))/(f.max()-f.min()+1e-12)
print('=== closed-loop (z=0) %s on %s ==='%(MD,SIM),flush=True)
print('  C_L: data exc=%.3f  model exc=%.3f  NRMSE=%.3f'%(CL_t.max()-CL_t[:20].mean(),CL.max()-CL[:20].mean(),nrmse(CL,CL_t)),flush=True)
print('  h:   data pp=%.5f  model pp=%.5f  NRMSE=%.3f'%(h_t.max()-h_t.min(),hh.max()-hh.min(),nrmse(hh,h_t)),flush=True)
print('  a:   data pp=%.5f  model pp=%.5f  NRMSE=%.3f'%(a_t.max()-a_t.min(),aa.max()-aa.min(),nrmse(aa,a_t)),flush=True)
print('  final h model=%.4f data=%.4f ; any nan=%s'%(hh[-1],h_t[-1],np.any(np.isnan(hh))),flush=True)
