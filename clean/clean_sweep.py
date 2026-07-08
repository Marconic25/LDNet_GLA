import numpy as np, csv, os
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
DS='/work/u10677113/NACA2312/dataset_v5'; MD='models_rollout/latent_10'; SIM=os.environ.get('SIM','sim_A_025_test')
DMAX=14.0; DDOT=300.0
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(f'{DS}/{SIM}/structural_trajectory.csv')))[1:]])
d=d[range(0,len(d),7)]; W=d[:,7]; t=d[:,0]; N=len(d); x0=np.array([d[0,1],d[0,2],d[0,3],d[0,4]])
a=LDNetAero(MD); a.reset(dt=DT); clt,_=a.predict(x0,0.,0.,U); CLTRIM=float(clt)
m=t<=1.6
def run(gain, lpf=0.0):
    a.reset(dt=DT); x=x0.copy(); dprev=0.; df=0.; CL=[];DD=[];HH=[]
    for i in range(N):
        clm,_=a.predict(x,dprev,W[i],U)
        dcmd=-gain*(float(clm)-CLTRIM)
        df = lpf*df + (1-lpf)*dcmd          # optional low-pass to kill chatter
        dt_cmd = df if lpf>0 else dcmd
        dt_cmd=np.clip(dt_cmd, dprev-DDOT*DT, dprev+DDOT*DT)
        dt_cmd=float(np.clip(dt_cmd,-DMAX,DMAX))
        cl,cm=a.predict(x,dt_cmd,W[i],U)
        a.advance(x,dt_cmd,W[i],U,DT); x=structure.step_dp45(x,q*cl,q*cm*C,DT)
        CL.append(float(cl));DD.append(dt_cmd);HH.append(x[0]); dprev=dt_cmd
    CL=np.array(CL);DD=np.array(DD);HH=np.array(HH)
    return CL[m].max()-CLTRIM, np.max(np.abs(HH[m])), np.abs(DD).max()
print(f'=== {SIM}  trim={CLTRIM:.3f} ===',flush=True)
e0,h0,_=run(0.0); print(f'  open: peak_exc={e0:.3f} h_pk={h0:.5f}',flush=True)
print('  -- plain proportional --',flush=True)
for G in [10,20,40,80,150]:
    e,h,dm=run(float(G)); print(f'  G={G:3d}: exc={e:.3f} ({(e0-e)/e0*100:+.0f}%) h_pk={h:.5f} flap_max={dm:.1f}',flush=True)
print('  -- proportional + LPF 0.9 (anti-chatter) --',flush=True)
for G in [40,80,150,300]:
    e,h,dm=run(float(G),lpf=0.9); print(f'  G={G:3d}: exc={e:.3f} ({(e0-e)/e0*100:+.0f}%) h_pk={h:.5f} flap_max={dm:.1f}',flush=True)
