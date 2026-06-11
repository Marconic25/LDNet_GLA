import numpy as np, csv, os
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; S=0.05; C=1.0; DT=0.002; q=0.5*RHO*U**2*S
DS='/work/u10677113/NACA2312/dataset_v5'
MD=os.environ.get('MD','models_rollout/latent_10')
SIM=os.environ.get('SIM','sim_A_025_test')
DMAX=14.0; DDOT=300.0   # flap saturation + rate limit (deg, deg/s)
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(f'{DS}/{SIM}/structural_trajectory.csv')))[1:]])
d=d[range(0,len(d),7)]
W=d[:,7]; N=len(d); x0=np.array([d[0,1],d[0,2],d[0,3],d[0,4]])
a=LDNetAero(MD); a.reset(dt=DT)
clt,_=a.predict(x0,0.,0.,U); CLTRIM=float(clt)   # trim C_L reference
gust_win = d[:,0] <= 1.6

def run(gain):
    a.reset(dt=DT); x=x0.copy(); dprev=0.; CL=[];HH=[];AA=[];DD=[]
    for i in range(N):
        clm,_=a.predict(x,dprev,W[i],U)     # measure C_L with current flap
        # proportional GLA: gust raises C_L -> negative flap (dC_L/dd>0) lowers it
        dcmd = -gain*(float(clm)-CLTRIM)
        dcmd = np.clip(dcmd, dprev-DDOT*DT, dprev+DDOT*DT)   # rate limit
        dcmd = float(np.clip(dcmd, -DMAX, DMAX))             # saturation
        cl,cm=a.predict(x,dcmd,W[i],U); CL.append(float(cl))
        Fy=q*cl; Mz=q*cm*C
        a.advance(x,dcmd,W[i],U,DT); x=structure.step_rk4(x,Fy,Mz,DT)
        HH.append(x[0]);AA.append(x[2]);DD.append(dcmd); dprev=dcmd
    CL=np.array(CL);HH=np.array(HH);AA=np.array(AA);DD=np.array(DD)
    exc=np.max(np.abs(CL[gust_win]-CLTRIM))
    hpk=np.max(np.abs(HH[gust_win])); apk=np.max(np.abs(AA[gust_win]))
    return exc,hpk,apk,np.max(np.abs(DD))
print('=== GLA control %s on %s (trim C_L=%.3f) ==='%(MD,SIM,CLTRIM),flush=True)
e0,h0,a0,_=run(0.0)
print('  open loop:   C_L exc=%.4f  h_pk=%.5f  a_pk=%.5f'%(e0,h0,a0),flush=True)
for G in [5,10,20,40,80]:
    e,h,ap,dm=run(float(G))
    print('  gain=%3d:    C_L exc=%.4f (%+.1f%%)  h_pk=%.5f (%+.1f%%)  a_pk=%.5f  flap_max=%.1f deg'%(
        G,e,(e0-e)/e0*100,h,(h0-h)/h0*100,ap,dm),flush=True)
