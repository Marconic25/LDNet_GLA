import numpy as np, os, csv
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
MD='models_rollout/latent_10'
X0=np.array([-6.49179e-3,0.,-8.76338e-4,0.])
a=LDNetAero(MD)
def openrun(Wt):
    a.reset(dt=DT); x=X0.copy(); AD=[]
    for W in Wt:
        cl,cm=a.predict(x,0.,W,U); Fy=q*cl;Mz=q*cm*C
        d=structure.rhs(x,Fy,Mz); AD.append(d[3])
        a.advance(x,0.,W,U,DT); x=structure.step_dp45(x,Fy,Mz,DT)
    return np.array(AD)
def ring(ad,t,t0,t1):
    m=(t>=t0)&(t<=t1); s=ad[m]
    sc=np.sum(np.abs(np.diff(np.sign(s))))/2   # zero crossings
    return np.sqrt(np.mean(s**2)), int(sc)
# synthetic 1-cos gust W0=11.46 Tg=1.12
TG=1.12; W0=11.46; N=1500; t=np.arange(N)*DT
Ws=np.array([(W0/2)*(1-np.cos(2*np.pi*tt/TG)) if 0<=tt<=TG else 0. for tt in t])
ad_s=openrun(Ws)
# data gust A_025 (subsampled)
d=np.array([[float(v) for v in r] for r in list(csv.reader(open('/work/u10677113/NACA2312/dataset_v5/sim_A_025_test/structural_trajectory.csv')))[1:]])
d=d[range(0,len(d),7)]; Wd=d[:,7]; td=d[:,0]
ad_d_model=openrun(Wd)                 # model open-loop on data gust
ad_d_cfd=np.gradient(d[:,4],td)*0 + d[:,4]   # data's true alpha_dot (col 4 = ad)
print('OPEN-LOOP alpha_dot ringing (RMS, zero-crossings):',flush=True)
print('  synthetic gust, MODEL : during[0,1.12] RMS=%.4g xc=%d | post[1.4,2.4] RMS=%.4g xc=%d'%(*ring(ad_s,t,0,1.12),*ring(ad_s,t,1.4,2.4)),flush=True)
print('  data gust,      MODEL : during        RMS=%.4g xc=%d | post[1.4,2.4] RMS=%.4g xc=%d'%(*ring(ad_d_model,td,0,1.12),*ring(ad_d_model,td,1.4,2.4)),flush=True)
print('  data gust,      CFD   : during        RMS=%.4g xc=%d | post[1.4,2.4] RMS=%.4g xc=%d'%(*ring(ad_d_cfd,td,0,1.12),*ring(ad_d_cfd,td,1.4,2.4)),flush=True)
