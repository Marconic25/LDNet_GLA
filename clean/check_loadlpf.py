import numpy as np, os, csv
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
MD='models_rollout/latent_10'
X0=np.array([-6.49179e-3,0.,-8.76338e-4,0.])
a=LDNetAero(MD)
TG=1.12; W0=11.46; N=1500; t=np.arange(N)*DT
Ws=np.array([(W0/2)*(1-np.cos(2*np.pi*tt/TG)) if 0<=tt<=TG else 0. for tt in t])
def openrun(Wt, lpf=0.0):
    a.reset(dt=DT); x=X0.copy(); AD=[]; Fyf=None; Mzf=None
    for W in Wt:
        cl,cm=a.predict(x,0.,W,U); Fy=q*cl; Mz=q*cm*C
        if lpf>0:
            Fyf = Fy if Fyf is None else lpf*Fyf+(1-lpf)*Fy
            Mzf = Mz if Mzf is None else lpf*Mzf+(1-lpf)*Mz
            Fy,Mz=Fyf,Mzf
        d=structure.rhs(x,Fy,Mz); AD.append(d[3])
        a.advance(x,0.,W,U,DT); x=structure.step_dp45(x,Fy,Mz,DT)
    return np.array(AD)
def ring(ad,t0,t1):
    m=(t>=t0)&(t<=t1); s=ad[m]
    return np.sqrt(np.mean(s**2)), int(np.sum(np.abs(np.diff(np.sign(s))))/2)
print('open-loop alpha_dot ringing vs load-LPF (synthetic gust); CFD ref RMS~0.0018 xc~2',flush=True)
for L in [0.0,0.8,0.9,0.95,0.97,0.99]:
    ad=openrun(Ws,L); r,xc=ring(ad,0,1.12); rp,xcp=ring(ad,1.4,2.4); pk=np.max(np.abs(ad[t<=1.6]))
    print('  lpf=%.2f : during RMS=%.4g xc=%d | post RMS=%.4g | ad_peak=%.4g'%(L,r,xc,rp,pk),flush=True)
