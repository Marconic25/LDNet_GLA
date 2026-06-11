import numpy as np
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
MD='models_rollout/latent_10'
X0=np.array([-6.49179e-3,0.,-8.76338e-4,0.])
a=LDNetAero(MD); a.reset(dt=DT)
TG=1.12; W0=11.46; N=1500; t=np.arange(N)*DT
W=np.array([(W0/2)*(1-np.cos(2*np.pi*tt/TG)) if 0<=tt<=TG else 0. for tt in t])
clt,_=a.predict(X0,0.,0.,U); CLTRIM=float(clt)
print('OPEN-LOOP (no control): C_L and ||z|| drift over time. trim C_L=%.3f'%CLTRIM,flush=True)
x=X0.copy(); CL=[];Z=[]
for i in range(N):
    cl,cm=a.predict(x,0.,W[i],U); CL.append(float(cl)); Z.append(np.linalg.norm(a._z))
    a.advance(x,0.,W[i],U,DT); x=structure.step_rk4(x,q*cl,q*cm*C,DT)
CL=np.array(CL);Z=np.array(Z)
for tt in [0.0,0.5,1.12,1.6,2.0,2.5,3.0-DT]:
    i=int(round(tt/DT)); print('  t=%.2f: C_L=%.4f (dev from trim %+.4f)  ||z||=%.1f'%(tt,CL[i],CL[i]-CLTRIM,Z[i]),flush=True)
# detect glitches: max |dC_L/step| after the gust
post=slice(int(1.6/DT),N); dcl=np.abs(np.diff(CL[post]))
print('  post-gust(>1.6s): max |dC_L/step|=%.4g at t=%.2f ; C_L range [%.3f,%.3f]'%(
    dcl.max(), 1.6+DT*np.argmax(dcl), CL[post].min(), CL[post].max()),flush=True)
