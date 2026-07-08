import numpy as np, structure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ldnet_aero import LDNetAero
from scipy.optimize import fsolve
from pathlib import Path
OUT=Path('results'); U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
CSV='/work/u10677113/NACA2312/dataset_v5/sim_A_000_train/structural_trajectory.csv'
a=LDNetAero('models/latent_10'); a.reset(dt=DT, warmup_csv=CSV); z_warm=a._z.copy()
def rr(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U); return [-q*cl-structure.K_H*xe[0],q*cm*C-structure.K_ALPHA*xe[1]]
a._z=z_warm.copy(); xe=fsolve(rr,[0.,0.]); a._z=z_warm.copy()
x0=np.array([xe[0],0.,xe[1],0.]); clt,cmt=a.predict(x0,0.,0.,U); CLt=float(clt); Fyt=q*CLt; Mzt=q*float(cmt)*C
def gust(t): return (10/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
def run(wf):
    a._z=z_warm.copy(); x=x0.copy(); tt=np.arange(0,1.8+DT,DT); CL=[];HD=[];ZN=[];WW=[]
    for t in tt:
        W=wf(t); cl,cm=a.predict(x,0.,W,U); a.advance(x,0.,W,U,DT)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz)
        CL.append(cl);HD.append(hdd);ZN.append(np.linalg.norm(a._z));WW.append(W); x=structure.step_dp45(x,Fy,Mz,DT)
    return tt,np.array(CL),np.array(HD),np.array(ZN),np.array(WW)
tg,CLg,HDg,ZNg,WW=run(gust)
tn,CLn,HDn,ZNn,_=run(lambda t:0.0)
fig,ax=plt.subplots(2,2,figsize=(13,8))
fig.suptitle('Why GLA fails in the physical (warmed) regime: the latent state drifts unbounded',fontsize=13)
ax[0,0].plot(tg,CLg,'crimson',label='with W0=10 gust'); ax[0,0].plot(tn,CLn,'gray',ls='--',label='no gust (drift only)')
ax[0,0].axhline(CLt,color='k',lw=.6,ls=':'); ax[0,0].set_ylabel('C_L'); ax[0,0].set_title('Lift coefficient (gust DOES move C_L ~0.13)')
ax[0,1].plot(tg,HDg,'crimson',label='with gust'); ax[0,1].plot(tn,HDn,'gray',ls='--',label='no gust')
ax[0,1].set_ylabel('h_ddot [m/s2]'); ax[0,1].set_title('Heave accel: IDENTICAL w/ and w/o gust = startup transient, not gust')
ax[1,0].plot(tg,ZNg,'navy'); ax[1,0].set_ylabel('||z||'); ax[1,0].set_title('Latent norm: drifts +4.4/step, NO equilibrium')
ax[1,1].plot(tg,WW,'green'); ax[1,1].set_ylabel('W gust [m/s]'); ax[1,1].set_title('Gust input')
for a_ in ax.flat: a_.set_xlabel('t [s]'); a_.grid(alpha=.3); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'figE_latent_drift.png',dpi=140); print('figE done',flush=True)
