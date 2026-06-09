import numpy as np, structure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
from pathlib import Path
OUT=Path('results'); U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
a=LDNetAero('models/latent_10');a.reset(dt=DT)
def res(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return [-q*cl-structure.K_H*xe[0], q*cm*C-structure.K_ALPHA*xe[1]]
xe=fsolve(res,[0.,0.]);x0=np.array([xe[0],0.,xe[1],0.])
cl0,cm0=a.predict(x0,0.,0.,U);CLt=float(cl0);Fyt=q*CLt;Mzt=q*float(cm0)*C
def sim(mode,W0=20.0):
    def gust(t): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
    a.reset(dt=DT);x=x0.copy();ctrl=None
    if mode=='opt':
        ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=15,causal_basin=True,target_lpf=0.95);ctrl.reset()
    elif mode=='mpc':
        ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=9,causal_basin=True,target_lpf=0.0,mpc_horizon=6,aero=a);ctrl.reset()
    tt=np.arange(0,2.0+DT,DT);CL=np.zeros_like(tt);HD=np.zeros_like(tt);AD=np.zeros_like(tt);D=np.zeros_like(tt)
    for i,t in enumerate(tt):
        W=gust(t); d=0.0 if mode=='open' else ctrl.compute(x,W); D[i]=d
        cl,cm=a.predict(x,d,W,U);a.advance(x,d,W,U,DT)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz)
        CL[i]=cl;HD[i]=hdd;AD[i]=np.rad2deg(x[3]);x=structure.step_rk4(x,Fy,Mz,DT)
    return tt,CL,HD,AD,D
R={}
for m in ['open','opt','mpc']:
    R[m]=sim(m); print(m,'done',flush=True)
sty={'open':('gray','-','Open loop'),'opt':('darkorange','--','One-step optimal'),'mpc':('crimson','-','MPC (N=6)')}
fig,ax=plt.subplots(2,2,figsize=(13,8)); fig.suptitle('Strong gust W0=20 m/s: one-step optimal DIVERGES, MPC stays stable',fontsize=13)
for m in ['open','opt','mpc']:
    tt,CL,HD,AD,D=R[m];c,ls,lb=sty[m]
    ax[0,0].plot(tt,CL,color=c,ls=ls,lw=1.6,label=lb); ax[0,1].plot(tt,HD,color=c,ls=ls,lw=1.6,label=lb)
    ax[1,0].plot(tt,AD,color=c,ls=ls,lw=1.6,label=lb); ax[1,1].plot(tt,D,color=c,ls=ls,lw=1.6,label=lb)
ax[0,0].axhline(CLt,color='k',lw=.6,ls=':'); ax[0,0].set_ylabel('C_L'); ax[0,0].set_title('Lift coefficient')
ax[0,1].set_ylabel('h_ddot [m/s2]'); ax[0,1].set_title('Heave acceleration')
ax[1,0].set_ylabel('alpha_dot [deg/s]'); ax[1,0].set_title('Pitch rate (the resonance)')
ax[1,1].set_ylabel('delta [deg]'); ax[1,1].set_title('Flap deflection')
for a_ in ax.flat: a_.set_xlabel('t [s]'); a_.grid(alpha=.3); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'figD_W20_contrast.png',dpi=140); print('figD done',flush=True)
