import numpy as np, structure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
from pathlib import Path
import csv
OUT=Path('results'); OUT.mkdir(exist_ok=True)
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
a=LDNetAero('models/latent_10');a.reset(dt=DT)
def res(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return [-q*cl-structure.K_H*xe[0], q*cm*C-structure.K_ALPHA*xe[1]]
xe=fsolve(res,[0.,0.]);x0=np.array([xe[0],0.,xe[1],0.])
cl0,cm0=a.predict(x0,0.,0.,U);CLt=float(cl0);Fyt=q*CLt;Mzt=q*float(cm0)*C
RR=DT*100.

W0s=[5,10,15,20,30]
prop={'CL':[16.7,21.6,25.3,22.3,17.2],'hd':[27.0,31.4,35.5,38.9,44.0],'ad':[-8.8,3.8,1.6,-8.5,-38.9]}
one ={'CL':[19.1,23.8,27.3,6.5,-35.7],'hd':[27.0,31.4,35.5,-6.2,-59.3],'ad':[-0.5,11.1,9.1,-168.5,-466.5]}
mpc ={'CL':[21.9,26.4,29.8,28.2,32.1],'hd':[29.6,33.8,37.8,41.1,46.0],'ad':[20.4,29.5,27.9,20.4,25.2]}
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
for k,(key,ttl,lo) in enumerate([('CL','C_L peak reduction',-80),('hd','h_ddot peak reduction',-80),('ad','alpha_dot peak reduction',-200)]):
    ax[k].axhline(0,color='k',lw=0.8)
    ax[k].plot(W0s,prop[key],'o-',color='steelblue',lw=2,label='Proportional')
    ax[k].plot(W0s,one[key],'s--',color='darkorange',lw=2,label='One-step optimal')
    ax[k].plot(W0s,mpc[key],'^-',color='crimson',lw=2.4,label='MPC (N=6)')
    ax[k].set_xlabel('gust W0 [m/s]'); ax[k].set_ylabel('reduction [%]'); ax[k].set_title(ttl)
    ax[k].grid(alpha=.3); ax[k].set_ylim(lo,60)
ax[0].legend(fontsize=9)
fig.suptitle('GLA robustness vs gust velocity (higher=better; below 0 = worse than no control)',fontsize=12)
fig.tight_layout(); fig.savefig(OUT/'figA_robustness.png',dpi=140); plt.close(fig); print('figA done',flush=True)

def sim(mode,W0=10.0):
    def gust(t): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
    a.reset(dt=DT);x=x0.copy();dprev=0.;clm=CLt;ctrl=None
    if mode=='opt':
        ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=15,causal_basin=True,target_lpf=0.95);ctrl.reset()
    tt=np.arange(0,2.5+DT,DT);CL=np.zeros_like(tt);HD=np.zeros_like(tt);AD=np.zeros_like(tt);D=np.zeros_like(tt)
    for i,t in enumerate(tt):
        W=gust(t)
        if mode=='prop': d=float(np.clip(np.clip(10*(clm-CLt),-20,20),dprev-RR,dprev+RR))
        elif mode=='open': d=0.0
        else: d=ctrl.compute(x,W)
        dprev=d;D[i]=d
        cl,cm=a.predict(x,d,W,U);a.advance(x,d,W,U,DT);clm=float(cl)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz)
        CL[i]=cl;HD[i]=hdd;AD[i]=np.rad2deg(x[3]);x=structure.step_rk4(x,Fy,Mz,DT)
    return tt,CL,HD,AD,D
R={m:sim(m) for m in ['open','prop','opt']}
sty={'open':('gray','-','Open loop'),'prop':('steelblue','-','Proportional'),'opt':('crimson','--','One-step optimal')}
fig,ax=plt.subplots(2,2,figsize=(13,8)); fig.suptitle('Closed-loop response at W0=10 m/s',fontsize=13)
for m in ['open','prop','opt']:
    tt,CL,HD,AD,D=R[m];c,ls,lb=sty[m]
    ax[0,0].plot(tt,CL,color=c,ls=ls,label=lb); ax[0,1].plot(tt,HD,color=c,ls=ls,label=lb)
    ax[1,0].plot(tt,AD,color=c,ls=ls,label=lb); ax[1,1].plot(tt,D,color=c,ls=ls,label=lb)
ax[0,0].axhline(CLt,color='k',lw=.6,ls=':'); ax[0,0].set_ylabel('C_L'); ax[0,0].set_title('Lift coefficient')
ax[0,1].set_ylabel('h_ddot [m/s2]'); ax[0,1].set_title('Heave acceleration')
ax[1,0].set_ylabel('alpha_dot [deg/s]'); ax[1,0].set_title('Pitch rate')
ax[1,1].set_ylabel('delta [deg]'); ax[1,1].set_title('Flap deflection')
for a_ in ax.flat: a_.set_xlabel('t [s]'); a_.grid(alpha=.3); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'figB_timehist_W10.png',dpi=140); plt.close(fig); print('figB done',flush=True)

CSV='/work/u10677113/NACA2312/dataset_v5/sim_A_025_test/structural_trajectory.csv'
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(CSV)))[1:]])
stride=7; a.reset(dt=DT); QD=0.5*1.225*U**2*0.05; tt=[];Fp=[];Fr=[]
for i in range(0,len(d),stride):
    st=np.array([d[i,1],d[i,2],d[i,3],d[i,4]]);delta=d[i,8];W=d[i,7]
    cl,cm=a.predict(st,delta,W,U); Fp.append(cl*QD); Fr.append(d[i,5]); tt.append(d[i,0]); a.advance(st,delta,W,U,DT)
tt=np.array(tt);Fp=np.array(Fp);Fr=np.array(Fr)
nrmse=np.sqrt(np.mean((Fp-Fr)**2))/(Fr.max()-Fr.min())
fig,ax=plt.subplots(figsize=(11,4.5))
ax.plot(tt,Fr,color='steelblue',lw=1.5,label='CFD reference')
ax.plot(tt,Fp,color='crimson',lw=1.1,ls='--',label='LDNet')
ax.set_xlabel('t [s]'); ax.set_ylabel('F_y [N]'); ax.grid(alpha=.3); ax.legend()
ax.set_title('LDNet replay validation on sim_A_025_test (teacher-forced NRMSE F_y=%.3f)'%nrmse)
fig.tight_layout(); fig.savefig(OUT/'figC_replay.png',dpi=140); plt.close(fig); print('figC done nrmse=%.3f'%nrmse,flush=True)
