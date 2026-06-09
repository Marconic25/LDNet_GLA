import numpy as np, structure, time
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S;TE=1.6
a=LDNetAero('models/latent_10');a.reset(dt=DT)
def res(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return [-q*cl-structure.K_H*xe[0], q*cm*C-structure.K_ALPHA*xe[1]]
xe=fsolve(res,[0.,0.]);x0=np.array([xe[0],0.,xe[1],0.])
cl0,cm0=a.predict(x0,0.,0.,U);CLt=float(cl0);Fyt=q*CLt;Mzt=q*float(cm0)*C
RR=DT*100.
def run(mode,W0,N=6):
    def gust(t): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
    a.reset(dt=DT);x=x0.copy();dprev=0.;clm=CLt;ctrl=None
    if mode=='mpc':
        ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=9,causal_basin=True,target_lpf=0.0,mpc_horizon=N,aero=a);ctrl.reset()
    tt=np.arange(0,TE+DT,DT);CL=np.zeros(len(tt));HD=np.zeros(len(tt));AD=np.zeros(len(tt));D=np.zeros(len(tt))
    for i,t in enumerate(tt):
        W=gust(t)
        if mode=='prop': d=float(np.clip(np.clip(10*(clm-CLt),-20,20),dprev-RR,dprev+RR))
        elif mode=='open': d=0.0
        else: d=ctrl.compute(x,W)
        dprev=d;D[i]=d
        cl,cm=a.predict(x,d,W,U);a.advance(x,d,W,U,DT);clm=float(cl)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz)
        CL[i]=cl;HD[i]=hdd;AD[i]=x[3];x=structure.step_rk4(x,Fy,Mz,DT)
    gw=tt<=1.6;return [np.max(np.abs(v[gw])) for v in (CL,HD,AD)]+[np.max(np.abs(D))]
print('  W0  ctrl   |  C_Lred  hddotred  adotred  flap', flush=True)
for W0 in [5,10,15,30]:
    o=run('open',W0)
    rp=run('prop',W0)
    print('%4d prop   | %+6.1f%%  %+7.1f%%  %+7.1f%%  %.1f'%(W0,(o[0]-rp[0])/o[0]*100,(o[1]-rp[1])/o[1]*100,(o[2]-rp[2])/o[2]*100,rp[3]),flush=True)
    rm=run('mpc',W0)
    print('%4d mpc-N6 | %+6.1f%%  %+7.1f%%  %+7.1f%%  %.1f'%(W0,(o[0]-rm[0])/o[0]*100,(o[1]-rm[1])/o[1]*100,(o[2]-rm[2])/o[2]*100,rm[3]),flush=True)
