import numpy as np, structure
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S;TE=2.0
a=LDNetAero('models/latent_10');a.reset(dt=DT)
def res(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return [-q*cl-structure.K_H*xe[0], q*cm*C-structure.K_ALPHA*xe[1]]
xe=fsolve(res,[0.,0.]);x0=np.array([xe[0],0.,xe[1],0.])
cl0,cm0=a.predict(x0,0.,0.,U);CLt=float(cl0);Fyt=q*CLt;Mzt=q*float(cm0)*C
def run(W0,lpf):
    def gust(t): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
    a.reset(dt=DT);x=x0.copy()
    ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=15,causal_basin=True,target_lpf=lpf);ctrl.reset()
    tt=np.arange(0,TE+DT,DT);CL=np.zeros(len(tt));HD=np.zeros(len(tt));AD=np.zeros(len(tt))
    o_cl=o_hd=o_ad=0
    for i,t in enumerate(tt):
        W=gust(t);d=ctrl.compute(x,W)
        cl,cm=a.predict(x,d,W,U);a.advance(x,d,W,U,DT)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz)
        CL[i]=cl;HD[i]=hdd;AD[i]=x[3];x=structure.step_rk4(x,Fy,Mz,DT)
    gw=tt<=1.6;return np.max(np.abs(CL[gw])),np.max(np.abs(HD[gw])),np.rad2deg(np.max(np.abs(AD[gw])))
# open-loop refs
def openrun(W0):
    def gust(t): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
    a.reset(dt=DT);x=x0.copy();tt=np.arange(0,TE+DT,DT);CL=np.zeros(len(tt));HD=np.zeros(len(tt));AD=np.zeros(len(tt))
    for i,t in enumerate(tt):
        W=gust(t);cl,cm=a.predict(x,0.,W,U);a.advance(x,0.,W,U,DT);Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz);CL[i]=cl;HD[i]=hdd;AD[i]=x[3];x=structure.step_rk4(x,Fy,Mz,DT)
    gw=tt<=1.6;return np.max(np.abs(CL[gw])),np.max(np.abs(HD[gw])),np.rad2deg(np.max(np.abs(AD[gw])))
for W0 in [20,30]:
    oc,oh,oa=openrun(W0)
    for lpf in [0.95,0.98,0.99,0.995]:
        c,h,ad=run(W0,lpf)
        print('W0=%d lpf=%.3f | C_L %+6.1f%%  hddot %+6.1f%%  adot %+7.1f%%'%(W0,lpf,(oc-c)/oc*100,(oh-h)/oh*100,(oa-ad)/oa*100), flush=True)
