import numpy as np, structure
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S;TE=1.5
a=LDNetAero('models/latent_10');a.reset(dt=DT)
def res(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return [-q*cl-structure.K_H*xe[0], q*cm*C-structure.K_ALPHA*xe[1]]
xe=fsolve(res,[0.,0.]);x0=np.array([xe[0],0.,xe[1],0.])
cl0,cm0=a.predict(x0,0.,0.,U);CLt=float(cl0);Fyt=q*CLt;Mzt=q*float(cm0)*C
def run(W0):
    def gust(t): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
    a.reset(dt=DT);x=x0.copy()
    ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=15,causal_basin=True,target_lpf=0.95);ctrl.reset()
    tt=np.arange(0,TE+DT,DT);rows=[]
    for i,t in enumerate(tt):
        W=gust(t);d=ctrl.compute(x,W)
        cl,cm=a.predict(x,d,W,U);a.advance(x,d,W,U,DT)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;x=structure.step_rk4(x,Fy,Mz,DT)
        if i%25==0: rows.append((t,W,d,cl,np.rad2deg(x[3])))
    return rows
for W0 in [10,20]:
    print('=== W0=%d (trim C_L=%.3f) ===  t: W  delta  C_L  adot[deg/s]'%(W0,CLt),flush=True)
    for t,W,d,cl,ad in run(W0): print('  %.2f: %4.1f  %+6.2f  %.3f  %+.2f'%(t,W,d,cl,ad),flush=True)
