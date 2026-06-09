import numpy as np, structure
from ldnet_aero import LDNetAero
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
CSV='/work/u10677113/NACA2312/dataset_v5/sim_A_000_train/structural_trajectory.csv'
a=LDNetAero('models/latent_10'); a.reset(dt=DT, warmup_csv=CSV); z_warm=a._z.copy()
def rr(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U); return [-q*cl-structure.K_H*xe[0],q*cm*C-structure.K_ALPHA*xe[1]]
a._z=z_warm.copy(); xe=fsolve(rr,[0.,0.]); a._z=z_warm.copy()
x0=np.array([xe[0],0.,xe[1],0.]); clt,cmt=a.predict(x0,0.,0.,U); CLt=float(clt); Fyt=q*CLt; Mzt=q*float(cmt)*C
def gust(t): return (10/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
# open-loop from warmed trim; also a NO-GUST run to isolate the startup/drift transient
for label,wf in [('WITH gust W0=10',gust),('NO gust (drift only)',lambda t:0.0)]:
    a._z=z_warm.copy(); x=x0.copy(); print('=== %s ===  t: C_L  h_ddot  adot[deg/s]  z_norm'%label,flush=True)
    tt=np.arange(0,1.6+DT,DT)
    for i,t in enumerate(tt):
        W=wf(t); cl,cm=a.predict(x,0.,W,U); a.advance(x,0.,W,U,DT)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz)
        if i%100==0: print('  %.2f: %.3f  %+7.2f  %+6.2f  %.0f'%(t,cl,hdd,np.rad2deg(x[3]),np.linalg.norm(a._z)),flush=True)
        x=structure.step_rk4(x,Fy,Mz,DT)
