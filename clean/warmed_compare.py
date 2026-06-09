import numpy as np, structure
from ldnet_aero import LDNetAero
from controller import Controller, ProportionalController
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
CSV='/work/u10677113/NACA2312/dataset_v5/sim_A_000_train/structural_trajectory.csv'
a=LDNetAero('models/latent_10')
a.reset(dt=DT, warmup_csv=CSV)       # PHYSICAL warmed latent state
z_warm=a._z.copy(); print('z_warm=%.1f'%np.linalg.norm(z_warm),flush=True)
def trim():
    a._z=z_warm.copy()
    def rr(xe):
        cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U); return [-q*cl-structure.K_H*xe[0],q*cm*C-structure.K_ALPHA*xe[1]]
    xe=fsolve(rr,[0.,0.]); a._z=z_warm.copy(); cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return np.array([xe[0],0.,xe[1],0.]),float(cl),q*float(cl),q*float(cm)*C
x0,CLt,Fyt,Mzt=trim(); print('warmed trim: C_L=%.3f Fy=%.1f'%(CLt,Fyt),flush=True)
RR=DT*100.
def run(mode,W0=10.0):
    def gust(t): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
    a._z=z_warm.copy(); x=x0.copy();dprev=0.;clm=CLt;ctrl=None
    if mode=='prop': ctrl=ProportionalController(C_L_trim=CLt,gain=10.,dt=DT,delta_max=20.,delta_dot_max=100.)
    elif mode=='opt': ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=15,causal_basin=True,target_lpf=0.95)
    elif mode=='mpc': ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=9,causal_basin=True,target_lpf=0.,mpc_horizon=6,aero=a)
    if ctrl and hasattr(ctrl,'reset'): ctrl.reset()
    tt=np.arange(0,1.8+DT,DT);CL=np.zeros_like(tt);HD=np.zeros_like(tt);AD=np.zeros_like(tt);D=np.zeros_like(tt)
    for i,t in enumerate(tt):
        W=gust(t)
        if mode=='open': d=0.0
        elif mode=='prop': d=ctrl.compute(clm)
        else: d=ctrl.compute(x,W)
        dprev=d;D[i]=d
        cl,cm=a.predict(x,d,W,U);a.advance(x,d,W,U,DT);clm=float(cl)
        Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;_,hdd,_,_=structure.rhs(x,Fy,Mz)
        CL[i]=cl;HD[i]=hdd;AD[i]=x[3];x=structure.step_rk4(x,Fy,Mz,DT)
    gw=tt<=1.6
    # peak EXCURSION from trim for C_L (since absolute is near trim)
    return [np.max(np.abs(CL[gw]-CLt)),np.max(np.abs(HD[gw])),np.max(np.abs(AD[gw])),np.max(np.abs(D))]
o=run('open'); print('open: dC_L_pk=%.3f hddot=%.3f adot=%.4f'%(o[0],o[1],np.rad2deg(o[2])),flush=True)
for m in ['prop','opt','mpc']:
    r=run(m); print('  %-4s: dC_L %+6.1f%%  hddot %+6.1f%%  adot %+6.1f%%  flap %.1f'%(m,(o[0]-r[0])/o[0]*100,(o[1]-r[1])/o[1]*100,(o[2]-r[2])/o[2]*100,r[3]),flush=True)
