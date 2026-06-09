#!/usr/bin/env python3
"""Compare GLA controllers on the LDNet model (no gust observer).
  open        : flap fixed at 0
  prop        : delta = +G*(C_L_meas - C_L_trim)   (pure C_L feedback)
  opt_frozen  : model-based one-step optimizer, frozen-z prediction (current code)
  opt_zaware  : same optimizer but prediction advances a copy of z (predict_step)
True structural state is used (isolates the controller from state estimation).
"""
import numpy as np, structure
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve

U=80.; RHO=1.225; S=0.05; C=1.0; DT=0.002; q=0.5*RHO*U**2*S; TE=3.0
def gust(t): return (10/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
aero=LDNetAero('models/latent_10'); aero.reset(dt=DT)
def res(xe):
    cl,cm=aero.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return [-q*cl-structure.K_H*xe[0], q*cm*C-structure.K_ALPHA*xe[1]]
xe=fsolve(res,[0.,0.]); x0=np.array([xe[0],0.,xe[1],0.])
cl0,cm0=aero.predict(x0,0.,0.,U); CLt=float(cl0); Fyt=q*CLt; Mzt=q*float(cm0)*C
RR=DT*100.0  # per-step rate limit (delta_dot_max*DT)

def run(mode, G=10.0):
    aero.reset(dt=DT); x=x0.copy(); dprev=0.; clm=CLt
    ctrl=None
    if mode in ('opt_frozen','opt_global'):
        gs = (mode=='opt_global')
        ctrl=Controller(aero.predict,U=U,dt=DT,Q_h=1e4,Q_alpha=1e4,Q_alpha_dot=1e4,Q_CL=1e3,R=1.0,
                        delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=gs); ctrl.reset()
    tt=np.arange(0,TE+DT,DT); CL=np.zeros(len(tt)); HD=np.zeros(len(tt)); AL=np.zeros(len(tt)); D=np.zeros(len(tt))
    for i,t in enumerate(tt):
        W=gust(t)
        if mode=='open': d=0.0
        elif mode=='prop': d=float(np.clip(G*(clm-CLt),-20.,20.)); d=float(np.clip(d,dprev-RR,dprev+RR))
        else: d=ctrl.compute(x,W)
        dprev=d; D[i]=d
        cl,cm=aero.predict(x,d,W,U); aero.advance(x,d,W,U,DT); clm=float(cl)
        Fy=q*cl-Fyt; Mz=q*cm*C-Mzt; _,hdd,_,_=structure.rhs(x,Fy,Mz)
        CL[i]=cl; HD[i]=hdd; AL[i]=x[2]; x=structure.step_rk4(x,Fy,Mz,DT)
    gw=tt<=1.6
    return dict(cl=np.max(np.abs(CL[gw])), hd=np.max(np.abs(HD[gw])), al=np.max(np.abs(AL[gw])), dmax=np.max(np.abs(D)))

o=run('open')
print(f"{'mode':<12}{'C_L_pk':>9}{'red%':>8}{'hddot_pk':>10}{'red%':>8}{'a_pk[deg]':>11}{'red%':>8}{'flap_max':>10}")
def line(name,r):
    rc=(o['cl']-r['cl'])/o['cl']*100; rh=(o['hd']-r['hd'])/o['hd']*100; ra=(o['al']-r['al'])/o['al']*100
    print(f"{name:<12}{r['cl']:>9.3f}{rc:>8.1f}{r['hd']:>10.3f}{rh:>8.1f}{np.rad2deg(r['al']):>11.4f}{ra:>8.1f}{r['dmax']:>10.1f}")
line('open',o)
for m in ['prop','opt_frozen','opt_global']:
    line(m, run(m))
