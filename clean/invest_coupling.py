import numpy as np, structure, json, os
from ldnet_aero import LDNetAero
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
def gust(t,W0=10.): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
MODELS=os.environ.get('MODELS','models_damped/latent_10,models_damped_l003_full/latent_10').split(',')
for MD in MODELS:
    lam=json.load(open(MD+'/config.json')).get('lambda_damp',0.0)
    a=LDNetAero(MD)
    print('=== %s (lambda=%.3f) ==='%(MD,lam),flush=True)
    # settle joint fixed point (no gust): relax z AND structure together
    a.reset(dt=DT); x=np.zeros(4)
    for _ in range(6000):
        cl,cm=a.predict(x,0.,0.,U); Fy=q*cl; Mz=q*cm*C
        a.advance(x,0.,0.,U,DT); x=structure.step_rk4(x,Fy,Mz,DT)
    xs=x.copy(); zs=a._z.copy()
    cls,cms=a.predict(xs,0.,0.,U)
    print('  joint fixed point: h=%.4f a=%.4f ||z||=%.1f  C_L=%.3f'%(xs[0],xs[2],np.linalg.norm(zs),float(cls)),flush=True)
    # residual h_ddot at fixed point (should be ~0 if truly settled)
    _,hdd0,_,add0=structure.rhs(xs,q*float(cls),q*float(cms)*C)
    print('  residual h_ddot=%.4f a_ddot=%.4f (settle check)'%(hdd0,add0),flush=True)
    # (1) DIRECT channel: at fixed xs,zs sweep W, C_L(W) (predict does not mutate z)
    a._z=zs.copy(); cl0,_=a.predict(xs,0.,0.,U); a._z=zs.copy(); cl10,_=a.predict(xs,0.,10.,U)
    print('  (1) DIRECT dC_L: C_L(W=0)=%.4f  C_L(W=10)=%.4f  delta=%.4f'%(float(cl0),float(cl10),float(cl10-cl0)),flush=True)
    # (2) INDIRECT channel: structure CLAMPED at xs, evolve z under gust, C_L(t)
    a._z=zs.copy(); clg=[]
    for t in np.arange(0,1.6+DT,DT):
        W=gust(t); cl,_=a.predict(xs,0.,W,U); clg.append(float(cl)); a.advance(xs,0.,W,U,DT)
    clg=np.array(clg); print('  (2) INDIRECT (struct clamped): C_L excursion via latent = %.4f'%(np.max(np.abs(clg-float(cls)))),flush=True)
    # (3) confound-free coupling: from joint fixed point, gust vs no-gust, both evolve
    def run(gon):
        a._z=zs.copy(); x=xs.copy(); HD=[];AD=[];CL=[]
        for t in np.arange(0,1.6+DT,DT):
            W=gust(t) if gon else 0.0
            cl,cm=a.predict(x,0.,W,U); Fy=q*cl;Mz=q*cm*C
            d=structure.rhs(x,Fy,Mz); HD.append(d[1]);AD.append(d[3]);CL.append(float(cl))
            a.advance(x,0.,W,U,DT); x=structure.step_rk4(x,Fy,Mz,DT)
        return np.array(CL),np.array(HD),np.array(AD)
    CLg,HDg,ADg=run(True); CLn,HDn,ADn=run(False)
    print('  (3) COUPLING (joint eq): C_L gust excursion=%.4f'%(np.max(np.abs(CLg-float(cls)))),flush=True)
    print('      h_ddot: gust peak=%.3f nogust peak=%.3f  GUST EFFECT=%.3f (%s)'%(np.max(np.abs(HDg)),np.max(np.abs(HDn)),np.max(np.abs(HDg-HDn)),'COUPLED' if np.max(np.abs(HDg-HDn))>0.5 else 'decoupled'),flush=True)
    print('      a_dot effect=%.4f'%(np.max(np.abs(ADg-ADn))),flush=True)
