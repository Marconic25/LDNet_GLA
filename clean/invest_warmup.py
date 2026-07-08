import numpy as np, structure, csv
from ldnet_aero import LDNetAero
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
a=LDNetAero('models/latent_10')
# inspect the warmup trajectory
CSV='/work/u10677113/NACA2312/dataset_v5/sim_A_000_train/structural_trajectory.csv'
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(CSV)))[1:]])
print('sim_A_000_train: nrows=%d  W_gust[min,max]=[%.3f,%.3f]  delta[min,max]=[%.2f,%.2f]'%(len(d),d[:,7].min(),d[:,7].max(),d[:,8].min(),d[:,8].max()),flush=True)
print('  Fy[min,max]=[%.1f,%.1f]  (so this trajectory %s a gust)'%(d[:,5].min(),d[:,5].max(),'HAS' if d[:,7].max()>1 else 'has NO'),flush=True)
# z growth during warmup (replicate _warmup_from_csv stepping)
a.reset(dt=DT); dt_ref=a._dt_ref; raw_dt=float(d[1,0]-d[0,0]); stride=max(1,round(dt_ref/raw_dt))
zn=[]; idxs=list(range(0,len(d)-stride,stride))
for j,i in enumerate(idxs):
    h,hd,al,ad=d[i,1],d[i,2],d[i,3],d[i,4]; W,delta=d[i,7],d[i,8]
    sg=a._normalize_signals(h,hd,al,ad,delta,W); Un=a._normalize_U(80.)
    a._z=a._step_z(a._z,sg,Un,dt_ref); zn.append(np.linalg.norm(a._z))
zn=np.array(zn)
print('z_norm during warmup: start=%.2f  @25%%=%.1f  @50%%=%.1f  @100%%=%.1f'%(zn[0],zn[len(zn)//4],zn[len(zn)//2],zn[-1]),flush=True)
# gust sensitivity vs initial z: warm to N steps, then open-loop W0=10, measure C_L excursion
def gust(t): return (10/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
def warm_to(n):
    a.reset(dt=DT)
    for i in idxs[:n]:
        h,hd,al,ad=d[i,1],d[i,2],d[i,3],d[i,4]; W,delta=d[i,7],d[i,8]
        sg=a._normalize_signals(h,hd,al,ad,delta,W); Un=a._normalize_U(80.)
        a._z=a._step_z(a._z,sg,Un,a._dt_ref)
print('init-z sweep (open-loop W0=10 gust from each warmed z):',flush=True)
for n in [0,100,400,800,len(idxs)]:
    warm_to(n); zinit=np.linalg.norm(a._z)
    # trim at this z
    def rr(xe):
        cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U); return [-q*cl-structure.K_H*xe[0],q*cm*C-structure.K_ALPHA*xe[1]]
    z_keep=a._z.copy(); xe=fsolve(rr,[0.,0.]); a._z=z_keep
    x=np.array([xe[0],0.,xe[1],0.]); clt,_=a.predict(x,0.,0.,U); Fyt=q*float(clt)
    tt=np.arange(0,1.6+DT,DT); CL=[]
    for t in tt:
        W=gust(t); cl,cm=a.predict(x,0.,W,U); a.advance(x,0.,W,U,DT); CL.append(float(cl))
        Fy=q*cl-Fyt; Mz=q*cm*C; x=structure.step_dp45(x,Fy,Mz,DT)
    CL=np.array(CL); exc=CL.max()-float(clt)
    print('  warm=%4d steps  z_init=%6.1f  C_L_trim=%.3f  C_L_peak=%.3f  excursion=%.3f'%(n,zinit,float(clt),CL.max(),exc),flush=True)
