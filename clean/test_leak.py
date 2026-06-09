import numpy as np, csv
from ldnet_aero import LDNetAero
U=80.;DT=0.002;QD=0.5*1.225*U**2*0.05
CSV='/work/u10677113/NACA2312/dataset_v5/sim_A_025_test/structural_trajectory.csv'
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(CSV)))[1:]])
a=LDNetAero('models/latent_10')
def replay(leak):
    a.reset(dt=DT); a._z_leak=leak; stride=7; Fp=[];Fr=[];zn=[]
    for i in range(0,len(d),stride):
        st=np.array([d[i,1],d[i,2],d[i,3],d[i,4]]); cl,cm=a.predict(st,d[i,8],d[i,7],U)
        Fp.append(cl*QD); Fr.append(d[i,5]); a.advance(st,d[i,8],d[i,7],U,DT); zn.append(np.linalg.norm(a._z))
    Fp=np.array(Fp);Fr=np.array(Fr)
    return np.sqrt(np.mean((Fp-Fr)**2))/(Fr.max()-Fr.min()), max(zn)
for leak in [0.0, 0.001, 0.003, 0.01, 0.03]:
    nr,zmax=replay(leak); print('leak=%.3f : replay NRMSE_F_y=%.3f   z_norm_max=%.0f'%(leak,nr,zmax),flush=True)
