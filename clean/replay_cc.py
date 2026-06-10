import numpy as np, csv, json, os, glob
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; S=0.05; C=1.0; DT=0.002; q=0.5*RHO*U**2*S
DS='/work/u10677113/NACA2312/dataset_v5'
MD=os.environ.get('MD','models_damped/latent_10')
SIM=os.environ.get('SIM','sim_Cc_002_train')
def load(f):
    return np.array([[float(v) for v in r] for r in list(csv.reader(open(f)))[1:]])
d=load(f'{DS}/{SIM}/structural_trajectory.csv')
# cols: t,h,hd,alpha,ad,Fy,Mz,W_gust,delta  (idx 0..8)
a=LDNetAero(MD)
print(f'=== teacher-forced replay: {MD} on {SIM} ===',flush=True)
print(f'  true delta range [{d[:,8].min():.2f},{d[:,8].max():.2f}] deg, W_max={d[:,7].max():.1f}',flush=True)

def replay(use_true_delta):
    a.reset(dt=DT)
    Fp=[]; Fr=[]
    for i in range(0,len(d),7):
        st=np.array([d[i,1],d[i,2],d[i,3],d[i,4]])
        dl=d[i,8] if use_true_delta else 0.0
        cl,cm=a.predict(st, dl, d[i,7], U)
        Fp.append(cl*q); Fr.append(d[i,5])
        a.advance(st, dl, d[i,7], U, DT)
    return np.array(Fp), np.array(Fr)

Fp,Fr=replay(True)
nrmse=np.sqrt(np.mean((Fp-Fr)**2))/(Fr.max()-Fr.min())
CLp=Fp/q; CLr=Fr/q; trim=CLr[:20].mean()
print(f'  [true delta]  replay NRMSE_Fy={nrmse:.3f}  pred C_L peak-exc={np.max(CLp)-trim:.3f}  true C_L peak-exc={np.max(CLr)-trim:.3f}',flush=True)

# Ablate flap: feed delta=0, keep true states+W. If model learned flap GLA,
# the predicted C_L peak should RISE (flap was suppressing it).
Fp0,_=replay(False)
CLp0=Fp0/q
print(f'  [delta=0  ]  pred C_L peak-exc={np.max(CLp0)-trim:.3f}  (vs {np.max(CLp)-trim:.3f} with true flap)',flush=True)
print(f'  => flap effect on predicted peak C_L: {(np.max(CLp0)-np.max(CLp)):+.3f}  ({"model SEES flap alleviation" if np.max(CLp0)>np.max(CLp)+0.005 else "model does NOT capture flap effect"})',flush=True)
