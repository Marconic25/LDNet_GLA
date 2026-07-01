"""
Focused sweep: one-step, Q_alpha_ddot = Q*(adot_next - adot_curr)^2
Skip R=1e-4 (always explosive) and R=0.1 (useless). NGRID=7, NSTEPS=800.
"""
import numpy as np, csv
import structure
from ldnet_aero import LDNetAero
from controller import Controller

U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
DS  = '/work/u10677113/NACA2312/dataset_v5'
MD  = 'models_rollout/latent_10'
NGRID = 7
NSTEPS = 800
DELTA_MAX = 14.

SIMS = ['sim_A_025_test', 'sim_A_027_test']
QADS = [5, 10, 30, 100, 300, 1000]
RS   = [5e-4, 1e-3, 5e-3, 1e-2]

aero = LDNetAero(MD)

def load_sim(sim):
    d = list(csv.reader(open(f'{DS}/{sim}/structural_trajectory.csv')))[1:]
    d = np.array([[float(v) for v in r] for r in d])[::7]
    return np.array([d[0,1],d[0,2],d[0,3],d[0,4]]), d[:, 7]

def simulate(x0, W, ctrl):
    aero.reset(dt=DT)
    if ctrl is not None: ctrl.reset()
    x = x0.copy()
    ad, add, hdd, CL, de = [], [], [], [], []
    for i in range(min(NSTEPS, len(W))):
        dv = ctrl.compute(x, W_hat=float(W[i])) if ctrl is not None else 0.0
        cl,cm = aero.predict(x,dv,W[i],U); Fy=q*cl; Mz=q*cm*C
        dr = structure.rhs(x,Fy,Mz)
        aero.advance(x,dv,W[i],U,DT); x=structure.step_rk4(x,Fy,Mz,DT)
        ad.append(x[3]); add.append(dr[3]); hdd.append(dr[1])
        CL.append(float(cl)); de.append(dv)
    return [np.array(v) for v in (ad,add,hdd,CL,de)]

def peaks(ad,add,hdd,CL,de,trim,ref=None):
    pk = dict(ad=np.max(np.abs(ad)), add=np.max(np.abs(add)),
              hdd=np.max(np.abs(hdd)), CLexc=np.max(np.abs(CL-trim)),
              dmax=np.max(np.abs(de)))
    flag=''
    if np.any(np.isnan(CL)): flag+='NaN '
    if ref:
        for k in ('ad','add','hdd'):
            if pk[k]>3.0*ref[k]+1e-9: flag+=f'{k}!! '
    return pk, flag

print('=== Design A: Q_alpha_ddot sweep (NH=1) ===', flush=True)

for sim in SIMS:
    x0, W = load_sim(sim)
    aero.reset(dt=DT); clt,_=aero.predict(x0,0.,0.,U); CLTRIM=float(clt)
    r0 = simulate(x0, W, None); pk0,_=peaks(*r0,CLTRIM)
    print(f'\n########## {sim}  trim={CLTRIM:.3f} ##########', flush=True)
    print(f'  open: CLexc={pk0["CLexc"]:.4f} ad={pk0["ad"]:.4g} add={pk0["add"]:.4g} hdd={pk0["hdd"]:.4g}', flush=True)
    for qad in QADS:
        print(f'  --- Q_ad={qad:.0e} ---', flush=True)
        for r in RS:
            ctrl=Controller(aero_predict=aero.predict,U=U,dt=DT,
                Q_h=0.,Q_alpha=0.,Q_alpha_dot=float(qad),Q_CL=1.0,
                R=r,R_du=0.,target_lpf=0.,e_ref=0.,
                n_grid=NGRID,causal_basin=False,global_search=True,
                mpc_horizon=1,aero=None,C_L_trim=CLTRIM,
                Fy_trim=0.,Mz_trim=0.,delta_max=DELTA_MAX,delta_dot_max=300.)
            res=simulate(x0,W,ctrl); pk,flag=peaks(*res,CLTRIM,pk0)
            red=(pk0['CLexc']-pk['CLexc'])/pk0['CLexc']*100
            print(f'  R={r:<6g} CLexc={pk["CLexc"]:.3f}({red:+.0f}%) dmax={pk["dmax"]:.1f}'
                  f' | ad={pk["ad"]:.4g} add={pk["add"]:.3g} hdd={pk["hdd"]:.3g}'
                  f'  {"EXPLODE: "+flag if flag else "stable"}', flush=True)

print('\nSWEEP_DONE', flush=True)
