import numpy as np, csv, os
import structure
from ldnet_aero import LDNetAero
from controller import Controller
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
DS='/work/u10677113/NACA2312/dataset_v5'; MD='models_rollout/latent_10'
SIM=os.environ.get('SIM','sim_A_025_test')
NSTEPS=int(os.environ.get('NSTEPS','1000'))   # truncate (gust window ~ first 1.6s); full=1592
NGRID=int(os.environ.get('NGRID','15'))
NH=int(os.environ.get('NH','1'))   # mpc_horizon (>1 = receding-horizon MPC)
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(f'{DS}/{SIM}/structural_trajectory.csv')))[1:]])
d=d[range(0,len(d),7)]; W=d[:,7]; t=d[:,0]; N=min(NSTEPS,len(d)); x0=np.array([d[0,1],d[0,2],d[0,3],d[0,4]])
a=LDNetAero(MD); a.reset(dt=DT); clt,_=a.predict(x0,0.,0.,U); CLTRIM=float(clt)
tt=t[:N]; m=tt<=1.6

def simulate(ctrl):
    a.reset(dt=DT)
    if ctrl is not None: ctrl.reset()
    x=x0.copy(); R={k:[] for k in ['h','hd','al','ad','hdd','add','de','CL']}
    for i in range(N):
        de = ctrl.compute(x, W_hat=float(W[i])) if ctrl is not None else 0.0
        cl,cm=a.predict(x,de,W[i],U); Fy=q*cl; Mz=q*cm*C
        der=structure.rhs(x,Fy,Mz)        # [hd, hdd, ad, add]
        a.advance(x,de,W[i],U,DT); x=structure.step_dp45(x,Fy,Mz,DT)
        R['h'].append(x[0]);R['hd'].append(x[1]);R['al'].append(x[2]);R['ad'].append(x[3])
        R['hdd'].append(der[1]);R['add'].append(der[3]);R['de'].append(de);R['CL'].append(float(cl))
    return {k:np.array(v) for k,v in R.items()}

def mkctrl(Rw, QH=0., QA=0., QAD=0.):
    return Controller(aero_predict=a.predict, U=U, dt=DT, Q_h=QH, Q_alpha=QA, Q_alpha_dot=QAD,
        Q_CL=1.0, R=Rw, R_du=0., target_lpf=0., e_ref=0., lpf_max=0., n_grid=NGRID,
        causal_basin=False, global_search=True, mpc_horizon=NH, aero=(a if NH>1 else None),
        C_L_trim=CLTRIM, Fy_trim=0., Mz_trim=0., delta_max=14., delta_dot_max=300.)

def peaks(r, ref=None):
    pk={k: np.max(np.abs(r[k][m])) for k in ['h','hd','al','ad','hdd','add']}
    pk['CLexc']=np.max(np.abs(r['CL'][m]-CLTRIM)); pk['dmax']=np.max(np.abs(r['de'][m]))
    # explosion flags: >3x open-loop peak, or NaN, or late-window growth
    flag=''
    if np.any(np.isnan(r['CL'])): flag+='NaN '
    if ref is not None:
        for k in ['ad','add','hdd']:
            if pk[k] > 3.0*ref[k]+1e-9: flag+=f'{k}!! '
    return pk, flag

print(f'=== {"MPC H="+str(NH) if NH>1 else "one-step OPTIMAL"} (Q_CL=1) on {SIM}  trim={CLTRIM:.3f}  N={N} grid={NGRID} ===',flush=True)
ro=simulate(None); pko,_=peaks(ro)
print('  open: '+' '.join(f'{k}={pko[k]:.4g}' for k in ['CLexc','h','al','ad','hdd','add']),flush=True)
QH=float(os.environ.get('QH','0')); QA=float(os.environ.get('QA','0')); QAD=float(os.environ.get('QAD','0'))
tag = f'(Q_h={QH:.0e},Q_a={QA:.0e},Q_ad={QAD:.0e})' if (QH>0 or QA>0 or QAD>0) else '(minimal: Q_CL+R only)'
print(f'  --- {tag} ---',flush=True)
for Rw in [float(x) for x in os.environ.get('RS','0.01,0.1,1,10,100').split(',')]:
    r=simulate(mkctrl(Rw,QH,QA,QAD)); pk,flag=peaks(r,pko)
    red=(pko['CLexc']-pk['CLexc'])/pko['CLexc']*100
    print(f'  R={Rw:<6g} CLexc={pk["CLexc"]:.3f}({red:+.0f}%) dmax={pk["dmax"]:.1f} | h={pk["h"]:.4g} al={pk["al"]:.4g} ad={pk["ad"]:.3g} hdd={pk["hdd"]:.3g} add={pk["add"]:.3g}  {("EXPLODE: "+flag) if flag else "stable"}',flush=True)
