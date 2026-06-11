import numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import structure
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
MD='models_rollout/latent_10'
W0=float(os.environ.get('W0','10')); TG=float(os.environ.get('TG','1.0'))
TEND=float(os.environ.get('TEND','3.0')); N=int(round(TEND/DT))+1
NH=int(os.environ.get('NH','6')); NGRID=int(os.environ.get('NGRID','15'))
PLOT=os.environ.get('PLOT','0')=='1'
SCHED=os.environ.get('SCHED','0')=='1'
QAD=float(os.environ.get('QAD','100')); RW=float(os.environ.get('RW','0.01'))
LPF=float(os.environ.get('LPF','0.0'))   # (unused for MPC batch path)
DLPF=float(os.environ.get('DLPF','0.95')) # 1st-order low-pass on the flap command in the
# harness loop -> smooth (sinusoidal) delta that does not excite the 14.5 Hz pitch mode.
# The Controller's own target_lpf is ignored on the vectorized MPC path, so we filter here.
OUT=os.environ.get('OUT','results/mpc_gust')

def gust(t): return (W0/2.0)*(1-np.cos(2*np.pi*t/TG)) if (0<=t<=TG) else 0.0

def schedule(w0):
    # feedforward gain schedule on gust peak velocity (a gust sensor provides w0).
    # Characterized on synthetic gusts: Qad=30,R=1e-2,LPF=0.7 is the robust optimum
    # (+39..+57% across W0=10..30); aggressive R<=1e-3 explodes. Very weak gusts get
    # gentler gains (loads negligible, avoid needless flap chatter).
    if w0 <= 8: return 100.0, 1e-2
    else:       return  30.0, 1e-2

a=LDNetAero(MD); a.reset(dt=DT)
# CFD pre-gust trim state (identical across all dataset sims; consistent with z=0,
# the regime the rollout model was trained in). Using the fsolve trim instead creates
# a spurious startup transient.
X0=np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
clt,cmt=a.predict(X0,0.,0.,U); CLTRIM=float(clt)
tg=np.arange(N)*DT; Wt=np.array([gust(t) for t in tg])
if SCHED: QAD,RW=schedule(W0)

def simulate(use_ctrl):
    a.reset(dt=DT); ctrl=None
    if use_ctrl:
        ctrl=Controller(aero_predict=a.predict,U=U,dt=DT,Q_h=0.,Q_alpha=0.,Q_alpha_dot=QAD,
            Q_CL=1.0,R=RW,n_grid=NGRID,global_search=True,causal_basin=False,mpc_horizon=NH,aero=a,
            target_lpf=LPF,C_L_trim=CLTRIM,Fy_trim=0.,Mz_trim=0.,delta_max=14.,delta_dot_max=300.)
        ctrl.reset()
    x=X0.copy(); de_f=0.0; R={k:[] for k in ['h','hd','al','ad','hdd','add','de','CL','CM','Fy']}
    for i in range(N):
        if ctrl:
            de_raw = ctrl.compute(x,W_hat=float(Wt[i]))
            de_f = DLPF*de_f + (1.0-DLPF)*de_raw      # smooth (sinusoidal) flap
            de = de_f
            ctrl._delta_prev = de                     # keep the rate-limit consistent
        else:
            de = 0.0
        cl,cm=a.predict(x,de,Wt[i],U); Fy=q*cl; Mz=q*cm*C
        der=structure.rhs(x,Fy,Mz)
        a.advance(x,de,Wt[i],U,DT); x=structure.step_rk4(x,Fy,Mz,DT)
        for k,v in zip(['h','hd','al','ad','hdd','add','de','CL','CM','Fy'],
                       [x[0],x[1],x[2],x[3],der[1],der[3],de,float(cl),float(cm),Fy]): R[k].append(v)
    return {k:np.array(v) for k,v in R.items()}

OL=simulate(False); CL=simulate(True)
mw = tg <= (TG+0.5)
def pk(r,k): return np.max(np.abs(r[k][mw]))
exo=np.max(np.abs(OL['CL'][mw]-CLTRIM))
exc=np.max(np.abs(CL['CL'][mw]-CLTRIM))
flag=''
for k in ['ad','add','hdd']:
    if pk(CL,k) > 3.0*pk(OL,k)+1e-9: flag+=k+'!! '
print(f'W0={W0:.1f} Tg={TG:.2f} Qad={QAD:g} R={RW:g} | CLexc open={exo:.3f} closed={exc:.3f} ({(exo-exc)/exo*100:+.0f}%) '
      f'| flap_max={pk(CL,"de"):.1f} ad {pk(OL,"ad"):.3g}->{pk(CL,"ad"):.3g} hdd {pk(OL,"hdd"):.3g}->{pk(CL,"hdd"):.3g} '
      f'{"EXPLODE: "+flag if flag else "stable"}', flush=True)

if PLOT:
    rows=[('h','h [mm]',1e3),('hd','h_dot [mm/s]',1e3),('al','alpha [deg]',180/np.pi),
          ('ad','alpha_dot [deg/s]',180/np.pi),('CL','C_L [-]',1.0),('Fy','Fy [N]',1.0),
          ('CM','C_M [-]',1.0),(None,'W_gust [m/s]',1.0),('de','delta [deg]',1.0)]
    fig,ax=plt.subplots(len(rows),1,figsize=(7,13),sharex=True)
    fig.suptitle(f'GLA: open vs MPC closed loop  -  Wg0={W0:.1f} m/s, Tg={TG:.2f} s, U={U:.0f} m/s  (Qad={QAD:g}, R={RW:g})',fontsize=10)
    for j,(k,lab,sc) in enumerate(rows):
        if k is None:
            ax[j].plot(tg,Wt,color='tab:green',lw=1.5)
        else:
            ax[j].plot(tg,OL[k]*sc,color='steelblue',lw=1.5,label='open loop')
            ax[j].plot(tg,CL[k]*sc,color='crimson',lw=1.5,label='MPC closed')
        ax[j].axvspan(0,TG,color='lightblue',alpha=0.25)
        ax[j].set_ylabel(lab,fontsize=8); ax[j].grid(alpha=0.3)
        if j==0: ax[j].legend(fontsize=8,loc='upper right')
    ax[-1].set_xlabel('Time [s]'); ax[-1].set_xlim(0,TEND)
    plt.tight_layout(rect=[0,0,1,0.985])
    fn=f'{OUT}_W{int(round(W0))}_Tg{int(round(TG*100))}.png'
    fig.savefig(fn,dpi=120); print('  saved '+fn,flush=True)
