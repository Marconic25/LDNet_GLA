import numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import structure
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
# Optional extra pitch damping (the LDNet aero under-represents aerodynamic pitch damping,
# so the lightly-damped structural pitch mode rings; a modest bump smooths alpha_dot).
structure.D_ALPHA *= float(os.environ.get('DAMULT','1'))
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
MD='models_rollout/latent_10'
W0=float(os.environ.get('W0','10')); TG=float(os.environ.get('TG','1.0'))
TEND=float(os.environ.get('TEND','3.0')); N=int(round(TEND/DT))+1
NH=int(os.environ.get('NH','6')); NGRID=int(os.environ.get('NGRID','15'))
PLOT=os.environ.get('PLOT','0')=='1'
SCHED=os.environ.get('SCHED','0')=='1'
ONESIDED=os.environ.get('ONESIDED','0')=='1'   # clamp flap to lift-reducing sign -> single bell delta
QAD=float(os.environ.get('QAD','100')); RW=float(os.environ.get('RW','0.01'))
LPF=float(os.environ.get('LPF','0.0'))   # (unused for MPC batch path)
RQUIET=float(os.environ.get('RQUIET','0.1')) # control penalty when the gust is gone: R is
# scheduled in time R(t)=RQUIET-(RQUIET-R)*W(t)/W0 -> aggressive at the gust peak, gentle when
# W->0 so the flap returns smoothly to 0 (no post-gust pulses) -> a smooth sinusoidal delta.
DLPF=float(os.environ.get('DLPF','0.95')) # 1st-order low-pass on the flap command in the
# harness loop -> smooth (sinusoidal) delta that does not excite the 14.5 Hz pitch mode.
# The Controller's own target_lpf is ignored on the vectorized MPC path, so we filter here.
OUT=os.environ.get('OUT','results/mpc_gust')

def gust(t): return (W0/2.0)*(1-np.cos(2*np.pi*t/TG)) if (0<=t<=TG) else 0.0

def schedule(w0, tg):
    # Feedforward schedule on gust peak velocity AND duration (gust sensor provides W0, Tg).
    # Returns (Q_alpha_dot, R, DLPF). R=1e-3 uses more flap amplitude; the flap-smoothing
    # DLPF is scaled to the gust timescale -> FAST flap for short gusts, smooth for long
    # ones (short needs speed; long needs smoothing or alpha_dot/h_ddot explode).
    if w0 <= 6: return 100.0, 1e-2, 0.9          # negligible gust: gentle, no needless flap
    if tg <= 0.6:   D, R = 0.6,  1e-3             # short/fast gust -> fast flap
    elif tg <= 1.5: D, R = 0.85, 1e-3            # medium gust
    else:           D, R = 0.9, (1e-2 if w0 >= 25 else 1e-3)  # long gust: gentler if strong
    return 30.0, R, D

a=LDNetAero(MD); a.reset(dt=DT)
# CFD pre-gust trim state (identical across all dataset sims; consistent with z=0,
# the regime the rollout model was trained in). Using the fsolve trim instead creates
# a spurious startup transient.
X0=np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
clt,cmt=a.predict(X0,0.,0.,U); CLTRIM=float(clt)
tg=np.arange(N)*DT; Wt=np.array([gust(t) for t in tg])
if SCHED: QAD,RW,DLPF=schedule(W0,TG)

def simulate(use_ctrl):
    a.reset(dt=DT); ctrl=None
    if use_ctrl:
        ctrl=Controller(aero_predict=a.predict,U=U,dt=DT,Q_h=0.,Q_alpha=0.,Q_alpha_dot=QAD,
            Q_CL=1.0,R=RW,n_grid=NGRID,global_search=True,causal_basin=False,mpc_horizon=NH,aero=a,
            target_lpf=LPF,C_L_trim=CLTRIM,Fy_trim=0.,Mz_trim=0.,delta_max=14.,delta_dot_max=300.)
        ctrl.reset()
    x=X0.copy(); de_f=0.0; de_f2=0.0; R={k:[] for k in ['h','hd','al','ad','hdd','add','de','CL','CM','Fy']}
    for i in range(N):
        if ctrl:
            wn = min(1.0, (Wt[i]/W0)/0.1) if W0 > 1e-6 else 0.0   # 1 while gust>10% peak, ramps to 0 at the tails
            ctrl.R = RQUIET - (RQUIET - RW)*wn             # aggressive THROUGHOUT the gust, gentle only after
            de_raw = ctrl.compute(x,W_hat=float(Wt[i]))
            if ONESIDED: de_raw = min(de_raw, 0.0)         # lift-reducing flap only -> single bell-shaped delta
            de_f  = DLPF*de_f  + (1.0-DLPF)*de_raw     # 2nd-order (cascaded) low-pass -> C1-smooth flap
            de_f2 = DLPF*de_f2 + (1.0-DLPF)*de_f       # (no corners, rounded transitions / raccordi)
            de = de_f2
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
def adrms(r): return np.sqrt(np.mean((r['ad'][mw])**2))*180/np.pi   # alpha_dot RMS deg/s, gust window
de_end=np.mean(np.abs(CL['de'][int(2.3/DT):int(2.7/DT)]))           # residual flap ~t=2.5 (should ->0)
flag=''
for k in ['ad','add','hdd']:
    if pk(CL,k) > 3.0*pk(OL,k)+1e-9: flag+=k+'!! '
print(f'W0={W0:.1f} Tg={TG:.2f} Qad={QAD:g} R={RW:g} DLPF={DLPF:g} ng={NGRID} | CLexc {exo:.3f}->{exc:.3f} ({(exo-exc)/exo*100:+.0f}%) '
      f'| flap_max={pk(CL,"de"):.1f} de@2.5={de_end:.2f} | adot_RMS deg/s open={adrms(OL):.3f} closed={adrms(CL):.3f} '
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
