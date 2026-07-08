import numpy as np, os, time
import structure
from ldnet_aero import LDNetAero
from controller import Controller, ProportionalController
from scipy.optimize import fsolve
# Optional extra pitch damping (the LDNet aero under-represents aerodynamic pitch damping,
# so the lightly-damped structural pitch mode rings; a modest bump smooths alpha_dot).
structure.D_ALPHA *= float(os.environ.get('DAMULT','1'))
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
MD='models_rollout/latent_10'

# --- model + CFD pre-gust trim state (regime-fixed, shared by every controller arm) ---
# Identical across all dataset sims; consistent with z=0, the regime the rollout model was
# trained in. Using the fsolve trim instead creates a spurious startup transient.
a=LDNetAero(MD); a.reset(dt=DT)
X0=np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
clt,cmt=a.predict(X0,0.,0.,U); CLTRIM=float(clt)

# Fixed reference weights for the *achieved-cost* score J (post-hoc, applied to every arm so
# open/prop/MPC are ranked on the SAME objective even though only the MPC optimizes it).
SCORE_W = dict(Q_CL=1.0, Q_ad=100.0, R=0.01)


def gust(t, W0, Tg):
    return (W0/2.0)*(1-np.cos(2*np.pi*t/Tg)) if (0<=t<=Tg) else 0.0


def schedule(w0, tg):
    # Returns (Q_alpha_dot, R, DLPF, R_du, NH, RQUIET, DMAX).
    # DLPF: 0.75 for strong or short gusts (need authority); 0.85 elsewhere (smoothness).
    # R_du=R only for medium-W + medium-Tg (over-shooting region). Others: R_du=0.
    # Weak+long (Tg>1.5, W<15): pitch reversal above ~1.7 deg → DMAX=1.7 hard limit.
    # RQUIET=R (no quiet zone): MPC naturally tracks CL bell shape and damps post-gust ring-down.
    if w0 <= 6: return 100.0, 1e-2, 0.85, 0.0, 6, 0.1, 14.0
    strong = w0 >= 25
    medium = 15 < w0 < 25
    D = 0.75 if (strong or tg <= 0.6) else 0.85
    DM = 14.0; QAD = 30.0
    if tg <= 0.6:
        R = 5e-3 if strong else 1e-3
        NH_s, RQ = 6, 0.1
    elif tg <= 1.5:
        R = 5e-4 if strong else (5e-3 if medium else 1e-3)
        NH_s, RQ = 6, 0.1
    else:  # slow gusts
        R = 1e-2 if strong else 1e-3
        NH_s, RQ = 6, 0.1
        if not (strong or medium):  # weak+long: amplitude-limit to below pitch-reversal threshold
            R = 1e-5; DM = 1.7; QAD = 100.0; RQ = 1e-5  # no quiet zone: CL-tracking + ring-down damping
        elif medium:  # medium+slow: RQUIET<RW inverts schedule → more aggressive at edges + ring-down damp
            RQ = 1e-5; QAD = 200.0
        elif strong:  # strong+slow: DLPF=0.85 reduces flap rate → less pitch coupling; RQUIET=1e-5 damps ring-down
            D = 0.85; RQ = 1e-5; QAD = 100.0
    R_du = R if (medium and 0.6 < tg <= 1.5) else 0.0
    return QAD, R, D, R_du, NH_s, RQ, DM


def simulate(mode, W0, Tg, gain=-40.0, TEND=3.0, NH=6, NGRID=15,
             QAD=100., RW=0.01, DLPF=0.95, RQUIET=0.1, LPF=0.0,
             ONESIDED=False, SCHED=False, prop_dlpf=True, DMAX=14.):
    """Run one closed-loop gust simulation in the physical (z=0, full-load rollout) regime.

    mode : 'open' (delta=0), 'prop' (model-free C_L feedback) or 'mpc' (receding-horizon).
    Returns the per-step trajectory dict (h,hd,al,ad,hdd,add,de,CL,CM,Fy) plus bookkeeping
    keys '_t' (time), '_Wt' (gust), '_comp_ms' (mean controller solve time, ms) and '_cfg'
    (the effective Qad/R/DLPF after scheduling).

    All arms share the SAME plant, IC, flap saturation (14 deg) / rate limit (300 deg/s) and
    flap-smoothing chain, so the comparison isolates the control law -- not the post-filter.
    The proportional gain is signed; in this regime dC_L/ddelta>0 so a *negative* gain reduces
    a positive C_L excursion (cf. taskC_gla).
    """
    N=int(round(TEND/DT))+1
    tg=np.arange(N)*DT; Wt=np.array([gust(t,W0,Tg) for t in tg])
    RDU = 0.0; NH_sim = NH; RQUIET_sim = RQUIET; DMAX_sim = DMAX
    if SCHED: QAD, RW, DLPF, RDU, NH_sim, RQUIET_sim, DMAX_sim = schedule(W0, Tg)

    a.reset(dt=DT); ctrl=None
    if mode=='mpc':
        ctrl=Controller(aero_predict=a.predict,U=U,dt=DT,Q_h=0.,Q_alpha=0.,Q_alpha_dot=QAD,
            Q_CL=1.0,R=RW,R_du=RDU,n_grid=NGRID,global_search=True,causal_basin=False,mpc_horizon=NH_sim,aero=a,
            target_lpf=LPF,C_L_trim=CLTRIM,Fy_trim=0.,Mz_trim=0.,delta_max=DMAX_sim,delta_dot_max=300.)
        ctrl.reset()
    elif mode=='prop':
        ctrl=ProportionalController(C_L_trim=CLTRIM,gain=gain,dt=DT,delta_max=14.,delta_dot_max=300.)
        ctrl.reset()

    x=X0.copy(); de_f=0.0; de_f2=0.0
    R={k:[] for k in ['h','hd','al','ad','hdd','add','de','CL','CM','Fy']}
    comp_t=0.0; comp_n=0
    for i in range(N):
        if mode=='mpc':
            wn = min(1.0, (Wt[i]/W0)/0.1) if W0 > 1e-6 else 0.0   # 1 while gust>10% peak, ramps to 0 at the tails
            ctrl.R = RQUIET_sim - (RQUIET_sim - RW)*wn      # aggressive THROUGHOUT the gust, gentle only after
            t0=time.perf_counter(); de_raw = ctrl.compute(x,W_hat=float(Wt[i])); comp_t+=time.perf_counter()-t0; comp_n+=1
            if ONESIDED: de_raw = min(de_raw, 0.0)         # lift-reducing flap only -> single bell-shaped delta
            de_f  = DLPF*de_f  + (1.0-DLPF)*de_raw     # 2nd-order (cascaded) low-pass -> C1-smooth flap
            de_f2 = DLPF*de_f2 + (1.0-DLPF)*de_f       # (no corners, rounded transitions / raccordi)
            de = de_f2
            ctrl._delta_prev = de                      # keep the rate-limit consistent
        elif mode=='prop':
            clm,_ = a.predict(x, de_f2, float(Wt[i]), U)   # measure C_L with the currently-applied flap
            t0=time.perf_counter(); de_raw = ctrl.compute(float(clm)); comp_t+=time.perf_counter()-t0; comp_n+=1
            if prop_dlpf:                                  # same smoothing chain as the MPC (fair post-filter)
                de_f  = DLPF*de_f  + (1.0-DLPF)*de_raw
                de_f2 = DLPF*de_f2 + (1.0-DLPF)*de_f
                de = de_f2
            else:
                de = de_raw; de_f2 = de_raw
            ctrl._delta_prev = de
        else:
            de = 0.0
        cl,cm=a.predict(x,de,Wt[i],U); Fy=q*cl; Mz=q*cm*C
        der=structure.rhs(x,Fy,Mz)
        a.advance(x,de,Wt[i],U,DT); x=structure.step_dp45(x,Fy,Mz,DT)
        for k,v in zip(['h','hd','al','ad','hdd','add','de','CL','CM','Fy'],
                       [x[0],x[1],x[2],x[3],der[1],der[3],de,float(cl),float(cm),Fy]): R[k].append(v)
    out={k:np.array(v) for k,v in R.items()}
    out['_t']=tg; out['_Wt']=Wt
    out['_comp_ms']=(comp_t/comp_n*1e3) if comp_n else 0.0
    out['_cfg']=dict(QAD=QAD, RW=RW, DLPF=DLPF, gain=gain)
    return out


# --------------------------- metrics (shared by the CLI and compare_grid) --------------------
def _win(t, Tg): return t <= (Tg+0.5)            # gust + ring-down window

def adrms(r, mw): return np.sqrt(np.mean((r['ad'][mw])**2))*180/np.pi   # alpha_dot RMS deg/s

def achieved_J(r, mw, w=SCORE_W):
    """Post-hoc objective score with FIXED weights, summed over the gust window. Lets us rank
    open/prop/MPC on the SAME J = Q_CL*sum(dC_L^2) + Q_ad*sum(alpha_dot^2) + R*sum(delta^2)."""
    dCL=r['CL'][mw]-CLTRIM; ad=r['ad'][mw]; de=r['de'][mw]
    return float(w['Q_CL']*np.sum(dCL**2) + w['Q_ad']*np.sum(ad**2) + w['R']*np.sum(de**2))

def metrics(r, r_open, Tg):
    """Per-arm metric dict vs the open-loop reference r_open (same W0,Tg)."""
    t=r['_t']; mw=_win(t,Tg)
    exo=float(np.max(np.abs(r_open['CL'][mw]-CLTRIM)))
    exc=float(np.max(np.abs(r['CL'][mw]-CLTRIM)))
    clred=(exo-exc)/exo*100.0 if exo>1e-12 else 0.0
    i0,i1=int(2.3/DT),int(2.7/DT)
    de_end=float(np.mean(np.abs(r['de'][i0:i1]))) if i1<=len(r['de']) else float('nan')
    flag=''
    for k in ['ad','add','hdd']:
        if np.max(np.abs(r[k][mw])) > 3.0*np.max(np.abs(r_open[k][mw]))+1e-9: flag+=k+'!'
    return dict(clexc=exc, exo=exo, clred=clred, adrms=adrms(r,mw),
                flap_max=float(np.max(np.abs(r['de'][mw]))), de_end=de_end,
                J=achieved_J(r,mw), comp_ms=float(r.get('_comp_ms',0.0)), flag=flag)


if __name__ == '__main__':
    W0=float(os.environ.get('W0','10')); TG=float(os.environ.get('TG','1.0'))
    TEND=float(os.environ.get('TEND','3.0'))
    NH=int(os.environ.get('NH','6')); NGRID=int(os.environ.get('NGRID','15'))
    PLOT=os.environ.get('PLOT','0')=='1'
    SCHED=os.environ.get('SCHED','0')=='1'
    ONESIDED=os.environ.get('ONESIDED','0')=='1'
    QAD=float(os.environ.get('QAD','100')); RW=float(os.environ.get('RW','0.01'))
    LPF=float(os.environ.get('LPF','0.0'))
    RQUIET=float(os.environ.get('RQUIET','0.1'))
    DLPF=float(os.environ.get('DLPF','0.95'))
    DMAX=float(os.environ.get('DMAX','14.'))
    OUT=os.environ.get('OUT','results/mpc_gust')

    kw=dict(TEND=TEND,NH=NH,NGRID=NGRID,QAD=QAD,RW=RW,DLPF=DLPF,RQUIET=RQUIET,LPF=LPF,
            ONESIDED=ONESIDED,SCHED=SCHED,DMAX=DMAX)
    OL=simulate('open',W0,TG,**kw); CL=simulate('mpc',W0,TG,**kw)
    cfg=CL['_cfg']; m=metrics(CL,OL,TG)
    print(f'W0={W0:.1f} Tg={TG:.2f} Qad={cfg["QAD"]:g} R={cfg["RW"]:g} DLPF={cfg["DLPF"]:g} ng={NGRID} | '
          f'CLexc {m["exo"]:.3f}->{m["clexc"]:.3f} ({m["clred"]:+.0f}%) '
          f'| flap_max={m["flap_max"]:.1f} de@2.5={m["de_end"]:.2f} | adot_RMS deg/s open={adrms(OL,_win(OL["_t"],TG)):.3f} '
          f'closed={m["adrms"]:.3f} {"EXPLODE: "+m["flag"] if m["flag"] else "stable"}', flush=True)
    gain=float(os.environ.get('GAIN','-40'))
    if os.environ.get('PROP','0')=='1':
        PL=simulate('prop',W0,TG,gain=gain,**kw); mp=metrics(PL,OL,TG)
        print(f'  PROP g={gain:g}: CLexc {mp["exo"]:.3f}->{mp["clexc"]:.3f} ({mp["clred"]:+.0f}%)'
              f' | flap_max={mp["flap_max"]:.1f} adot_RMS={mp["adrms"]:.3f}', flush=True)

    if PLOT:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        # prop arm for plot: always use standard DLPF=0.95, gain=-40 (fair baseline)
        prop_kw = {**kw, 'DLPF': 0.95}
        PL=simulate('prop',W0,TG,gain=gain,**prop_kw); mp=metrics(PL,OL,TG)
        tg=OL['_t']; Wt=OL['_Wt']
        rows=[('h','h [mm]',1e3),('hd','h_dot [mm/s]',1e3),('al','alpha [deg]',180/np.pi),
              ('ad','alpha_dot [deg/s]',180/np.pi),('CL','C_L [-]',1.0),('Fy','Fy [N]',1.0),
              ('CM','C_M [-]',1.0),(None,'W_gust [m/s]',1.0),('de','delta [deg]',1.0)]
        fig,ax=plt.subplots(len(rows),1,figsize=(7,13),sharex=True)
        fig.suptitle(
            f'GLA  Wg0={W0:.1f} m/s  Tg={TG:.2f} s  U={U:.0f} m/s\n'
            f'MPC: Qad={cfg["QAD"]:g} R={cfg["RW"]:g} DLPF={cfg["DLPF"]:g}  |  '
            f'MPC CLred={m["clred"]:+.0f}%   Prop CLred={mp["clred"]:+.0f}%', fontsize=9)
        for j,(k,lab,sc) in enumerate(rows):
            if k is None:
                ax[j].plot(tg,Wt,color='tab:green',lw=1.5)
            else:
                ax[j].plot(tg,OL[k]*sc,color='steelblue',lw=1.5,label='open loop')
                ax[j].plot(tg,CL[k]*sc,color='crimson',lw=1.5,label=f'MPC ({m["clred"]:+.0f}%)')
                ax[j].plot(tg,PL[k]*sc,color='darkorange',lw=1.2,ls='--',label=f'Prop g={gain:g} ({mp["clred"]:+.0f}%)')
            ax[j].axvspan(0,TG,color='lightblue',alpha=0.25)
            ax[j].set_ylabel(lab,fontsize=8); ax[j].grid(alpha=0.3)
            if j==0: ax[j].legend(fontsize=7,loc='upper right')
        ax[-1].set_xlabel('Time [s]'); ax[-1].set_xlim(0,TEND)
        plt.tight_layout(rect=[0,0,1,0.97])
        fn=f'{OUT}_W{int(round(W0))}_Tg{int(round(TG*100))}.png'
        fig.savefig(fn,dpi=120); print('  saved '+fn,flush=True)
