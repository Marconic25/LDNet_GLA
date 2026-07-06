"""
Closed-loop GLA simulation harness.

Two arms:
  'open'    — no control (delta = 0), open-loop reference
  'optimal' — final one-step optimal controller (wnext + refine, optimal.py),
              receives the true gust W(t) and its next sample W(t+dt)

Both arms share the same plant, initial condition, flap saturation, rate limit
and optional 2nd-order LPF smoothing chain, so comparisons isolate the control
law.

Usage (cluster):
    DAMULT=3 W0=30 TG=0.4 python3 -s -u run.py
"""
import numpy as np, os, time
import structure
from ldnet_aero import LDNetAero
from optimal import OptimalController

structure.D_ALPHA *= float(os.environ.get('DAMULT', '1'))

U = 80.; RHO = 1.225; C = 1.0; DT = 0.002; S = 0.05; q = 0.5*RHO*U**2*S
MD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  '..', 'clean', 'models_rollout', 'latent_10')

aero = LDNetAero(MD); aero.reset(dt=DT)
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
CLTRIM = float(aero.predict(X0, 0., 0., U)[0])

SCORE_W = dict(Q_CL=1.0, Q_ad=100.0, R=0.01)


# ---------------------------------------------------------------------------
# Gust profile
# ---------------------------------------------------------------------------

def gust(t, W0, Tg):
    """1-cosine gust: W(t) = (W0/2)(1 - cos(2pi t / Tg)) for 0 <= t <= Tg."""
    return (W0/2.0)*(1 - np.cos(2*np.pi*t/Tg)) if (0 <= t <= Tg) else 0.0


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def simulate(mode, W0, Tg, TEND=3.0, R=3e-4, DLPF=0.0, DMAX=14., NGRID=161):
    """
    Run one closed-loop gust simulation.

    mode    : 'open' | 'optimal'
    Returns : trajectory dict with keys h, hd, al, ad, hdd, add, de, CL, CM, Fy,
              plus _t, _Wt, _comp_ms, _cfg.
    """
    N  = int(round(TEND/DT)) + 1
    ts = np.arange(N)*DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])

    aero.reset(dt=DT)
    ctrl = None
    if mode == 'optimal':
        ctrl = OptimalController(
            aero, U=U, dt=DT, R=R, n_grid=NGRID,
            C_L_trim=CLTRIM, delta_max=DMAX, delta_dot_max=300.)
        ctrl.reset()

    x = X0.copy(); de_f = 0.0; de_f2 = 0.0
    rec = {k: [] for k in ['h','hd','al','ad','hdd','add','de','CL','CM','Fy']}
    comp_t = 0.0; comp_n = 0

    for i in range(N):
        Wi = float(Wt[i])
        Wn = float(Wt[i+1]) if i + 1 < N else 0.0

        if mode == 'optimal':
            t0 = time.perf_counter()
            de_raw = ctrl.compute(x, Wi, Wn)
            comp_t += time.perf_counter()-t0; comp_n += 1
        else:
            de_raw = 0.0

        # 2nd-order LPF smoothing (only when DLPF > 0)
        if mode != 'open':
            if DLPF > 0.0:
                de_f  = DLPF*de_f  + (1.0-DLPF)*de_raw
                de_f2 = DLPF*de_f2 + (1.0-DLPF)*de_f
                de = de_f2
            else:
                de = de_raw
            ctrl._delta_prev = de
        else:
            de = 0.0

        cl, cm = aero.predict(x, de, Wi, U)
        Fy = q*cl; Mz = q*cm*C
        der = structure.rhs(x, Fy, Mz)
        aero.advance(x, de, Wi, U, DT)
        x = structure.step_rk4(x, Fy, Mz, DT)
        for k, v in zip(['h','hd','al','ad','hdd','add','de','CL','CM','Fy'],
                        [x[0],x[1],x[2],x[3],der[1],der[3],de,float(cl),float(cm),Fy]):
            rec[k].append(v)

    out = {k: np.array(v) for k, v in rec.items()}
    out['_t']       = ts
    out['_Wt']      = Wt
    out['_comp_ms'] = (comp_t/comp_n*1e3) if comp_n else 0.0
    out['_cfg']     = dict(R=R, DLPF=DLPF)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _gust_window(t, Tg):
    return t <= (Tg + 0.5)

def metrics(r, r_open, Tg):
    """Metric dict for one arm vs the open-loop reference."""
    t  = r['_t']; mw = _gust_window(t, Tg)
    exo = float(np.max(np.abs(r_open['CL'][mw] - CLTRIM)))
    exc = float(np.max(np.abs(r['CL'][mw]      - CLTRIM)))
    clred = (exo - exc)/exo*100.0 if exo > 1e-12 else 0.0
    i0, i1 = int(2.3/DT), int(2.7/DT)
    de_end = float(np.mean(np.abs(r['de'][i0:i1]))) if i1 <= len(r['de']) else float('nan')
    flag = ''
    for k in ['ad','add','hdd']:
        if np.max(np.abs(r[k][mw])) > 3.0*np.max(np.abs(r_open[k][mw]))+1e-9:
            flag += k+'!'
    adrms = float(np.sqrt(np.mean(r['ad'][mw]**2))*180/np.pi)
    pitchpk = float(np.max(np.abs(r['al'][mw])))
    dCL = r['CL'][mw]-CLTRIM; ad = r['ad'][mw]; de = r['de'][mw]
    J = float(SCORE_W['Q_CL']*np.sum(dCL**2)
              + SCORE_W['Q_ad']*np.sum(ad**2)
              + SCORE_W['R']*np.sum(de**2))
    return dict(clexc=exc, exo=exo, clred=clred, adrms=adrms,
                flap_max=float(np.max(np.abs(r['de'][mw]))),
                de_end=de_end, J=J, pitchpk=pitchpk,
                comp_ms=float(r.get('_comp_ms', 0.0)), flag=flag)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    W0    = float(os.environ.get('W0',    '30'))
    TG    = float(os.environ.get('TG',    '0.4'))
    TEND  = float(os.environ.get('TEND',  '3.0'))
    RW    = float(os.environ.get('RW',    '3e-4'))
    DLPF  = float(os.environ.get('DLPF',  '0.0'))
    DMAX  = float(os.environ.get('DMAX',  '14.'))

    kw = dict(TEND=TEND, R=RW, DLPF=DLPF, DMAX=DMAX)

    OL  = simulate('open',    W0, TG, **kw)
    OPT = simulate('optimal', W0, TG, **kw)
    mo  = metrics(OPT, OL, TG)
    cfg = OPT['_cfg']

    print(f'W0={W0:.1f} Tg={TG:.2f} | R={cfg["R"]:g} DLPF={cfg["DLPF"]:g}', flush=True)
    print(f'  optimal : CLexc {mo["exo"]:.3f}->{mo["clexc"]:.3f} ({mo["clred"]:+.0f}%)'
          f'  flap_max={mo["flap_max"]:.1f}  adot_RMS={mo["adrms"]:.3f}'
          f'  {"EXPLODE: "+mo["flag"] if mo["flag"] else "stable"}', flush=True)
