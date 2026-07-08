"""
Noise-study harness: scalar B=1 rollout where the PLANT advances with the true
gust W(t) while the CONTROLLER receives a corrupted gust signal Wc(t).

light/run.py cannot express this separation (both arms see the true W), so this
module replicates its loop (== 76/preview_study.py::rollout_customW) importing
the physics and the controller from light/ — nothing is duplicated or modified:
structure.py, ldnet_aero.py, optimal.py are the verified originals.

Controller API used by rollout():
    ctrl.reset()
    ctrl.compute(state, W_true, Wc) -> delta [deg]
light/optimal.py's OptimalController matches positionally (state, W, W_next):
with use_wnext=True it reads only W_next = Wc, so the corrupted preview drives
BOTH the candidate scan and the causal gate — the controller has no clean W.
Reference laws in controllers_ref.py follow the same signature and document
which of (W_true, Wc) each one is allowed to read.

DAMULT must be in the environment BEFORE importing this module (structure.
D_ALPHA is scaled here exactly once; run.py is NOT imported to avoid scaling
twice).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import structure
from ldnet_aero import LDNetAero
from optimal import OptimalController  # noqa: F401  (re-exported for the axis scripts)

structure.D_ALPHA *= float(os.environ.get('DAMULT', '1'))

# ---- constants (identical to light/run.py) ----------------------------------
U    = 80.0
RHO  = 1.225
C    = 1.0
DT   = 0.002
S    = 0.05
q    = 0.5 * RHO * U**2 * S
DMAX = 14.0
DDOT_MAX = 300.0

MD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  '..', '..', 'clean', 'models_rollout', 'latent_10')
if not os.path.isdir(MD):
    raise FileNotFoundError(f"LDNet model dir not found: {MD} "
                            "(expected <repo>/clean/models_rollout/latent_10)")

X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])

aero = LDNetAero(MD); aero.reset(dt=DT)
LAM = float(aero._z_leak)
CLTRIM = float(aero.predict(X0, 0., 0., U)[0])

SCORE_W = dict(Q_CL=1.0, Q_ad=100.0, R=0.01)


# ---------------------------------------------------------------------------
def gust(t, W0, Tg):
    """1-cosine gust: W(t) = (W0/2)(1 - cos(2pi t / Tg)) for 0 <= t <= Tg."""
    return (W0 / 2.0) * (1 - np.cos(2 * np.pi * t / Tg)) if (0 <= t <= Tg) else 0.0


def gust_array(W0, Tg, TEND=3.0):
    N = int(round(TEND / DT)) + 1
    ts = np.arange(N) * DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])
    return ts, Wt


def make_optimal(R=3e-4, refine=True):
    """Fresh final controller (defaults of light/optimal.py untouched)."""
    return OptimalController(aero, U=U, dt=DT, R=R, C_L_trim=CLTRIM,
                             delta_max=DMAX, delta_dot_max=DDOT_MAX,
                             n_grid=161, use_wnext=True, refine=refine)


# ---------------------------------------------------------------------------
def rollout(ctrl, W0, Tg, wc_fun=None, TEND=3.0):
    """
    One closed-loop simulation. ctrl=None -> open loop.

    wc_fun(i, Wt, N) -> the gust signal Wc the controller sees at step i
    (each axis script defines what Wc means: noisy preview W(t+dt), noisy
    W(t), filtered estimate, ...). Default: oracle preview Wt[i+1].
    The plant ALWAYS uses the true Wt[i].
    """
    ts, Wt = gust_array(W0, Tg, TEND)
    N = len(ts)
    aero.reset(dt=DT)
    if ctrl is not None:
        ctrl.reset()
    if wc_fun is None:
        wc_fun = lambda i, Wt, N: Wt[min(i + 1, N - 1)]

    x = X0.copy()
    rec = {k: [] for k in ['h','hd','al','ad','hdd','add','de','CL','CM','Fy']}
    Wc_seen = np.zeros(N)
    comp_t = 0.0; comp_n = 0

    for i in range(N):
        Wi = float(Wt[i])
        Wc = float(wc_fun(i, Wt, N))
        Wc_seen[i] = Wc
        if ctrl is None:
            de = 0.0
        else:
            t0 = time.perf_counter()
            de = ctrl.compute(x, Wi, Wc)
            comp_t += time.perf_counter() - t0; comp_n += 1
        cl, cm = aero.predict(x, de, Wi, U)     # plant uses the TRUE gust
        Fy = q * cl; Mz = q * cm * C
        der = structure.rhs(x, Fy, Mz)
        aero.advance(x, de, Wi, U, DT)
        x = structure.step_dp45(x, Fy, Mz, DT)
        for k, v in zip(['h','hd','al','ad','hdd','add','de','CL','CM','Fy'],
                        [x[0], x[1], x[2], x[3], der[1], der[3], de,
                         float(cl), float(cm), Fy]):
            rec[k].append(v)

    out = {k: np.array(v) for k, v in rec.items()}
    out['_t']  = ts
    out['_Wt'] = Wt
    out['_Wc'] = Wc_seen
    out['_comp_ms'] = (comp_t / comp_n * 1e3) if comp_n else 0.0
    return out


# ---------------------------------------------------------------------------
def _gust_window(t, Tg):
    return t <= (Tg + 0.5)


def metrics(r, r_open, Tg):
    """Metric dict for one arm vs the open-loop reference (== light/run.py)."""
    t  = r['_t']; mw = _gust_window(t, Tg)
    exo = float(np.max(np.abs(r_open['CL'][mw] - CLTRIM)))
    exc = float(np.max(np.abs(r['CL'][mw]      - CLTRIM)))
    clred = (exo - exc) / exo * 100.0 if exo > 1e-12 else 0.0
    i0, i1 = int(2.3 / DT), int(2.7 / DT)
    de_end = float(np.mean(np.abs(r['de'][i0:i1]))) if i1 <= len(r['de']) else float('nan')
    flag = ''
    for k in ['ad', 'add', 'hdd']:
        if np.max(np.abs(r[k][mw])) > 3.0 * np.max(np.abs(r_open[k][mw])) + 1e-9:
            flag += k + '!'
    adrms = float(np.sqrt(np.mean(r['ad'][mw]**2)) * 180 / np.pi)
    pitchpk = float(np.max(np.abs(r['al'][mw])))
    dCL = r['CL'][mw] - CLTRIM; ad = r['ad'][mw]; de = r['de'][mw]
    J = float(SCORE_W['Q_CL'] * np.sum(dCL**2)
              + SCORE_W['Q_ad'] * np.sum(ad**2)
              + SCORE_W['R'] * np.sum(de**2))
    return dict(clexc=exc, exo=exo, clred=clred, adrms=adrms,
                flap_max=float(np.max(np.abs(r['de'][mw]))),
                de_end=de_end, J=J, pitchpk=pitchpk,
                comp_ms=float(r.get('_comp_ms', 0.0)), flag=flag)


def seed_stats(ms):
    """Aggregate a list of metric dicts (one per seed) -> summary dict."""
    crs = np.array([m['clred'] for m in ms])
    nflag = sum(1 for m in ms if m['flag'])
    return dict(mean=float(crs.mean()), lo=float(crs.min()), hi=float(crs.max()),
                std=float(crs.std()), nflag=nflag, n=len(ms), clred=crs)


def fmt_stats(s):
    return (f"{s['mean']:+7.1f}% [{s['lo']:+6.1f},{s['hi']:+6.1f}] "
            f"std={s['std']:5.2f} flags={s['nflag']}/{s['n']}")


# ---------------------------------------------------------------------------
# npz record schema shared by all axis scripts. Every record is a plain dict
# with a 'kind' field:
#   'open'  : open-loop reference     (cex0, t, W, CL, + cell keys)
#   'point' : one study point         (seeds, clred, flags, flap_max, pitchpk,
#                                      mean/lo/hi/std/nflag, + config keys)
#   'traj'  : one stored trajectory   (label, seed, clred, flag,
#                                      t, W, Wc, CL, de, ad, + config keys)
# Files: results/<axis>.npz, key 'records' (object array, allow_pickle).
# allow_pickle is safe here: these npz are written and read exclusively by the
# scripts in this folder (trusted, locally generated — never third-party data).
# ---------------------------------------------------------------------------
def point_record(ms, **cfg):
    """Build a 'point' record from a list of per-seed metric dicts."""
    s = seed_stats(ms)
    rec = dict(kind='point',
               seeds=list(range(len(ms))),
               clred=s['clred'],
               flags=[m['flag'] for m in ms],
               flap_max=np.array([m['flap_max'] for m in ms]),
               pitchpk=np.array([m['pitchpk'] for m in ms]),
               mean=s['mean'], lo=s['lo'], hi=s['hi'], std=s['std'],
               nflag=s['nflag'])
    rec.update(cfg)
    return rec


def traj_record(r, m, **cfg):
    """Build a 'traj' record from a rollout dict + its metric dict."""
    rec = dict(kind='traj', clred=m['clred'], flag=m['flag'],
               t=r['_t'], W=r['_Wt'], Wc=r['_Wc'],
               CL=r['CL'], de=r['de'], ad=r['ad'])
    rec.update(cfg)
    return rec


def save_records(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, records=np.array(recs, dtype=object))
    print(f"# saved {len(recs)} records -> {path}", flush=True)


def load_records(path):
    return list(np.load(path, allow_pickle=True)['records'])
