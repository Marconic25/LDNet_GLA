"""
Shared harness for the 76/ investigation: robustly reaching the good closed-loop
branch (CLred ~ +76%) at W0=30, Tg=0.4, DAMULT=3.

Everything here is byte-faithful to the verified reference:
  - physics: ldnet_aero.py / structure.py (copied from light/, == clean/)
  - scalar rollout loop: identical to light/run.py 'optimal'/'prop_w'/'open' arms
  - batched oracle: identical to clean/propw.py batch_optW (leak-corrected)
Because dt == dt_ref (=0.002), n_sub=1, so the scalar advance() and the
leak-corrected batch_step() are the SAME map up to TF batch-position rounding.

DAMULT must be in the environment BEFORE importing this module (structure.D_ALPHA
is scaled at import, exactly as run.py / mpc_gust.py do).
"""
import os, time
import numpy as np
import structure
from ldnet_aero import LDNetAero

# ---- plant scaling (must happen at import, before any sim) -------------------
structure.D_ALPHA *= float(os.environ.get('DAMULT', '1'))

# ---- constants (match run.py / mpc_gust.py) ---------------------------------
U    = 80.0
RHO  = 1.225
C    = 1.0
DT   = 0.002
S    = 0.05
q    = 0.5 * RHO * U**2 * S
DMAX = 14.0
DDOT_MAX = 300.0
G_DEFAULT = 161

MD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  '..', 'clean', 'models_rollout', 'latent_10')

X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])

# Single shared aero instance (loading TF weights is the expensive part).
aero = LDNetAero(MD); aero.reset(dt=DT)
LAM = float(aero._z_leak)
CLTRIM = float(aero.predict(X0, 0., 0., U)[0])

SCORE_W = dict(Q_CL=1.0, Q_ad=100.0, R=0.01)


# ---------------------------------------------------------------------------
def gust(t, W0, Tg):
    """1-cosine gust."""
    return (W0 / 2.0) * (1 - np.cos(2 * np.pi * t / Tg)) if (0 <= t <= Tg) else 0.0


def gust_array(W0, Tg, TEND=3.0):
    N = int(round(TEND / DT)) + 1
    ts = np.arange(N) * DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])
    return ts, Wt


def window(t, Tg):
    return t <= (Tg + 0.5)


# ---------------------------------------------------------------------------
def metrics(r, r_open, Tg):
    """Per-arm metric dict vs open-loop reference (identical to light/run.py)."""
    t  = r['_t']; mw = window(t, Tg)
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
    dCL = r['CL'][mw] - CLTRIM; ad = r['ad'][mw]; de = r['de'][mw]
    J = float(SCORE_W['Q_CL'] * np.sum(dCL**2)
              + SCORE_W['Q_ad'] * np.sum(ad**2)
              + SCORE_W['R'] * np.sum(de**2))
    pitchpk = float(np.max(np.abs(r['al'][mw])))
    return dict(clexc=exc, exo=exo, clred=clred, adrms=adrms,
                flap_max=float(np.max(np.abs(r['de'][mw]))),
                de_end=de_end, J=J, pitchpk=pitchpk,
                comp_ms=float(r.get('_comp_ms', 0.0)), flag=flag)


# ---------------------------------------------------------------------------
# Scalar rollout — one controller object, B=1, no batching, no smoothing.
# Loop body is identical to light/run.py.  The controller must expose
#   .reset()
#   .compute(aero, state, prev_delta, Wi, Wnext) -> delta   (already saturated
#                                                            and rate-limited)
# Pass ctrl=None for the open-loop arm.
# ---------------------------------------------------------------------------
def scalar_rollout(ctrl, W0, Tg, TEND=3.0, X0_override=None, W0_scale=1.0):
    ts, Wt = gust_array(W0, Tg, TEND)
    Wt = Wt * W0_scale
    N  = len(ts)
    aero.reset(dt=DT)
    if ctrl is not None:
        ctrl.reset()
    x = (X0.copy() if X0_override is None else np.asarray(X0_override, float).copy())
    prev = 0.0
    rec = {k: [] for k in ['h','hd','al','ad','hdd','add','de','CL','CM','Fy']}
    comp_t = 0.0; comp_n = 0
    for i in range(N):
        Wi = float(Wt[i]); Wn = float(Wt[i+1]) if i + 1 < N else 0.0
        if ctrl is None:
            de = 0.0
        else:
            t0 = time.perf_counter()
            de = ctrl.compute(aero, x, prev, Wi, Wn)
            comp_t += time.perf_counter() - t0; comp_n += 1
        cl, cm = aero.predict(x, de, Wi, U)
        Fy = q * cl; Mz = q * cm * C
        der = structure.rhs(x, Fy, Mz)
        aero.advance(x, de, Wi, U, DT)
        x = structure.step_dp45(x, Fy, Mz, DT)
        for k, v in zip(['h','hd','al','ad','hdd','add','de','CL','CM','Fy'],
                        [x[0], x[1], x[2], x[3], der[1], der[3], de,
                         float(cl), float(cm), Fy]):
            rec[k].append(v)
        prev = de
    out = {k: np.array(v) for k, v in rec.items()}
    out['_t'] = ts; out['_Wt'] = Wt
    out['_comp_ms'] = (comp_t / comp_n * 1e3) if comp_n else 0.0
    return out


# ---------------------------------------------------------------------------
# Batched RK4 structural step (identical to clean/controller.py _rk4_batch).
# ---------------------------------------------------------------------------
def rk4_batch(x_b, Fy_b, Mz_b, dt):
    def rhs(xx):
        h = xx[:,0]; hd = xx[:,1]; a = xx[:,2]; ad = xx[:,3]
        rh = -Fy_b - structure.D_H * hd - structure.K_H * h
        ra =  Mz_b - structure.D_ALPHA * ad - structure.K_ALPHA * a
        hdd = (structure.M_AA * rh - structure.M_HA * ra) / structure.DET
        add = (structure.M_HH * ra - structure.M_HA * rh) / structure.DET
        return np.stack([hd, hdd, ad, add], axis=1)
    k1 = rhs(x_b); k2 = rhs(x_b + 0.5*dt*k1); k3 = rhs(x_b + 0.5*dt*k2); k4 = rhs(x_b + dt*k3)
    return x_b + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Batched single-step-optimal oracle with FULL instrumentation.
# Identical algorithm to clean/propw.py batch_optW (leak-corrected), but records
# the per-step trajectory + argmin diagnostics for every row so we can dissect
# the good vs bad branch split.
# ---------------------------------------------------------------------------
def batch_optW_trace(W0, Tg, R_GRID, G=G_DEFAULT, TEND=3.0):
    R_GRID = np.asarray(R_GRID, float)
    B = len(R_GRID)
    ts, Wt = gust_array(W0, Tg, TEND)
    N = len(ts)
    aero.reset(dt=DT)
    z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (B, 1))
    x_b = np.tile(np.asarray(X0, float).reshape(1, -1), (B, 1))
    prev = np.zeros(B); reach = DDOT_MAX * DT
    dg = np.linspace(-DMAX, DMAX, G)

    T = dict(de=np.zeros((B, N)), CL=np.zeros((B, N)), al=np.zeros((B, N)),
             ad=np.zeros((B, N)), hdd=np.zeros((B, N)), add=np.zeros((B, N)),
             cl0=np.zeros((B, N)), jarg=np.zeros((B, N), int),
             margin=np.zeros((B, N)), nfeas=np.zeros((B, N), int),
             gate=np.zeros((B, N), int))
    for i in range(N):
        Wi = float(Wt[i])
        cl0, _, _ = aero.batch_step(z_b, x_b, np.zeros(B), Wi, U, DT)
        CLg, _, _ = aero.batch_step(np.repeat(z_b, G, 0), np.repeat(x_b, G, 0),
                                    np.tile(dg, B), Wi, U, DT)
        CLg = CLg.reshape(B, G)
        cost = (CLg - CLTRIM)**2 + R_GRID[:, None] * dg[None, :]**2
        neg = dg[None, :] <= 0.0
        gate_up = cl0[:, None] >= CLTRIM          # True -> use negative flap
        causal = np.where(gate_up, neg, ~neg)
        ratem = np.abs(dg[None, :] - prev[:, None]) <= reach + 1e-9
        feas = causal & ratem
        cost_m = np.where(feas, cost, np.inf)
        jarg = np.argmin(cost_m, axis=1)
        d = dg[jarg]

        # argmin diagnostics (near-tie detection)
        for b in range(B):
            row = cost_m[b]
            fin = np.isfinite(row)
            T['nfeas'][b, i] = int(fin.sum())
            if fin.sum() >= 2:
                s = np.sort(row[fin])
                T['margin'][b, i] = float(s[1] - s[0])
            else:
                T['margin'][b, i] = np.inf
        T['jarg'][:, i] = jarg
        T['cl0'][:, i] = cl0
        T['gate'][:, i] = gate_up[:, 0].astype(int)

        cl, cm, z_new = aero.batch_step(z_b, x_b, d, Wi, U, DT)
        z_b = z_new - LAM * z_b
        # structural accelerations at this step (pre-integration forces)
        Fy = q * cl; Mz = q * cm * C
        der = rk4_batch(x_b, Fy, Mz, DT)  # not used for hdd; compute rhs directly:
        hh = x_b[:,0]; hd = x_b[:,1]; aa = x_b[:,2]; ad_ = x_b[:,3]
        rh = -Fy - structure.D_H*hd - structure.K_H*hh
        ra =  Mz - structure.D_ALPHA*ad_ - structure.K_ALPHA*aa
        T['hdd'][:, i] = (structure.M_AA*rh - structure.M_HA*ra)/structure.DET
        T['add'][:, i] = (structure.M_HH*ra - structure.M_HA*rh)/structure.DET
        x_b = rk4_batch(x_b, Fy, Mz, DT)
        T['de'][:, i] = d
        T['CL'][:, i] = cl
        T['al'][:, i] = x_b[:, 2]
        T['ad'][:, i] = x_b[:, 3]
        prev = d
    T['_t'] = ts; T['_Wt'] = Wt; T['_R'] = R_GRID
    return T


def cex_of(CL_row, ts, Tg):
    mw = window(ts, Tg)
    return float(np.max(np.abs(CL_row[mw] - CLTRIM)))
