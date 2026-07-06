"""
Is W(t+dt) legal?  Physical-realizability study of the wnext controller.

The winner feeds the one-step-ahead gust W(t+dt) to the optimizer. That is a
gust PREVIEW of exactly one step = dt = 0.002 s. At U=80 m/s that is a look-ahead
distance of 80*0.002 = 0.16 m. Forward-looking Doppler LIDAR — the standard GLA
gust sensor — previews the wind field 30-200 m ahead (~0.4-2.5 s at 80 m/s), i.e.
190-1250 steps. So a 1-step preview is the MILDEST possible preview assumption.

Also: the study already assumes W(t) is known (prop-W and opt-W are gust-oracle
controllers; C_L(W) is non-invertible so W is not observable from C_L -> a sensor
is required either way). So W(t+dt) is the SAME class of assumption (a gust
sensor), extended by one 2 ms step.

This script quantifies three things:
  A) PREVIEW HORIZON  : CLred vs k for W(t+k*dt), k=0..20 (oracle preview).
  B) CAUSAL EXTRAP    : replace the future sample by a CAUSAL linear extrapolation
                        of the known W: Wpred = 2*W(t) - W(t-dt) (uses only present
                        + past). If this recovers +76%, NO future knowledge is
                        needed at all — only the gust RATE.
  C) PREVIEW NOISE    : LIDAR-like measurement error on the 1-step preview
                        (sigma = frac * W0), several seeds. Literature flags preview
                        feedforward as noise-sensitive, so we measure the graceful
                        degradation.
"""
import numpy as np
import harness as H
from controllers import OptGrid

W0, Tg = 30.0, 0.4
OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f"# W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}  (dt={H.DT}s, U={H.U} -> 1 step = {H.U*H.DT:.2f} m ahead)", flush=True)


def rollout_customW(wfun, R=3e-4):
    """Rollout where the controller is fed a custom 'measured' gust wfun(i,Wt,N);
    the PLANT is always advanced with the TRUE gust Wt[i]."""
    ctrl = OptGrid(R=R, G=161, gate='hard', use_wnext=True)  # use_wnext -> uses the Wn we pass
    ts, Wt = H.gust_array(W0, Tg); N = len(ts)
    H.aero.reset(dt=H.DT); ctrl.reset()
    x = H.X0.copy(); prev = 0.0
    CL = np.zeros(N); al = np.zeros(N); ad = np.zeros(N); de = np.zeros(N); hdd = np.zeros(N); add = np.zeros(N)
    import structure
    for i in range(N):
        Wi = float(Wt[i])
        Wc = float(wfun(i, Wt, N))                 # what the controller "sees"
        d = ctrl.compute(H.aero, x, prev, Wi, Wc)
        cl, cm = H.aero.predict(x, d, Wi, H.U)     # PLANT uses the TRUE gust
        Fy = H.q*cl; Mz = H.q*cm*H.C
        der = structure.rhs(x, Fy, Mz)
        H.aero.advance(x, d, Wi, H.U, H.DT); x = structure.step_rk4(x, Fy, Mz, H.DT)
        CL[i]=cl; al[i]=x[2]; ad[i]=x[3]; de[i]=d; hdd[i]=der[1]; add[i]=der[3]
        prev = d
    r = dict(CL=CL, al=al, ad=ad, de=de, hdd=hdd, add=add, _t=ts, _Wt=Wt)
    return r


def clred(r):
    m = H.metrics(r, OL, Tg)
    return m['clred'], m['flap_max'], m['pitchpk']*180/np.pi, m['adrms'], m['flag']


# ---- A) preview horizon -----------------------------------------------------
print("\n=== A) preview horizon  W(t+k*dt)  (k=0 is the plain W(t) controller) ===", flush=True)
print(f"{'k':>3s} {'dist[m]':>8s} {'preview[ms]':>11s} {'CLred':>7s} {'flap':>5s} {'pitch':>6s} {'flag':>6s}", flush=True)
for k in (0, 1, 2, 3, 5, 10, 20):
    r = rollout_customW(lambda i, Wt, N, k=k: Wt[min(i+k, N-1)])
    cr, fm, pk, ar, fl = clred(r)
    print(f"{k:3d} {H.U*H.DT*k:8.2f} {1000*H.DT*k:11.1f} {cr:+6.1f}% {fm:5.1f} {pk:6.3f} {fl:>6s}", flush=True)

# ---- B) causal extrapolation (NO future) ------------------------------------
print("\n=== B) causal predictors of W(t+dt) from present+past only ===", flush=True)
def w_lin(i, Wt, N):    # linear: 2W(t)-W(t-1)
    return max(0.0, 2*Wt[i] - Wt[i-1]) if i >= 1 else Wt[i]
def w_quad(i, Wt, N):   # quadratic extrapolation from last 3 samples
    if i >= 2: return max(0.0, 3*Wt[i] - 3*Wt[i-1] + Wt[i-2])
    return Wt[i]
for name, f in [('W(t) [k=0 ref]', lambda i,Wt,N: Wt[i]),
                ('causal linear', w_lin),
                ('causal quad',   w_quad),
                ('oracle W(t+dt)', lambda i,Wt,N: Wt[min(i+1,N-1)])]:
    r = rollout_customW(f); cr, fm, pk, ar, fl = clred(r)
    print(f"  {name:18s} CLred={cr:+6.1f}%  flap={fm:5.1f} pitch={pk:6.3f} {fl}", flush=True)

# ---- C) preview noise on the 1-step oracle preview --------------------------
print("\n=== C) LIDAR-like preview noise on W(t+dt): Wc = W(t+dt) + N(0, frac*W0) ===", flush=True)
print(f"{'noise':>7s} {'sigma[m/s]':>10s} {'CLred mean':>11s} {'min':>7s} {'max':>7s} {'std':>6s}", flush=True)
for frac in (0.0, 0.02, 0.05, 0.10, 0.20):
    crs = []
    for seed in range(6):
        rng = np.random.default_rng(100+seed)
        def wf(i, Wt, N, rng=rng, frac=frac):
            return max(0.0, Wt[min(i+1, N-1)] + rng.normal(0.0, frac*W0))
        r = rollout_customW(wf); crs.append(clred(r)[0])
    crs = np.array(crs)
    print(f"{frac:7.2f} {frac*W0:10.2f} {crs.mean():+10.1f}% {crs.min():+6.1f} {crs.max():+6.1f} {crs.std():6.2f}", flush=True)
print("# DONE", flush=True)
