"""
Does the MPC path need a preview, and is it noise-robust?

Single-step wnext needs an ACCURATE 1-step preview (preview_study.py part C: 2%
noise collapses it). The MPC N4/gate=none controller reached +79% using only the
CURRENT gust W(t) held over the horizon (NO preview). Its horizon cost integrates
over 4 steps, so it should be far less sensitive to per-step near-ties.

Here we add gust MEASUREMENT noise to the gust the CONTROLLER sees (plant always
uses the true gust) and compare the two controllers' graceful degradation:
  - MPC N4 none, controller sees W(t)+noise      (NO preview)
  - SS wnext,   controller sees W(t+dt)+noise     (1-step preview) [ref, already fragile]
"""
import numpy as np
import structure
import harness as H
from controllers import OptGrid, MPCConst

W0, Tg = 30.0, 0.4
OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f"# W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}", flush=True)


def rollout_customW(make_ctrl, wfun):
    """Controller sees wfun(i,Wt,N) as its gust (passed as both Wi and Wn); the
    PLANT is advanced with the TRUE gust Wt[i]."""
    ctrl = make_ctrl()
    ts, Wt = H.gust_array(W0, Tg); N = len(ts)
    H.aero.reset(dt=H.DT); ctrl.reset()
    x = H.X0.copy(); prev = 0.0
    CL = np.zeros(N); al = np.zeros(N); ad = np.zeros(N); de = np.zeros(N)
    hdd = np.zeros(N); add = np.zeros(N)
    for i in range(N):
        Wi = float(Wt[i]); Wc = float(wfun(i, Wt, N))
        d = ctrl.compute(H.aero, x, prev, Wc, Wc)     # controller sees Wc (as Wi and Wn)
        cl, cm = H.aero.predict(x, d, Wi, H.U)         # plant uses TRUE gust
        Fy = H.q*cl; Mz = H.q*cm*H.C
        der = structure.rhs(x, Fy, Mz)
        H.aero.advance(x, d, Wi, H.U, H.DT); x = structure.step_dp45(x, Fy, Mz, H.DT)
        CL[i]=cl; al[i]=x[2]; ad[i]=x[3]; de[i]=d; hdd[i]=der[1]; add[i]=der[3]
        prev = d
    return dict(CL=CL, al=al, ad=ad, de=de, hdd=hdd, add=add, _t=ts, _Wt=Wt)


def sweep(label, make_ctrl, preview_k):
    print(f"\n=== {label} (controller sees W(t+{preview_k}*dt)+noise) ===", flush=True)
    print(f"{'noise':>7s} {'sigma':>6s} {'CLred mean':>11s} {'min':>7s} {'max':>7s} {'std':>6s} {'flag_frac':>9s}", flush=True)
    for frac in (0.0, 0.02, 0.05, 0.10):
        crs = []; nflag = 0
        S = 1 if frac == 0.0 else 6
        for seed in range(S):
            rng = np.random.default_rng(200+seed)
            def wf(i, Wt, N, rng=rng, frac=frac, k=preview_k):
                return max(0.0, Wt[min(i+k, N-1)] + rng.normal(0.0, frac*W0))
            r = rollout_customW(make_ctrl, wf)
            m = H.metrics(r, OL, Tg); crs.append(m['clred'])
            if m['flag']: nflag += 1
        crs = np.array(crs)
        print(f"{frac:7.2f} {frac*W0:6.2f} {crs.mean():+10.1f}% {crs.min():+6.1f} {crs.max():+6.1f} "
              f"{crs.std():6.2f} {nflag}/{S}", flush=True)


sweep("MPC N4 none (NO preview)", lambda: MPCConst(N=4, R=3e-4, G=161, gate='none'), 0)
sweep("SS wnext (1-step preview)", lambda: OptGrid(R=3e-4, G=161, gate='hard', use_wnext=True), 1)
sweep("MPC N4 none + 1-step preview", lambda: MPCConst(N=4, R=3e-4, G=161, gate='none'), 1)
print("# DONE", flush=True)
