"""
Fair (band-limited) LIDAR-noise test.

mpc_noise.py used WHITE noise (independent every 2 ms = 500 Hz) — the worst case:
it makes the gust estimate jump every step, so the flap chatters and the pitch
mode rings. A real Doppler LIDAR delivers a temporally-smoothed, low-bandwidth
wind estimate (range-gate averaged, updated ~1-10 Hz). We model that by low-pass
filtering the noisy estimate the controller sees, time constant tau:

    est[i] = a*est[i-1] + (1-a)*(W_true(t+k) + white_noise),   a = exp(-dt/tau)

The PLANT always uses the true gust. We sweep tau at a fixed 5% white-noise level
for the two candidates and check whether band-limiting recovers robustness.
"""
import numpy as np
import structure
import harness as H
from controllers import OptGrid, MPCConst

W0, Tg = 30.0, 0.4
OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f"# W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}  (band-limited LIDAR noise, 5% white + LPF)", flush=True)
FRAC = 0.05


def rollout_filtered(make_ctrl, preview_k, tau, seed):
    ctrl = make_ctrl()
    ts, Wt = H.gust_array(W0, Tg); N = len(ts)
    H.aero.reset(dt=H.DT); ctrl.reset()
    rng = np.random.default_rng(300 + seed)
    a = float(np.exp(-H.DT / tau)) if tau > 0 else 0.0
    x = H.X0.copy(); prev = 0.0; est = 0.0
    CL = np.zeros(N); al = np.zeros(N); ad = np.zeros(N); de = np.zeros(N)
    hdd = np.zeros(N); add = np.zeros(N)
    for i in range(N):
        Wi = float(Wt[i])
        raw = max(0.0, float(Wt[min(i+preview_k, N-1)]) + rng.normal(0.0, FRAC*W0))
        est = a*est + (1.0 - a)*raw                 # band-limited estimate
        d = ctrl.compute(H.aero, x, prev, est, est) # controller sees the filtered estimate
        cl, cm = H.aero.predict(x, d, Wi, H.U)       # plant uses TRUE gust
        Fy = H.q*cl; Mz = H.q*cm*H.C
        der = structure.rhs(x, Fy, Mz)
        H.aero.advance(x, d, Wi, H.U, H.DT); x = structure.step_rk4(x, Fy, Mz, H.DT)
        CL[i]=cl; al[i]=x[2]; ad[i]=x[3]; de[i]=d; hdd[i]=der[1]; add[i]=der[3]
        prev = d
    return dict(CL=CL, al=al, ad=ad, de=de, hdd=hdd, add=add, _t=ts, _Wt=Wt)


def sweep(label, make_ctrl, preview_k):
    print(f"\n=== {label} (5% white noise + LPF) ===", flush=True)
    print(f"{'tau[ms]':>7s} {'CLred mean':>11s} {'min':>7s} {'max':>7s} {'std':>6s} {'flag':>5s}", flush=True)
    for tau_ms in (0.0, 5.0, 10.0, 20.0, 40.0):
        tau = tau_ms/1000.0
        crs = []; nflag = 0
        for seed in range(6):
            r = rollout_filtered(make_ctrl, preview_k, tau, seed)
            m = H.metrics(r, OL, Tg); crs.append(m['clred'])
            if m['flag']: nflag += 1
        crs = np.array(crs)
        print(f"{tau_ms:7.0f} {crs.mean():+10.1f}% {crs.min():+6.1f} {crs.max():+6.1f} "
              f"{crs.std():6.2f} {nflag}/6", flush=True)


sweep("MPC N4 none (no preview)", lambda: MPCConst(N=4, R=3e-4, G=161, gate='none'), 0)
sweep("SS wnext (1-step preview)", lambda: OptGrid(R=3e-4, G=161, gate='hard', use_wnext=True), 1)
print("# DONE", flush=True)
