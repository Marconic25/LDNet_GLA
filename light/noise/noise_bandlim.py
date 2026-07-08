"""
Axis B — band-limited (LIDAR-like) noise on the 1-step gust preview.

    raw(i) = max(0, W(t+dt) + N(0, sigma)),   sigma = 0.05 * W0  (5% white)
    est(i) = a*est(i-1) + (1-a)*raw(i),       a = exp(-DT/tau),  est(0) = 0
    Wc(i)  = est(i)

The 1st-order LPF models a range-gate-averaged Doppler-LIDAR wind estimate:
the instrument returns a spatially/temporally averaged value, not an
instantaneous point sample, so the 5% white component is band-limited with
time constant tau. tau=0 means a=0 -> unfiltered white noise (continuity
check against axis A frac=0.05; rng base differs — 300+seed here vs 100+seed
there — so agreement is statistical, not bitwise).

Home cell W30/Tg0.4, DAMULT=3. tau_ms in {0, 2, 5, 10, 20, 40},
R in {3e-4, 1e-4}, 6 seeds (rng 300+seed, one draw per step — the exact
convention of 76/mpc_noise2.py, so numbers are comparable).

Known 76/ result (SS wnext, refine=False OptGrid, mean CLred): the filter
trades white-noise attenuation against gust lag — sweet spot tau ~ 5-10 ms,
collapse at tau >= 40 by gust lag (a 40 ms filter lags a ~100 ms gust ->
wrong-phase flap -> pitch excitation, 6/6 explosion flags in the 76/ sweep):

    tau=0: -14.2% (3/6 flags), 5: -1.0%, 10: +19.8% (1/6),
    20: +12.5%, 40: -9.3%

Here the same noise model drives the FINAL light controller (use_wnext=True,
refine=True): does the parabolic sub-cell refinement shift the sweet spot?

Trajectories stored for the money plot: all 6 seeds at the best tau of the
R=3e-4 sweep (max mean CLred, preferring zero-flag points).

Output: results/B_bandlim.npz  (schema: harness_noise.py record helpers)
"""
import os
import numpy as np
import harness_noise as H

W0, Tg = 30.0, 0.4
FRAC    = 0.05
TAUS_MS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
R_LIST  = (3e-4, 1e-4)
NSEED   = 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'B_bandlim.npz')

OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis B | W30/Tg0.4 DAMULT=3 | frac={FRAC:.2f} band-limited preview | "
      f"open cex0={cex0:.4f}", flush=True)
recs = [dict(kind='open', axis='B', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def wc_bandlim(rng, tau_ms):
    """Noisy 1-step preview through a 1st-order LPF, est(0)=0 (== 76/mpc_noise2.py)."""
    tau = tau_ms / 1000.0
    a = float(np.exp(-H.DT / tau)) if tau > 0 else 0.0
    state = {'est': 0.0}
    def f(i, Wt, N):
        raw = max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, FRAC * W0))
        state['est'] = a * state['est'] + (1.0 - a) * raw
        return state['est']
    return f


def run_point(make_ctrl, tau_ms, **cfg):
    """One study point: NSEED seeds (noise is always on — FRAC is fixed)."""
    ms, rs = [], []
    for seed in range(NSEED):
        rng = np.random.default_rng(300 + seed)     # 300-base == 76/mpc_noise2.py
        r = H.rollout(make_ctrl(), W0, Tg, wc_fun=wc_bandlim(rng, tau_ms))
        ms.append(H.metrics(r, OL, Tg)); rs.append(r)
    rec = H.point_record(ms, tau_ms=tau_ms, **cfg)
    print(f"  {cfg.get('arm','?'):8s} R={cfg.get('R', float('nan')):g} "
          f"tau={tau_ms:4.0f}ms: {H.fmt_stats(H.seed_stats(ms))}", flush=True)
    return rec, ms, rs


# ---- optimal (final controller), tau sweep at each R -------------------------
points_R3 = {}   # tau_ms -> (rec, ms, rs), kept for the traj pick at R=3e-4

for R in R_LIST:
    print(f"\n=== optimal wnext+refine R={R:g} | frac={FRAC:.2f} ===", flush=True)
    for tau_ms in TAUS_MS:
        rec, ms, rs = run_point(lambda R=R: H.make_optimal(R=R), tau_ms,
                                axis='B', arm='optimal', W0=W0, Tg=Tg,
                                R=R, frac=FRAC)
        recs.append(rec)
        if R == 3e-4:
            points_R3[tau_ms] = (rec, ms, rs)

# ---- best tau at R=3e-4: max mean CLred, preferring zero-flag points ---------
cand  = list(points_R3.items())
clean = [(tau, v) for tau, v in cand if v[0]['nflag'] == 0]
pool  = clean if clean else cand
tau_best, (rec_b, ms_b, rs_b) = max(pool, key=lambda kv: kv[1][0]['mean'])
print(f"\n# best tau at R=3e-4: {tau_best:g} ms, mean {rec_b['mean']:+.1f}% "
      f"flags={rec_b['nflag']}/{NSEED}", flush=True)
for seed, (m, r) in enumerate(zip(ms_b, rs_b)):
    recs.append(H.traj_record(r, m, axis='B', arm='optimal',
                              W0=W0, Tg=Tg, R=3e-4, frac=FRAC,
                              tau_ms=tau_best, seed=seed,
                              label=f'B_opt_tau{tau_best:g}_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
