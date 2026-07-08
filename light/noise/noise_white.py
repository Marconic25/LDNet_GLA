"""
Axis A — baseline degradation: white Gaussian noise on the 1-step preview.

    Wc(i) = max(0, W(t+dt) + N(0, sigma)),   sigma = frac * W0

Home cell W30/Tg0.4, DAMULT=3. frac in {0, .01, .02, .05, .10},
R in {3e-4, 1e-4}, 12 seeds (rng 100+seed, one draw per step — seeds 0..5 are
bit-comparable with 76/preview_study.py section C; 12 because the regime is
chaotic at seed level with per-seed std ~19 pts, so a 6-seed mean has SE ~8 pts
— two 76/ runs of this very cell differ by 5.7 pts in the mean on rng base
alone, preview.log -0.5% vs mpc_noise.log +5.2%).

Plus the equal-info prop-W reference (g_CL=-60, g_W=-0.5, the honest_home
no-flag best) fed the SAME noisy preview, frac in {0, .02, .05}: the break-even
"how much noise before the optimal loses its edge over prop" is only honest if
prop degrades under the same sensor.

Trajectories stored for the money plot: all 6 seeds at (R=3e-4, frac=0.02)
and the clean run at R=1e-4.

Output: results/A_white.npz  (schema: harness_noise.py record helpers)
"""
import os
import numpy as np
import harness_noise as H
from controllers_ref import PropWRef

W0, Tg = 30.0, 0.4
FRACS      = (0.0, 0.01, 0.02, 0.05, 0.10)
FRACS_PROP = (0.0, 0.02, 0.05)
R_LIST     = (3e-4, 1e-4)
NSEED      = 12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'A_white.npz')

OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis A | W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}", flush=True)
recs = [dict(kind='open', axis='A', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def wf_noisy_preview(rng, frac):
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


def run_point(make_ctrl, frac, **cfg):
    """One study point: NSEED seeds (or 1 deterministic run when frac=0)."""
    ms, rs = [], []
    nseed = 1 if frac == 0.0 else NSEED
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        r = H.rollout(make_ctrl(), W0, Tg, wc_fun=wf_noisy_preview(rng, frac))
        ms.append(H.metrics(r, OL, Tg)); rs.append(r)
    rec = H.point_record(ms, frac=frac, **cfg)
    Rv = cfg.get('R')
    Rs = f"{Rv:g}" if Rv is not None else "-"
    print(f"  {cfg.get('arm','?'):8s} R={Rs} "
          f"frac={frac:.2f}: {H.fmt_stats(H.seed_stats(ms))}", flush=True)
    return rec, ms, rs


# ---- optimal (final controller) ---------------------------------------------
for R in R_LIST:
    print(f"\n=== optimal wnext+refine R={R:g} ===", flush=True)
    for frac in FRACS:
        rec, ms, rs = run_point(lambda R=R: H.make_optimal(R=R), frac,
                                axis='A', arm='optimal', W0=W0, Tg=Tg, R=R)
        recs.append(rec)
        if R == 3e-4 and frac == 0.02:          # money-plot candidates (worst case)
            for seed, (m, r) in enumerate(zip(ms, rs)):
                recs.append(H.traj_record(r, m, axis='A', arm='optimal',
                                          W0=W0, Tg=Tg, R=R, frac=frac,
                                          seed=seed, label=f'A_opt_f0.02_s{seed}'))
        if R == 1e-4 and frac == 0.0:           # clean reference trajectory
            recs.append(H.traj_record(rs[0], ms[0], axis='A', arm='optimal',
                                      W0=W0, Tg=Tg, R=R, frac=frac,
                                      seed=0, label='A_opt_clean_R1e-4'))

# ---- prop-W equal-info reference under the same noisy preview ---------------
print(f"\n=== prop-W (g_CL=-60, g_W=-0.5), same noisy preview ===", flush=True)
for frac in FRACS_PROP:
    rec, ms, rs = run_point(lambda: PropWRef(), frac,
                            axis='A', arm='prop', W0=W0, Tg=Tg, R=None)
    recs.append(rec)

H.save_records(OUT, recs)
print("# DONE", flush=True)
