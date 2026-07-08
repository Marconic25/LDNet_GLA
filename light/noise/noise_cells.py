"""
Axis D — cell dependence of the preview-noise sensitivity.

Is the white-noise fragility measured at the sharp home cell W30/Tg0.4 (axis A)
specific to that knife-edge, or general? Repeat axis A on a reduced sigma grid
at two gentler cells:

    (W0, Tg) in {(10, 0.7), (30, 0.7)}
    Wc(i) = max(0, W(t+dt) + N(0, sigma)),   sigma = frac * W0_cell

sigma scales with the cell's OWN W0, so frac means the same relative sensor
quality everywhere. frac in {0, .02, .05}, R in {3e-4, 1e-4}, 6 seeds
(rng 100+seed, one draw per step — the exact convention of axis A /
76/preview_study.py section C, so numbers are comparable across axes).

Each cell gets its own open-loop reference (kind='open' record with its cex0).
Noise-free anchors from the CS-25 sweep (76/NOTES.md, no-flag R pick):
W10/Tg0.7 -> +95.8%, W30/Tg0.7 -> +90.6%. Those were the best over an R sweep;
our two fixed R values may come in somewhat lower — that is expected and fine,
the object of study here is the degradation with frac, not the clean peak.

No trajectory records — points only.

Output: results/D_cells.npz  (schema: harness_noise.py record helpers)
"""
import os
import numpy as np
import harness_noise as H

CELLS  = ((10.0, 0.7), (30.0, 0.7))
FRACS  = (0.0, 0.02, 0.05)
R_LIST = (3e-4, 1e-4)
NSEED  = 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'D_cells.npz')

recs = []


def wf_noisy_preview(rng, frac, W0):
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


def run_point(make_ctrl, OL, W0c, Tgc, frac, **cfg):
    """One study point: NSEED seeds (or 1 deterministic run when frac=0).
    Cell passed as W0c/Tgc to avoid colliding with the W0/Tg record keys in cfg."""
    ms = []
    nseed = 1 if frac == 0.0 else NSEED
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        r = H.rollout(make_ctrl(), W0c, Tgc, wc_fun=wf_noisy_preview(rng, frac, W0c))
        ms.append(H.metrics(r, OL, Tgc))
    rec = H.point_record(ms, frac=frac, **cfg)
    print(f"  {cfg.get('arm','?'):8s} R={cfg.get('R', float('nan')):g} "
          f"frac={frac:.2f}: {H.fmt_stats(H.seed_stats(ms))}", flush=True)
    return rec


for W0, Tg in CELLS:
    OL = H.rollout(None, W0, Tg)
    cex0 = H.metrics(OL, OL, Tg)['exo']
    print(f"\n=== axis D | cell W{W0:g}/Tg{Tg:g} | open cex0={cex0:.4f} ===", flush=True)
    recs.append(dict(kind='open', axis='D', W0=W0, Tg=Tg, cex0=cex0,
                     t=OL['_t'], W=OL['_Wt'], CL=OL['CL']))
    for R in R_LIST:
        for frac in FRACS:
            recs.append(run_point(lambda R=R: H.make_optimal(R=R), OL, W0, Tg, frac,
                                  axis='D', arm='optimal', W0=W0, Tg=Tg, R=R))

H.save_records(OUT, recs)
print("# DONE", flush=True)
