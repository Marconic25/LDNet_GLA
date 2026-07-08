"""
Axis C — structured (non-random) errors on the 1-step gust preview W(t+dt).

Which SYSTEMATIC sensor imperfections break the good-branch advantage
(+76.6% at R=3e-4, +80.7% at R=1e-4, clean preview) and which are tolerated?
Unlike axis A/B noise, these errors have structure and do not average out.
Home cell W30/Tg0.4, DAMULT=3, one error family at a time:

    bias    : Wc = max(0, W(t+dt) + beta*W0),  beta in {+-0.05, +-0.10}
    scale   : Wc = s * W(t+dt),                s    in {0.8, 0.9, 1.1, 1.2}
    latency : Wc = W(t)        (one full step of sensor latency, no preview)
    jitter  : Wc = W(t + k_i*dt), k_i ~ U{0,1,2} drawn per step (preview age)

bias/scale/latency are deterministic -> single rollout per point; jitter is
random -> 6 seeds (rng 100+seed, same base as axis A). R in {3e-4, 1e-4}.

Known anchor: latency = feeding the controller W(t) instead of W(t+dt) is the
known bad branch, ~+17% (76/preview_study.py section A, k=0, refine=False) —
here it is re-measured with the final controller (refine=True) at both R.

Trajectories stored: the latency case at both R, and the worst bias case
(most negative mean CLred) at R=3e-4.

Output: results/C_struct.npz  (schema: harness_noise.py record helpers)
"""
import os
import numpy as np
import harness_noise as H

W0, Tg = 30.0, 0.4
BETAS  = (+0.05, -0.05, +0.10, -0.10)
SCALES = (0.8, 0.9, 1.1, 1.2)
R_LIST = (3e-4, 1e-4)
NSEED  = 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'C_struct.npz')

OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis C | W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}", flush=True)
recs = [dict(kind='open', axis='C', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def rtag(R):
    return 'R' + f"{R:.0e}".replace('e-0', 'e-')       # 3e-4 -> 'R3e-4'


# ---- error models (wc_fun for H.rollout) -------------------------------------
def wf_bias(b):
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + b)


def wf_scale(s):
    return lambda i, Wt, N: s * Wt[min(i + 1, N - 1)]


def wf_latency():
    return lambda i, Wt, N: Wt[i]


def wf_jitter(rng):
    return lambda i, Wt, N: Wt[min(i + int(rng.integers(0, 3)), N - 1)]


def run_det(R, wf, tag, **cfg):
    """One deterministic point: a single rollout, 1-element metrics list."""
    r = H.rollout(H.make_optimal(R=R), W0, Tg, wc_fun=wf)
    m = H.metrics(r, OL, Tg)
    rec = H.point_record([m], axis='C', W0=W0, Tg=Tg, R=R, **cfg)
    print(f"  {cfg.get('arm', '?'):8s} R={R:g} {tag:11s}: "
          f"{H.fmt_stats(H.seed_stats([m]))}", flush=True)
    return rec, m, r


# ------------------------------------------------------------------------------
for R in R_LIST:
    print(f"\n=== optimal wnext+refine R={R:g} ===", flush=True)

    # ---- 1. bias: constant offset on the preview -----------------------------
    bias_worst = None                                  # (clred, beta, m, r)
    for beta in BETAS:
        rec, m, r = run_det(R, wf_bias(beta * W0), f"beta={beta:+.2f}",
                            arm='bias', beta=beta)
        recs.append(rec)
        if R == 3e-4 and (bias_worst is None or m['clred'] < bias_worst[0]):
            bias_worst = (m['clred'], beta, m, r)
    if R == 3e-4:                                      # store worst bias traj
        _, beta, m, r = bias_worst
        recs.append(H.traj_record(r, m, axis='C', arm='bias',
                                  W0=W0, Tg=Tg, R=R, beta=beta, seed=0,
                                  label=f'C_bias_worst_{rtag(R)}'))

    # ---- 2. scale: multiplicative preview error ------------------------------
    for s in SCALES:
        rec, m, r = run_det(R, wf_scale(s), f"scale={s:.2f}",
                            arm='scale', scale=s)
        recs.append(rec)

    # ---- 3. latency: controller sees W(t), the known bad branch --------------
    rec, m, r = run_det(R, wf_latency(), "latency", arm='latency')
    recs.append(rec)
    recs.append(H.traj_record(r, m, axis='C', arm='latency',
                              W0=W0, Tg=Tg, R=R, seed=0,
                              label=f'C_latency_{rtag(R)}'))

    # ---- 4. jitter: random per-step preview age, 6 seeds ----------------------
    ms = []
    for seed in range(NSEED):
        rng = np.random.default_rng(100 + seed)
        r = H.rollout(H.make_optimal(R=R), W0, Tg, wc_fun=wf_jitter(rng))
        ms.append(H.metrics(r, OL, Tg))
    recs.append(H.point_record(ms, axis='C', arm='jitter', W0=W0, Tg=Tg, R=R))
    print(f"  {'jitter':8s} R={R:g} {'k=U(0..2)':11s}: "
          f"{H.fmt_stats(H.seed_stats(ms))}", flush=True)

H.save_records(OUT, recs)
print("# DONE", flush=True)
