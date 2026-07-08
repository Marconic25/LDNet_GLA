"""
Axis E2-sensor — DLR-style massed-measurement gust reconstruction (Technique 1
of LITERATURE.md), home cell W30/Tg0.4, DAMULT=3.

Ref: Cavaliere, Fezans, Kiehn, "Method to Account for Estimator-Induced
Previewed Information Losses ...", CEAS EuroGNC 2022, CEAS-GNC-2022-063 — the
DLR massed-measurement Tikhonov-regularized wind reconstruction. A real Doppler
lidar does NOT deliver one sample of W(t+dt): it re-measures the whole
approaching frozen gust profile every pulse (500 Hz, 9 range gates, raw
line-of-sight noise 1-3 m/s per shot), so every spatial point is measured
hundreds of times as it approaches, and a weighted-least-squares estimator
(weights 1/sigma_i^2, Tikhonov smoothness penalties) fuses the redundancy into
a delivered noise of 0.02-0.27 m/s.

This script replaces the axes A/E single-sample corruption Wc = W(t+dt) + eps
with an honest simulation of that redundancy: at every step the sensor takes
ONE new noisy sample of EACH of the Jmax future gust points and fuses per
SPATIAL point (inverse-variance running sums; optional Tikhonov re-fit over
the lookahead window). Estimates of a given point combine only measurements OF
that point, so — unlike the centred time-window avgK of axis E — the estimator
introduces no phase lag by construction (verified by the frac=0 anchor).
PHYSICAL HONESTY RULE: the true Wt[m] is read ONLY to synthesize a noisy
measurement of it; the controller sees nothing but the fused Wc.

  none    unmitigated single-sample baseline (== axis E 'none'), frac {.02,.05}
  fuse    flat per-sample sigma = frac*W0 (the SAME raw per-sample noise as
          'none', so the comparison isolates the pure redundancy gain,
          expected ~1/sqrt(Jmax)), lam=0, Jmax {10,25,50}
  fuseT   fusion + Tikhonov smoothing, Jmax=50, lam {1,10}: does DLR-style
          smoothing add anything over pure fusion at these noise levels?
  dlr     the headline arm: RAW line-of-sight noise 1->3 m/s linear in
          lookahead (3.3-10% of W0=30 — the real-lidar numbers) through the
          fusion, Jmax=50, lam {0,1}. Does the DLR sensor architecture rescue
          the +76.6% that collapses at raw sigma >= 1%*W0 with a single sample?
  anchor  frac=0 pipeline check (per-sample sigma=1e-6, fuse Jmax=50, lam=0):
          must reproduce the clean +76.58% — verifies the no-lag claim.

REQUIRED extra metric per point: the DELIVERED noise
    sigma_del = std(Wc_seen - W_true(t+dt))  over the gust window t <= Tg+0.5,
mean over seeds — THE raw -> delivered number for the thesis. Stored in the
point record and printed next to the stats line.

Controller: always the final optimal (R=3e-4). NSEED=6, rng 100+seed fresh per
rollout. Trajectories stored: all 6 seeds of dlr lam=0 + paired 'none' at
frac=0.05 (money plot: raw-lidar noise fused vs the same order of raw noise
unfused).

Output: results/E2_sensor.npz          Smoke: python3 e2_sensor.py --smoke
"""
import os
import sys
import numpy as np
import harness_noise as H

W0, Tg = 30.0, 0.4
SMOKE = '--smoke' in sys.argv
NSEED = 2 if SMOKE else 6
FRACS = (0.02,) if SMOKE else (0.02, 0.05)
JMAXS = (25,) if SMOKE else (10, 25, 50)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'E2_sensor.npz')

OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis E2-sensor | W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)
recs = [dict(kind='open', axis='E2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


# ---- Wc generators (controller-side sensor models) ---------------------------
def wc_plain(rng, frac):
    """Noisy 1-step preview (== axis A / axis E 'none'): the paired baseline."""
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


def wc_fuse(rng, sigma_fun, Jmax, lam=0.0):
    """
    Rolling massed-measurement fusion database (DLR Technique 1).

    At each step i the sensor takes ONE new noisy sample of each of the Jmax
    future gust points m = i+1 .. i+Jmax (clipped to N-1), with per-sample
    noise sigma_fun(j), j = lookahead in steps. Raw samples are NEVER clamped
    (clamping the raw stream would bias the fusion toward >0 at W=0); only the
    OUTPUT estimate is clamped at 0. Running inverse-variance sums per spatial
    node:  num[m] += y/sig^2,  den[m] += 1/sig^2   (allocated lazily, length N
    from the wc_fun N argument).

    Estimate of the needed preview W(t+dt) = node i+1:
      lam == 0: Wc = num[i+1]/den[i+1]  — by query time node i+1 has fused up
                to Jmax samples, taken at steps i+1-Jmax .. i.
      lam  > 0: Tikhonov re-fit over the window m in [i+1, min(i+Jmax, N-1)]:
                minimize  sum_m den_m (u_m - ybar_m)^2
                        + lam_eff * sum (u_{m+1} - 2 u_m + u_{m-1})^2
                with ybar_m = num_m/den_m (nodes with den_m == 0 get zero data
                weight), lam_eff = lam * mean(den_m over the window). Dense
                solve (window <= 50 nodes — trivial); take u at m = i+1.

    PRE-WARM: a real lidar measures the approaching field continuously from
    before t=0, so on the first call the database is seeded with the Jmax-1
    virtual steps ii = -(Jmax-1)..-1 (same measurement model, nodes clipped to
    the sim window). Without this the first Jmax nodes are under-averaged
    exactly during the gust onset — the branch-selection window. The pre-gust
    field is W=0 (true for the 1-cos gust), so no future information leaks.
    The extra rng draws shift the stream vs the paired arms (documented; the
    fusion arms are not draw-paired with 'none' anyway, sample counts differ).
    Early steps: den == 0 -> return 0.0 (defensive; cannot trigger after the
    pre-warm). Tail: m clipped to N-1 accumulates duplicate samples of
    the last node (den[N-1] inflated -> over-confident there); acceptable
    since W = 0 at the end and t_end is far outside the metric window.
    """
    js = np.arange(1, Jmax + 1)
    sigs = np.array([float(sigma_fun(j)) for j in js])
    inv2 = 1.0 / sigs**2
    st = {'num': None, 'den': None}

    def f(i, Wt, N):
        if st['num'] is None:
            st['num'] = np.zeros(N)
            st['den'] = np.zeros(N)
            # pre-warm: virtual steps before t=0 (see docstring)
            for ii in range(-(Jmax - 1), 0):
                mm = ii + js
                keep = mm >= 0
                mk = np.minimum(mm[keep], N - 1)
                yk = Wt[mk] + rng.normal(0.0, sigs[keep])
                np.add.at(st['num'], mk, yk * inv2[keep])
                np.add.at(st['den'], mk, inv2[keep])
        num, den = st['num'], st['den']
        # one new noisy shot of each approaching point (honesty rule: Wt[m] is
        # read only to synthesize the measurement)
        m_idx = np.minimum(i + js, N - 1)
        y = Wt[m_idx] + rng.normal(0.0, sigs)        # NO clamp on raw samples
        np.add.at(num, m_idx, y * inv2)              # add.at: tail duplicates
        np.add.at(den, m_idx, inv2)                  # accumulate correctly
        tgt = min(i + 1, N - 1)
        if den[tgt] == 0.0:
            return 0.0
        if lam == 0.0:
            wc = num[tgt] / den[tgt]
        else:
            hi = min(i + Jmax, N - 1)
            w = den[tgt:hi + 1].copy()               # data weights = den_m
            n = w.size
            ybar = np.zeros(n)
            good = w > 0.0
            ybar[good] = num[tgt:hi + 1][good] / w[good]
            if n < 3:                                # no curvature term possible
                wc = ybar[0]
            else:
                lam_eff = lam * float(np.mean(w))
                D = (np.eye(n - 2, n, 0) - 2.0 * np.eye(n - 2, n, 1)
                     + np.eye(n - 2, n, 2))          # second-difference operator
                A = np.diag(w) + lam_eff * (D.T @ D)
                u = np.linalg.solve(A, w * ybar)
                wc = u[0]
        return max(0.0, wc)                          # clamp the OUTPUT only
    return f


# ---- per-sample sigma models --------------------------------------------------
def sig_flat(frac):
    """sigma_j = frac*W0 for all j — same per-sample noise as the single-sample
    axis A/E baseline, isolating the pure redundancy gain (~1/sqrt(Jmax))."""
    s = frac * W0
    return lambda j: s


def sig_dlr(Jmax=50):
    """DLR-realistic RAW line-of-sight noise: sigma_j linear in lookahead from
    1.0 m/s at j=1 to 3.0 m/s at j=Jmax (Cavaliere et al. 2022: sigma grows
    with range). Delivered noise ~ sqrt(1/sum_j 1/sig_j^2) ~ 0.24 m/s at
    Jmax=50 — inside the published 0.02-0.27 m/s band."""
    return lambda j: 1.0 + 2.0 * (j - 1) / (Jmax - 1)


# ---- delivered-noise metric ----------------------------------------------------
def delivered_sigma(r):
    """std(Wc_seen - W_true(t+dt)) over the gust window t <= Tg+0.5."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    N = len(t)
    Wnext = Wt[np.minimum(np.arange(N) + 1, N - 1)]
    mw = t <= (Tg + 0.5)
    return float(np.std(Wc[mw] - Wnext[mw]))


# ---- one study point -----------------------------------------------------------
def run_point(make_wc, nseed=None, **cfg):
    ms, rs, sds = [], [], []
    for seed in range(nseed if nseed is not None else NSEED):
        rng = np.random.default_rng(100 + seed)
        r = H.rollout(H.make_optimal(R=3e-4), W0, Tg, wc_fun=make_wc(rng))
        ms.append(H.metrics(r, OL, Tg))
        rs.append(r)
        sds.append(delivered_sigma(r))
    sigma_del = float(np.mean(sds))
    rec = H.point_record(ms, sigma_del=sigma_del, **cfg)
    fs = f"{cfg['frac']:.2f}" if cfg.get('frac') is not None else ' dlr'
    print(f"  {cfg['arm']:6s} {cfg.get('detail', ''):12s} frac={fs}: "
          f"{H.fmt_stats(H.seed_stats(ms))}  sig_del={sigma_del:.3g} m/s",
          flush=True)
    return rec, ms, rs


points = {}   # (arm, detail, frac) -> (rec, ms, rs)

# ---- frac=0 anchor: pure pipeline / no-lag check (deterministic, 1 seed) -------
print("\n=== anchor: per-sample sigma=1e-6, fuse Jmax=50 lam=0 "
      "(must reproduce ~+76.58%) ===", flush=True)
k = ('fuse', 'anchor', 0.0)
points[k] = run_point(lambda rng: wc_fuse(rng, lambda j: 1e-6, 50, lam=0.0),
                      nseed=1, axis='E2', arm='fuse', detail='anchor',
                      W0=W0, Tg=Tg, R=3e-4, Jmax=50, lam=0.0, frac=0.0)

# ---- flat-sigma arms ------------------------------------------------------------
for frac in FRACS:
    print(f"\n=== frac={frac:.2f} (raw sigma={frac * W0:.2f} m/s per sample) ===",
          flush=True)

    k = ('none', '', frac)
    points[k] = run_point(lambda rng, f=frac: wc_plain(rng, f),
                          axis='E2', arm='none', detail='',
                          W0=W0, Tg=Tg, R=3e-4, Jmax=None, lam=None, frac=frac)

    for J in JMAXS:
        k = ('fuse', f'J={J}', frac)
        points[k] = run_point(
            lambda rng, f=frac, J=J: wc_fuse(rng, sig_flat(f), J, lam=0.0),
            axis='E2', arm='fuse', detail=f'J={J}',
            W0=W0, Tg=Tg, R=3e-4, Jmax=J, lam=0.0, frac=frac)

    if not SMOKE:
        for lam in (1.0, 10.0):
            k = ('fuseT', f'lam={lam:g}', frac)
            points[k] = run_point(
                lambda rng, f=frac, lam=lam: wc_fuse(rng, sig_flat(f), 50, lam=lam),
                axis='E2', arm='fuseT', detail=f'J=50 lam={lam:g}',
                W0=W0, Tg=Tg, R=3e-4, Jmax=50, lam=lam, frac=frac)

# ---- dlr headline arm: real raw lidar noise through the DLR architecture --------
if not SMOKE:
    print("\n=== dlr: raw LOS sigma 1->3 m/s (3.3-10% of W0), Jmax=50 ===",
          flush=True)
    for lam in (0.0, 1.0):
        k = ('dlr', f'lam={lam:g}', None)
        points[k] = run_point(
            lambda rng, lam=lam: wc_fuse(rng, sig_dlr(50), 50, lam=lam),
            axis='E2', arm='dlr', detail=f'lam={lam:g}',
            W0=W0, Tg=Tg, R=3e-4, Jmax=50, lam=lam, frac=None,
            raw_sigma='1-3m/s')

# ---- collect records + money-plot trajectories -----------------------------------
for key, (rec, ms, rs) in points.items():
    recs.append(rec)

if not SMOKE:
    rec_d, ms_d, rs_d = points[('dlr', 'lam=0', None)]
    for seed, (m, r) in enumerate(zip(ms_d, rs_d)):
        recs.append(H.traj_record(r, m, axis='E2', arm='dlr', detail='lam=0',
                                  W0=W0, Tg=Tg, R=3e-4, Jmax=50, lam=0.0,
                                  frac=None, raw_sigma='1-3m/s',
                                  seed=seed, label=f'E2S_dlr_s{seed}'))
    rec_n, ms_n, rs_n = points[('none', '', 0.05)]
    for seed, (m, r) in enumerate(zip(ms_n, rs_n)):
        recs.append(H.traj_record(r, m, axis='E2', arm='none', detail='',
                                  W0=W0, Tg=Tg, R=3e-4, frac=0.05,
                                  seed=seed, label=f'E2S_none_f0.05_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
