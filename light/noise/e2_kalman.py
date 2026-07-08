"""
Axis E2-KF — disturbance-model Kalman filtering of the noisy gust preview,
home cell W30/Tg0.4, DAMULT=3 (LITERATURE.md Technique 6).

Reference: Forte, Nguyen, Xiong, "Gust Load Alleviation Control and Gust
Estimation for a High Aspect Ratio Wing Wind Tunnel Model", AIAA SciTech
2023 (NASA), disturbance-state estimation with an internal gust model:
the gust is not consumed as a measured signal but as a STATE of an assumed
disturbance model estimated by an extended-state Kalman filter, so sensor
noise cannot command gust trajectories the internal dynamics cannot produce
(their wing-root strain alleviation: -69.07% clean vs -68.45/-68.51% noisy —
essentially free graceful degradation). This script is the cheap scalar
variant as a pure preprocessing wrapper (wc_fun): the noisy 1-step preview
measurement y_i = W((i+1)*dt) + eps, eps ~ N(0, (frac*W0)^2), is tracked by
a discrete constant-velocity (CV) Kalman filter and the controller receives
the filtered W-state instead of the raw sample. Causal and phase-aware —
unlike the centred moving average (rings) and the plain LPF (lag kills).

Filter (one fresh instance per rollout, plain 2x2 numpy in the closure):
    state s = [W, Wdot]
    F  = [[1, dt], [0, 1]]
    Q  = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]    exact discretisation of
         continuous white-noise-acceleration PSD q, units m^2/s^3
         (i.e. (m/s)^2 per s)
    Rm = (frac*W0)^2                             measurement noise, frac > 0
    init s = [0, 0], P = diag(Rm, 300.0^2)
    per step: raw = max(0, W_next_true + eps); predict; update with raw;
    return max(0, s[0]) — the OUTPUT is clamped like every sensor model in
    this study, the INTERNAL state is deliberately left unclamped so the
    filter can represent (and reject) below-zero excursions consistently.
    frac=0 anchor: the KF needs Rm > 0 to be defined, so the clean-gust
    anchors use Rm = (0.001*W0)^2 (0.03 m/s — negligible vs the gust).

Why the q sweep {1e4, 1e5, 1e6, 1e7}: the 1-cos gust
W(t) = (W0/2)(1 - cos(2 pi t/Tg)) at W0=30, Tg=0.4 reaches
    max |Wdot|  = W0*pi/Tg        ~  236 (m/s)/s
    max |Wddot| = 2*W0*(pi/Tg)^2  ~ 3701 (m/s)/s^2.
A CV model has no acceleration state: the only way the filter can follow the
gust's deterministic acceleration is by allowing it through Q, otherwise it
lags the ramp — but the more acceleration Q admits, the more measurement
noise passes to the controller. The q sweep maps exactly this
lag-vs-delivered-noise trade-off; the frac=0 anchors isolate the lag end
(pure distortion of the clean gust, to compare against the plain clean
anchor +76.58%), sigma_del quantifies the noise end.

Arms (controller always H.make_optimal(R=3e-4), NSEED=6, rng 100+seed fresh
per rollout, frac in {0.02, 0.05}):
    none     unmitigated noisy preview (paired in-file baseline, == axis E)
    kf       wc_kalman, q in {1e4, 1e5, 1e6, 1e7}
    anchors  frac=0 through the KF, one rollout per q (deterministic)

Extra metric per point: sigma_del = std(Wc_seen - W_true_next) over the gust
window t <= Tg+0.5, W_true_next[i] = Wt[min(i+1, N-1)], averaged over seeds
— the noise the filter DELIVERS to the controller (raw sigma at frac=0.02
is 0.6 m/s); for the frac=0 anchors it is the pure lag distortion in m/s.

Trajectories: all seeds of the best q per frac (pick rule: max mean CLred
s.t. zero flags, fallback max mean) + the paired 'none' seeds
(labels E2K_best_f<frac>_s<seed> / E2K_none_f<frac>_s<seed>).

Output: results/E2_kalman.npz  (schema: harness_noise.py record helpers)
--smoke: frac=0 anchor for q=1e6 + arms none and kf(q=1e6) at frac=0.02
with 2 seeds; same OUT.
"""
import os
import sys
import numpy as np
import harness_noise as H

W0, Tg = 30.0, 0.4
SMOKE  = '--smoke' in sys.argv[1:]
FRACS  = (0.02,) if SMOKE else (0.02, 0.05)
QS     = (1e6,)  if SMOKE else (1e4, 1e5, 1e6, 1e7)
NSEED  = 2 if SMOKE else 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'E2_kalman.npz')

OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis E2-KF | W30/Tg0.4 DAMULT={os.environ.get('DAMULT', '1')} | "
      f"open cex0={cex0:.4f}{' | SMOKE' if SMOKE else ''}", flush=True)
recs = [dict(kind='open', axis='E2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


# ---- Wc generators (controller-side sensor models) ---------------------------
def wc_plain(rng, frac):
    """Noisy 1-step preview (== axis A)."""
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


def wc_kalman(rng, frac, q):
    """Constant-velocity Kalman filter on the noisy preview (see module doc).

    frac=0 anchor uses Rm=(0.001*W0)^2 so the filter stays defined; the rng
    draw is kept (it returns exactly 0 at sigma=0) for stream parity.
    """
    dt = H.DT
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    Q = q * np.array([[dt**3 / 3.0, dt**2 / 2.0],
                      [dt**2 / 2.0, dt]])
    Rm = ((frac if frac > 0.0 else 0.001) * W0) ** 2
    st = {'s': np.zeros(2), 'P': np.diag([Rm, 300.0**2])}

    def f(i, Wt, N):
        raw = max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))
        s = F @ st['s']                              # predict
        P = F @ st['P'] @ F.T + Q
        K = P[:, 0] / (P[0, 0] + Rm)                 # update, Hobs = [1, 0]
        s = s + K * (raw - s[0])
        P = P - np.outer(K, P[0, :])
        st['s'], st['P'] = s, P
        return max(0.0, s[0])       # clamp the OUTPUT only, never the state
    return f


# ---- delivered-noise metric ---------------------------------------------------
def delivered_sigma(r):
    """std(Wc_seen - W_true_next) over the gust window t <= Tg+0.5."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw = t <= (Tg + 0.5)
    return float(np.std(Wc[mw] - Wnext[mw]))


# ---- one study point ----------------------------------------------------------
def run_point(make_ctrl, make_wc, frac, **cfg):
    """NSEED seeds (1 deterministic run when frac=0); adds sigma_del."""
    ms, rs = [], []
    nseed = 1 if frac == 0.0 else NSEED
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        r = H.rollout(make_ctrl(), W0, Tg, wc_fun=make_wc(rng, frac))
        ms.append(H.metrics(r, OL, Tg)); rs.append(r)
    sig = float(np.mean([delivered_sigma(r) for r in rs]))
    rec = H.point_record(ms, frac=frac, sigma_del=sig, **cfg)
    print(f"  {cfg['arm']:6s} {cfg.get('detail',''):8s} frac={frac:.2f}: "
          f"{H.fmt_stats(H.seed_stats(ms))} | sigma_del={sig:.3f} m/s", flush=True)
    return rec, ms, rs


points = {}   # (arm, detail, frac) -> (rec, ms, rs)

# ---- frac=0 anchors: pure KF lag on the clean gust ----------------------------
print("\n=== frac=0 anchors (KF pure lag; plain clean anchor is +76.58%) ===",
      flush=True)
for qv in QS:
    k = ('kf', f'q={qv:g}', 0.0)
    points[k] = run_point(lambda: H.make_optimal(R=3e-4),
                          lambda rng, f, qv=qv: wc_kalman(rng, f, qv), 0.0,
                          axis='E2', arm='kf', detail=f'q={qv:g}',
                          W0=W0, Tg=Tg, R=3e-4, q=qv)

# ---- noisy points -------------------------------------------------------------
for frac in FRACS:
    print(f"\n=== frac={frac:.2f} (raw sigma={frac*W0:.2f} m/s) ===", flush=True)

    k = ('none', '', frac)
    points[k] = run_point(lambda: H.make_optimal(R=3e-4), wc_plain, frac,
                          axis='E2', arm='none', detail='', W0=W0, Tg=Tg, R=3e-4)

    for qv in QS:
        k = ('kf', f'q={qv:g}', frac)
        points[k] = run_point(lambda: H.make_optimal(R=3e-4),
                              lambda rng, f, qv=qv: wc_kalman(rng, f, qv), frac,
                              axis='E2', arm='kf', detail=f'q={qv:g}',
                              W0=W0, Tg=Tg, R=3e-4, q=qv)


# ---- collect records + trajectories of the best q per frac --------------------
def best_kf(frac):
    """Pick rule: max mean CLred s.t. zero flags; fallback max mean."""
    cand = [(key, v[0]) for key, v in points.items()
            if key[0] == 'kf' and key[2] == frac]
    clean = [(key, rec) for key, rec in cand if rec['nflag'] == 0]
    pool = clean if clean else cand
    return max(pool, key=lambda kv: kv[1]['mean'])


for key, (rec, ms, rs) in points.items():
    recs.append(rec)

for frac in FRACS:
    key_b, rec_b = best_kf(frac)
    _, ms_b, rs_b = points[key_b]
    print(f"# best kf frac={frac:.2f}: {key_b[1]} mean {rec_b['mean']:+.1f}% "
          f"(sigma_del={rec_b['sigma_del']:.3f} m/s)", flush=True)
    for seed, (m, r) in enumerate(zip(ms_b, rs_b)):
        recs.append(H.traj_record(r, m, axis='E2', arm='kf', detail=key_b[1],
                                  W0=W0, Tg=Tg, frac=frac, q=rec_b['q'],
                                  seed=seed, label=f'E2K_best_f{frac:g}_s{seed}'))
    # unmitigated trajectories for the same frac (paired money plot)
    rec_n, ms_n, rs_n = points[('none', '', frac)]
    for seed, (m, r) in enumerate(zip(ms_n, rs_n)):
        recs.append(H.traj_record(r, m, axis='E2', arm='none', detail='',
                                  W0=W0, Tg=Tg, frac=frac,
                                  seed=seed, label=f'E2K_none_f{frac:g}_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
