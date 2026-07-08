"""
Axis E — mitigations under preview noise, home cell W30/Tg0.4, DAMULT=3.

All arms see white noise sigma = frac*W0 on the gust signal (rng 100+seed,
identical streams across arms -> paired comparison), frac in {0.02, 0.05},
6 seeds, same metrics/flags/pick rule as everywhere else.

  none    unmitigated baseline: noisy preview straight into the controller
  lpf     1st-order LPF on the noisy preview, tau in {5,10} ms (the 76/ sweet
          spot; >=20 ms is known to collapse by lag — full sweep = axis B)
  R       effort weight up: R in {1e-3, 3e-3, 1e-2} (baseline 3e-4)
  Rdu     move suppression: R_du in {1e-4, 1e-3, 1e-2} at R=3e-4
  avgK    multi-sample preview averaging: a real LIDAR measures the whole
          approaching profile, so each step it has K noisy samples
          W(t + j*dt) + eps_j on a window centred on the needed k=1;
          Wc = mean_j max(0, W_j + eps_j), K in {3,5,9}. Noise falls ~1/sqrt(K)
          at the price of smoothing the true gust over (K-1)*dt (<=16 ms).
  combo   best avgK  +  best of (R | Rdu) family (picked by max mean CLred
          s.t. zero flags, else max mean) — run last, adaptively
  mpc     MPC N4 gate-none reference (76/'s noise-tolerant alternative):
          NO preview, sees max(0, W(t) + eps)
  prop    prop-W under the same noise lives in axis A (results/A_white.npz)

Trajectories stored: all 6 seeds of the overall-best mitigated config at each
frac (money plot: worst unmitigated seed vs best mitigated).

Output: results/E_mitigation.npz
"""
import os
import numpy as np
import harness_noise as H
from controllers_ref import make_optimal_rdu, MPCConstRef

W0, Tg = 30.0, 0.4
FRACS = (0.02, 0.05)
NSEED = 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'E_mitigation.npz')

OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis E | W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}", flush=True)
recs = [dict(kind='open', axis='E', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


# ---- Wc generators (controller-side sensor models) ---------------------------
def wc_plain(rng, frac):
    """Noisy 1-step preview (== axis A)."""
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


def wc_lpf(rng, frac, tau_ms):
    """Noisy preview through a 1st-order LPF, est(0)=0."""
    a = float(np.exp(-H.DT / (tau_ms / 1000.0)))
    state = {'est': 0.0}
    def f(i, Wt, N):
        raw = max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))
        state['est'] = a * state['est'] + (1.0 - a) * raw
        return state['est']
    return f


def wc_avgK(rng, frac, K):
    """Mean of K independent noisy samples on a window centred on k=1."""
    js = np.arange(1 - K // 2, 1 + K // 2 + 1)      # e.g. K=5 -> [-1..3]
    def f(i, Wt, N):
        idx = np.clip(i + js, 0, N - 1)
        return float(np.mean(np.maximum(
            0.0, Wt[idx] + rng.normal(0.0, frac * W0, size=K))))
    return f


def wc_current(rng, frac):
    """Noisy CURRENT gust (no preview) — for the MPC reference."""
    return lambda i, Wt, N: max(0.0, Wt[i] + rng.normal(0.0, frac * W0))


# ---- one study point ----------------------------------------------------------
def run_point(make_ctrl, make_wc, frac, keep_traj=False, **cfg):
    ms, rs = [], []
    for seed in range(NSEED):
        rng = np.random.default_rng(100 + seed)
        r = H.rollout(make_ctrl(), W0, Tg, wc_fun=make_wc(rng, frac))
        ms.append(H.metrics(r, OL, Tg)); rs.append(r)
    rec = H.point_record(ms, frac=frac, **cfg)
    print(f"  {cfg['arm']:6s} {cfg.get('detail',''):12s} frac={frac:.2f}: "
          f"{H.fmt_stats(H.seed_stats(ms))}", flush=True)
    return rec, ms, rs


points = {}   # (arm, detail, frac) -> (rec, ms, rs)

for frac in FRACS:
    print(f"\n=== frac={frac:.2f} (sigma={frac*W0:.2f} m/s) ===", flush=True)

    k = ('none', '', frac)
    points[k] = run_point(lambda: H.make_optimal(R=3e-4), wc_plain, frac,
                          axis='E', arm='none', detail='', W0=W0, Tg=Tg, R=3e-4)

    for tau in (5.0, 10.0):
        k = ('lpf', f'tau={tau:g}ms', frac)
        points[k] = run_point(lambda: H.make_optimal(R=3e-4),
                              lambda rng, f, tau=tau: wc_lpf(rng, f, tau), frac,
                              axis='E', arm='lpf', detail=f'tau={tau:g}ms',
                              W0=W0, Tg=Tg, R=3e-4, tau_ms=tau)

    for R in (1e-3, 3e-3, 1e-2):
        k = ('R', f'R={R:g}', frac)
        points[k] = run_point(lambda R=R: H.make_optimal(R=R), wc_plain, frac,
                              axis='E', arm='R', detail=f'R={R:g}',
                              W0=W0, Tg=Tg, R=R)

    for Rdu in (1e-4, 1e-3, 1e-2):
        k = ('Rdu', f'Rdu={Rdu:g}', frac)
        points[k] = run_point(lambda Rdu=Rdu: make_optimal_rdu(R=3e-4, R_du=Rdu),
                              wc_plain, frac,
                              axis='E', arm='Rdu', detail=f'Rdu={Rdu:g}',
                              W0=W0, Tg=Tg, R=3e-4, R_du=Rdu)

    for K in (3, 5, 9):
        k = ('avgK', f'K={K}', frac)
        points[k] = run_point(lambda: H.make_optimal(R=3e-4),
                              lambda rng, f, K=K: wc_avgK(rng, f, K), frac,
                              axis='E', arm='avgK', detail=f'K={K}',
                              W0=W0, Tg=Tg, R=3e-4, K=K)

    k = ('mpc', 'N4-none', frac)
    points[k] = run_point(lambda: MPCConstRef(N=4, R=3e-4), wc_current, frac,
                          axis='E', arm='mpc', detail='N4-none',
                          W0=W0, Tg=Tg, R=3e-4)


# ---- adaptive combo: best avgK + best (R | Rdu) -------------------------------
def best_in(arm, frac):
    """Pick rule: max mean CLred s.t. zero flags; fallback max mean."""
    cand = [(key, v[0]) for key, v in points.items()
            if key[0] == arm and key[2] == frac]
    clean = [(key, rec) for key, rec in cand if rec['nflag'] == 0]
    pool = clean if clean else cand
    return max(pool, key=lambda kv: kv[1]['mean'])


for frac in FRACS:
    (k_avg, rec_avg) = best_in('avgK', frac)
    (k_reg, rec_reg) = max([best_in('R', frac), best_in('Rdu', frac)],
                           key=lambda kv: (kv[1]['nflag'] == 0, kv[1]['mean']))
    K = rec_avg['K']
    R_c   = rec_reg['R'] if k_reg[0] == 'R' else 3e-4
    Rdu_c = rec_reg.get('R_du', 0.0) if k_reg[0] == 'Rdu' else 0.0
    detail = f'K={K}+{k_reg[1]}'
    print(f"\n=== combo frac={frac:.2f}: avgK {k_avg[1]} + {k_reg[1]} ===", flush=True)
    k = ('combo', detail, frac)
    points[k] = run_point(lambda: make_optimal_rdu(R=R_c, R_du=Rdu_c),
                          lambda rng, f, K=K: wc_avgK(rng, f, K), frac,
                          axis='E', arm='combo', detail=detail,
                          W0=W0, Tg=Tg, R=R_c, R_du=Rdu_c, K=K)


# ---- collect records + trajectories of the overall best mitigation ------------
for key, (rec, ms, rs) in points.items():
    recs.append(rec)

for frac in FRACS:
    mit = [(key, v) for key, v in points.items()
           if key[2] == frac and key[0] not in ('none', 'mpc')]
    clean = [(key, v) for key, v in mit if v[0]['nflag'] == 0]
    pool = clean if clean else mit
    key_best, (rec_b, ms_b, rs_b) = max(pool, key=lambda kv: kv[1][0]['mean'])
    print(f"# best mitigation frac={frac:.2f}: {key_best[0]} {key_best[1]} "
          f"mean {rec_b['mean']:+.1f}%", flush=True)
    for seed, (m, r) in enumerate(zip(ms_b, rs_b)):
        recs.append(H.traj_record(r, m, axis='E', arm=key_best[0],
                                  detail=key_best[1], W0=W0, Tg=Tg, frac=frac,
                                  seed=seed, label=f'E_best_f{frac:g}_s{seed}'))
    # unmitigated trajectories for the same frac (paired money plot)
    rec_n, ms_n, rs_n = points[('none', '', frac)]
    for seed, (m, r) in enumerate(zip(ms_n, rs_n)):
        recs.append(H.traj_record(r, m, axis='E', arm='none', detail='',
                                  W0=W0, Tg=Tg, frac=frac,
                                  seed=seed, label=f'E_none_f{frac:g}_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
