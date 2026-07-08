"""
White-noise robustness of the E2-combo pipeline vs the one-step argmin.

sigma/W0 in {0, 0.01, 0.02, 0.05, 0.10, 0.20} -- white Gaussian noise on each
individual sensor measurement BEFORE fusion (i.e. the raw-shot sigma, not the
delivered sigma). With Jmax=50 the fusion already reduces the effective preview
noise dramatically; sigma_del = std(Wc - W_true_next) is printed and logged.

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, 6 seeds (rng 100+seed), home cell
W30/Tg0.4, DAMULT=3. Paired baseline: one-step none (wc_plain, same frac).
Metrics and t<=Tg+0.5 window identical to harness_noise axes.

Output: results/W_combo.npz
--smoke: sigma in {0, 0.02}, 2 seeds, same OUT schema.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import harness_noise as H
from optimal import FusedPreviewSensor, MPCPreviewController

W0, Tg   = 30.0, 0.4
JMAX, N  = 50, 8
R        = 3e-4
R_DU     = 0.0
LAM      = 0.0
SMOKE    = '--smoke' in sys.argv
NSEED    = 2 if SMOKE else 6
FRACS    = [0.0, 0.02] if SMOKE else [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'W_combo.npz')


def _delivered_sigma(r):
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw    = t <= (Tg + 0.5)
    return float(np.std(Wc[mw] - Wnext[mw]))


# ---- harness adapter: wc_fun sets sensor.last; compute reads it ----------------
class _ComboCtrl:
    def __init__(self, sensor, mpc):
        self._sensor = sensor
        self._mpc    = mpc
        self._delta_prev = 0.0

    def reset(self):
        self._sensor.reset()
        self._mpc.reset()
        self._delta_prev = 0.0

    def compute(self, state, W_true, Wc):
        # sensor.last set by wc_fun before this call (harness protocol)
        return self._mpc.compute(state, self._sensor.last)


def make_combo(rng, frac):
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor    = FusedPreviewSensor(rng, sigma_fun, JMAX, N, lam=LAM)
    mpc       = MPCPreviewController(
        H.aero, U=H.U, dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=H.CLTRIM, N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    return _ComboCtrl(sensor, mpc), sensor.wc_fun


def wc_plain(rng, frac):
    return lambda i, Wt, Nsteps: max(0.0, Wt[min(i+1, Nsteps-1)]
                                     + rng.normal(0.0, frac * W0))


OL   = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# W_combo | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='Wco', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]

for frac in FRACS:
    ms_c, rs_c, sds_c = [], [], []
    ms_n, rs_n, sds_n = [], [], []
    for seed in range(NSEED):
        rng_c = np.random.default_rng(100 + seed)
        rng_n = np.random.default_rng(100 + seed)

        # combo arm
        ctrl, wc = make_combo(rng_c, frac)
        rc = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms_c.append(H.metrics(rc, OL, Tg)); rs_c.append(rc)
        sds_c.append(_delivered_sigma(rc))

        # one-step baseline (paired, same frac)
        if frac > 0.0:
            none_ctrl = H.make_optimal(R=R)
            rc_n = H.rollout(none_ctrl, W0, Tg, wc_fun=wc_plain(rng_n, frac))
            ms_n.append(H.metrics(rc_n, OL, Tg)); rs_n.append(rc_n)
            sds_n.append(_delivered_sigma(rc_n))

    sig_c = float(np.mean(sds_c))
    rec_c = H.point_record(ms_c, axis='Wco', arm='combo',
                           W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM,
                           R_du=R_DU, frac=frac, sigma_del=sig_c)
    recs.append(rec_c)
    print(f"  combo frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms_c))}  "
          f"sig_del={sig_c:.3g} m/s", flush=True)

    if frac > 0.0 and ms_n:
        sig_n = float(np.mean(sds_n))
        rec_n = H.point_record(ms_n, axis='Wco', arm='none',
                               W0=W0, Tg=Tg, R=R, frac=frac, sigma_del=sig_n)
        recs.append(rec_n)
        print(f"  none  frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms_n))}  "
              f"sig_del={sig_n:.3g} m/s", flush=True)

H.save_records(OUT, recs)
print("# DONE", flush=True)
