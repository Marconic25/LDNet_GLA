"""
White-noise robustness of the E2-combo pipeline.

sigma/W0 in {0, 0.01, 0.02, 0.05, 0.10, 0.20} -- white Gaussian noise on each
individual sensor measurement BEFORE fusion (i.e. the raw-shot sigma, not the
delivered sigma). With Jmax=50 the fusion already reduces the effective preview
noise dramatically; sigma_del = std(Wc - W_true_next) is printed and logged.

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, 6 seeds (rng 100+seed), DAMULT=3.
Cell defaults to the home cell W30/Tg0.4; override with CELL_W0 / CELL_TG
(R=3e-4 is R* for both W30/Tg0.4 and W10/Tg0.4 -- check summary.md before
running a cell with a different R*).
Metrics and t<=Tg+0.5 window identical to harness_noise axes.

Output: results/W_combo.npz (home cell) / results/W_combo_W<W0>[T<10Tg>].npz.
--smoke: sigma in {0, 0.02}, 2 seeds, same OUT schema, '_smoke' suffix on the
filename so a smoke run can never clobber a production .npz.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import harness_noise as H
from optimal import FusedPreviewSensor, MPCPreviewController

# Gust cell: defaults to the home cell W30/Tg0.4 so that running this script
# with no environment override reproduces the published thesis artifact
# (results/W_combo.npz) exactly. Override with CELL_W0 / CELL_TG.
W0       = float(os.environ.get('CELL_W0', 30.0))
Tg       = float(os.environ.get('CELL_TG', 0.4))
JMAX, N  = 50, 8
R        = 3e-4
R_DU     = 0.0
LAM      = 0.0
SMOKE    = '--smoke' in sys.argv
NSEED    = 2 if SMOKE else 6
FRACS    = [0.0, 0.02] if SMOKE else [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]


def cell_tag(W0, Tg):
    """'' for the home cell (W30/Tg0.4), else '_W<W0>[T<10*Tg>]'.

    Empty for the home cell so the default run keeps writing W_combo.npz;
    non-home cells get their own file (naming precedent: E2_combo_cells_W10T07).
    """
    if (W0, Tg) == (30.0, 0.4):
        return ''
    tag = f'_W{W0:g}'
    if Tg != 0.4:
        tag += f'T{round(Tg * 10):02d}'
    return tag


TAG = cell_tag(W0, Tg)
# smoke output never collides with a production .npz
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                   f"W_combo{TAG}{'_smoke' if SMOKE else ''}.npz")


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
        return self._mpc.compute(state, self._sensor.last, self._sensor.cur)


def make_combo(rng, frac):
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor    = FusedPreviewSensor(rng, sigma_fun, JMAX, N, lam=LAM)
    mpc       = MPCPreviewController(
        H.aero, U=H.U, dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=H.CLTRIM, N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    return _ComboCtrl(sensor, mpc), sensor.wc_fun


OL   = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# W_combo | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='Wco', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]

for frac in FRACS:
    ms_c, rs_c, sds_c = [], [], []
    for seed in range(NSEED):
        rng_c = np.random.default_rng(100 + seed)

        # combo arm
        ctrl, wc = make_combo(rng_c, frac)
        rc = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms_c.append(H.metrics(rc, OL, Tg)); rs_c.append(rc)
        sds_c.append(_delivered_sigma(rc))

    sig_c = float(np.mean(sds_c))
    rec_c = H.point_record(ms_c, axis='Wco', arm='combo',
                           W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM,
                           R_du=R_DU, frac=frac, sigma_del=sig_c)
    recs.append(rec_c)
    print(f"  combo frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms_c))}  "
          f"sig_del={sig_c:.3g} m/s", flush=True)

H.save_records(OUT, recs)
print("# DONE", flush=True)
