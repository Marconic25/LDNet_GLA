"""
Calibration robustness of the E2-combo pipeline: additive bias + gain error.

The sensor measures a mis-calibrated field Wt_meas = gain*Wt + bias
(bias = value*W0) with per-shot white noise sigma_fun(j). Inverse-variance
fusion removes VARIANCE, not systematic error: bias and gain pass through
untouched (bias_del = mean(Wc - Wnext) is logged to verify), so this axis
measures the MPC's own calibration tolerance -- the lidar bias/gain spec.

Clean sweep (sigma=1e-9, deterministic, 1 rollout/point):
  anchor bias=0 ; bias value in {+-0.02,+-0.05,+-0.10} ; gain in {0.8,0.9,1.1,1.2}
Noisy spot-checks (sigma=2%*W0 flat, 6 seeds rng 100+seed):
  bias +-0.05 ; gain 1.2

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, home cell W30/Tg0.4, DAMULT=3.
Output: results/A2_calib.npz
--smoke: clean {bias 0, bias +0.05} + noisy {bias +0.05} 2 seeds.
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'A2_calib.npz')

if SMOKE:
    CLEAN_PTS = [('bias', 0.0), ('bias', 0.05)]
    NOISY_PTS = [('bias', 0.05)]
else:
    CLEAN_PTS = ([('bias', 0.0)]
                 + [('bias', b) for b in (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)]
                 + [('gain', g) for g in (0.8, 0.9, 1.1, 1.2)])
    NOISY_PTS = [('bias', -0.05), ('bias', 0.05), ('gain', 1.2)]


def _delivered(r):
    """(sigma_del, bias_del) of the Wc channel vs the true next-step gust."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw  = t <= (Tg + 0.5)
    err = Wc[mw] - Wnext[mw]
    return float(np.std(err)), float(np.mean(err))


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


def make_combo(rng, frac, bias=0.0, gain=1.0):
    """Combo whose sensor measures the mis-calibrated field gain*Wt + bias."""
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor    = FusedPreviewSensor(rng, sigma_fun, JMAX, N, lam=LAM)
    mpc       = MPCPreviewController(
        H.aero, U=H.U, dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=H.CLTRIM, N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    cache = {}
    def wc(i, Wt, Nsteps):
        if 'Wm' not in cache:
            cache['Wm'] = gain * Wt + bias      # plant keeps the true Wt
        return sensor.wc_fun(i, cache['Wm'], Nsteps)
    return _ComboCtrl(sensor, mpc), wc


OL   = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# A2_calib | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='A2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def run_point(arm, val, frac, nseed):
    bias = val * W0 if arm == 'bias' else 0.0
    gain = val      if arm == 'gain' else 1.0
    ms, sds, bds = [], [], []
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        ctrl, wc = make_combo(rng, frac, bias=bias, gain=gain)
        r = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms.append(H.metrics(r, OL, Tg))
        sd, bd = _delivered(r)
        sds.append(sd); bds.append(bd)
    rec = H.point_record(ms, axis='A2', arm=arm, value=float(val),
                         W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM, R_du=R_DU,
                         frac=frac, sigma_del=float(np.mean(sds)),
                         bias_del=float(np.mean(bds)))
    print(f"  {arm}={val:+.2f} frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms))}  "
          f"sig_del={np.mean(sds):.3g}  bias_del={np.mean(bds):+.3g} m/s", flush=True)
    return rec


for arm, val in CLEAN_PTS:
    recs.append(run_point(arm, val, 0.0, 1))

for arm, val in NOISY_PTS:
    recs.append(run_point(arm, val, 0.02, NSEED))

H.save_records(OUT, recs)
print("# DONE", flush=True)
