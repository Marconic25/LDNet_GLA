"""
Timing robustness of the E2-combo pipeline: preview shift + estimator refit rate.

shift: the sensor measures a time-shifted field Wt_meas[m] = Wt[m+d]
  (frozen-field advection error). d>0 -> the gust appears d steps EARLY in
  the preview (controller acts early); d<0 -> late (latency analogue; the
  one-step argmin died at d=-1). Passes through fusion untouched.
refit: measurements accumulate EVERY step (DLR lidar fires at 500 Hz) but the
  profile is re-SOLVED only every K steps (DLR refits at 10 Hz); between
  refits the cached window is re-sliced by the frozen-field offset i-i0.
  K-1+N <= Jmax keeps the slice inside the solved window -> K <= 42 with
  Jmax=50 (true DLR 10 Hz = K=50 would need Jmax >= 58; K=42 ~ 12 Hz is the
  honest comparison point). Clean refit is trivially flat (exact measurements
  make staleness invisible), so the refit arm runs at sigma=2% only.

Clean sweep (deterministic): d in {-10,-5,-2,-1,0,+1,+2,+5,+10}   (+-20 ms)
Noisy spot-checks (sigma=2%*W0, 6 seeds): d = +-5
Refit (sigma=2%*W0, 6 seeds): K in {1,5,10,25,42}
  K=1 == FusedPreviewSensor exactly (same rng stream): per-seed regression
  vs W_combo frac=2% [80.3860, 80.5235, 80.5510, 80.4897, 80.5364, 80.5024].

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, home cell W30/Tg0.4, DAMULT=3.
Output: results/B2_timing.npz
--smoke: clean {d=0, d=+2} + noisy {K=1, K=5} 2 seeds.
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'B2_timing.npz')

if SMOKE:
    SHIFT_CLEAN = [0, 2]
    SHIFT_NOISY = []
    REFIT_KS    = [1, 5]
else:
    SHIFT_CLEAN = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    SHIFT_NOISY = [-5, 5]
    REFIT_KS    = [1, 5, 10, 25, 42]


def shift_field(Wt, d):
    """Wt_meas[m] = Wt[m+d]; edge padding is exact (W=0 outside the gust)."""
    if d == 0:
        return Wt.copy()
    if d > 0:
        return np.concatenate([Wt[d:], np.full(d, Wt[-1])])
    return np.concatenate([np.full(-d, Wt[0]), Wt[:d]])


def _delivered(r):
    """(sigma_del, bias_del) of the Wc channel vs the true next-step gust."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw  = t <= (Tg + 0.5)
    err = Wc[mw] - Wnext[mw]
    return float(np.std(err)), float(np.mean(err))


class RefitSensor(FusedPreviewSensor):
    """
    FusedPreviewSensor that re-solves the fused profile only every K steps.

    Measurements accumulate into (num, den) EVERY step; the window solve is
    cached at step _i0 and re-sliced by the frozen-field offset i-_i0 between
    refits. wc_fun body duplicated from optimal.FusedPreviewSensor (kept in
    sync by eye) with the solve made conditional. K=1 reproduces the parent
    bit-exactly (same rng draw sequence).
    """

    def __init__(self, rng, sigma_fun, Jmax, N, lam=0.0, K=1):
        super().__init__(rng, sigma_fun, Jmax, N, lam=lam)
        self.K   = int(K)
        self._i0 = None
        self._u0 = None

    def wc_fun(self, i, Wt, Nsteps):
        js, sigs, inv2 = self.js, self.sigs, self.inv2
        if self.num is None:
            self.num = np.zeros(Nsteps)
            self.den = np.zeros(Nsteps)
            for ii in range(-(self.Jmax - 1), 0):
                mm = ii + js
                keep = mm >= 0
                mk = np.minimum(mm[keep], Nsteps - 1)
                yk = Wt[mk] + self.rng.normal(0.0, sigs[keep])
                np.add.at(self.num, mk, yk * inv2[keep])
                np.add.at(self.den, mk, inv2[keep])

        m_idx = np.minimum(i + js, Nsteps - 1)
        y = Wt[m_idx] + self.rng.normal(0.0, sigs)
        np.add.at(self.num, m_idx, y * inv2)
        np.add.at(self.den, m_idx, inv2)

        if self._i0 is None or (i - self._i0) >= self.K:
            lo = min(i + 1, Nsteps - 1)
            hi = min(i + self.Jmax, Nsteps - 1)
            w = self.den[lo:hi + 1].copy()
            n = w.size
            ybar = np.zeros(n)
            good = w > 0.0
            ybar[good] = self.num[lo:hi + 1][good] / w[good]
            if self.lam == 0.0 or n < 3:
                u = ybar
            else:
                lam_eff = self.lam * float(np.mean(w))
                D = (np.eye(n - 2, n, 0) - 2.0 * np.eye(n - 2, n, 1)
                     + np.eye(n - 2, n, 2))
                A = np.diag(w) + lam_eff * (D.T @ D)
                u = np.linalg.solve(A, w * ybar)
            self._i0 = i
            self._u0 = u

        off = i - self._i0
        idx = np.minimum(off + np.arange(self.N), self._u0.size - 1)
        self.last = np.maximum(0.0, self._u0[idx])
        return float(self.last[0])

    def reset(self):
        super().reset()
        self._i0 = None
        self._u0 = None


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


def make_mpc():
    return MPCPreviewController(
        H.aero, U=H.U, dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=H.CLTRIM, N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)


def make_combo_shift(rng, frac, d):
    """Combo whose sensor measures the time-shifted field Wt[m+d]."""
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor = FusedPreviewSensor(rng, sigma_fun, JMAX, N, lam=LAM)
    cache = {}
    def wc(i, Wt, Nsteps):
        if 'Wm' not in cache:
            cache['Wm'] = shift_field(Wt, d)    # plant keeps the true Wt
        return sensor.wc_fun(i, cache['Wm'], Nsteps)
    return _ComboCtrl(sensor, make_mpc()), wc


def make_combo_refit(rng, frac, K):
    """Combo whose sensor re-solves the profile only every K steps."""
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor = RefitSensor(rng, sigma_fun, JMAX, N, lam=LAM, K=K)
    return _ComboCtrl(sensor, make_mpc()), sensor.wc_fun


OL   = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# B2_timing | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='B2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def run_point(arm, val, frac, nseed):
    ms, sds, bds = [], [], []
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        if arm == 'shift':
            ctrl, wc = make_combo_shift(rng, frac, int(val))
        else:
            ctrl, wc = make_combo_refit(rng, frac, int(val))
        r = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms.append(H.metrics(r, OL, Tg))
        sd, bd = _delivered(r)
        sds.append(sd); bds.append(bd)
    rec = H.point_record(ms, axis='B2', arm=arm, value=int(val),
                         W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM, R_du=R_DU,
                         frac=frac, sigma_del=float(np.mean(sds)),
                         bias_del=float(np.mean(bds)))
    print(f"  {arm}={int(val):+d} frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms))}  "
          f"sig_del={np.mean(sds):.3g}  bias_del={np.mean(bds):+.3g} m/s", flush=True)
    return rec


for d in SHIFT_CLEAN:
    recs.append(run_point('shift', d, 0.0, 1))

for d in SHIFT_NOISY:
    recs.append(run_point('shift', d, 0.02, NSEED))

for K in REFIT_KS:
    recs.append(run_point('refit', K, 0.02, NSEED))

H.save_records(OUT, recs)
print("# DONE", flush=True)
