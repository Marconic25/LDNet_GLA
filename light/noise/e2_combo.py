"""
Axis E2-combo — the literature pipeline COMPOSED (NOTES.md, E2 verdict #5):
T1 DLR massed-measurement fusion  ->  T7 N-step preview-horizon MPC  ->
R_du move suppression for the residual acceleration flags.

E2 measured the halves separately at the home cell (W30/Tg0.4, DAMULT=3):
    one-step none            sigma=2%:  -0.1% [-26,+19] 3/6
    mpcp N=8, white preview  sigma=2%: +75.1% [+45,+83] 4/6   (value, no safety)
    fuseT J50 lam=10 one-step sigma=2%: +36.7% [+16,+77] 0/6  (safety, no value)
    dlr lam=1 one-step (raw 1-3 m/s):  +34.8% [+17,+76] 1/6
This script composes them: the fusion database (verbatim mechanics of
e2_sensor.wc_fuse, incl. the pre-warm) delivers the fused N-node preview
VECTOR to the MPC via the e2_mpc.PreviewSensor coupling (harness calls
wc_fun BEFORE ctrl.compute each step), and the MPC cost gets the OptimalRdu
move-suppression term on its single horizon move (constant-flap MPC).

Parts (run as two concurrent jobs; both share the script):
    python3 e2_combo.py flat   anchors (sigma=1e-6, R_du in {0, max}) +
                               paired one-step 'none' baseline (wc_plain,
                               frac=0.02) + fused flat-sigma frac=0.02,
                               lam=0, R_du in {0, 1e-2, 1e-1}
    python3 e2_combo.py dlr    raw LOS sigma 1->3 m/s (3.3-10% W0), Jmax=50:
                               lam=1 x R_du in {0, 1e-2, 1e-1} + lam=0
                               R_du=0 (does the horizon replace smoothing?)
R_du scale note: the horizon holds N tracking terms, so the same R_du weighs
~N-times less against tracking than in the one-step cost (axis E used
1e-4..1e-2 there); hence the sweep {0, 1e-2, 1e-1}.

Extra metric: sigma_del = std(Wc - W_true(t+dt)) over the gust window
(the scalar channel only; the horizon nodes see the same fusion database).

--smoke: N=4, part flat, anchor + one fused config, 2 seeds, same OUT.
Output: results/E2_combo_<part>.npz     Runtime: ~2.8 h (flat) / ~3.2 h (dlr)
at ~8 min per N=8 rollout.
"""
import os
import sys
import numpy as np
import harness_noise as H
from controllers_ref import MPCConstRef, rk4_batch

W0, Tg = 30.0, 0.4
SMOKE = '--smoke' in sys.argv
PART  = 'dlr' if ('dlr' in sys.argv and not SMOKE) else 'flat'
NSEED = 2 if SMOKE else 6
NH    = 4 if SMOKE else 8          # horizon; smoke only validates the glue
JMAX  = 50
RDUS  = (0.0,) if SMOKE else (0.0, 1e-2, 1e-1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                   f'E2_combo_{PART}.npz')


# ---- T1 fusion sensor delivering an N-node preview vector ----------------------
class FusedSensor:
    """
    Rolling massed-measurement fusion database — verbatim mechanics of
    e2_sensor.wc_fuse (inverse-variance running sums per spatial node, raw
    samples NEVER clamped, pre-warmed with the Jmax-1 virtual steps before
    t=0, optional Tikhonov re-fit over the lookahead window) — extended to
    cache the fused estimates of nodes i+1 .. i+N in self.last for the MPC
    (e2_mpc.PreviewSensor coupling). wc_fun returns node i+1 (the scalar the
    harness logs as r['_Wc']). Honesty rule: Wt[m] is read only to synthesize
    a noisy measurement of it.
    """

    def __init__(self, rng, sigma_fun, Jmax, N, lam=0.0):
        self.rng = rng
        self.js = np.arange(1, Jmax + 1)
        self.sigs = np.array([float(sigma_fun(j)) for j in self.js])
        self.inv2 = 1.0 / self.sigs**2
        self.Jmax = int(Jmax); self.N = int(N); self.lam = float(lam)
        self.num = None; self.den = None; self.last = None

    def wc_fun(self, i, Wt, Nsteps):
        js, sigs, inv2 = self.js, self.sigs, self.inv2
        if self.num is None:
            self.num = np.zeros(Nsteps); self.den = np.zeros(Nsteps)
            for ii in range(-(self.Jmax - 1), 0):      # pre-warm (see e2_sensor)
                mm = ii + js
                keep = mm >= 0
                mk = np.minimum(mm[keep], Nsteps - 1)
                yk = Wt[mk] + self.rng.normal(0.0, sigs[keep])
                np.add.at(self.num, mk, yk * inv2[keep])
                np.add.at(self.den, mk, inv2[keep])
        m_idx = np.minimum(i + js, Nsteps - 1)
        y = Wt[m_idx] + self.rng.normal(0.0, sigs)     # NO clamp on raw samples
        np.add.at(self.num, m_idx, y * inv2)
        np.add.at(self.den, m_idx, inv2)

        lo = min(i + 1, Nsteps - 1); hi = min(i + self.Jmax, Nsteps - 1)
        w = self.den[lo:hi + 1].copy(); n = w.size
        ybar = np.zeros(n)
        good = w > 0.0
        ybar[good] = self.num[lo:hi + 1][good] / w[good]
        if self.lam == 0.0 or n < 3:
            u = ybar
        else:
            lam_eff = self.lam * float(np.mean(w))
            D = (np.eye(n - 2, n, 0) - 2.0 * np.eye(n - 2, n, 1)
                 + np.eye(n - 2, n, 2))                # second-difference op
            A = np.diag(w) + lam_eff * (D.T @ D)
            u = np.linalg.solve(A, w * ybar)
        idx = np.minimum(np.arange(self.N), n - 1)     # tail: hold last node
        self.last = np.maximum(0.0, u[idx])            # clamp the OUTPUT only
        return float(self.last[0])


# ---- T7 preview-horizon MPC + R_du move suppression ----------------------------
class MPCPrevRdu(MPCConstRef):
    """
    e2_mpc.MPCPrevRef (verbatim copy — the e2_*.py axis files are scripts,
    importing them would execute the whole axis) + the OptimalRdu-style
    move-suppression term R_du*(dg - prev)^2 on the single horizon move
    (constant-flap MPC: one move per step). Horizon step k is evaluated at
    sensor.last[k] = fused estimate of W(t+(k+1)*dt) — the wnext convention
    over the horizon. compute() ignores the scalar Wc harness argument.
    """

    def __init__(self, sensor, N=8, R=3e-4, R_du=0.0, **kw):
        super().__init__(N=N, R=R, **kw)
        self.sensor = sensor
        self.R_du = float(R_du)

    def compute(self, state, W_true, Wc):
        last = self.sensor.last
        assert last is not None and len(last) >= self.N, \
            "FusedSensor.wc_fun must have run before compute (harness ordering)"
        aero = H.aero
        dg = self._dg; G = self.G
        reach = self.delta_dot_max * H.DT
        ratem = np.abs(dg - self._prev) <= reach + 1e-9

        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2 + self.R_du * (dg - self._prev) ** 2
        for k in range(self.N):
            CL, CM, z_new = aero.batch_step(z_b, x_b, dg, float(last[k]), H.U, H.DT)
            z_b = z_new - H.LAM * z_b
            Fy = H.q * CL; Mz = H.q * CM * H.C
            x_b = rk4_batch(x_b, Fy, Mz, H.DT)
            J = J + (CL - H.CLTRIM) ** 2
        J = np.where(ratem, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._prev - reach, self._prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._prev = d
        return d


# ---- sigma models + wc_plain (== e2_sensor) ------------------------------------
def sig_flat(frac):
    s = frac * W0
    return lambda j: s


def sig_dlr(Jmax=50):
    """Raw LOS noise linear in lookahead, 1.0 m/s at j=1 to 3.0 m/s at Jmax."""
    return lambda j: 1.0 + 2.0 * (j - 1) / (Jmax - 1)


def wc_plain(rng, frac):
    """Noisy 1-step preview (== axis A/E) for the paired one-step baseline."""
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


def delivered_sigma(r):
    """std(Wc - W_true(t+dt)) over the gust window t <= Tg+0.5."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw = t <= (Tg + 0.5)
    return float(np.std(Wc[mw] - Wnext[mw]))


# ---- study body ------------------------------------------------------------------
OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis E2-combo [{PART}] N={NH} | W30/Tg0.4 DAMULT="
      f"{os.environ.get('DAMULT', '1')} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)
recs = [dict(kind='open', axis='E2C', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def run_point(make_pair, nseed=None, **cfg):
    """make_pair(rng) -> (ctrl, wc_fun); sensor+controller built together
    per seed (rng 100+seed fresh per rollout); adds sigma_del."""
    ms, rs, sds = [], [], []
    for seed in range(nseed if nseed is not None else NSEED):
        rng = np.random.default_rng(100 + seed)
        ctrl, wc = make_pair(rng)
        r = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms.append(H.metrics(r, OL, Tg)); rs.append(r)
        sds.append(delivered_sigma(r))
    sig = float(np.mean(sds))
    rec = H.point_record(ms, sigma_del=sig, **cfg)
    print(f"  {cfg['arm']:6s} {cfg.get('detail', ''):22s}: "
          f"{H.fmt_stats(H.seed_stats(ms))}  sig_del={sig:.3g} m/s", flush=True)
    return rec, ms, rs


def pair_combo(rng, sigma_fun, lam, rdu):
    sensor = FusedSensor(rng, sigma_fun, JMAX, NH, lam=lam)
    return MPCPrevRdu(sensor, N=NH, R=3e-4, R_du=rdu), sensor.wc_fun


points = {}   # (arm, detail) -> (rec, ms, rs)

if PART == 'flat':
    # anchors: clean gust through the full pipeline (fusion sigma=1e-6 + MPC)
    print("\n=== anchors: sigma=1e-6 through fusion+MPC "
          f"(mpcp N=8 white-clean anchor is +80.51%) ===", flush=True)
    for rdu in ((0.0,) if SMOKE else (0.0, RDUS[-1])):
        k = ('anchor', f'Rdu={rdu:g}')
        points[k] = run_point(
            lambda rng, rdu=rdu: pair_combo(rng, lambda j: 1e-6, 0.0, rdu),
            nseed=1, axis='E2C', arm='anchor', detail=f'Rdu={rdu:g}',
            W0=W0, Tg=Tg, R=3e-4, N=NH, Jmax=JMAX, lam=0.0, R_du=rdu, frac=0.0)

    # paired one-step baseline (cheap, 1 min/rollout)
    if not SMOKE:
        print("\n=== one-step baseline, white frac=0.02 ===", flush=True)
        k = ('none', '')
        points[k] = run_point(
            lambda rng: (H.make_optimal(R=3e-4), wc_plain(rng, 0.02)),
            axis='E2C', arm='none', detail='', W0=W0, Tg=Tg, R=3e-4, frac=0.02)

    print(f"\n=== combo: fused flat sigma frac=0.02, lam=0, N={NH} ===", flush=True)
    for rdu in RDUS:
        k = ('combo', f'flat Rdu={rdu:g}')
        points[k] = run_point(
            lambda rng, rdu=rdu: pair_combo(rng, sig_flat(0.02), 0.0, rdu),
            axis='E2C', arm='combo', detail=f'flat Rdu={rdu:g}',
            W0=W0, Tg=Tg, R=3e-4, N=NH, Jmax=JMAX, lam=0.0, R_du=rdu, frac=0.02)

else:  # PART == 'dlr'
    print(f"\n=== combo: dlr raw 1-3 m/s, Jmax={JMAX}, N={NH} ===", flush=True)
    for lam, rdu in [(1.0, r) for r in RDUS] + [(0.0, 0.0)]:
        k = ('combo', f'dlr lam={lam:g} Rdu={rdu:g}')
        points[k] = run_point(
            lambda rng, lam=lam, rdu=rdu: pair_combo(rng, sig_dlr(JMAX), lam, rdu),
            axis='E2C', arm='combo', detail=f'dlr lam={lam:g} Rdu={rdu:g}',
            W0=W0, Tg=Tg, R=3e-4, N=NH, Jmax=JMAX, lam=lam, R_du=rdu,
            frac=None, raw_sigma='1-3m/s')


# ---- collect records + trajectories of the best combo ---------------------------
for key, (rec, ms, rs) in points.items():
    recs.append(rec)

combo = [(key, v) for key, v in points.items() if key[0] == 'combo']
if combo:
    clean = [(key, v) for key, v in combo if v[0]['nflag'] == 0]
    pool = clean if clean else combo
    key_b, (rec_b, ms_b, rs_b) = max(pool, key=lambda kv: kv[1][0]['mean'])
    print(f"# best combo [{PART}]: {key_b[1]} mean {rec_b['mean']:+.1f}% "
          f"flags {rec_b['nflag']}/{len(ms_b)}", flush=True)
    for seed, (m, r) in enumerate(zip(ms_b, rs_b)):
        recs.append(H.traj_record(r, m, axis='E2C', arm='combo',
                                  detail=key_b[1], W0=W0, Tg=Tg, N=NH,
                                  seed=seed, label=f'E2C_{PART}_best_s{seed}'))
if PART == 'flat' and ('none', '') in points:
    rec_n, ms_n, rs_n = points[('none', '')]
    for seed, (m, r) in enumerate(zip(ms_n, rs_n)):
        recs.append(H.traj_record(r, m, axis='E2C', arm='none', detail='',
                                  W0=W0, Tg=Tg, frac=0.02, seed=seed,
                                  label=f'E2C_none_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
