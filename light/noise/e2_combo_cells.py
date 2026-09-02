"""
Axis E2CC — generality check of the composed pipeline on the other two cells.

e2_combo.py SOLVED the noise problem at the home cell W30/Tg0.4 (DAMULT=3):
T1 DLR massed-measurement fusion (Jmax=50, pre-warmed database) feeding the
fused 8-node preview vector to the constant-flap preview-MPC (N=8, R=3e-4,
R_du=0): flat sigma=2%*W0 -> +80.5% [80.4,80.6] 0/6 flags (== clean anchor);
DLR-realistic raw LOS noise 1-3 m/s with Tikhonov lam=1 -> +81.1% [80.3,83.7]
0/6 flags. This script repeats THE WINNER (fixed N=8, Jmax=50, R=3e-4,
R_du=0 — no sweeps) at the two other study cells:

    python3 e2_combo_cells.py W10T07     W0=10, Tg=0.7  (gentle)
    python3 e2_combo_cells.py W30T07     W0=30, Tg=0.7  (strong but slower)

Default cell (no recognized arg): W30T07.

Honesty note: the dlr LOS noise is ABSOLUTE (1-3 m/s), so at W0=10 it is
10-30% of the gust amplitude — a much harsher relative test than at W30
(3.3-10%); the flat arm scales with the cell (sigma = 2%*W0 = 0.2 m/s at W10).

Arms (NSEED=6, rng 100+seed fresh per rollout):
    anchor       fusion sigma=1e-6 + MPC                  (1 rollout)
                 — the cell's clean combo reference
    combo-flat   fused flat sigma frac=0.02, lam=0        (6 seeds)
    combo-dlr    fused raw LOS 1-3 m/s, lam=1             (6 seeds)

--smoke: N=4, 2 seeds, anchor + combo-flat only, same OUT.
Output: results/E2_combo_cells_<cell>.npz  (schema: harness_noise.py helpers)
Runtime at ~8 min per N=8 rollout: per cell
1 (OL) + 8 (anchor) + 48 (flat) + 48 (dlr) = ~105 min; the two cells run
as two concurrent cluster jobs.
DAMULT must be in the environment before launch (the study uses DAMULT=3).
"""
import os
import sys
import numpy as np
import harness_noise as H
from controllers_ref import MPCConstRef
try:
    from controllers_ref import rk4_batch            # name on the cluster tree
except ImportError:                                  # local tree: dp45_batch
    from controllers_ref import dp45_batch as rk4_batch

# ---- cell selection ------------------------------------------------------------
# W0/Tg are MODULE GLOBALS read by sig_flat / delivered_sigma below
# (same convention as e2_combo.py), so they are set from argv before those
# definitions and the copied code works unchanged.
CELLS = {'W10T07': (10.0, 0.7), 'W30T07': (30.0, 0.7)}
cellname = next((a for a in sys.argv[1:] if a in CELLS), 'W30T07')
W0, Tg = CELLS[cellname]
SMOKE = '--smoke' in sys.argv
NSEED = 2 if SMOKE else 6
NH    = 4 if SMOKE else 8          # horizon; smoke only validates the glue
JMAX  = 50
RDU   = 0.0                        # the home-cell winner: R_du=0 throughout
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                   f'E2_combo_cells_{cellname}.npz')


# ---- T1 fusion sensor delivering an N-node preview vector ----------------------
# verbatim copy of e2_combo.py — kept in sync by eye
class FusedSensor:
    """
    Rolling massed-measurement fusion database (DLR Technique 1): inverse-
    variance running sums per spatial node, raw samples NEVER clamped,
    pre-warmed with the Jmax-1 virtual steps before t=0, optional Tikhonov
    re-fit over the lookahead window — extended to cache the fused estimates
    of nodes i+1 .. i+N in self.last for the MPC (e2_mpc.PreviewSensor
    coupling). wc_fun returns node i+1 (the scalar the harness logs as
    r['_Wc']). Honesty rule: Wt[m] is read only to synthesize a noisy
    measurement of it.
    """

    def __init__(self, rng, sigma_fun, Jmax, N, lam=0.0):
        self.rng = rng
        self.js = np.arange(1, Jmax + 1)
        self.sigs = np.array([float(sigma_fun(j)) for j in self.js])
        self.inv2 = 1.0 / self.sigs**2
        self.Jmax = int(Jmax); self.N = int(N); self.lam = float(lam)
        self.num = None; self.den = None; self.last = None; self.cur = 0.0

    def wc_fun(self, i, Wt, Nsteps):
        js, sigs, inv2 = self.js, self.sigs, self.inv2
        if self.num is None:
            self.num = np.zeros(Nsteps); self.den = np.zeros(Nsteps)
            for ii in range(-(self.Jmax - 1), 0):      # pre-warm (see class doc)
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

        di = self.den[i]                               # fused CURRENT-gust node
        self.cur = max(0.0, float(self.num[i] / di)) if di > 0.0 else 0.0

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
# verbatim copy of e2_combo.py — kept in sync by eye
class MPCPrevRdu(MPCConstRef):
    """
    e2_mpc.MPCPrevRef (verbatim copy — the e2_*.py axis files are scripts,
    importing them would execute the whole axis) + a move-suppression term
    R_du*(dg - prev)^2 on the single horizon move (constant-flap MPC: one
    move per step). Horizon step k is evaluated at sensor.last[k] = fused
    estimate of W(t+(k+1)*dt) — the wnext convention over the horizon.
    compute() ignores the scalar Wc harness argument.
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

        z_b = np.tile(np.asarray(self._z_ctrl, float).reshape(1, -1), (G, 1))
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
        # advance the controller's OWN latent with the fused current-gust estimate
        self._z_ctrl = aero.advance_z(self._z_ctrl, state, d, float(self.sensor.cur),
                                      H.U, H.DT)
        return d


# ---- sigma models ----------------------------------------------------------------
# verbatim copies of e2_combo.py — kept in sync by eye
def sig_flat(frac):
    s = frac * W0
    return lambda j: s


def sig_dlr(Jmax=50):
    """Raw LOS noise linear in lookahead, 1.0 m/s at j=1 to 3.0 m/s at Jmax."""
    return lambda j: 1.0 + 2.0 * (j - 1) / (Jmax - 1)


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
print(f"# axis E2-combo-cells [{cellname}] N={NH} | W{W0:g}/Tg{Tg:g} DAMULT="
      f"{os.environ.get('DAMULT', '1')} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)
recs = [dict(kind='open', axis='E2CC', cell=cellname, W0=W0, Tg=Tg, cex0=cex0,
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
    print(f"  {cfg['arm']:12s}: {H.fmt_stats(H.seed_stats(ms))}  "
          f"sig_del={sig:.3g} m/s", flush=True)
    return rec, ms, rs


def pair_combo(rng, sigma_fun, lam, rdu):
    sensor = FusedSensor(rng, sigma_fun, JMAX, NH, lam=lam)
    return MPCPrevRdu(sensor, N=NH, R=3e-4, R_du=rdu), sensor.wc_fun


points = {}   # arm -> (rec, ms, rs)

print("\n=== anchor: sigma=1e-6 through fusion+MPC "
      "(home-cell anchor is +80.5%) ===", flush=True)
points['anchor'] = run_point(
    lambda rng: pair_combo(rng, lambda j: 1e-6, 0.0, RDU),
    nseed=1, axis='E2CC', arm='anchor', cell=cellname, W0=W0, Tg=Tg,
    R=3e-4, N=NH, Jmax=JMAX, lam=0.0, R_du=RDU, frac=0.0)

print(f"\n=== combo-flat: fused flat sigma frac=0.02 "
      f"(= {0.02 * W0:.2g} m/s), lam=0, N={NH} ===", flush=True)
points['combo-flat'] = run_point(
    lambda rng: pair_combo(rng, sig_flat(0.02), 0.0, RDU),
    axis='E2CC', arm='combo-flat', cell=cellname, W0=W0, Tg=Tg,
    R=3e-4, N=NH, Jmax=JMAX, lam=0.0, R_du=RDU, frac=0.02)

if not SMOKE:
    print(f"\n=== combo-dlr: raw LOS 1-3 m/s, lam=1, Jmax={JMAX}, N={NH} ===",
          flush=True)
    print(f"# note: dlr noise is ABSOLUTE -> {100 * 1.0 / W0:.1f}-"
          f"{100 * 3.0 / W0:.1f}% of W0={W0:g} m/s at this cell", flush=True)
    points['combo-dlr'] = run_point(
        lambda rng: pair_combo(rng, sig_dlr(JMAX), 1.0, RDU),
        axis='E2CC', arm='combo-dlr', cell=cellname, W0=W0, Tg=Tg,
        R=3e-4, N=NH, Jmax=JMAX, lam=1.0, R_du=RDU,
        frac=None, raw_sigma='1-3m/s')


# ---- collect records + trajectories (all combo-dlr seeds) -----------------------
for arm, (rec, ms, rs) in points.items():
    recs.append(rec)

if 'combo-dlr' in points:
    rec_d, ms_d, rs_d = points['combo-dlr']
    for seed, (m, r) in enumerate(zip(ms_d, rs_d)):
        recs.append(H.traj_record(r, m, axis='E2CC', arm='combo-dlr',
                                  cell=cellname, W0=W0, Tg=Tg, N=NH,
                                  Jmax=JMAX, lam=1.0, R_du=RDU, seed=seed,
                                  label=f'E2CC_{cellname}_dlr_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
