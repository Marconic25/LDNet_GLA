"""
Axis E2-zctrl — does the preview-MPC keep its performance when it propagates
its OWN latent instead of reading the plant's z(t)?

In the archived pipeline (e2_combo.MPCPrevRdu) plant and controller SHARE one
LDNetAero: every step the MPC starts its horizon recursion from the plant's
current latent, `z_b = tile(H.aero._z)`. That is an idealization — on a real
aircraft the plant is NOT the LDNet, and the controller would carry its own
LDNet copy, propagated in closed loop, whose latent z_ctrl is never
re-synchronised with reality. This axis quantifies the drift of that
controller-side latent and whether the controller still works from it.

The copy is NOT a free run: it is fed the TRUE structural state x(t) (a stated
thesis assumption) and the APPLIED flap delta* each step, and only the gust it
sees is a surrogate. The four arms differ only in (i) which latent starts the
MPC horizon and (ii) which gust drives the copy's own advance:

  A  shared     horizon starts from H.aero._z (plant latent) — the archived
                MPCPrevRdu baseline, RE-RUN here (not read from old npz).
  B  true       copy advanced with the TRUE gust W(t). z_ctrl == z_plant to
                roundoff, so B must reproduce A bit-for-bit. This is the
                implementation sanity check.
  C  fused_cur  copy advanced with the fused estimate of the CURRENT gust
                (node m=i of the FusedSensor database, num[i]/den[i]).
                The realistic arm.
  D  prev_w1    copy advanced with the previous step's Ŵ_1 (a simpler proxy
                of the current estimate). Optional.

Everything else is identical to MPCPrevRdu: N=8, R=3e-4, R_du=0, G=161 flap
grid, FusedSensor N-node preview horizon (same rng streams per seed across
arms -> paired comparison), leak-corrected batch_step horizon. The controller's
own model is a SEPARATE LDNetAero(MD) instance (`aero_ctrl`), reset identically
to the plant each rollout; its z is advanced with aero_ctrl.advance (leak
included, exactly the plant's scheme), NOT batch_step.

Diagnostics stored per point (over the gust window trajectory):
  drift  = max_t || z_ctrl - z_plant ||_2
  clerr  = | CL_pred(z_ctrl, ĝ) - CL_plant(z_plant, W_true) | one-step

Grid (faithful to e2_combo):
  home  W30/Tg0.4 : clean + white sweep sigma/W0 in {1,2,5,10,20}% (flat, lam0)
                    + dlr lidar 1-3 m/s (lam1). Arms A,B,C,D (B on clean +
                    one noisy level as sanity; D home only).
  cells W10/Tg0.7 and W30/Tg0.7 : dlr lidar (lam1). Arms A,C.

Parts (run concurrently, both share the script):
  python3 e2_zctrl.py home    -> results/E2_zctrl_home.npz
  python3 e2_zctrl.py cells   -> results/E2_zctrl_cells.npz
--smoke: home cell, clean, N=4, arms A+B, 1 seed; asserts B == A and prints
the max |CL_A - CL_B|. Saves to results/E2_zctrl_smoke.npz.

Runtime: ~N x a single-step 161-grid scan per rollout (same as e2_combo).
"""
import os
import sys
import numpy as np
import harness_noise as H
from ldnet_aero import LDNetAero
from controllers_ref import MPCConstRef
try:
    from controllers_ref import rk4_batch          # cluster tree (matches the
except ImportError:                                # recorded E2 results)
    from controllers_ref import dp45_batch as rk4_batch  # local tree, f92d8975

SMOKE = '--smoke' in sys.argv
PART  = 'cells' if ('cells' in sys.argv and not SMOKE) else 'home'
NSEED = 6
NH    = 4 if SMOKE else 8
JMAX  = 50
SUFX  = 'smoke' if SMOKE else PART
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                   f'E2_zctrl_{SUFX}.npz')

CELLS = {'home': (30.0, 0.4), 'w10t07': (10.0, 0.7), 'w30t07': (30.0, 0.7)}

# The controller's OWN LDNet copy (same weights, own latent). One shared
# instance is enough: rollouts run sequentially and ctrl.reset() re-zeroes it.
aero_ctrl = LDNetAero(H.MD); aero_ctrl.reset(dt=H.DT)


# ---- T1 fusion sensor (verbatim from e2_combo, + current-node estimate) --------
class FusedSensor:
    """
    Rolling massed-measurement fusion (DLR Technique 1): inverse-variance
    running sums per spatial node, raw samples never clamped, pre-warmed with
    the Jmax-1 virtual steps before t=0, optional Tikhonov re-fit over the
    lookahead window; caches the fused nodes i+1..i+N in self.last for the MPC.
    wc_fun returns node i+1 (logged as r['_Wc']). Honesty rule: Wt[m] is read
    only to synthesize a noisy measurement of it.

    Extension for this axis: self.cur = fused estimate of the CURRENT gust
    node m=i (num[i]/den[i], clamped >= 0). Node i is untouched by step i's
    update (which only adds to nodes i+1..i+Jmax), so it aggregates every prior
    measurement of the current gust — the natural current-gust estimate.
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

        # current-gust node m=i (all prior measurements of node i)
        di = self.den[i]
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


# ---- preview-horizon MPC with a controller-side latent -------------------------
class MPCPrevZctrl(MPCConstRef):
    """
    MPCPrevRdu (e2_combo, verbatim structure) with a switchable latent source.
    gust_src is None -> horizon starts from the PLANT latent H.aero._z (arm A,
    the archived baseline). Otherwise the horizon starts from the controller's
    own copy aero_ctrl._z, and after choosing delta* the copy is advanced with
    (true x(t), delta*, W_src, U, DT) via aero_ctrl.advance (leak included):
        'true'      W_src = W(t)                 (arm B, sanity)
        'fused_cur' W_src = sensor.cur           (arm C, realistic)
        'prev_w1'   W_src = previous step's Ŵ_1  (arm D, proxy)
    R_du fixed 0. compute() ignores the scalar Wc harness argument.
    """

    def __init__(self, sensor, aero_ctrl=None, gust_src=None,
                 N=8, R=3e-4, R_du=0.0, **kw):
        super().__init__(N=N, R=R, **kw)
        self.sensor = sensor
        self.aero_ctrl = aero_ctrl
        self.gust_src = gust_src
        self.R_du = float(R_du)
        self.z_drift = []; self.cl_err = []; self._prev_w1 = 0.0

    def reset(self):
        self._prev = 0.0; self._prev_w1 = 0.0
        self.z_drift = []; self.cl_err = []
        if self.aero_ctrl is not None:
            self.aero_ctrl.reset(dt=H.DT)

    def compute(self, state, W_true, Wc):
        last = self.sensor.last
        assert last is not None and len(last) >= self.N, \
            "FusedSensor.wc_fun must have run before compute (harness ordering)"
        # horizon start: plant latent (arm A) or controller's own copy
        if self.aero_ctrl is None:
            z0 = np.asarray(H.aero._z, float)
        else:
            z0 = np.asarray(self.aero_ctrl._z, float)

        dg = self._dg; G = self.G
        reach = self.delta_dot_max * H.DT
        ratem = np.abs(dg - self._prev) <= reach + 1e-9

        z_b = np.tile(z0.reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2 + self.R_du * (dg - self._prev) ** 2
        for k in range(self.N):
            CL, CM, z_new = H.aero.batch_step(z_b, x_b, dg, float(last[k]),
                                              H.U, H.DT)
            z_b = z_new - H.LAM * z_b
            Fy = H.q * CL; Mz = H.q * CM * H.C
            x_b = rk4_batch(x_b, Fy, Mz, H.DT)
            J = J + (CL - H.CLTRIM) ** 2
        J = np.where(ratem, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._prev - reach, self._prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._prev = d

        # diagnostics + propagate the controller's own latent (arms B/C/D)
        if self.aero_ctrl is not None:
            zc = np.asarray(self.aero_ctrl._z, float)
            zp = np.asarray(H.aero._z, float)
            self.z_drift.append(float(np.linalg.norm(zc - zp)))
            if self.gust_src == 'true':
                w_src = float(W_true)
            elif self.gust_src == 'fused_cur':
                w_src = float(self.sensor.cur)
            else:                                       # 'prev_w1'
                w_src = float(self._prev_w1)
            cl_pred  = float(self.aero_ctrl.predict(state, d, w_src, H.U)[0])
            cl_plant = float(H.aero.predict(state, d, float(W_true), H.U)[0])
            self.cl_err.append(abs(cl_pred - cl_plant))
            self.aero_ctrl.advance(state, d, w_src, H.U, H.DT)
        self._prev_w1 = float(self.sensor.last[0])
        return d


# ---- sigma models (from e2_combo) ----------------------------------------------
def sig_flat(sigma):
    return lambda j: sigma


def sig_dlr(Jmax=50):
    """Raw LOS noise linear in lookahead, 1.0 m/s at j=1 to 3.0 m/s at Jmax."""
    return lambda j: 1.0 + 2.0 * (j - 1) / (Jmax - 1)


def delivered_sigma(r, Tg):
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw = t <= (Tg + 0.5)
    return float(np.std(Wc[mw] - Wnext[mw]))


GSRC = {'A': None, 'B': 'true', 'C': 'fused_cur', 'D': 'prev_w1'}


# ---- per-cell open-loop references ----------------------------------------------
OLs, CEX0 = {}, {}
_cells = ['home'] if PART in ('home',) or SMOKE else ['w10t07', 'w30t07']
for _cell in _cells:
    _W0, _Tg = CELLS[_cell]
    OLs[_cell] = H.rollout(None, _W0, _Tg)
    CEX0[_cell] = H.metrics(OLs[_cell], OLs[_cell], _Tg)['exo']
print(f"# axis E2-zctrl [{SUFX}] N={NH} DAMULT={os.environ.get('DAMULT','1')} "
      + " ".join(f"| {c} cex0={CEX0[c]:.4f}" for c in _cells)
      + (" | SMOKE" if SMOKE else ""), flush=True)

recs = [dict(kind='open', axis='E2Z', cell=c, W0=CELLS[c][0], Tg=CELLS[c][1],
             cex0=CEX0[c], t=OLs[c]['_t'], W=OLs[c]['_Wt'], CL=OLs[c]['CL'])
        for c in _cells]


# ---- one study point ------------------------------------------------------------
def run_point(cell, sigma_fun, lam, arm, nseed=NSEED, **cfg):
    W0c, Tgc = CELLS[cell]; OL = OLs[cell]
    ms, rs, sds, drifts, ce_max, ce_mean = [], [], [], [], [], []
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        sensor = FusedSensor(rng, sigma_fun, JMAX, NH, lam=lam)
        ctrl = MPCPrevZctrl(sensor, aero_ctrl=(None if arm == 'A' else aero_ctrl),
                            gust_src=GSRC[arm], N=NH, R=3e-4, R_du=0.0)
        r = H.rollout(ctrl, W0c, Tgc, wc_fun=sensor.wc_fun)
        ms.append(H.metrics(r, OL, Tgc)); rs.append(r)
        sds.append(delivered_sigma(r, Tgc))
        # drift diagnostics are recorded over ALL steps; window to the gust
        mw = r['_t'] <= (Tgc + 0.5)
        nw = int(mw.sum())
        if ctrl.z_drift:
            zd = np.asarray(ctrl.z_drift[:nw]); ce = np.asarray(ctrl.cl_err[:nw])
            drifts.append(float(zd.max())); ce_max.append(float(ce.max()))
            ce_mean.append(float(ce.mean()))
    sig = float(np.mean(sds))
    rec = H.point_record(
        ms, sigma_del=sig, cell=cell, arm=arm,
        drift_max=(float(np.max(drifts)) if drifts else 0.0),
        drift_arr=np.array(drifts) if drifts else np.zeros(0),
        clerr_max=(float(np.max(ce_max)) if ce_max else 0.0),
        clerr_mean=(float(np.mean(ce_mean)) if ce_mean else 0.0), **cfg)
    extra = (f"  drift={rec['drift_max']:.3g} clerr={rec['clerr_max']:.2e}"
             if arm != 'A' else "")
    print(f"  {cell:7s} {arm} {cfg.get('detail',''):16s}: "
          f"{H.fmt_stats(H.seed_stats(ms))}  sig_del={sig:.3g}{extra}", flush=True)
    return rec, ms, rs


# ---- study configs --------------------------------------------------------------
# (label, sigma_fun, lam, [arms], nseed, detail, cfg-extras)
def home_configs():
    cfgs = []
    cfgs.append(('clean', sig_flat(1e-6), 0.0, ['A', 'B', 'C', 'D'], 1,
                 'clean', dict(noise='clean', frac=0.0)))
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        arms = ['A', 'C'] + (['B'] if frac == 0.05 else [])
        cfgs.append((f'white{frac:g}', sig_flat(frac * 30.0), 0.0, arms, NSEED,
                     f'white {frac*100:g}%', dict(noise='white', frac=frac)))
    cfgs.append(('dlr', sig_dlr(JMAX), 1.0, ['A', 'C', 'D'], NSEED,
                 'dlr 1-3m/s', dict(noise='dlr', frac=None, raw_sigma='1-3m/s')))
    return cfgs


points = {}   # (cell, noise-label, arm) -> (rec, ms, rs)

if SMOKE:
    print("\n=== SMOKE: home clean, arms A,B (B must reproduce A) ===", flush=True)
    for arm in ['A', 'B']:
        points[('home', 'clean', arm)] = run_point(
            'home', sig_flat(1e-6), 0.0, arm, nseed=1,
            axis='E2Z', detail='clean', noise='clean', frac=0.0)
    rA = points[('home', 'clean', 'A')][2][0]
    rB = points[('home', 'clean', 'B')][2][0]
    dCL = float(np.max(np.abs(rA['CL'] - rB['CL'])))
    dde = float(np.max(np.abs(rA['de'] - rB['de'])))
    print(f"\n# SANITY B vs A: max|CL_A-CL_B|={dCL:.3e}  max|de_A-de_B|={dde:.3e}",
          flush=True)
    assert dCL < 1e-9 and dde < 1e-9, \
        f"B did not reproduce A (dCL={dCL:.2e}, dde={dde:.2e}) -> bug in the copy"
    print("# SANITY PASS: z_ctrl tracks z_plant to roundoff", flush=True)

elif PART == 'home':
    for label, sfun, lam, arms, nseed, detail, extra in home_configs():
        print(f"\n=== home {detail} (lam={lam:g}) ===", flush=True)
        for arm in arms:
            points[('home', label, arm)] = run_point(
                'home', sfun, lam, arm, nseed=nseed,
                axis='E2Z', detail=detail, **extra)

else:  # cells
    for cell in ['w10t07', 'w30t07']:
        print(f"\n=== {cell} dlr 1-3m/s (lam=1) ===", flush=True)
        for arm in ['A', 'C']:
            points[(cell, 'dlr', arm)] = run_point(
                cell, sig_dlr(JMAX), 1.0, arm, nseed=NSEED,
                axis='E2Z', detail='dlr 1-3m/s', noise='dlr',
                frac=None, raw_sigma='1-3m/s')


# ---- collect records + trajectories of arm C at each cell/noise ----------------
for key, (rec, ms, rs) in points.items():
    recs.append(rec)

if not SMOKE:
    for (cell, label, arm), (rec, ms, rs) in points.items():
        if arm != 'C':
            continue
        for seed, (m, r) in enumerate(zip(ms, rs)):
            recs.append(H.traj_record(
                r, m, axis='E2Z', arm='C', cell=cell, detail=label,
                W0=CELLS[cell][0], Tg=CELLS[cell][1], N=NH, seed=seed,
                label=f'E2Z_{cell}_{label}_C_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
