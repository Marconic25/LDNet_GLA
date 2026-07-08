"""
Axis E2-MPC — N-step preview-horizon MPC as intrinsic noise dilution
(LITERATURE.md Technique 7), home cell W30/Tg0.4, DAMULT=3.

The one-step wnext optimal consumes a SINGLE noisy preview sample W(t+dt) and
collapses at sigma ~ 1%*W0 (branch lottery). The literature's most consistent
finding is that controllers integrating the preview over an N-step horizon
inside the cost tolerate order-100% preview errors, because the optimal move
depends on a weighted integral of the disturbance and per-sample noise
averages down [Mendez, Whidborne, Chen, "Wind Preview-Based Model Predictive
Control of Multi-Rotor UAVs Using LiDAR", Sensors 23(7):3711, 2023:
preview-MPC beats no-preview-MPC up to ~120% magnitude error; Sinner et al.,
"Insensitivity to propagation timing in a preview-enabled wind turbine control
experiment", Frontiers in Mech. Eng. 9, 2023: feedforward benefit retained
across +-20% timing error]. Our own axis E pointed the same way: MPCConst N=4
WITHOUT preview (noisy current gust held over the horizon) was the most
noise-tolerant arm. This script tests whether an N-step NOISY PREVIEW consumed
through the integrated cost beats BOTH the one-step controller and the
no-preview MPC.

All arms see white noise sigma = frac*W0 on the gust signal (rng 100+seed,
identical streams across arms -> paired comparison), frac in {0.02, 0.05},
6 seeds, same metrics/flags/pick rule as everywhere else.

  none     one-step optimal (wnext) on the noisy 1-step preview — the
           unmitigated baseline reproduced in-file for pairing (== axis E)
  mpc-cur  MPCConstRef N=4 gate-none on the noisy CURRENT gust (no preview)
           — the axis-E noise-tolerant reference reproduced in-file
  mpcp     MPCPrevRef N in {2,4,8}: constant-flap MPC whose horizon step k is
           evaluated at an INDEPENDENT noisy sample of W(t+(k+1)*dt) from the
           PreviewSensor (the wnext convention of light/optimal.py extended to
           the horizon: step k advances the plant from t+k*dt into
           t+(k+1)*dt, so it is costed at the gust one step later)
  anchors  mpcp N in {2,4,8} at frac=0 (clean preview through the sensor,
           single rollout each): the one-step clean anchor is +76.58%; we need
           the integrated-cost MPC's clean-preview level before judging its
           noise tolerance.

Sensor/controller coupling: harness_noise.rollout calls wc_fun BEFORE
ctrl.compute at every step, so PreviewSensor.wc_fun draws the N noisy future
samples once per step, caches them in sensor.last for the MPC, and returns
element j=1 — r['_Wc'] therefore logs exactly the 1-step preview a one-step
controller would see.

Runtime: each horizon step costs one batch_step over the 161-point flap grid,
so a rollout costs ~N x the one-step controller's (~1 min): N=8 ~ 8 min per
rollout. Arms are ordered so the cheap ones print results first (anchors,
none, mpc-cur, then mpcp by ascending N). Full study: 63 controlled rollouts
(~240 one-step-equivalents, ~4 h). DAMULT=3 must be set in the environment by
the runner, as for all axis scripts.

Trajectories stored: all 6 seeds of the best mpcp N at each frac (pick rule:
max mean CLred s.t. zero flags, else max mean) + the paired 'none' seeds.

Output: results/E2_mpc.npz.
--smoke: frac=0 anchor for N=4 + arms none and mpcp N=4 at frac=0.02 with
2 seeds; saves to the SAME output and prints # DONE.
"""
import os
import sys
import numpy as np
import harness_noise as H
from controllers_ref import MPCConstRef
try:
    from controllers_ref import rk4_batch          # cluster tree (all recorded
except ImportError:                                # E2 results used this RK4)
    from controllers_ref import dp45_batch as rk4_batch  # local tree, f92d8975

W0, Tg = 30.0, 0.4
SMOKE = '--smoke' in sys.argv[1:]
FRACS = (0.02,) if SMOKE else (0.02, 0.05)
NSEED = 2 if SMOKE else 6
NS_MPCP = (4,) if SMOKE else (2, 4, 8)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'E2_mpc.npz')


# ---- multi-sample noisy lidar (controller-side sensor model) ------------------
class PreviewSensor:
    """Generates, once per step, N independent noisy samples of the future
    gust W(t+j*dt)+eps_j, j=1..N (each clamped >= 0), caches them in
    self.last; wc_fun returns element j=1 (so r['_Wc'] logs the 1-step
    preview the one-step controller would see), the MPC reads self.last.
    For frac=0 the noise term vanishes — same code path, clean preview."""

    def __init__(self, rng, frac, N):
        self.rng = rng
        self.frac = float(frac)
        self.N = int(N)
        self.last = None

    def wc_fun(self, i, Wt, Nsteps):
        js = np.clip(i + np.arange(1, self.N + 1), 0, Nsteps - 1)
        self.last = np.maximum(0.0, Wt[js] + self.rng.normal(0.0, self.frac * W0,
                                                             size=self.N))
        return float(self.last[0])


# ---- N-step constant-flap MPC WITH preview ------------------------------------
class MPCPrevRef(MPCConstRef):
    """
    Constant-flap N-step MPC consuming the PreviewSensor's noisy N-sample
    preview through the integrated cost. Structure == MPCConstRef (constant
    flap over the horizon, G=161 grid, J = R*dg^2 + sum_k (CL_k - trim)^2,
    z leak correction each step, rate mask vs prev, argmin, clip, no causal
    gate) with ONE change: horizon step k (k = 0..N-1) is evaluated at
    sensor.last[k] — the noisy sample of W(t+(k+1)*dt) — instead of the held
    scalar Wc. This is the wnext convention of the one-step controller
    extended to the horizon: step k advances the batch plant from t+k*dt and
    the wnext finding says evaluate that step at the gust one step LATER.

    compute() IGNORES the scalar Wc argument of the harness API and reads
    self.sensor.last instead (fresh each step: harness_noise.rollout calls
    wc_fun before ctrl.compute).
    """

    def __init__(self, sensor, N=4, R=3e-4, **kw):
        super().__init__(N=N, R=R, **kw)
        self.sensor = sensor

    def compute(self, state, W_true, Wc):
        last = self.sensor.last
        assert last is not None and len(last) >= self.N, \
            "PreviewSensor.wc_fun must have run before compute (harness ordering)"
        aero = H.aero
        dg = self._dg; G = self.G
        reach = self.delta_dot_max * H.DT
        ratem = np.abs(dg - self._prev) <= reach + 1e-9

        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2
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


# ---- Wc generators for the paired in-file baselines (== noise_mitigation.py) --
def wc_plain(rng, frac):
    """Noisy 1-step preview (== axis A / axis E 'none')."""
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


def wc_current(rng, frac):
    """Noisy CURRENT gust (no preview) — for the MPC reference."""
    return lambda i, Wt, N: max(0.0, Wt[i] + rng.normal(0.0, frac * W0))


# ---- open-loop reference -------------------------------------------------------
OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis E2-MPC | W30/Tg0.4 DAMULT=3 | open cex0={cex0:.4f}"
      + (" | SMOKE" if SMOKE else ""), flush=True)
recs = [dict(kind='open', axis='E2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


# ---- one study point -----------------------------------------------------------
def run_point(make_pair, frac, **cfg):
    """make_pair(rng, frac) -> (ctrl, wc_fun): sensor + controller are built
    TOGETHER per seed so the controller reads exactly the sensor whose wc_fun
    drives the rollout (fresh rng 100+seed per rollout)."""
    ms, rs = [], []
    for seed in range(NSEED):
        rng = np.random.default_rng(100 + seed)
        ctrl, wc = make_pair(rng, frac)
        r = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms.append(H.metrics(r, OL, Tg)); rs.append(r)
    rec = H.point_record(ms, frac=frac, **cfg)
    print(f"  {cfg['arm']:7s} {cfg.get('detail',''):6s} frac={frac:.2f}: "
          f"{H.fmt_stats(H.seed_stats(ms))}", flush=True)
    return rec, ms, rs


def pair_mpcp(rng, frac, N):
    sensor = PreviewSensor(rng, frac, N)
    return MPCPrevRef(sensor, N=N, R=3e-4), sensor.wc_fun


points = {}   # (arm, detail-or-N, frac) -> (rec, ms, rs)


# ---- frac=0 anchors: clean preview through the sensor --------------------------
print("\n=== frac=0.00 anchors (clean preview; one-step clean anchor +76.58%) ===",
      flush=True)
for N in NS_MPCP:
    rng = np.random.default_rng(100)
    ctrl, wc = pair_mpcp(rng, 0.0, N)
    r = H.rollout(ctrl, W0, Tg, wc_fun=wc)
    m = H.metrics(r, OL, Tg)
    rec = H.point_record([m], axis='E2', arm='mpcp', detail=f'N={N}',
                         frac=0.0, N=N, W0=W0, Tg=Tg, R=3e-4)
    points[('mpcp', N, 0.0)] = (rec, [m], [r])
    print(f"  mpcp    N={N}    frac=0.00 (anchor): clred={m['clred']:+7.2f}% "
          f"flap_max={m['flap_max']:5.1f} flag='{m['flag']}'", flush=True)


# ---- noisy arms (cheap first: none, mpc-cur, then mpcp by ascending N) ---------
for frac in FRACS:
    print(f"\n=== frac={frac:.2f} (sigma={frac*W0:.2f} m/s) ===", flush=True)

    k = ('none', '', frac)
    points[k] = run_point(lambda rng, f: (H.make_optimal(R=3e-4), wc_plain(rng, f)),
                          frac, axis='E2', arm='none', detail='',
                          W0=W0, Tg=Tg, R=3e-4)

    if not SMOKE:
        k = ('mpc-cur', 4, frac)
        points[k] = run_point(lambda rng, f: (MPCConstRef(N=4, R=3e-4),
                                              wc_current(rng, f)),
                              frac, axis='E2', arm='mpc-cur', detail='N=4',
                              N=4, W0=W0, Tg=Tg, R=3e-4)

    for N in NS_MPCP:
        k = ('mpcp', N, frac)
        points[k] = run_point(lambda rng, f, N=N: pair_mpcp(rng, f, N),
                              frac, axis='E2', arm='mpcp', detail=f'N={N}',
                              N=N, W0=W0, Tg=Tg, R=3e-4)


# ---- collect records + trajectories of the best mpcp per frac ------------------
for key, (rec, ms, rs) in points.items():
    recs.append(rec)

for frac in FRACS:
    cand = [(key, v) for key, v in points.items()
            if key[0] == 'mpcp' and key[2] == frac]
    clean = [(key, v) for key, v in cand if v[0]['nflag'] == 0]
    pool = clean if clean else cand
    key_b, (rec_b, ms_b, rs_b) = max(pool, key=lambda kv: kv[1][0]['mean'])
    print(f"# best mpcp frac={frac:.2f}: N={key_b[1]} mean {rec_b['mean']:+.1f}%",
          flush=True)
    for seed, (m, r) in enumerate(zip(ms_b, rs_b)):
        recs.append(H.traj_record(r, m, axis='E2', arm='mpcp',
                                  detail=f'N={key_b[1]}', N=key_b[1],
                                  W0=W0, Tg=Tg, frac=frac, seed=seed,
                                  label=f'E2M_best_f{frac:g}_s{seed}'))
    # paired one-step baseline trajectories for the same frac
    rec_n, ms_n, rs_n = points[('none', '', frac)]
    for seed, (m, r) in enumerate(zip(ms_n, rs_n)):
        recs.append(H.traj_record(r, m, axis='E2', arm='none', detail='',
                                  W0=W0, Tg=Tg, frac=frac, seed=seed,
                                  label=f'E2M_none_f{frac:g}_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
