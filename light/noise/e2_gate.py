"""
Axis E2 (gate) — chattering-proof causal gate + command washout under preview
noise, home cell W30/Tg0.4, DAMULT=3.

The final controller (light/optimal.py, use_wnext=True) collapses under white
preview noise mainly through ONE mechanism (failure F1, LITERATURE.md): after
the gust has passed (true W ~ 0) the noisy Wc flips the causal sign gate
`cl0 >= trim` every step, the argmin alternates between the two half-grids,
and the rate-limited flap winds up to the -14 deg stop and stays there. This
script evaluates the two literature cures aimed exactly at that mechanism:

  (A) hysteresis / deadband / dwell on the switching gate — Technique 8 of
      LITERATURE.md: Hespanha, Liberzon & Morse, "Hysteresis-based switching
      algorithms for supervisory control of uncertain systems", Automatica
      39:263-272, 2003 (hysteresis switching gives bounded switch counts
      under bounded noise); applied instance Schlipf et al., Wind Energy
      Science 8:149, 2023 (feedforward conditionally ACTIVATED only where
      the wind signal is strong enough to be trusted).
  (B) washout (high-pass) on the flap COMMAND — Technique 3 of LITERATURE.md:
      Wallace & Fezans et al., EUCASS 2023 (DLR band-pass on every control-
      surface command; the high-pass leg "ensures that no constant or quasi-
      constant deflections can be commanded" — the wound-up-flap state
      becomes impossible by construction, without punishing fast transient
      action the way R_du did).

All arms see white noise sigma = frac*W0 on the 1-step preview (rng 100+seed,
identical streams across arms -> paired comparison), frac in {0.02, 0.05},
6 seeds, same metrics/flags/pick rule as the other axes.

  none    unmitigated baseline: noisy preview straight into the controller
  db      OptimalGated deadband, eps_db in {0.005, 0.01} C_L units: inside the
          band the flap DECAYS toward 0 (tau 30 ms) and the argmin is skipped
  hyst    OptimalGated hysteresis margin, eps_on in {0.005, 0.01}: the gate
          flips only when |C_L,pred - trim| clears the margin
  hd      hysteresis + dwell WITHOUT deadband (0.005 / 5 steps): hysteresis
          never delays the ONSET action (the gate is seeded at the first
          compute and the first flip is never dwell-blocked) — it only slows
          gate CHATTERING, so it dodges the deadband's onset problem below
  dhd     deadband + hysteresis + dwell combined (0.005 / 0.005 / 5 steps)
  wash    Washout(final controller), tau_w in {200, 500, 1000, 2000} ms
  gw      Washout(best gate config) + best tau_w, both picked adaptively per
          frac (max mean CLred s.t. zero flags, else max mean; db/hyst/hd/dhd
          pooled for the gate pick) — run last

Deadband scale sanity (SMOKE RESULT — structural tension): eps_db must gate
the noise-induced |C_L - trim| fluctuation at W = 0, sigma_e ~ dCL/dW * sigma
~ (0.46/30) * 0.6 ~ 0.009 at frac=0.02 — but the smoke anchor showed
eps_db=0.01 already costs the CLEAN case the good branch (+6.2% vs +76.6%):
the ~7-step onset delay while the real gust climbs out of the band loses the
branch selection. The db/dhd arms therefore probe the *small*-band end
{0.005, 0.01}; if the tension is real neither can win, and that finding is
the point of the arm.

Washout tau scale (SMOKE RESULT): tau_w in the tens of ms bleeds LEGITIMATE
flap inside the gust (tau=100 ms clean anchor: -62%, flagged). DLR's command
high-pass sits at 0.05 Hz against ~0.5-1 Hz gusts — a cut ~10x slower than
the disturbance. Scaled to our 2.5 Hz (Tg=0.4 s) gust that is a cut around
0.1-0.8 Hz, i.e. tau_w ~ 0.2-2 s: slow enough to spare the in-gust flap,
fast enough to bleed the post-gust DC wind-up (the F1 damage path) within
the 0.9 s metric window. Hence {200, 500, 1000, 2000} ms.

frac=0 ANCHORS: one clean-preview rollout (oracle wc) each for db(0.005),
db(0.01), hyst(0.005), hd, dhd, wash(200..2000) and the adaptive gw combo.
The plain clean anchor is +76.58% CLred — a gate/washout variant that damages
the clean case must be visible here.

Trajectories stored: all 6 seeds of the best config per frac (best_in over
all arms except none) + the paired none seeds.

Output: results/E2_gate.npz            Smoke test: python e2_gate.py --smoke
"""
import os
import sys
import numpy as np
import harness_noise as H
from optimal import OptimalController


class OptimalGated(OptimalController):
    """
    Final controller + chattering-proof causal gate (mitigation E2-gate).

    compute() is a verbatim copy of OptimalController.compute with the plain
    sign gate  `causal = neg if cl0 >= trim else ~neg`  replaced by a
    deadband / hysteresis / dwell state machine — kept in sync by eye with
    light/optimal.py (same convention as controllers_ref.OptimalRdu). The
    single reordering: cl0 is predicted BEFORE the batched candidate scan
    (both calls are read-only on aero._z, so this is numerically neutral)
    so the deadband branch can skip the expensive scan entirely.

    Gate state machine, driven by e = cl0 - C_L_trim (cl0 predicted at flap
    0 from the noisy Wc — the controller has no clean gust signal):

      (a) deadband  |e| < eps_db : no gust signal worth acting on — the flap
          decays toward 0 with time constant tau_decay_ms (rate-limited so
          the decay cannot exceed delta_dot_max), argmin skipped, gate
          memory unchanged.
      (b) hysteresis + dwell     : the allowed half-grid flips to the side
          requested by sign(e) only if |e| > eps_on AND at least `dwell`
          compute() calls have passed since the last flip.

    With eps_db=0, eps_on=0, dwell=0 the behavior is IDENTICAL to the parent
    except on the measure-zero event of a requested sign flip with |e|
    exactly 0.0 (parent: gate follows sign(e) instantly; here: the flip is
    blocked because |e| > 0 fails).

    _steps counts compute() calls since the last gate flip (deadband steps
    included); reset() sets it to a large int so the first flip is never
    dwell-blocked.
    """

    def __init__(self, eps_db=0.0, eps_on=0.0, dwell=0, tau_decay_ms=30.0, **kw):
        super().__init__(**kw)
        self.eps_db    = float(eps_db)
        self.eps_on    = float(eps_on)
        self.dwell     = int(dwell)
        self.tau_decay = float(tau_decay_ms) / 1000.0
        self._decay    = float(np.exp(-self.dt / self.tau_decay))
        self._gate     = None
        self._steps    = 10**9

    def compute(self, state, W, W_next=0.0):
        aero = self.aero
        dg = self._dg; G = self.n_grid
        reach = self.delta_dot_max * self.dt
        Wc = float(W_next) if self.use_wnext else float(W)

        # gate signal (hoisted before the candidate scan; read-only on z)
        cl0 = float(aero.predict(state, 0.0, Wc, self.U)[0])
        e = cl0 - self._C_L_trim

        # (a) deadband: no gust signal worth acting on -> decay flap toward
        # 0, skip the argmin (and the batched scan) entirely
        if self.eps_db > 0.0 and abs(e) < self.eps_db:
            d = self._delta_prev * self._decay
            d = float(np.clip(d, self._delta_prev - reach, self._delta_prev + reach))
            d = float(np.clip(d, -self.delta_max, self.delta_max))
            self._delta_prev = d
            self._steps += 1
            return d

        # frozen-z candidate C_L for all deltas (batched reconstruction only,
        # does not touch aero._z)
        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        CLg = aero.batch_step(z_b, x_b, dg, Wc, self.U, self.dt)[0]

        # (b) hysteresis + dwell replacing `causal = neg if cl0 >= trim else ~neg`
        want = -1 if e >= 0.0 else +1   # -1 -> negative half allowed (dg <= 0)
        if self._gate is None:
            self._gate = want
        elif want != self._gate and abs(e) > self.eps_on and self._steps >= self.dwell:
            self._gate = want
            self._steps = 0
        causal = (dg <= 0.0) if self._gate == -1 else (dg > 0.0)
        self._steps += 1

        ratem = np.abs(dg - self._delta_prev) <= reach + 1e-9

        cost = (CLg - self._C_L_trim) ** 2 + self.R * dg ** 2
        cost = np.where(causal & ratem, cost, np.inf)
        j = int(np.argmin(cost))
        d = float(dg[j])

        if self.refine and 0 < j < G - 1 \
                and np.isfinite(cost[j - 1]) and np.isfinite(cost[j + 1]):
            c0, c1, c2 = cost[j - 1], cost[j], cost[j + 1]
            denom = c0 - 2.0 * c1 + c2
            if denom > 1e-30:
                d += 0.5 * (c0 - c2) / denom * (dg[1] - dg[0])

        d = float(np.clip(d, self._delta_prev - reach, self._delta_prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._delta_prev = d
        return d

    def reset(self):
        super().reset()
        self._gate = None
        self._steps = 10**9


class Washout:
    """
    d_out = d_raw - LP(d_raw): high-passes the flap COMMAND of ANY inner
    controller so a DC flap decays with time constant tau_w regardless of
    what the noisy preview commands (DLR command band-pass, Wallace & Fezans
    EUCASS 2023 — high-pass leg only; the low-pass leg is axis B/E's LPF).

    OPEN-LOOP series filter, the DLR arrangement: the inner controller does
    NOT see the filtered command (its _delta_prev keeps tracking its own raw
    command), exactly as DLR's FF gain never re-reads the post-filter
    actuator signal. The first smoke closed this loop (re-synced the inner's
    _delta_prev to the applied flap each step) and the argmin FOUGHT the
    filter — it re-commanded the bled-off flap, the LP wound up further, and
    the pair rang at the rate limit: tau=100 ms clean anchor -62% flagged.
    The price of the open-loop form is a model-mismatch of size LP(d_raw)
    between the flap the inner assumes and the flap the plant gets — same
    price DLR pays. The washed command is rate- and position-limited in the
    wrapper (actuator constraints hold on the APPLIED command).

    Trade-off measured by the frac=0 anchors: the washout also bleeds off
    LEGITIMATE flap held for times ~tau_w — too small a tau_w damages even
    the clean response (the gust lasts Tg = 0.4 s).
    """

    def __init__(self, inner, tau_w_ms):
        self.inner = inner
        self.tau_w = float(tau_w_ms) / 1000.0
        self.a = float(np.exp(-H.DT / self.tau_w))
        self._s = 0.0
        self._applied = 0.0

    def reset(self):
        self.inner.reset()
        self._s = 0.0
        self._applied = 0.0

    def compute(self, state, W_true, Wc):
        d_raw = self.inner.compute(state, W_true, Wc)
        self._s = self.a * self._s + (1.0 - self.a) * d_raw
        d = d_raw - self._s
        reach = H.DDOT_MAX * H.DT
        d = float(np.clip(d, self._applied - reach, self._applied + reach))
        d = float(np.clip(d, -H.DMAX, H.DMAX))
        self._applied = d
        return d


# ------------------------------------------------------------------------------
W0, Tg = 30.0, 0.4
SMOKE = '--smoke' in sys.argv
FRACS = (0.02,) if SMOKE else (0.02, 0.05)
NSEED = 2 if SMOKE else 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'E2_gate.npz')

HD  = dict(eps_db=0.0, eps_on=0.005, dwell=5)
HD_DETAIL = 'h=0.005+dw=5'
DHD = dict(eps_db=0.005, eps_on=0.005, dwell=5)
DHD_DETAIL = 'db=0.005+h=0.005+dw=5'
WASH_TAUS = (200.0, 500.0, 1000.0, 2000.0)

OL = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# axis E2-gate | W30/Tg0.4 DAMULT={os.environ.get('DAMULT', '1')} | "
      f"open cex0={cex0:.4f}{' | SMOKE' if SMOKE else ''}", flush=True)
recs = [dict(kind='open', axis='E2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def make_gated(eps_db=0.0, eps_on=0.0, dwell=0):
    """OptimalGated with exactly the constructor args H.make_optimal builds."""
    return OptimalGated(eps_db=eps_db, eps_on=eps_on, dwell=dwell,
                        tau_decay_ms=30.0,
                        aero=H.aero, U=H.U, dt=H.DT, R=3e-4,
                        C_L_trim=H.CLTRIM, delta_max=H.DMAX,
                        delta_dot_max=H.DDOT_MAX, n_grid=161,
                        use_wnext=True, refine=True)


# ---- Wc generator (controller-side sensor model) ------------------------------
def wc_plain(rng, frac):
    """Noisy 1-step preview (== axis A/E)."""
    return lambda i, Wt, N: max(0.0, Wt[min(i + 1, N - 1)] + rng.normal(0.0, frac * W0))


# ---- one study point / one clean anchor ----------------------------------------
def run_point(make_ctrl, make_wc, frac, **cfg):
    ms, rs = [], []
    for seed in range(NSEED):
        rng = np.random.default_rng(100 + seed)
        r = H.rollout(make_ctrl(), W0, Tg, wc_fun=make_wc(rng, frac))
        ms.append(H.metrics(r, OL, Tg)); rs.append(r)
    rec = H.point_record(ms, frac=frac, **cfg)
    print(f"  {cfg['arm']:6s} {cfg.get('detail',''):30s} frac={frac:.2f}: "
          f"{H.fmt_stats(H.seed_stats(ms))}", flush=True)
    return rec, ms, rs


def run_anchor(make_ctrl, **cfg):
    """frac=0 point: one clean rollout (wc_fun=None -> oracle preview)."""
    r = H.rollout(make_ctrl(), W0, Tg)
    m = H.metrics(r, OL, Tg)
    rec = H.point_record([m], frac=0.0, **cfg)
    print(f"  {cfg['arm']:6s} {cfg.get('detail',''):30s} frac=0.00: "
          f"clred={m['clred']:+7.2f}% flag='{m['flag']}'"
          f"  (plain clean anchor +76.58%)", flush=True)
    return rec, [m], [r]


points = {}   # (arm, detail, frac) -> (rec, ms, rs)

# ---- frac=0 anchors: does the cure damage the clean case? ----------------------
print("\n=== frac=0 anchors (oracle preview) ===", flush=True)
anchors = [
    ('db',   'db=0.005',   dict(eps_db=0.005, eps_on=0.0, dwell=0),
     lambda: make_gated(eps_db=0.005)),
    ('db',   'db=0.01',    dict(eps_db=0.01, eps_on=0.0, dwell=0),
     lambda: make_gated(eps_db=0.01)),
    ('hyst', 'hyst=0.005', dict(eps_db=0.0, eps_on=0.005, dwell=0),
     lambda: make_gated(eps_on=0.005)),
    ('hd',   HD_DETAIL,    dict(**HD),
     lambda: make_gated(**HD)),
    ('dhd',  DHD_DETAIL,   dict(**DHD),
     lambda: make_gated(**DHD)),
] + [
    ('wash', f'tau={tau:g}ms', dict(tau_w=tau),
     (lambda t=tau: Washout(H.make_optimal(R=3e-4), t)))
    for tau in WASH_TAUS
]
if SMOKE:
    anchors = [a for a in anchors
               if (a[0], a[1]) in (('hd', HD_DETAIL), ('wash', 'tau=500ms'))]
for arm, detail, extra, mk in anchors:
    points[(arm, detail, 0.0)] = run_anchor(
        mk, axis='E2', arm=arm, detail=detail, W0=W0, Tg=Tg, R=3e-4, **extra)


# ---- noisy arms ----------------------------------------------------------------
for frac in FRACS:
    print(f"\n=== frac={frac:.2f} (sigma={frac*W0:.2f} m/s) ===", flush=True)

    k = ('none', '', frac)
    points[k] = run_point(lambda: H.make_optimal(R=3e-4), wc_plain, frac,
                          axis='E2', arm='none', detail='', W0=W0, Tg=Tg, R=3e-4)

    if not SMOKE:
        for eps_db in (0.005, 0.01):
            k = ('db', f'db={eps_db:g}', frac)
            points[k] = run_point(lambda e=eps_db: make_gated(eps_db=e),
                                  wc_plain, frac,
                                  axis='E2', arm='db', detail=f'db={eps_db:g}',
                                  W0=W0, Tg=Tg, R=3e-4,
                                  eps_db=eps_db, eps_on=0.0, dwell=0)

        for eps_on in (0.005, 0.01):
            k = ('hyst', f'hyst={eps_on:g}', frac)
            points[k] = run_point(lambda e=eps_on: make_gated(eps_on=e),
                                  wc_plain, frac,
                                  axis='E2', arm='hyst', detail=f'hyst={eps_on:g}',
                                  W0=W0, Tg=Tg, R=3e-4,
                                  eps_db=0.0, eps_on=eps_on, dwell=0)

    k = ('hd', HD_DETAIL, frac)
    points[k] = run_point(lambda: make_gated(**HD), wc_plain, frac,
                          axis='E2', arm='hd', detail=HD_DETAIL,
                          W0=W0, Tg=Tg, R=3e-4, **HD)

    if not SMOKE:
        k = ('dhd', DHD_DETAIL, frac)
        points[k] = run_point(lambda: make_gated(**DHD), wc_plain, frac,
                              axis='E2', arm='dhd', detail=DHD_DETAIL,
                              W0=W0, Tg=Tg, R=3e-4, **DHD)

        for tau in WASH_TAUS:
            k = ('wash', f'tau={tau:g}ms', frac)
            points[k] = run_point(
                lambda t=tau: Washout(H.make_optimal(R=3e-4), t),
                wc_plain, frac,
                axis='E2', arm='wash', detail=f'tau={tau:g}ms',
                W0=W0, Tg=Tg, R=3e-4, tau_w=tau)


# ---- adaptive gw: best gate config (db/hyst/dhd pooled) + best washout tau -----
def best_in(arms, frac):
    """Pick rule: max mean CLred s.t. zero flags; fallback max mean."""
    cand = [(key, v[0]) for key, v in points.items()
            if key[0] in arms and key[2] == frac]
    clean = [(key, rec) for key, rec in cand if rec['nflag'] == 0]
    pool = clean if clean else cand
    return max(pool, key=lambda kv: kv[1]['mean'])


if not SMOKE:
    gw_anchored = set()
    for frac in FRACS:
        k_gate, rec_gate = best_in(('db', 'hyst', 'hd', 'dhd'), frac)
        k_wash, rec_wash = best_in(('wash',), frac)
        eb = float(rec_gate.get('eps_db', 0.0))
        eo = float(rec_gate.get('eps_on', 0.0))
        dw = int(rec_gate.get('dwell', 0))
        tau = float(rec_wash['tau_w'])
        detail = f'{k_gate[1]}+tau={tau:g}ms'
        print(f"\n=== gw frac={frac:.2f}: gate [{k_gate[0]} {k_gate[1]}] "
              f"+ wash tau={tau:g}ms ===", flush=True)
        k = ('gw', detail, frac)
        points[k] = run_point(
            lambda eb=eb, eo=eo, dw=dw, tau=tau:
                Washout(make_gated(eps_db=eb, eps_on=eo, dwell=dw), tau),
            wc_plain, frac,
            axis='E2', arm='gw', detail=detail, W0=W0, Tg=Tg, R=3e-4,
            eps_db=eb, eps_on=eo, dwell=dw, tau_w=tau)
        # frac=0 anchor of the combo (once per distinct config)
        if (eb, eo, dw, tau) not in gw_anchored:
            gw_anchored.add((eb, eo, dw, tau))
            points[('gw', detail, 0.0)] = run_anchor(
                lambda eb=eb, eo=eo, dw=dw, tau=tau:
                    Washout(make_gated(eps_db=eb, eps_on=eo, dwell=dw), tau),
                axis='E2', arm='gw', detail=detail, W0=W0, Tg=Tg, R=3e-4,
                eps_db=eb, eps_on=eo, dwell=dw, tau_w=tau)


# ---- collect records + trajectories of the best cure per frac ------------------
for key, (rec, ms, rs) in points.items():
    recs.append(rec)

if not SMOKE:
    for frac in FRACS:
        mit = [(key, v) for key, v in points.items()
               if key[2] == frac and key[0] != 'none']
        clean = [(key, v) for key, v in mit if v[0]['nflag'] == 0]
        pool = clean if clean else mit
        key_best, (rec_b, ms_b, rs_b) = max(pool, key=lambda kv: kv[1][0]['mean'])
        print(f"# best cure frac={frac:.2f}: {key_best[0]} {key_best[1]} "
              f"mean {rec_b['mean']:+.1f}%", flush=True)
        for seed, (m, r) in enumerate(zip(ms_b, rs_b)):
            recs.append(H.traj_record(r, m, axis='E2', arm=key_best[0],
                                      detail=key_best[1], W0=W0, Tg=Tg, frac=frac,
                                      seed=seed, label=f'E2G_best_f{frac:g}_s{seed}'))
        # unmitigated trajectories for the same frac (paired money plot)
        rec_n, ms_n, rs_n = points[('none', '', frac)]
        for seed, (m, r) in enumerate(zip(ms_n, rs_n)):
            recs.append(H.traj_record(r, m, axis='E2', arm='none', detail='',
                                      W0=W0, Tg=Tg, frac=frac,
                                      seed=seed, label=f'E2G_none_f{frac:g}_s{seed}'))

H.save_records(OUT, recs)
print("# DONE", flush=True)
