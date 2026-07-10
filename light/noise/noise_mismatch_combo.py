"""
Structural model-mismatch robustness of the E2-combo pipeline.

All previous axes corrupt what the controller SEES; here plant and internal
model stop being twins: the PLANT flies with perturbed parameters while the
MPC predicts its horizon with the nominal ones (Forte/NASA 2023: 2.5%
disturbance-frequency mismatch collapsed their GLA 69->39%; Fournier 2022
makes the synthesis robust to model uncertainty by construction).

Arms (deterministic, oracle-clean preview -- the axis isolates model error):
  dalpha  plant D_ALPHA x {0.67,0.83,1.17,1.33}  (plant DAMULT ~ 2,2.5,3.5,4
          vs the controller's 3; DAMULT=3 is applied at import by the
          harness, multipliers are relative to that nominal)
  kalpha  plant K_ALPHA x {0.90,0.95,1.05,1.10}  (pitch stiffness -> natural
          frequency, the Forte analogue)
  uinf    controller U in {76,84} m/s vs plant 80 (airspeed estimate error:
          the MPC's q AND its aero evaluations use the believed U)
  cltrim  controller C_L_trim x {0.95,1.05} (trim estimate error)
  anchor  all nominal (gate +80.51%)

Mechanics: structure.D_ALPHA / structure.K_ALPHA are module globals read at
call time by BOTH the plant step (structure.rhs via step_dp45) and the MPC
horizon (optimal.dp45_batch). The adapter sets them to NOMINAL for the
duration of mpc.compute and restores the PERTURBED values before returning,
so the plant integrates with the perturbed structure while the controller
predicts with the nominal one. Each perturbed plant is compared against ITS
OWN open-loop rollout (cex0 logged per point). Declared limit: the LDNet
aero model is shared plant/controller -- this axis tests STRUCTURAL
mismatch only. X0 is the nominal equilibrium; the residual initial
transient under perturbed stiffness is negligible vs the gust response and
cancels in the relative metric.

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, home cell W30/Tg0.4, DAMULT=3.
Output: results/D2_mismatch.npz
--smoke: anchor + dalpha x1.33.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import harness_noise as H
import structure as S
from optimal import FusedPreviewSensor, MPCPreviewController

W0, Tg   = 30.0, 0.4
JMAX, N  = 50, 8
R        = 3e-4
R_DU     = 0.0
LAM      = 0.0
SMOKE    = '--smoke' in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'D2_mismatch.npz')

DA_NOM = float(S.D_ALPHA)      # after the harness applied DAMULT=3
KA_NOM = float(S.K_ALPHA)

if SMOKE:
    PTS = [('anchor', 1.0), ('dalpha', 1.33)]
else:
    PTS = ([('anchor', 1.0)]
           + [('dalpha', m) for m in (0.67, 0.83, 1.17, 1.33)]
           + [('kalpha', m) for m in (0.90, 0.95, 1.05, 1.10)]
           + [('uinf', u) for u in (76.0, 84.0)]
           + [('cltrim', m) for m in (0.95, 1.05)])


def _delivered(r):
    """(sigma_del, bias_del) of the Wc channel vs the true next-step gust."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw  = t <= (Tg + 0.5)
    err = Wc[mw] - Wnext[mw]
    return float(np.std(err)), float(np.mean(err))


class _MismatchCtrl:
    """
    Combo adapter that runs mpc.compute under NOMINAL structure globals and
    restores the PERTURBED (plant) values before returning, so the plant
    step that follows in the harness loop integrates the perturbed system.
    pert: list of (attr, plant_value, nominal_value); empty for uinf/cltrim
    arms (those mis-set the controller's own constructor args instead).
    """

    def __init__(self, sensor, mpc, pert):
        self._sensor = sensor
        self._mpc    = mpc
        self._pert   = list(pert)

    def reset(self):
        self._sensor.reset()
        self._mpc.reset()

    def compute(self, state, W_true, Wc):
        for a, pv, nv in self._pert:
            setattr(S, a, nv)
        try:
            return self._mpc.compute(state, self._sensor.last)
        finally:
            for a, pv, nv in self._pert:
                setattr(S, a, pv)


def build(arm, val):
    """-> (pert list, mpc kwargs overrides)"""
    if arm == 'dalpha':
        return [('D_ALPHA', val * DA_NOM, DA_NOM)], {}
    if arm == 'kalpha':
        return [('K_ALPHA', val * KA_NOM, KA_NOM)], {}
    if arm == 'uinf':
        return [], dict(U=float(val))
    if arm == 'cltrim':
        return [], dict(C_L_trim=float(val) * H.CLTRIM)
    return [], {}                                    # anchor


def make_combo(rng, mpc_kw):
    sensor = FusedPreviewSensor(rng, lambda j: 1e-9, JMAX, N, lam=LAM)
    kw = dict(U=H.U, C_L_trim=H.CLTRIM)
    kw.update(mpc_kw)
    mpc = MPCPreviewController(
        H.aero, U=kw['U'], dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=kw['C_L_trim'], N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    return sensor, mpc


OL_NOM   = H.rollout(None, W0, Tg)
cex0_nom = H.metrics(OL_NOM, OL_NOM, Tg)['exo']
print(f"# D2_mismatch | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0_nom:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='D2', W0=W0, Tg=Tg, cex0=cex0_nom,
             t=OL_NOM['_t'], W=OL_NOM['_Wt'], CL=OL_NOM['CL'])]

for arm, val in PTS:
    pert, mpc_kw = build(arm, val)

    # plant state = perturbed for the whole point (OL + closed loop)
    for a, pv, nv in pert:
        setattr(S, a, pv)
    try:
        OLp = H.rollout(None, W0, Tg) if pert else OL_NOM
        cex0p = H.metrics(OLp, OLp, Tg)['exo']

        rng = np.random.default_rng(100)
        sensor, mpc = make_combo(rng, mpc_kw)
        ctrl = _MismatchCtrl(sensor, mpc, pert)
        r = H.rollout(ctrl, W0, Tg, wc_fun=sensor.wc_fun)
        m = H.metrics(r, OLp, Tg)
    finally:
        for a, pv, nv in pert:
            setattr(S, a, nv)

    sd, bd = _delivered(r)
    rec = H.point_record([m], axis='D2', arm=arm, value=float(val),
                         W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM, R_du=R_DU,
                         frac=0.0, cex0=float(cex0p),
                         sigma_del=sd, bias_del=bd)
    recs.append(rec)
    print(f"  {arm}={val:g}: {H.fmt_stats(H.seed_stats([m]))}  "
          f"cex0={cex0p:.4f}", flush=True)

H.save_records(OUT, recs)
print("# DONE", flush=True)
