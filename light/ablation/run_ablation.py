"""
Internal-model ablation: LDNet vs linear unsteady aerodynamics inside the
SAME GLA loop, at equal preview and equal cost function.

Design
------
The plant is the LDNet in BOTH arms. Only the controller's internal model
changes. This is the comparison the thesis paragraph asks for: it isolates
what the internal model contributes to the alleviation, with everything else
-- preview, horizon, cost weights, flap grid, rate limit, saturation,
structural integrator, gust -- held identical.

Using the LDNet as the plant in both arms is the conservative choice and
needs stating plainly: the LDNet-controlled arm is then running with a
perfect internal model, which flatters it. The alternative (each controller
simulated against its own model) would be self-consistent but would measure
nothing, since each controller would be perfect in its own world and the
comparison would reduce to which model is easier to control. A third option,
CFD as the common plant, is what the thesis would ideally use but costs
~370x more per run and is out of reach here; this is recorded as a caveat
rather than hidden.

Arms
----
  open    : no control (delta = 0) -- the reference excursion
  ldnet   : MPCPreviewController with the LDNet as internal model
  linear  : MPCPreviewController with LinearUnsteadyAero as internal model

Usage
-----
  W0=30 TG=0.4 RW=3e-4 ARM=linear python3 -u run_ablation.py
  (or import simulate_arm / run_cell from a sweep driver)
"""
import os
import sys
import time
import json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import structure
structure.D_ALPHA *= float(os.environ.get('DAMULT', '3'))

from ldnet_aero import LDNetAero
from linear_aero import LinearUnsteadyAero
from optimal import MPCPreviewController

U, RHO, C, DT, S = 80.0, 1.225, 1.0, 0.002, 0.05
q = 0.5 * RHO * U ** 2 * S
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')

SCORE_W = dict(Q_CL=1.0, Q_ad=100.0, R=0.01)

# module-level plant, loaded once (TF model construction is slow)
_PLANT = None


def plant():
    global _PLANT
    if _PLANT is None:
        _PLANT = LDNetAero(MD)
        _PLANT.reset(dt=DT)
    return _PLANT


def make_internal(kind):
    """Build the controller's internal model."""
    if kind == 'ldnet':
        # a SEPARATE LDNet instance, so the controller cannot share the
        # plant's latent state by accident; it carries its own z_ctrl anyway
        m = LDNetAero(MD)
        m.reset(dt=DT)
        return m
    if kind == 'linear':
        coeffs = os.environ.get('COEFFS', 'linear_coeffs.json')
        m = LinearUnsteadyAero(coeff_path=os.path.join(HERE, coeffs))
        m.reset(dt=DT)
        return m
    raise ValueError(kind)


def gust(t, W0, Tg):
    return (W0 / 2.0) * (1 - np.cos(2 * np.pi * t / Tg)) if (0 <= t <= Tg) else 0.0


def simulate_arm(arm, W0, Tg, TEND=3.0, R=3e-4, DMAX=14.0, NGRID=161, NH=8,
                 R_du=0.0):
    """
    Run one arm. arm in {'open','ldnet','linear'}.

    The plant is always the LDNet. For the controlled arms the MPC is given
    the oracle preview w_seq[k] = W(t+(k+1)*dt), exactly as run.py does.
    """
    n = int(round(TEND / DT)) + 1
    ts = np.arange(n) * DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])

    p = plant()
    p.reset(dt=DT)
    CLTRIM = float(p.predict(X0, 0.0, 0.0, U)[0])

    ctrl = None
    if arm != 'open':
        internal = make_internal(arm)
        ctrl = MPCPreviewController(
            internal, U=U, dt=DT, rho=RHO, S=S, C=C,
            C_L_trim=CLTRIM, N=NH, R=R, R_du=R_du,
            G=NGRID, delta_max=DMAX, delta_dot_max=300.)
        ctrl.reset()

    x = X0.copy()
    rec = {k: [] for k in ['h', 'hd', 'al', 'ad', 'hdd', 'add', 'de', 'CL', 'CM']}
    comp_t, comp_n = 0.0, 0

    for i in range(n):
        Wi = float(Wt[i])
        if arm != 'open':
            t0 = time.perf_counter()
            lo = i + 1
            hi = min(i + 1 + NH, n)
            w_seq = np.zeros(NH)
            w_seq[:hi - lo] = Wt[lo:hi]
            de = ctrl.compute(x, w_seq, Wi)
            comp_t += time.perf_counter() - t0
            comp_n += 1
            ctrl._delta_prev = de
        else:
            de = 0.0

        cl, cm = p.predict(x, de, Wi, U)
        Fy = q * cl
        Mz = q * cm * C
        der = structure.rhs(x, Fy, Mz)
        p.advance(x, de, Wi, U, DT)
        x = structure.step_dp45(x, Fy, Mz, DT)
        for k, v in zip(['h', 'hd', 'al', 'ad', 'hdd', 'add', 'de', 'CL', 'CM'],
                        [x[0], x[1], x[2], x[3], der[1], der[3], de,
                         float(cl), float(cm)]):
            rec[k].append(v)

    out = {k: np.array(v) for k, v in rec.items()}
    out['_t'] = ts
    out['_Wt'] = Wt
    out['_CLTRIM'] = CLTRIM
    out['_comp_ms'] = (comp_t / comp_n * 1e3) if comp_n else 0.0
    return out


def metrics(r, r_open, Tg):
    """Same metric definitions as light/run.py, so numbers are comparable."""
    t = r['_t']
    mw = t <= (Tg + 0.5)
    trim = r_open['_CLTRIM']
    exo = float(np.max(np.abs(r_open['CL'][mw] - trim)))
    exc = float(np.max(np.abs(r['CL'][mw] - trim)))
    clred = (exo - exc) / exo * 100.0 if exo > 1e-12 else 0.0
    flag = ''
    for k in ['ad', 'add', 'hdd']:
        if np.max(np.abs(r[k][mw])) > 3.0 * np.max(np.abs(r_open[k][mw])) + 1e-9:
            flag += k + '!'
    dCL = r['CL'][mw] - trim
    J = float(SCORE_W['Q_CL'] * np.sum(dCL ** 2)
              + SCORE_W['Q_ad'] * np.sum(r['ad'][mw] ** 2)
              + SCORE_W['R'] * np.sum(r['de'][mw] ** 2))
    pitch_ol = float(np.max(np.abs(r_open['al'][mw])))
    pitch = float(np.max(np.abs(r['al'][mw])))
    return dict(
        clexc=exc, exo=exo, clred=clred,
        flap_max=float(np.max(np.abs(r['de'][mw]))),
        adrms=float(np.sqrt(np.mean(r['ad'][mw] ** 2)) * 180 / np.pi),
        pitchpk=pitch,
        pitch_ratio=(pitch / pitch_ol if pitch_ol > 1e-12 else float('nan')),
        J=J, comp_ms=float(r.get('_comp_ms', 0.0)), flag=flag)


def run_cell(W0, Tg, R, arms=('ldnet', 'linear'), **kw):
    """Run the open reference plus the requested arms for one gust cell."""
    ol = simulate_arm('open', W0, Tg, **kw)
    res = {}
    for a in arms:
        r = simulate_arm(a, W0, Tg, R=R, **kw)
        res[a] = metrics(r, ol, Tg)
    return res


if __name__ == '__main__':
    W0 = float(os.environ.get('W0', '30'))
    TG = float(os.environ.get('TG', '0.4'))
    RW = float(os.environ.get('RW', '3e-4'))
    ARMS = os.environ.get('ARMS', 'ldnet,linear').split(',')
    TEND = float(os.environ.get('TEND', '3.0'))

    t0 = time.time()
    res = run_cell(W0, TG, RW, arms=tuple(ARMS), TEND=TEND)
    print(f'W0={W0} Tg={TG} R={RW:g}  ({time.time()-t0:.0f}s)')
    for a, m in res.items():
        print(f'  {a:7s}: CLexc {m["exo"]:.4f}->{m["clexc"]:.4f} '
              f'({m["clred"]:+.1f}%)  flap={m["flap_max"]:.1f}deg  '
              f'pitch_ratio={m["pitch_ratio"]:.2f}  '
              f'{"FLAG:"+m["flag"] if m["flag"] else "stable"}')
    print(json.dumps({a: {k: v for k, v in m.items()} for a, m in res.items()}))
