"""
Symmetric-plant control: does the LDNet still win when it is NOT the plant?

The main ablation uses the LDNet as the plant in both arms, which gives the
LDNet-controlled arm a perfect internal model. That is the conservative
reading's weak point, and it deserves a direct answer rather than a caveat.

Ideally the common plant would be CFD, but a closed-loop CFD run costs ~370x
a ROM run and cannot be afforded per cell. The next best thing is to make the
asymmetry work AGAINST the LDNet: run the loop with the LINEAR model as the
plant. Now it is the linear controller that has the perfect internal model
and the LDNet controller that suffers the mismatch.

Interpretation
--------------
  * If the linear arm wins here by roughly the margin the LDNet won by in the
    main ablation, the result is purely an artefact of who is the plant, and
    the ablation says nothing.
  * If the LDNet remains competitive -- or wins -- while steering a plant it
    was never trained on, that cannot be explained by plant privilege.

This is a falsification test for the main result, not a headline number, and
is reported as such.
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
import run_ablation as RA

U, RHO, C, DT, S = 80.0, 1.225, 1.0, 0.002, 0.05
q = 0.5 * RHO * U ** 2 * S
X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')


def build(kind):
    if kind == 'ldnet':
        m = LDNetAero(MD)
    else:
        m = LinearUnsteadyAero(
            coeff_path=os.path.join(HERE,
                                    os.environ.get('COEFFS', 'linear_coeffs.json')))
    m.reset(dt=DT)
    return m


def simulate(plant_kind, ctrl_kind, W0, Tg, TEND=3.0, R=3e-4, NH=8):
    n = int(round(TEND / DT)) + 1
    ts = np.arange(n) * DT
    Wt = np.array([RA.gust(t, W0, Tg) for t in ts])

    p = build(plant_kind)
    CLTRIM = float(p.predict(X0, 0.0, 0.0, U)[0])

    ctrl = None
    if ctrl_kind is not None:
        ctrl = MPCPreviewController(build(ctrl_kind), U=U, dt=DT, rho=RHO, S=S,
                                    C=C, C_L_trim=CLTRIM, N=NH, R=R, R_du=0.0,
                                    G=161, delta_max=14., delta_dot_max=300.)
        ctrl.reset()

    x = X0.copy()
    rec = {k: [] for k in ['al', 'ad', 'hdd', 'add', 'de', 'CL']}
    for i in range(n):
        Wi = float(Wt[i])
        if ctrl is not None:
            lo = i + 1; hi = min(i + 1 + NH, n)
            w = np.zeros(NH); w[:hi - lo] = Wt[lo:hi]
            de = ctrl.compute(x, w, Wi)
            ctrl._delta_prev = de
        else:
            de = 0.0
        cl, cm = p.predict(x, de, Wi, U)
        der = structure.rhs(x, q * cl, q * cm * C)
        p.advance(x, de, Wi, U, DT)
        x = structure.step_dp45(x, q * cl, q * cm * C, DT)
        for k, v in zip(['al', 'ad', 'hdd', 'add', 'de', 'CL'],
                        [x[2], x[3], der[1], der[3], de, float(cl)]):
            rec[k].append(v)
    out = {k: np.array(v) for k, v in rec.items()}
    out['_t'] = ts
    out['_CLTRIM'] = CLTRIM
    return out


def score(r, ol, Tg):
    mw = r['_t'] <= (Tg + 0.5)
    trim = ol['_CLTRIM']
    exo = float(np.max(np.abs(ol['CL'][mw] - trim)))
    exc = float(np.max(np.abs(r['CL'][mw] - trim)))
    flag = ''
    for k in ['ad', 'add', 'hdd']:
        if np.max(np.abs(r[k][mw])) > 3.0 * np.max(np.abs(ol[k][mw])) + 1e-9:
            flag += k + '!'
    return dict(clred=(exo - exc) / exo * 100.0 if exo > 1e-12 else 0.0,
                exo=exo, clexc=exc, flap_max=float(np.max(np.abs(r['de'][mw]))),
                flag=flag)


def main():
    cells = [tuple(float(x) for x in c.split(':'))
             for c in os.environ.get('CELLS', '30:0.4,30:0.3,20:0.7,10:0.4'
                                     ).split(',')]
    ladder = [float(x) for x in os.environ.get(
        'RLADDER', '1e-4,3e-4,1e-3,3e-3,1e-2').split(',')]
    plant_kind = os.environ.get('PLANT', 'linear')

    print(f'=== plant = {plant_kind} (the LINEAR controller now has the '
          f'perfect model) ===\n')
    results = {}
    for (W0, Tg) in cells:
        ol = simulate(plant_kind, None, W0, Tg)
        row = {}
        for ck in ['ldnet', 'linear']:
            best = None
            for R in ladder:
                s = score(simulate(plant_kind, ck, W0, Tg, R=R), ol, Tg)
                s['R'] = R
                if not s['flag'] and (best is None or s['clred'] > best['clred']):
                    best = s
            if best is None:
                best = dict(clred=float('nan'), flag='all-flagged', R=None,
                            flap_max=float('nan'))
            row[ck] = best
            print(f'  W{W0:g}/T{Tg:g}  ctrl={ck:6s}  CLred={best["clred"]:+7.2f}%'
                  f'  R={best["R"]}  flap={best["flap_max"]:.1f}'
                  f'  {best["flag"] or "ok"}', flush=True)
        results[f'{W0:g}:{Tg:g}'] = row
        print(f'    -> gap (ldnet - linear) = '
              f'{row["ldnet"]["clred"] - row["linear"]["clred"]:+.2f} pp\n',
              flush=True)

    with open(os.path.join(HERE, f'symmetric_{plant_kind}.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
