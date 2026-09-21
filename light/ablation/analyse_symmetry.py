"""
Quantify how much of the closed-loop margin is plant privilege.

The symmetric test came back with an uncomfortable answer: with the linear
model as the controlled system, the LINEAR controller wins by 28-51 pp. In
the main ablation, with the LDNet as the controlled system, the LDNET
controller wins by 31-348 pp. Whichever model is the controlled system, its
own controller wins. That is the signature of self-consistency advantage,
and it means the raw closed-loop margin cannot be read as model quality.

This script separates the two effects with the standard 2x2 decomposition:

                        controller
                     LDNet     linear
  system  LDNet      A          B
          linear     C          D

  self-consistency effect = ((A - C) + (D - B)) / 2
      how much a controller gains purely from matching the system

  model effect            = ((A - B) + (C - D)) / 2
      how much the LDNet controller gains ACROSS both systems, i.e. the part
      that does not depend on who the system is

If the model effect is ~0, the closed-loop comparison says nothing about
model quality and the thesis claim must rest entirely on the held-out
full-order forecast comparison (run_cfd_horizon.py), which has no such
confound.

Reporting this honestly is the point. A 2x2 that shows the margin is mostly
privilege is a result about the experiment's design, not a failure of the
work, and it changes which number belongs in the thesis.
"""
import os
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_main():
    res = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'shards', 'cell_*.json'))):
        with open(p) as f:
            res.update(json.load(f))
    return res


def main():
    sym_path = os.path.join(HERE, 'symmetric_linear.json')
    if not os.path.exists(sym_path):
        print('symmetric_linear.json missing')
        return
    with open(sym_path) as f:
        sym = json.load(f)
    main_res = load_main()

    print('2x2 decomposition of the closed-loop margin\n')
    print(f'{"cell":>10s}{"A ld/ld":>10s}{"B ld/lin":>10s}'
          f'{"C lin/ld":>10s}{"D lin/lin":>11s}'
          f'{"self":>9s}{"model":>9s}')

    rows = []
    for cell, row in sym.items():
        if cell not in main_res:
            continue
        m = main_res[cell]
        if 'ldnet' not in m or 'linear' not in m:
            continue
        A = m['ldnet']['best']['clred']     # LDNet system, LDNet controller
        B = m['linear']['best']['clred']    # LDNet system, linear controller
        C = row['ldnet']['clred']           # linear system, LDNet controller
        D = row['linear']['clred']          # linear system, linear controller
        self_eff = ((A - C) + (D - B)) / 2.0
        model_eff = ((A - B) + (C - D)) / 2.0
        rows.append((cell, A, B, C, D, self_eff, model_eff))
        print(f'{cell:>10s}{A:>10.1f}{B:>10.1f}{C:>10.1f}{D:>11.1f}'
              f'{self_eff:>9.1f}{model_eff:>9.1f}')

    if not rows:
        print('\nno overlapping cells yet')
        return

    se = np.array([r[5] for r in rows])
    me = np.array([r[6] for r in rows])
    print(f'\n  mean self-consistency effect : {se.mean():+.1f} pp')
    print(f'  mean model effect            : {me.mean():+.1f} pp')
    print(f'\n  share of the raw margin attributable to matching the '
          f'controlled system: {se.mean()/(se.mean()+abs(me.mean()))*100:.0f}%')

    print('\ninterpretation:')
    if me.mean() > 10:
        print('  the LDNet controller retains a real advantage ACROSS both')
        print('  controlled systems; the closed-loop comparison is informative')
    elif me.mean() > -10:
        print('  the model effect is within noise: the closed-loop margin is')
        print('  dominated by self-consistency and must NOT be quoted as')
        print('  evidence of model quality. The thesis claim rests on the')
        print('  held-out full-order forecast comparison instead.')
    else:
        print('  the LDNet controller is WORSE across systems once privilege')
        print('  is removed -- the closed-loop result would be an artefact')

    with open(os.path.join(HERE, 'symmetry_decomposition.json'), 'w') as f:
        json.dump(dict(rows=[dict(cell=r[0], A=r[1], B=r[2], C=r[3], D=r[4],
                                  self_effect=r[5], model_effect=r[6])
                             for r in rows],
                       mean_self=float(se.mean()),
                       mean_model=float(me.mean())), f, indent=1)


if __name__ == '__main__':
    main()
