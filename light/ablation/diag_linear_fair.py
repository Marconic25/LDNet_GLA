"""
Is the linear controller's collapse a fair loss, or a fixable mis-specification?

The linear arm returns CLred of -150% to -266% at EVERY tuning on the mild
W10 cells, with the pitch ratio above 4 throughout. A baseline that fails
even at the gentlest gust and the heaviest effort penalty is suspicious: the
thesis' own prediction is that the linear model should be ADEQUATE at low
r_g and small deflections, and only fail at the sharp/severe corner. A
uniform collapse suggests something structural, and reporting it without
checking would overstate the case for the LDNet.

Candidate causes, in order of how much they would undermine the comparison:

C1  The R ladder is simply too aggressive for this model. The MPC cost is
    R*delta^2 + sum (C_L - trim)^2; if the linear model mis-predicts the
    achievable C_L it will always ask for too much flap. A much larger R
    would restrain it. Chapter 3 selected R per cell from a fixed ladder,
    but that ladder was chosen around the LDNet's needs -- extending it for
    the linear arm is fairer, not softer.
C2  The controller latent z_ctrl is being advanced with the leak term
    `z_b = z_new - self.lam * z_b` inside the MPC horizon. lam is read from
    aero._z_leak, which is 0.003 for the LDNet but 0 for the linear model --
    correct, but worth confirming it is not doing something odd.
C3  A genuine feedback instability: the linear model under-predicts flap
    effectiveness (measured ratio 1.11-1.49), so the MPC over-commands,
    overshoots, and rings. This IS the honest finding if C1 is excluded.

This script extends the R ladder far beyond chapter 3's range and reports the
best achievable linear performance, to establish whether the collapse
survives a fair tuning search.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_ablation as RA


def main():
    W0 = float(os.environ.get('W0', '10'))
    Tg = float(os.environ.get('TG', '0.4'))
    # far wider than chapter 3's ladder: up to a 1e4x heavier effort penalty
    ladder = [float(x) for x in os.environ.get(
        'RLADDER',
        '1e-4,1e-3,1e-2,3e-2,1e-1,3e-1,1e0,3e0,1e1,1e2').split(',')]

    ol = RA.simulate_arm('open', W0, Tg)
    print(f'=== W0={W0:g} Tg={Tg:g}: extended R search for the linear arm ===')
    print(f'{"R":>10s}{"CLred":>10s}{"flap":>8s}{"pitch_r":>9s}{"adrms":>9s}'
          f'{"flag":>12s}')
    best = None
    for R in ladder:
        m = RA.metrics(RA.simulate_arm('linear', W0, Tg, R=R), ol, Tg)
        print(f'{R:>10g}{m["clred"]:>10.2f}{m["flap_max"]:>8.2f}'
              f'{m["pitch_ratio"]:>9.2f}{m["adrms"]:>9.3f}'
              f'{(m["flag"] or "ok"):>12s}', flush=True)
        if not m['flag'] and (best is None or m['clred'] > best[1]['clred']):
            best = (R, m)

    print()
    if best is None:
        print('  no tuning passes the stability check even with R up to '
              f'{max(ladder):g}')
    else:
        print(f'  best stable linear tuning: R={best[0]:g} -> '
              f'CLred {best[1]["clred"]:+.2f}%  flap {best[1]["flap_max"]:.1f}deg')

    # reference: the LDNet on the same cell at chapter-3's ladder
    ldb = None
    for R in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        m = RA.metrics(RA.simulate_arm('ldnet', W0, Tg, R=R), ol, Tg)
        if not m['flag'] and (ldb is None or m['clred'] > ldb[1]['clred']):
            ldb = (R, m)
    if ldb:
        print(f'  LDNet best on same cell: R={ldb[0]:g} -> '
              f'CLred {ldb[1]["clred"]:+.2f}%  flap {ldb[1]["flap_max"]:.1f}deg')


if __name__ == '__main__':
    main()
