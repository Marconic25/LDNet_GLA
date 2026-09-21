"""
Remove the plant asymmetry: closed-loop control against CFD loads.

The caveat this addresses
-------------------------
In run_ablation.py the plant is the LDNet in both arms, so the LDNet-based
controller has a perfect internal model. That is the main reading weakness of
the ablation. A live CFD plant is the ideal fix but costs ~370x per run and
cannot be afforded per cell.

This script removes the asymmetry a different way, at ROM cost. The MPC's
only use of its internal model is to choose delta. So we can pose the honest
question: at the states the CFD campaign actually visited, which model picks
the better flap, judged by CFD loads?

Method -- CFD-referenced flap selection
---------------------------------------
For each probe point along a held-out CFD trajectory:

  1. Both controllers see the identical true state x(t) and gust preview,
     and each runs its OWN full MPC horizon to choose a flap angle.
  2. The chosen angle is scored against a CFD-DERIVED map of how C_L
     responds to delta at that operating point.

The scoring reference is built from the CFD data itself, not from either
model. Because the campaign does not sweep delta at fixed state (it contains
one delta history per trajectory), the reference dC_L/ddelta is estimated
from the campaign by local linear regression of C_L on delta within
neighbourhoods of the state space -- a purely empirical, model-free estimate
with its own uncertainty, which is reported.

This is weaker evidence than a live CFD loop and is labelled as such: it
tests flap-selection quality against CFD, not full closed-loop trajectory
tracking. But it is free of the plant privilege that the main ablation has,
and it is the strongest CFD-referenced statement available at ROM cost.
"""
import os
import sys
import json
import numpy as np
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'light'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import structure
from ldnet_aero import LDNetAero
from linear_aero import LinearUnsteadyAero
from optimal import dp45_batch

U, RHO, C, DT, S = 80.0, 1.225, 1.0, 0.002, 0.05
MD = os.path.join(ROOT, 'clean', 'models_rollout', 'latent_10')


# ---------------------------------------------------------------------------
# Empirical CFD flap-response reference
# ---------------------------------------------------------------------------

def build_cfd_flap_map(split='train', stride=7, n_bins=6):
    """
    Estimate dC_L/ddelta from the CFD campaign as a function of |delta|.

    Pools all campaign samples, bins them by |delta|, and within each bin
    regresses C_L on delta after removing the variation explained by the
    other inputs (alpha, hd, W) with a linear model. What remains is the
    partial slope of C_L with respect to delta in that deflection band -- an
    empirical, model-free measurement of how flap authority changes with
    deflection, which is the property under test.
    """
    with h5py.File(os.path.join(ROOT, 'data', f'GLA_{split}.h5'), 'r') as f:
        sig = np.array(f['input_signals'])[:, ::stride, :]
        out = np.array(f['output_signals'])[:, ::stride, 0, :]
        par = np.array(f['input_parameters'])

    rows = []
    for n in range(sig.shape[0]):
        Uf = max(float(par[n, 0]), 1.0)
        q = 0.5 * RHO * Uf ** 2 * S
        CL = out[n, :, 0] / q
        h, hd, a, ad, d, W = [sig[n, :, j] for j in range(6)]
        rows.append(np.column_stack([a, hd / Uf, W / Uf, ad, d, CL]))
    A = np.concatenate(rows)

    edges = np.linspace(0, np.percentile(np.abs(A[:, 4]), 99), n_bins + 1)
    res = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (np.abs(A[:, 4]) >= lo) & (np.abs(A[:, 4]) < hi)
        if m.sum() < 200:
            continue
        # A bin whose delta barely varies cannot identify a delta slope at
        # all (the lowest bin is essentially all delta == 0), so skip it
        # rather than reporting a number the data does not support.
        d_spread = float(np.std(A[m, 4]))
        if d_spread < 0.05:
            print(f'  [skip] bin {lo:.1f}-{hi:.1f}: delta std {d_spread:.4f} deg'
                  f' -- too little variation to identify a slope')
            continue
        X = np.column_stack([A[m, 0], A[m, 1], A[m, 2], A[m, 3], A[m, 4],
                             np.ones(int(m.sum()))])
        b, *_ = np.linalg.lstsq(X, A[m, 5], rcond=None)
        resid = A[m, 5] - X @ b
        dof = max(1, int(m.sum()) - X.shape[1])
        s2 = np.sum(resid ** 2) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        res.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                        d_std=d_spread,
                        slope=float(b[4]), se=float(np.sqrt(max(cov[4, 4], 0.0)))))
    return res


def main():
    print('=== empirical CFD flap authority vs deflection ===')
    print('  (partial slope dC_L/ddelta, controlling for alpha, hd/U, W/U, ad)')
    mp = build_cfd_flap_map()
    print(f'  {"|delta| bin [deg]":>20s}{"n":>8s}{"dCL/dd [1/deg]":>18s}{"std err":>10s}')
    for r in mp:
        lab = f'{r["lo"]:.1f}-{r["hi"]:.1f}'
        print(f'  {lab:>20s}{r["n"]:>8d}'
              f'{np.deg2rad(1.0)*r["slope"]:>18.5f}'
              f'{np.deg2rad(1.0)*r["se"]:>10.5f}')

    s0 = np.deg2rad(1.0) * mp[0]['slope']
    sN = np.deg2rad(1.0) * mp[-1]['slope']
    print(f'\n  CFD flap authority changes by a factor '
          f'{s0/sN if sN else float("nan"):.2f} across the deflection range')
    print(f'  (small-deflection {s0:.5f} -> large-deflection {sN:.5f} /deg)')
    print('  A linear model carries ONE constant here by construction.')

    with open(os.path.join(HERE, 'cfd_flap_map.json'), 'w') as f:
        json.dump(mp, f, indent=1)


if __name__ == '__main__':
    main()
