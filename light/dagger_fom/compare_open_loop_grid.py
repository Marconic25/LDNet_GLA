#!/usr/bin/env python3
"""
Pure open-loop (delta=0) prediction error, ROM vs FOM, across the full 3x4
(W0 x Tg) grid. No controller, no flap, no closed-loop retroaction -- the
cleanest possible isolation of the LDNet surrogate's aerodynamic error, as
established for W20/Tg0.70 (28% underestimate) and W30/Tg0.40 (74%).

exo_ROM comes from traces_W{W0}.npz's own '{tag}_cex0' field (the ROM's
open-loop excursion, already computed the same way the R* grid search used).
exo_FOM comes from a real FOM open-loop run (Rsweep_.../OpenLoop_W{W0}_Tg{Tg}/
structural_trajectory.csv), same metric: max|Fy/q - CLTRIM| over t-t0<=Tg+0.5.

Purpose: decide whether the underestimate scales systematically with gust
amplitude W0 (points to a model capacity/extrapolation limit -- the never-tried
lever of widening NNdyn) or is scattered across the grid (points to a training
data coverage gap -- fixable by a cheap, clean fine-tune on open-loop FOM
trajectories, decoupled from every closed-loop confound found in this study).

Pure numpy, no TensorFlow.

Usage:
    python3 compare_open_loop_grid.py --npz-dir DIR --openloop-dir DIR
"""
import argparse
import csv as _csv
import os

import numpy as np

RHO, U, S = 1.225, 80.0, 0.05
Q = 0.5 * RHO * U ** 2 * S
CLTRIM_DEFAULT = 0.8683425957523628

W_LIST = [10, 20, 30]
TG_LIST = [0.30, 0.40, 0.70, 1.20]


def exo_fom(csv_path, tg, cltrim):
    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    d = np.array([[float(v) for v in r] for r in rows[1:]])
    t = d[:, 0] - d[0, 0]
    m = t <= tg + 0.5
    cl = d[m, 5] / Q
    return float(np.max(np.abs(cl - cltrim))), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--openloop-dir', required=True,
                    help='dir containing OpenLoop_W{W0}_Tg{Tg}/structural_trajectory.csv')
    ap.add_argument('--cltrim', type=float, default=CLTRIM_DEFAULT)
    args = ap.parse_args()

    print(f'q = {Q:.4f} N   CLTRIM = {args.cltrim:.10f}\n')
    print(f'{"cell":>14s}  {"exo_ROM":>10s}  {"exo_FOM":>10s}  {"gap%":>7s}  {"n_fom":>6s}')

    rows = []
    for W0 in W_LIST:
        npz_path = os.path.join(args.npz_dir, f'traces_W{W0}.npz')
        d = np.load(npz_path)
        for Tg in TG_LIST:
            tag = f'Tg{Tg:.2f}'
            exo_rom = float(d[f'{tag}_cex0'])
            csv_path = os.path.join(args.openloop_dir, f'OpenLoop_W{W0}_Tg{Tg:.2f}',
                                    'structural_trajectory.csv')
            if not os.path.exists(csv_path):
                print(f'  [MISSING] {csv_path}')
                continue
            exo_f, n = exo_fom(csv_path, Tg, args.cltrim)
            gap = (exo_f - exo_rom) / exo_f * 100.0
            rows.append((W0, Tg, exo_rom, exo_f, gap))
            print(f'  W{W0:2d}/Tg{Tg:.2f}    {exo_rom:10.6f}  {exo_f:10.6f}  '
                  f'{gap:+6.1f}%  {n:6d}')

    print('\n=== per ampiezza W0 (media e range del gap%) ===')
    for W0 in W_LIST:
        gaps = [g for (w, tg, er, ef, g) in rows if w == W0]
        if gaps:
            print(f'  W0={W0:2d}:  mean={np.mean(gaps):+6.1f}%  '
                  f'range=[{min(gaps):+.1f}, {max(gaps):+.1f}]  n={len(gaps)}')

    print('\n=== per k / durata Tg (media e range del gap%) ===')
    for Tg in TG_LIST:
        gaps = [g for (w, tg, er, ef, g) in rows if tg == Tg]
        if gaps:
            print(f'  Tg={Tg:.2f}:  mean={np.mean(gaps):+6.1f}%  '
                  f'range=[{min(gaps):+.1f}, {max(gaps):+.1f}]  n={len(gaps)}')

    if len(rows) >= 3:
        W0s = np.array([r[0] for r in rows], float)
        gaps = np.array([r[4] for r in rows], float)
        r_w0 = float(np.corrcoef(W0s, gaps)[0, 1])
        print(f'\nPearson r(gap%, W0) = {r_w0:+.3f}  (n={len(rows)})')


if __name__ == '__main__':
    main()
