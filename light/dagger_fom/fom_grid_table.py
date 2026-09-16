#!/usr/bin/env python3
"""
Full-grid CLred table for the MPC-FOM verification campaign.

For each (W0, Tg) cell of the CS-25 grid, compare the closed-loop co-simulation
(controller predicts with the LDNet surrogate, is fed the true CFD structural
state every coupling window) against the open-loop run on the SAME plant.

Both excursions are measured the same way and on the same plant:
    exc/exo = max |C_L - C_L,trim|   over   t - t[0] <= Tg + 0.5
    CLred   = (exo - exc) / exo

This deliberately does NOT reuse validate_iteration.py's baseline: that script
takes exo from the ROM (validate_iteration.py:90-93), which divides two
different plants. NOTES.md flags that as an error; the ROM underestimates the
open-loop peak by up to 74%, so a ROM-referenced CLred flatters the FOM.
The ROM's own self-consistent CLred is reported alongside, for the gap.

No import of light/run.py: everything needed from the ROM side is already
stored in traces_W{W0}.npz, which avoids run.py's import-time DAMULT coupling.

Usage:
    python3 fom_grid_table.py [--base DIR] [--traces DIR]
"""
import argparse
import csv as _csv
import glob
import os
import re

import numpy as np

U, RHO, S = 80.0, 1.225, 0.05
Q = 0.5 * RHO * U ** 2 * S
W_LIST = [10, 20, 30]
TG_LIST = [0.30, 0.40, 0.70, 1.20]


def load_csv(path):
    with open(path) as f:
        rows = list(_csv.reader(f))
    d = np.array([[float(v) for v in r] for r in rows[1:]])
    return dict(t=d[:, 0], Fy=d[:, 5], delta=d[:, 8])


def count_oscillation(t, delta, tg):
    d = delta[t - t[0] <= tg + 0.5]
    signs = np.sign(np.diff(d))
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def metrics(path, tg, cltrim):
    """(peak excursion, flap_max, osc_count) over the evaluation window."""
    tr = load_csv(path)
    t = tr['t'] - tr['t'][0]
    m = t <= tg + 0.5
    exc = float(np.max(np.abs(tr['Fy'][m] / Q - cltrim)))
    return exc, float(np.max(np.abs(tr['delta'][m]))), count_oscillation(tr['t'], tr['delta'], tg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/work/u10677113/NACA2312/mpc_fom_dagger')
    ap.add_argument('--traces', default='/work/u10677113/LDNet_GLA/light/results_cs25_combo')
    args = ap.parse_args()

    rows = []
    for W0 in W_LIST:
        npz = np.load(os.path.join(args.traces, f'traces_W{W0}.npz'))
        cltrim = float(npz['CLTRIM'])
        for Tg in TG_LIST:
            tag = f'Tg{Tg:.2f}'
            ol = os.path.join(args.base, f'OpenLoop_W{W0}_Tg{Tg:.2f}',
                              'structural_trajectory.csv')
            hits = sorted(glob.glob(os.path.join(
                args.base, f'Rsweep_W{W0}_Tg{Tg:.2f}_R*_win29_OLDMODEL_backup',
                'structural_trajectory.csv')))
            if not os.path.exists(ol) or not hits:
                rows.append(dict(W0=W0, Tg=Tg, miss=True))
                continue
            cl = hits[0]
            R = re.search(r'_R([0-9.]+)_win29', cl).group(1)

            exo, _, _ = metrics(ol, Tg, cltrim)
            exc, fmax, osc = metrics(cl, Tg, cltrim)
            rows.append(dict(
                W0=W0, Tg=Tg, miss=False, R=R,
                exo=exo, exc=exc, clred=100 * (exo - exc) / exo,
                fmax=fmax, osc=osc,
                exo_rom=float(npz[f'{tag}_cex0']),
                clred_rom=float(npz[f'{tag}_clred']),
                n_extra=len(hits) - 1))

    print(f'CLTRIM = {cltrim:.4f}   window = t <= Tg+0.5   plant-consistent exo\n')
    hdr = (f'{"W0":>3} {"Tg":>5} {"R*":>7} | {"exo_FOM":>8} {"exc_FOM":>8} '
           f'{"CLred_FOM":>10} {"|d|max":>7} {"osc":>4} | {"exo_ROM":>8} '
           f'{"CLred_ROM":>10} | {"gap":>7}')
    print(hdr)
    print('-' * len(hdr))
    gaps = []
    for r in rows:
        if r['miss']:
            print(f'{r["W0"]:>3} {r["Tg"]:>5.2f} {"--":>7} |  (no run)')
            continue
        gap = r['clred_rom'] - r['clred']
        gaps.append((gap, r))
        print(f'{r["W0"]:>3} {r["Tg"]:>5.2f} {r["R"]:>7} | {r["exo"]:>8.4f} '
              f'{r["exc"]:>8.4f} {r["clred"]:>+9.1f}% {r["fmax"]:>6.2f}° '
              f'{r["osc"]:>4} | {r["exo_rom"]:>8.4f} {r["clred_rom"]:>+9.1f}% '
              f'{gap:>+6.1f}')

    if gaps:
        g = np.array([x[0] for x in gaps])
        print(f'\ngap ROM-FOM: mean {g.mean():+.1f} pt   '
              f'min {g.min():+.1f}   max {g.max():+.1f}')
        wins = [r for gg, r in gaps if gg < 0]
        print(f'celle dove il FOM batte il ROM: '
              f'{[(r["W0"], r["Tg"]) for r in wins] or "nessuna"}')
        sat = [r for _, r in gaps if r['fmax'] >= 13.99]
        print(f'celle con flap saturo (>=14 deg): '
              f'{[(r["W0"], r["Tg"]) for r in sat] or "nessuna"}')


if __name__ == '__main__':
    main()
