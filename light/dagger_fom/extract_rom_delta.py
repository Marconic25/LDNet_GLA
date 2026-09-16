#!/usr/bin/env python3
"""
Extract the ROM closed-loop flap history delta(t) for one (W0, Tg) cell from
light/results_cs25_combo/traces_W{W0}.npz, and emit it as a prescribed-schedule
table for cosim_driver_extract.py's --delta-times / --delta-angles.

Purpose: run the FOM with the commands the ROM's MPC decided for itself, i.e.
the FOM(u_ROM) cell of the plant x commands table. The controller is then fully
decoupled from the FOM (its inputs are the ROM-propagated structural state and
the analytic gust), so delta(t) is determined offline and no live co-simulation
of the controller is needed.

Prints the full npz key inventory with shapes first, so the caller can verify
that the trace being exported really is the one at R* for this cell, before any
CFD time is spent on it.

Usage:
    python3 extract_rom_delta.py --w0 20 --tg 0.70 --t-cover 1.5 \
        --npz  .../results_cs25_combo/traces_W20.npz \
        --out  delta_table_W20_Tg0.70.txt
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--w0', type=float, required=True)
    ap.add_argument('--tg', type=float, required=True)
    ap.add_argument('--t-cover', type=float, required=True,
                    help='extend the table (holding the last value) out to at '
                         'least this time [s], so np.interp never extrapolates '
                         'inside the run; pass t_end_rel + window_dt or more')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    tag = f'Tg{args.tg:.2f}'

    print('=== npz inventory ===')
    for k in sorted(d.files):
        v = d[k]
        mark = ' <<<' if k.startswith(tag) else ''
        print(f'  {k:28s} shape={str(v.shape):16s} dtype={v.dtype}{mark}')

    t = np.asarray(d[f'{tag}_t'], float)
    de = np.asarray(d[f'{tag}_opt_de'], float)
    print(f'\n=== cell {tag} ===')
    print(f'  t : n={t.size}  t0={t[0]:.6f}  dt={t[1]-t[0]:.6f}  tmax={t[-1]:.4f}')
    print(f'  delta: shape={de.shape}  min={de.min():+.4f}  max={de.max():+.4f}  '
          f'|max|={np.abs(de).max():.4f}')
    if de.ndim != 1:
        raise SystemExit(f'ERROR: {tag}_opt_de is not 1-D (shape {de.shape}) — '
                         'it is probably an R-sweep, not the trace at R*. '
                         'Inspect the inventory above before proceeding.')
    if de.shape != t.shape:
        raise SystemExit(f'ERROR: delta shape {de.shape} != t shape {t.shape}')

    # Report the R* metadata the npz carries, if any, so the caller can confirm
    # this trace matches the R used by the FOM baseline run.
    for k in (f'{tag}_jb', f'{tag}_clred', f'{tag}_Rgrid', f'{tag}_fmax'):
        if k in d.files:
            v = d[k]
            print(f'  {k} = {v if v.size <= 12 else str(v.shape) + " (array)"}')

    # Hold the last commanded value out to t_cover so the FOM never interpolates
    # past the end of the table mid-run.
    if t[-1] < args.t_cover:
        t = np.append(t, args.t_cover)
        de = np.append(de, de[-1])

    with open(args.out, 'w') as f:
        f.write(' '.join(f'{v:.6f}' for v in t) + '\n')
        f.write(' '.join(f'{v:.6f}' for v in de) + '\n')
    print(f'\nwrote {args.out}: {t.size} knots, covering t=[{t[0]:.4f}, {t[-1]:.4f}]s')


if __name__ == '__main__':
    main()
