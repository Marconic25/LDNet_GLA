#!/usr/bin/env python3
"""
Evaluate ONE candidate LDNet model's own pure open-loop (delta=0) prediction
across the full 3x4 grid, and compare against the real FOM open-loop
excursion already measured (compare_open_loop_grid.py). No controller
involved at any point -- this is the cleanest possible test of whether a
retraining attempt actually fixed the aerodynamic prediction, decoupled from
every closed-loop confound (R1 flap discontinuity, R2 explicit coupling
splitting, R3/R4 filtering asymmetry) that doomed teacher-forcing as a proxy
in this study.

Must run inside the TF container: imports light/run.py, which loads an
LDNetAero model at MODULE IMPORT TIME via the MD_OVERRIDE env var. So this
script sets MD_OVERRIDE from --model BEFORE importing run, and evaluates
exactly ONE model per process invocation -- run it twice (once per model) to
compare two candidates, rather than trying to swap models within one process.

Usage (inside tensorflow_gpu.sif, light/ on sys.path):
    python3 verify_openloop_model.py --model /path/to/candidate/latent_10 \
        --fom-exo "10,0.30=0.335126" "10,0.40=0.303847" ...
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='candidate model dir (config.json + weights), sets MD_OVERRIDE')
    ap.add_argument('--fom-exo', nargs='+', required=True,
                    help='"W0,Tg=exo_FOM_value" pairs, the real FOM open-loop '
                         'excursions already measured for each grid cell')
    ap.add_argument('--held-out', nargs='*', default=[],
                    help='"W0,Tg" pairs to flag as held-out (never seen in this '
                         'model\'s training set) in the report')
    ap.add_argument('--damult', type=float, default=3.0,
                    help='structure.D_ALPHA multiplier, read by light/run.py at '
                         'import time. MUST match the value the reference exo_ROM '
                         'was generated with (cs25_combo_study.py / summary.md '
                         'header: DAMULT=3) or the two are different plants -- '
                         'default 1.0 in light/run.py would silently compare '
                         'against an under-damped ROM.')
    args = ap.parse_args()

    os.environ['MD_OVERRIDE'] = args.model
    os.environ['DAMULT'] = str(args.damult)
    _THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_THIS, '..'))  # light/
    import numpy as np
    import run as Rn  # noqa: E402  (import AFTER MD_OVERRIDE is set)

    fom = {}
    for spec in args.fom_exo:
        cell, _, val = spec.partition('=')
        w0s, _, tgs = cell.partition(',')
        fom[(float(w0s), float(tgs))] = float(val)

    held = set()
    for spec in args.held_out:
        w0s, _, tgs = spec.partition(',')
        held.add((float(w0s), float(tgs)))

    print(f'model = {args.model}')
    print(f'DAMULT = {args.damult}')
    print(f'CLTRIM (this model) = {Rn.CLTRIM:.10f}\n')
    print(f'{"cell":>14s}  {"exo_model":>10s}  {"exo_FOM":>10s}  {"gap%":>7s}  held-out')

    rows = []
    for (w0, tg), exo_f in sorted(fom.items()):
        OL = Rn.simulate('open', w0, tg, TEND=3.0)
        t = OL['_t']
        mw = Rn._gust_window(t, tg)
        exo_m = float(np.max(np.abs(OL['CL'][mw] - Rn.CLTRIM)))
        gap = (exo_f - exo_m) / exo_f * 100.0
        ho = (w0, tg) in held
        rows.append((w0, tg, exo_m, exo_f, gap, ho))
        print(f'  W{w0:.0f}/Tg{tg:.2f}    {exo_m:10.6f}  {exo_f:10.6f}  '
              f'{gap:+6.1f}%  {"HELD-OUT" if ho else ""}')

    if rows:
        all_gaps = [r[4] for r in rows]
        ho_gaps = [r[4] for r in rows if r[5]]
        print(f'\nmean gap% (tutte le celle)  = {np.mean(all_gaps):+.1f}%')
        if ho_gaps:
            print(f'mean gap% (solo held-out)   = {np.mean(ho_gaps):+.1f}%')


if __name__ == '__main__':
    main()
