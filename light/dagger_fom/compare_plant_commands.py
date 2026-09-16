#!/usr/bin/env python3
"""
Fill in the plant x commands table for one cell and report the decomposition.

                    u_ROM                u_FOM
    plant ROM       CLred_ROM            --
    plant FOM       FOM(u_ROM)           CLred_FOM

  gap_plant    = CLred_ROM  - FOM(u_ROM)   same commands, two plants, no feedback
  gap_feedback = FOM(u_ROM) - CLred_FOM    what closing the loop on FOM state does

Metrics follow validate_iteration.py exactly (window t-t0 <= Tg+0.5,
exc = max|CL - CLTRIM|, osc_count = sign changes of diff(delta) in that window)
but take exo from a REAL FOM open-loop CSV instead of the ROM's own open-loop
prediction, and can report against several exo references at once -- the study
has two, measured on different plants (see NOTES.md, "Difetto 2").

Pure numpy: no TensorFlow, no model load.

Usage:
    python3 compare_plant_commands.py --tg 0.70 \
        --run "FOM(u_FOM) baseline=/path/Rsweep_.../structural_trajectory.csv" \
        --run "FOM(u_ROM) replay=/path/Replay_.../structural_trajectory.csv" \
        --exo "legacy win50 dam1.0=/path/OpenLoop_W20_Tg0.70/structural_trajectory.csv" \
        --exo "coerente win29 dam3.0=/path/OpenLoop_..._win29_dam3.0/structural_trajectory.csv" \
        --rom-clred 91.9
"""
import argparse
import csv as _csv

import numpy as np

RHO, U, S = 1.225, 80.0, 0.05
Q = 0.5 * RHO * U ** 2 * S
CLTRIM_DEFAULT = 0.8683425957523628


def load(path):
    with open(path) as f:
        rows = list(_csv.reader(f))
    d = np.array([[float(v) for v in r] for r in rows[1:]])
    return dict(t=d[:, 0], Fy=d[:, 5], delta=d[:, 8])


def metrics(path, tg, cltrim):
    d = load(path)
    t = d['t'] - d['t'][0]
    m = t <= tg + 0.5
    cl = d['Fy'] / Q
    exc = float(np.max(np.abs(cl[m] - cltrim)))
    flap = float(np.max(np.abs(d['delta'][m])))
    dd = np.diff(d['delta'][m])
    sg = np.sign(dd)
    sg = sg[sg != 0]
    osc = int(np.sum(sg[1:] * sg[:-1] < 0))
    return dict(exc=exc, flap=flap, osc=osc, n=int(m.sum()), tmax=float(t[m][-1]))


def split(spec):
    label, _, path = spec.partition('=')
    if not path:
        raise SystemExit(f'ERROR: --run/--exo needs "label=path", got {spec!r}')
    return label.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tg', type=float, required=True)
    ap.add_argument('--cltrim', type=float, default=CLTRIM_DEFAULT)
    ap.add_argument('--run', action='append', required=True,
                    help='"label=path" of a closed-loop / replay trajectory')
    ap.add_argument('--exo', action='append', required=True,
                    help='"label=path" of an open-loop trajectory giving exo')
    ap.add_argument('--rom-clred', type=float, default=None,
                    help='CLred of the ROM at R* for this cell, for the gap split')
    args = ap.parse_args()

    print(f'q = {Q:.4f} N   CLTRIM = {args.cltrim:.10f}   window: t <= {args.tg + 0.5:.2f} s\n')

    exos = {}
    print('=== exo references (open loop) ===')
    for spec in args.exo:
        lab, p = split(spec)
        r = metrics(p, args.tg, args.cltrim)
        exos[lab] = r['exc']
        print(f'  {lab:28s} exo={r["exc"]:.6f}   (n={r["n"]}, t_max={r["tmax"]:.4f}s)')

    print('\n=== runs ===')
    runs = {}
    for spec in args.run:
        lab, p = split(spec)
        r = metrics(p, args.tg, args.cltrim)
        runs[lab] = r
        print(f'  {lab:28s} exc={r["exc"]:.6f}  flap_max={r["flap"]:.2f} deg  '
              f'osc={r["osc"]:2d}   (n={r["n"]}, t_max={r["tmax"]:.4f}s)')

    for elab, exo in exos.items():
        print(f'\n=== CLred con exo "{elab}" (exo={exo:.6f}) ===')
        vals = {}
        for lab, r in runs.items():
            clred = (exo - r['exc']) / exo * 100.0
            vals[lab] = clred
            print(f'  {lab:28s} CLred = {clred:+6.1f}%')
        if args.rom_clred is not None:
            rom = args.rom_clred
            urom = next((v for k, v in vals.items() if 'u_ROM' in k), None)
            ufom = next((v for k, v in vals.items() if 'u_FOM' in k), None)
            print(f'  {"ROM (riferimento)":28s} CLred = {rom:+6.1f}%')
            if urom is not None:
                print(f'    gap_impianto    = {rom - urom:+6.1f} pt '
                      f'(stessi comandi, due impianti)')
            if urom is not None and ufom is not None:
                print(f'    gap_retroazione = {urom - ufom:+6.1f} pt '
                      f'(cosa fa il chiudere l anello sul FOM)')
                print(f'    gap_totale      = {rom - ufom:+6.1f} pt')


if __name__ == '__main__':
    main()
