"""
Envelope sweep driver for the internal-model ablation.

Runs the CS-25.341 grid of chapter 3 -- W0 in {10,20,30} m/s, Tg in
{0.3,0.4,0.5,0.7,1.0,1.2} s -- for both internal models, and selects the
effort weight R per cell by the SAME rule chapter 3 used:

    R* = the R maximising CLred among tunings that pass the stability check;
    if every tuning flags, fall back to the one with the smallest pitch ratio.

Applying that rule to BOTH arms is what makes the comparison fair: the linear
controller gets its own best tuning per cell rather than inheriting the
LDNet's. Without this the linear arm could lose merely because it was run at
someone else's operating point.

Results are written incrementally to a JSON file so a long sweep can be
resumed and inspected while running.

Usage
-----
  CELLS=all ARMS=ldnet,linear python3 -u sweep.py out.json
  CELLS=30:0.4,30:0.3 python3 -u sweep.py probe.json
"""
import os
import sys
import json
import time
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_ablation as RA

W0_LIST = [10.0, 20.0, 30.0]
TG_LIST = [0.3, 0.4, 0.5, 0.7, 1.0, 1.2]
R_LADDER = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def parse_cells():
    spec = os.environ.get('CELLS', 'all')
    if spec == 'all':
        return [(w, t) for w in W0_LIST for t in TG_LIST]
    out = []
    for tok in spec.split(','):
        w, t = tok.split(':')
        out.append((float(w), float(t)))
    return out


def select_R(per_R):
    """chapter-3 rule: max CLred among unflagged; else min pitch ratio."""
    ok = [(R, m) for R, m in per_R.items() if not m['flag']]
    if ok:
        R, m = max(ok, key=lambda kv: kv[1]['clred'])
    else:
        R, m = min(per_R.items(), key=lambda kv: kv[1]['pitch_ratio'])
    return R, m


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'sweep_results.json'
    if not os.path.isabs(out_path):
        out_path = os.path.join(HERE, out_path)
    arms = os.environ.get('ARMS', 'ldnet,linear').split(',')
    ladder = [float(x) for x in os.environ.get(
        'RLADDER', ','.join(str(r) for r in R_LADDER)).split(',')]
    cells = parse_cells()

    results = {}
    if os.path.exists(out_path) and os.environ.get('RESUME', '1') == '1':
        with open(out_path) as f:
            results = json.load(f)
        print(f'[sweep] resuming, {len(results)} cells already done')

    for (W0, Tg) in cells:
        key = f'{W0:g}:{Tg:g}'
        if key in results and all(a in results[key] for a in arms):
            print(f'[sweep] skip {key} (done)')
            continue
        t0 = time.time()
        cell = results.get(key, {})
        # open-loop reference once per cell
        ol = RA.simulate_arm('open', W0, Tg)
        for arm in arms:
            per_R = {}
            for R in ladder:
                r = RA.simulate_arm(arm, W0, Tg, R=R)
                m = RA.metrics(r, ol, Tg)
                per_R[R] = m
                print(f'  [{key}] {arm:6s} R={R:<8g} CLred={m["clred"]:+7.2f}%'
                      f'  flap={m["flap_max"]:5.1f}  pr={m["pitch_ratio"]:.2f}'
                      f'  {m["flag"] or "ok"}', flush=True)
            Rstar, best = select_R(per_R)
            cell[arm] = dict(R_star=Rstar, best=best,
                             per_R={str(k): v for k, v in per_R.items()})
        results[key] = cell
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=1)
        dt = time.time() - t0
        line = f'[sweep] {key} done in {dt:.0f}s |'
        for a in arms:
            line += f' {a}={cell[a]["best"]["clred"]:+.1f}%'
        print(line, flush=True)

    print(f'\n[sweep] wrote {out_path}')


if __name__ == '__main__':
    main()
