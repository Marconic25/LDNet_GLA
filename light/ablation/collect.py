"""
Merge the per-cell shards into one table and report the ablation verdict.

Produces:
  * a per-cell comparison at each arm's OWN best tuning R* (the chapter-3
    selection rule applied independently to both arms, so neither is judged
    at the other's operating point);
  * the envelope map -- where the linear internal model is adequate and where
    it is not -- which is what the thesis paragraph asks to locate;
  * a LaTeX table body ready to paste into chapter 3.

The LDNet column is cross-checked against the published chapter-3 numbers
(results_cs25_combo/summary.md) so any drift in the harness is caught rather
than silently reported.
"""
import os
import sys
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Published chapter-3 CLred, for regression-checking the ldnet arm.
CH3 = {
    '10:0.3': 81.2, '10:0.4': 89.1, '10:0.5': 91.5,
    '10:0.7': 93.5, '10:1': 95.2, '10:1.2': 93.4,
    '20:0.3': 86.0, '20:0.4': 87.8, '20:0.5': 88.8,
    '20:0.7': 91.9, '20:1': 81.2, '20:1.2': 58.2,
    '30:0.3': 39.4, '30:0.4': 80.5, '30:0.5': 81.0,
    '30:0.7': 91.8, '30:1': 57.3, '30:1.2': 41.7,
}
K_ORDER = [f'{w:g}:{t:g}' for w in (10, 20, 30)
           for t in (0.3, 0.4, 0.5, 0.7, 1.0, 1.2)]


def load():
    res = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'shards', 'cell_*.json'))):
        with open(p) as f:
            res.update(json.load(f))
    return res


def reduced_freq(Tg, U=80.0, c=1.0):
    return np.pi * c / (U * Tg)


def main():
    res = load()
    if not res:
        print('no shards yet')
        return

    print(f'{"cell":>10s}{"k":>7s}{"LDNet":>9s}{"ch3":>7s}{"lin":>9s}'
          f'{"gap":>8s}{"R*ld":>8s}{"R*lin":>8s}{"flap_l":>8s}'
          f'{"pr_l":>7s}{"flag_l":>8s}')
    rows = []
    for k in K_ORDER:
        if k not in res or 'ldnet' not in res[k] or 'linear' not in res[k]:
            continue
        ld = res[k]['ldnet']; li = res[k]['linear']
        bl, bi = ld['best'], li['best']
        W0, Tg = [float(x) for x in k.split(':')]
        kk = reduced_freq(Tg)
        ch3 = CH3.get(k, float('nan'))
        gap = bl['clred'] - bi['clred']
        rows.append((k, W0, Tg, kk, bl, bi, gap))
        print(f'{k:>10s}{kk:>7.3f}{bl["clred"]:>9.1f}{ch3:>7.1f}'
              f'{bi["clred"]:>9.1f}{gap:>8.1f}'
              f'{ld["R_star"]:>8.0e}{li["R_star"]:>8.0e}'
              f'{bi["flap_max"]:>8.1f}{bi["pitch_ratio"]:>7.2f}'
              f'{(bi["flag"] or "ok"):>8s}')

    if not rows:
        print('\n(no complete cells yet)')
        return

    # --- regression check against chapter 3 ---------------------------------
    devs = [abs(r[4]['clred'] - CH3[r[0]]) for r in rows if r[0] in CH3]
    print(f'\n[check] LDNet arm vs published chapter 3: '
          f'max |dev| = {max(devs):.2f} pp, mean {np.mean(devs):.2f} pp')
    if max(devs) > 1.5:
        print('[check] WARNING: harness deviates from the published table')

    # --- verdict ------------------------------------------------------------
    gaps = np.array([r[6] for r in rows])
    lin_cl = np.array([r[5]['clred'] for r in rows])
    ld_cl = np.array([r[4]['clred'] for r in rows])
    print(f'\n=== verdict over {len(rows)} cells ===')
    print(f'  LDNet mean CLred  {ld_cl.mean():+.1f}%   '
          f'range [{ld_cl.min():+.1f}, {ld_cl.max():+.1f}]')
    print(f'  linear mean CLred {lin_cl.mean():+.1f}%   '
          f'range [{lin_cl.min():+.1f}, {lin_cl.max():+.1f}]')
    print(f'  LDNet better in {int((gaps > 0).sum())}/{len(rows)} cells;'
          f'  mean gap {gaps.mean():+.1f} pp, max {gaps.max():+.1f} pp')
    nflag = sum(1 for r in rows if r[5]['flag'])
    print(f'  linear arm violates the stability check in {nflag}/{len(rows)} cells')
    nfl_ld = sum(1 for r in rows if r[4]['flag'])
    print(f'  LDNet arm violates it in {nfl_ld}/{len(rows)} cells')

    # --- where is the linear model adequate? --------------------------------
    print(f'\n=== envelope map (gap = LDNet - linear, pp) ===')
    print(f'{"W0\\Tg":>8s}' + ''.join(f'{t:>8g}' for t in (0.3, 0.4, 0.5, 0.7, 1.0, 1.2)))
    for W0 in (10, 20, 30):
        line = f'{W0:>8g}'
        for Tg in (0.3, 0.4, 0.5, 0.7, 1.0, 1.2):
            k = f'{W0:g}:{Tg:g}'
            m = [r for r in rows if r[0] == k]
            line += f'{m[0][6]:>8.1f}' if m else f'{"-":>8s}'
        print(line)

    # --- LaTeX body ---------------------------------------------------------
    tex = []
    for (k, W0, Tg, kk, bl, bi, gap) in rows:
        tex.append(
            f'    {W0:g} & {Tg:.2f} & {kk:.3f} & {bl["clred"]:.1f} & '
            f'{bi["clred"]:.1f} & {gap:+.1f} & {bi["flap_max"]:.1f} & '
            f'{bi["pitch_ratio"]:.2f} \\\\')
    out = os.path.join(HERE, 'table_body.tex')
    with open(out, 'w') as f:
        f.write('\n    \\hline\n'.join(tex) + '\n')
    print(f'\n[collect] LaTeX body -> {out}')

    with open(os.path.join(HERE, 'merged.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
