"""
Compare the linear-baseline variants and report the STRONGEST one.

The ablation must be reported against the best linear model that could
reasonably be built, not the first one fitted. The closed-loop failure traces
to a flap-gain error (constant dC_L/ddelta vs the plant's state-dependent
one), so variants with corrected gains are the fair competitors:

  linear_coeffs          CL_d = 0.02531 /deg  (global least squares)
  linear_coeffs_ctrlgain CL_d = 0.03830 /deg  (matched to plant near trim)
  linear_coeffs_secant   CL_d = 0.02246 /deg  (secant over +-14 deg)
  linear_coeffs_lsqflap  CL_d = 0.02507 /deg  (refit on |delta|>2 deg only)

Whichever wins is the number that belongs in the thesis.
"""
import os
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL = {
    'variant_linear_coeffs': 'global lsq (0.0253/deg)',
    'variant_linear_coeffs_ctrlgain': 'plant-matched (0.0383/deg)',
    'variant_linear_coeffs_secant': 'secant +-14deg (0.0225/deg)',
    'variant_linear_coeffs_lsqflap': '|d|>2deg refit (0.0251/deg)',
}


def main():
    files = sorted(glob.glob(os.path.join(HERE, 'variant_*.json')))
    if not files:
        print('no variant results yet')
        return

    cells = set()
    data = {}
    for p in files:
        tag = os.path.basename(p)[:-5]
        with open(p) as f:
            d = json.load(f)
        data[tag] = d
        cells |= set(d.keys())

    cells = sorted(cells)
    print(f'{"variant":>32s}' + ''.join(f'{c:>12s}' for c in cells) + f'{"best":>9s}')
    best_overall = None
    for tag, d in data.items():
        row = f'{LABEL.get(tag, tag):>32s}'
        vals, complete = [], True
        for c in cells:
            if c in d and 'linear' in d[c]:
                v = d[c]['linear']['best']['clred']
                fl = d[c]['linear']['best']['flag']
                vals.append(v)
                row += f'{v:>10.1f}{"*" if fl else " ":>2s}'
            else:
                complete = False
                row += f'{"-":>12s}'
        m = np.mean(vals) if vals else float('nan')
        row += f'{m:>9.1f}'
        print(row)
        # Only a variant evaluated on the SAME cells can be compared on the
        # mean; a partial row would otherwise win by having skipped the hard
        # cell rather than by performing better.
        if vals and complete and (best_overall is None or m > best_overall[1]):
            best_overall = (tag, m)

    print('\n  (* = violates the stability check at its selected tuning)')
    if best_overall:
        print(f'\n  strongest linear variant: {LABEL.get(best_overall[0], best_overall[0])}'
              f'  mean CLred {best_overall[1]:+.1f}%')
        print('  -> this is the number to report against the LDNet')


if __name__ == '__main__':
    main()
