"""
Report for axis E2-zctrl: A (shared plant z) vs B (true-gust copy, sanity) vs
C (fused-current-gust copy) vs D (prev Ŵ_1 copy). Reads the point records from
results/E2_zctrl_home.npz and results/E2_zctrl_cells.npz and prints, per
cell/noise level, the paired CLred (mean [min,max], flags) for each arm plus
the controller-latent drift and one-step CL prediction error diagnostics.

Usage: python3 zctrl_report.py
"""
import os
import numpy as np
import harness_noise as H

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [os.path.join(HERE, 'results', f'E2_zctrl_{p}.npz')
         for p in ('home', 'cells')]

recs = []
for f in FILES:
    if os.path.exists(f):
        recs += H.load_records(f)
    else:
        print(f"# MISSING {f}")

pts = [r for r in recs if r.get('kind') == 'point']

# order noise levels for a stable table
NOISE_ORDER = {'clean': 0, 'white': 1, 'dlr': 2}
def frac_key(r):
    fr = r.get('frac', None)
    return fr if fr is not None else 9.9

def nseed(r):
    return len(r.get('flags', r.get('clred', [])))

def level_label(r):
    n = r.get('noise')
    if n == 'clean':
        return 'clean'
    if n == 'white':
        return f"white {r['frac']*100:g}%"
    return 'dlr 1-3m/s'

cells = []
for r in pts:
    c = r['cell']
    if c not in cells:
        cells.append(c)

ARMS = ['A', 'B', 'C', 'D']
ARMNAME = {'A': 'shared z_plant', 'B': 'copy true-W (sanity)',
           'C': 'copy fused-cur', 'D': 'copy prev-W1'}

print("=" * 92)
print("E2-zctrl : does the preview-MPC keep performance on its OWN latent?")
print("CLred = gust-load excursion reduction vs open loop (higher=better); "
      "paired seeds")
print("=" * 92)

for cell in cells:
    cps = [r for r in pts if r['cell'] == cell]
    # group by noise level
    levels = []
    seen = set()
    for r in sorted(cps, key=lambda r: (NOISE_ORDER.get(r.get('noise'), 9),
                                        frac_key(r))):
        lab = level_label(r)
        if lab not in seen:
            seen.add(lab); levels.append(lab)
    w0 = next((r['W0'] for r in recs if r.get('kind') == 'open'
               and r.get('cell') == cell), '?')
    print(f"\n### cell {cell}  (W0={w0})")
    print(f"{'level':13s} {'arm':22s} {'CLred mean[min,max]':26s} "
          f"{'flags':6s} {'sig_del':8s} {'drift_max':10s} {'clerr_max':10s}")
    print("-" * 92)
    for lab in levels:
        for arm in ARMS:
            rr = [r for r in cps if level_label(r) == lab and r['arm'] == arm]
            if not rr:
                continue
            r = rr[0]
            cl = f"{r['mean']:+6.1f} [{r['lo']:+5.1f},{r['hi']:+5.1f}]"
            fl = f"{r['nflag']}/{nseed(r)}"
            sd = f"{r.get('sigma_del', 0):.2g}"
            dr = ("  --" if arm == 'A' else f"{r.get('drift_max', 0):.2e}")
            ce = ("  --" if arm == 'A' else f"{r.get('clerr_max', 0):.2e}")
            print(f"{lab:13s} {ARMNAME[arm]:22s} {cl:26s} {fl:6s} "
                  f"{sd:8s} {dr:10s} {ce:10s}")
        # paired A vs C delta
        rA = [r for r in cps if level_label(r) == lab and r['arm'] == 'A']
        rC = [r for r in cps if level_label(r) == lab and r['arm'] == 'C']
        if rA and rC:
            dmean = rC[0]['mean'] - rA[0]['mean']
            print(f"{'':13s} {'-> C - A (pts)':22s} {dmean:+6.1f}")
        print()

# ---- verdict helper --------------------------------------------------------------
print("=" * 92)
print("VERDICT SUMMARY (C = realistic own-latent controller vs A = shared z)")
print("=" * 92)
for cell in cells:
    cps = [r for r in pts if r['cell'] == cell]
    for r in sorted(cps, key=lambda r: (NOISE_ORDER.get(r.get('noise'), 9),
                                         frac_key(r))):
        if r['arm'] != 'C':
            continue
        lab = level_label(r)
        rA = [q for q in cps if level_label(q) == lab and q['arm'] == 'A']
        if not rA:
            continue
        d = r['mean'] - rA[0]['mean']
        tag = ('OK' if abs(d) < 3 and r['nflag'] <= rA[0]['nflag']
               else ('DEGRADE' if d < -3 else 'FLAGS' if r['nflag'] > rA[0]['nflag']
                     else 'OK'))
        print(f"  {cell:7s} {lab:13s}: A={rA[0]['mean']:+6.1f}%({rA[0]['nflag']}/{nseed(rA[0])}) "
              f"C={r['mean']:+6.1f}%({r['nflag']}/{nseed(r)}) dC-A={d:+5.1f}pt "
              f"drift={r.get('drift_max',0):.2e} -> {tag}")
