"""
CS-25.341-style gust study driver — final wnext controller.

3x6 grid (W0 in {10,20,30} x Tg in {0.30..1.20}), R swept per cell with the
scalar rollout of run.py. Per cell, R* = MAX CLred subject to NO explosion
flag (alpha_dot / alpha_ddot / h_ddot < 3x open loop — the physical
instability gate; pitch is reported for transparency). If every R flags,
fall back to min pitch. Same rule as 76/cs25_wnext2.py.

CS-25.341 framing: H = U*Tg/2 (gust gradient distance), k = pi/(U*Tg).

Usage (cluster, all 18 cells in one job; optionally W0=<10|20|30> for one row):
    DAMULT=3 python3 -s -u cs25_study.py

Outputs: ../results_cs25/traces_W{10,20,30}.npz (open + best-optimal traces
per cell, per-R metric tables) + summary table in the log. Plot with
cs25_plots.py.
"""
import os, sys
import numpy as np

# Allow imports from light/ (this file lives in light/tests/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import run as Rn

TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
W0_LIST = [10.0, 20.0, 30.0]
if 'W0' in os.environ:
    W0_LIST = [float(os.environ['W0'])]
R_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
TEND = float(os.environ.get('TEND', '3.0'))
QTY = ['CL', 'de', 'al', 'ad']

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_cs25')
os.makedirs(OUT_DIR, exist_ok=True)

print(f'# cs25_study (wnext+refine, no-flag pick)  W0_LIST={W0_LIST}  '
      f'TG_LIST={TG_LIST}  R_GRID={R_GRID}  TEND={TEND}  '
      f'DAMULT={os.environ.get("DAMULT", "1")}', flush=True)

for W0 in W0_LIST:
    out = {'CLTRIM': Rn.CLTRIM, 'TG_LIST': np.array(TG_LIST), 'W0': W0,
           'R_GRID': np.array(R_GRID)}
    for Tg in TG_LIST:
        tag = f'Tg{Tg:.2f}'
        OL = Rn.simulate('open', W0, Tg, TEND=TEND)
        t = OL['_t']; mw = t <= (Tg + 0.5)
        cex0 = float(np.max(np.abs(OL['CL'][mw] - Rn.CLTRIM)))
        pk0 = float(np.max(np.abs(OL['al'][mw])))
        print(f'  open  W{W0:g}/Tg{Tg:.2f}  cex0={cex0:.4f}', flush=True)

        runs = []; ms = []
        for r in R_GRID:
            OPT = Rn.simulate('optimal', W0, Tg, TEND=TEND, R=float(r))
            m = Rn.metrics(OPT, OL, Tg)
            m['pr'] = m['pitchpk'] / max(pk0, 1e-12)
            runs.append(OPT); ms.append(m)
            print(f'    R={r:g}  CLred={m["clred"]:+.1f}%  pitch={m["pr"]:.2f}  '
                  f'flap={m["flap_max"]:.1f}  {m["flag"]}', flush=True)

        crs = np.array([m['clred'] for m in ms])
        prs = np.array([m['pr'] for m in ms])
        noflag = np.array([m['flag'] == '' for m in ms])
        idx = np.where(noflag)[0]
        jb = int(idx[np.argmax(crs[idx])]) if len(idx) else int(np.argmin(prs))
        OPT = runs[jb]; m = ms[jb]
        print(f'  BEST  W{W0:g}/Tg{Tg:.2f}: R*={R_GRID[jb]:g}  '
              f'CLred={m["clred"]:+.1f}%  pitch={m["pr"]:.2f}  '
              f'flap={m["flap_max"]:.1f}  {m["flag"]}', flush=True)

        out[f'{tag}_t'] = t; out[f'{tag}_Wt'] = OL['_Wt']
        for k in QTY:
            out[f'{tag}_open_{k}'] = OL[k]
            out[f'{tag}_opt_{k}'] = OPT[k]
        out[f'{tag}_cex0'] = cex0
        out[f'{tag}_crs'] = crs; out[f'{tag}_prs'] = prs
        out[f'{tag}_fmax'] = np.array([m['flap_max'] for m in ms])
        out[f'{tag}_flags'] = np.array([m['flag'] for m in ms])
        out[f'{tag}_Rstar'] = float(R_GRID[jb]); out[f'{tag}_jb'] = jb
        out[f'{tag}_clred'] = float(m['clred']); out[f'{tag}_pitch'] = float(m['pr'])

    fn = os.path.join(OUT_DIR, f'traces_W{int(W0)}.npz')
    np.savez_compressed(fn, **out)
    print(f'# saved {fn}', flush=True)
    print(f'# ROW W{int(W0)} DONE', flush=True)

print('# DONE', flush=True)
