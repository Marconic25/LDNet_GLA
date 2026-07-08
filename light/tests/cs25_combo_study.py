"""
CS-25.341-style gust study — one-step optimal OR E2-combo controller.

Parametrised by env:
  MODE=optimal  (default) -> results_cs25/         (dp45 re-run)
  MODE=combo              -> results_cs25_combo/   (new)

Env controls identical to cs25_study.py (W0, TEND, DAMULT).
R_GRID and pick rule (no-flag max CLred, fallback min-pitch) unchanged.
NH=8 and R_du=0 are hard-coded for the combo arm.

Usage:
  DAMULT=3 MODE=combo W0=30 python3 -s -u cs25_combo_study.py
  DAMULT=3 MODE=optimal W0=10 python3 -s -u cs25_combo_study.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import run as Rn

TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
W0_LIST = [10.0, 20.0, 30.0]
if 'W0' in os.environ:
    W0_LIST = [float(os.environ['W0'])]
R_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
TEND   = float(os.environ.get('TEND', '3.0'))
MODE   = os.environ.get('MODE', 'optimal')
NH     = int(os.environ.get('NH', '8'))
QTY    = ['CL', 'de', 'al', 'ad']

_THIS = os.path.dirname(os.path.abspath(__file__))
OUT_DIRS = {
    'optimal': os.path.join(_THIS, '..', 'results_cs25'),
    'combo':   os.path.join(_THIS, '..', 'results_cs25_combo'),
}
if MODE not in OUT_DIRS:
    raise ValueError(f'MODE must be optimal|combo, got {MODE!r}')
OUT_DIR = OUT_DIRS[MODE]
os.makedirs(OUT_DIR, exist_ok=True)

print(f'# cs25_combo_study MODE={MODE} NH={NH}  W0_LIST={W0_LIST}  '
      f'TG_LIST={TG_LIST}  R_GRID={R_GRID}  TEND={TEND}  '
      f'DAMULT={os.environ.get("DAMULT", "1")}', flush=True)

for W0 in W0_LIST:
    out = {'CLTRIM': Rn.CLTRIM, 'TG_LIST': np.array(TG_LIST), 'W0': W0,
           'R_GRID': np.array(R_GRID)}
    for Tg in TG_LIST:
        tag = f'Tg{Tg:.2f}'
        OL = Rn.simulate('open', W0, Tg, TEND=TEND)
        t  = OL['_t']; mw = t <= (Tg + 0.5)
        cex0 = float(np.max(np.abs(OL['CL'][mw] - Rn.CLTRIM)))
        pk0  = float(np.max(np.abs(OL['al'][mw])))
        print(f'  open  W{W0:g}/Tg{Tg:.2f}  cex0={cex0:.4f}', flush=True)

        runs = []; ms = []
        for r in R_GRID:
            kw = dict(TEND=TEND, R=float(r))
            if MODE == 'combo':
                kw['NH'] = NH; kw['R_du'] = 0.0
            RES = Rn.simulate(MODE, W0, Tg, **kw)
            m   = Rn.metrics(RES, OL, Tg)
            m['pr'] = m['pitchpk'] / max(pk0, 1e-12)
            runs.append(RES); ms.append(m)
            print(f'    R={r:g}  CLred={m["clred"]:+.1f}%  pitch={m["pr"]:.2f}  '
                  f'flap={m["flap_max"]:.1f}  {m["flag"]}', flush=True)

        crs    = np.array([m['clred'] for m in ms])
        prs    = np.array([m['pr']    for m in ms])
        noflag = np.array([m['flag'] == '' for m in ms])
        idx    = np.where(noflag)[0]
        jb     = int(idx[np.argmax(crs[idx])]) if len(idx) else int(np.argmin(prs))
        RES_B  = runs[jb]; m = ms[jb]
        print(f'  BEST  W{W0:g}/Tg{Tg:.2f}: R*={R_GRID[jb]:g}  '
              f'CLred={m["clred"]:+.1f}%  pitch={m["pr"]:.2f}  '
              f'flap={m["flap_max"]:.1f}  {m["flag"]}', flush=True)

        out[f'{tag}_t']      = t;   out[f'{tag}_Wt']   = OL['_Wt']
        for k in QTY:
            out[f'{tag}_open_{k}'] = OL[k]
            out[f'{tag}_opt_{k}']  = RES_B[k]
        out[f'{tag}_cex0']   = cex0
        out[f'{tag}_crs']    = crs; out[f'{tag}_prs']   = prs
        out[f'{tag}_fmax']   = np.array([m['flap_max'] for m in ms])
        out[f'{tag}_flags']  = np.array([m['flag'] for m in ms])
        out[f'{tag}_Rstar']  = float(R_GRID[jb]); out[f'{tag}_jb'] = jb
        out[f'{tag}_clred']  = float(m['clred']); out[f'{tag}_pitch'] = float(m['pr'])

    fn = os.path.join(OUT_DIR, f'traces_W{int(W0)}.npz')
    np.savez_compressed(fn, **out)
    print(f'# saved {fn}', flush=True)
    print(f'# ROW W{int(W0)} DONE', flush=True)

print('# DONE', flush=True)
