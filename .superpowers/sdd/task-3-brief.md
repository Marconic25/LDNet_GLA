## Task 3 â€“ dp45 baseline regression on cluster + create CS-25 combo study

### Part A: Cluster baseline sync and smoke regression

**Context:** After syncing the local dp45 tree to the cluster, re-derive the open/optimal/combo anchors. The cluster's old rk4 anchor was open=0.4600, optimal R=3e-4 +76.58%, R=1e-4 +80.67%. The dp45 values may differ slightly; document and use as new anchors.

- [ ] **Step 1: Sync local light/ to cluster**
```bash
# From WSL: /home/marco/LDNet_OF
scp light/optimal.py light/run.py light/structure.py \
    u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/

scp light/noise/harness_noise.py \
    u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/noise/
```

- [ ] **Step 2: Write cluster smoke script**

Create `light/smoke_dp45_baseline.sh` locally then scp:

```bash
#!/bin/bash
# Smoke regression: open + optimal (R=3e-4, R=1e-4) + combo oracle clean
# at W30/Tg0.4 DAMULT=3, dp45 tree. Run from cluster light/ dir.
cd /work/u10677113/LDNet_GLA/light
APP="apptainer exec --writable-tmpfs --env PYTHONNOUSERSITE=1 --env DAMULT=3 \
  --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"
$APP bash -c "pip install -q scipy h5py matplotlib; python3 -s -u -c \"
import run as R
OL  = R.simulate('open', 30, 0.4)
OPT3 = R.simulate('optimal', 30, 0.4, R=3e-4)
OPT1 = R.simulate('optimal', 30, 0.4, R=1e-4)
COM  = R.simulate('combo',   30, 0.4, R=3e-4, NH=8)
mo3 = R.metrics(OPT3, OL, 0.4); mo1 = R.metrics(OPT1, OL, 0.4)
mc  = R.metrics(COM,  OL, 0.4)
import numpy as np
cex0 = float(np.max(np.abs(OL['CL'] - R.CLTRIM)))
print(f'open cex0     = {cex0:.4f}   (rk4 ref: 0.4600)')
print(f'optimal R=3e-4: {mo3[\"clred\"]:+.2f}%  (rk4 ref: +76.58%)')
print(f'optimal R=1e-4: {mo1[\"clred\"]:+.2f}%  (rk4 ref: +80.67%)')
print(f'combo   R=3e-4: {mc[\"clred\"]:+.2f}%   (rk4 ref: +80.5%)')
\""
```

Scp and run:
```bash
scp light/smoke_dp45_baseline.sh \
    u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/

ssh -n u10677113@10.78.18.100 'chmod +x /work/u10677113/LDNet_GLA/light/smoke_dp45_baseline.sh && nohup /work/u10677113/LDNet_GLA/light/smoke_dp45_baseline.sh > /work/u10677113/LDNet_GLA/light/smoke_dp45.log 2>&1 &'
```

- [ ] **Step 3: Poll and collect dp45 baseline values**

Poll via (run from WSL, no `$` in the command, use the script approach):
```bash
ssh -n u10677113@10.78.18.100 'tail -10 /work/u10677113/LDNet_GLA/light/smoke_dp45.log'
```

Collect the four numbers: cex0, optimal R=3e-4, optimal R=1e-4, combo clean.

- [ ] **Step 4: Fill in dp45 baseline table in NOTES.md**

Update the "TO BE FILLED" section added in Task 0 with the actual dp45 values. If any non-chaotic value (cex0, combo clean) differs >1 pt from rk4, STOP and run systematic-debugging before continuing.

### Part B: CS-25 combo study script

**Files:**
- Create: `light/tests/cs25_combo_study.py`
- Create: `light/tests/launch_cs25_combo.sh`
- Create: `light/tests/status_cs25_combo.sh`

- [ ] **Step 5: Write cs25_combo_study.py**

Create `light/tests/cs25_combo_study.py`:

```python
"""
CS-25.341-style gust study â€” one-step optimal OR E2-combo controller.

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
```

- [ ] **Step 6: Write launch_cs25_combo.sh**

Create `light/tests/launch_cs25_combo.sh`:

```bash
#!/bin/bash
# Launch CS-25 combo + optimal (dp45) rows in parallel (cluster only).
# Usage:
#   ./launch_cs25_combo.sh smoke   -> W30 combo, 1 config, quick smoke
#   ./launch_cs25_combo.sh full    -> 3 combo rows + 3 optimal rows
cd /work/u10677113/LDNet_GLA/light/tests || exit 1
APP="apptainer exec --writable-tmpfs --env PYTHONNOUSERSITE=1 --env DAMULT=3 \
  --env OMP_NUM_THREADS=3 --env TF_NUM_INTRAOP_THREADS=3 --env TF_NUM_INTEROP_THREADS=1 \
  --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif"

run_row() {
    local MODE=$1 W0=$2 LOG=$3
    nohup bash -c "pip install -q scipy h5py matplotlib; $APP bash -c \
        'pip install -q scipy h5py matplotlib; cd /work/u10677113/LDNet_GLA/light/tests && \
         MODE=$MODE W0=$W0 python3 -s -u cs25_combo_study.py'" > "$LOG" 2>&1 &
    echo "launched MODE=$MODE W0=$W0 -> $LOG pid $!"
}

# Wrap in apptainer properly
run_row_apptainer() {
    local MODE=$1 W0=$2 LOG=$3
    nohup $APP bash -c \
        "pip install -q scipy h5py matplotlib; \
         cd /work/u10677113/LDNet_GLA/light/tests && \
         MODE=$MODE W0=$W0 python3 -s -u cs25_combo_study.py" > "$LOG" 2>&1 &
    echo "launched MODE=$MODE W0=$W0 -> $LOG pid $!"
}

if [ "$1" = "smoke" ]; then
    nohup $APP bash -c \
        "pip install -q scipy h5py matplotlib; \
         cd /work/u10677113/LDNet_GLA/light/tests && \
         MODE=combo W0=30 TG_SMOKE=1 python3 -s -u cs25_combo_study.py" \
        > cs25combo.smoke.log 2>&1 &
    echo "smoke pid $!"
else
    for W0 in 10 20 30; do
        run_row_apptainer combo   $W0 "cs25combo_c${W0}.log"
        run_row_apptainer optimal $W0 "cs25combo_o${W0}.log"
    done
fi
```

Note: since this script runs ON the cluster (not via ssh), it can use `$` directly.

- [ ] **Step 7: Write status_cs25_combo.sh**

Create `light/tests/status_cs25_combo.sh`:

```bash
#!/bin/bash
# Status of CS-25 combo jobs (run ON the cluster; called via ssh -n).
cd /work/u10677113/LDNet_GLA/light/tests 2>/dev/null || exit 1
for pair in \
    "cs25combo.smoke:cs25_combo_study.py" \
    "cs25combo_c10:cs25_combo_study.py" \
    "cs25combo_c20:cs25_combo_study.py" \
    "cs25combo_c30:cs25_combo_study.py" \
    "cs25combo_o10:cs25_combo_study.py" \
    "cs25combo_o20:cs25_combo_study.py" \
    "cs25combo_o30:cs25_combo_study.py"; do
  L=${pair%%:*}; P=${pair##*:}
  LOG="${L}.log"
  [ -f "$LOG" ] || continue
  if grep -q "^# DONE" "$LOG"; then st="DONE"
  elif grep -qE "Traceback|Error|Killed" "$LOG"; then st="ERROR"
  elif pgrep -f "python3 -s -u $P" >/dev/null 2>&1; then st="RUNNING"
  else st="DEAD"; fi
  last=$(grep -E "^  BEST|^# ROW|^# DONE|^#" "$LOG" 2>/dev/null | tail -1)
  echo "$L: $st | $last"
done
```

- [ ] **Step 8: Sync and launch CS-25 combo jobs**

Sync to cluster:
```bash
scp light/tests/cs25_combo_study.py \
    light/tests/launch_cs25_combo.sh \
    light/tests/status_cs25_combo.sh \
    u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/tests/

ssh -n u10677113@10.78.18.100 \
    'chmod +x /work/u10677113/LDNet_GLA/light/tests/launch_cs25_combo.sh \
              /work/u10677113/LDNet_GLA/light/tests/status_cs25_combo.sh'
```

First run a smoke (1 row, should finish in ~5 min combo vs 1 min optimal):
```bash
# From WSL:
ssh -n u10677113@10.78.18.100 '/work/u10677113/LDNet_GLA/light/tests/launch_cs25_combo.sh smoke'
# Poll:
ssh -n u10677113@10.78.18.100 '/work/u10677113/LDNet_GLA/light/tests/status_cs25_combo.sh'
```

After smoke passes (cex0 matches dp45 anchor Â±0.001, CLred plausible), launch full:
```bash
ssh -n u10677113@10.78.18.100 '/work/u10677113/LDNet_GLA/light/tests/launch_cs25_combo.sh full'
```

Expected run time: combo rows ~4h each (6 cells Ã— 5 R values Ã— ~8 min), optimal rows ~30 min each. All 6 jobs run in parallel. Total wall time â‰ˆ 4h.

- [ ] **Step 9: Commit new scripts**
```bash
git add light/tests/cs25_combo_study.py light/tests/launch_cs25_combo.sh \
        light/tests/status_cs25_combo.sh light/smoke_dp45_baseline.sh
git commit -m "feat(cs25): MODE-parametrised cs25_combo_study.py + cluster scripts"
```

---

