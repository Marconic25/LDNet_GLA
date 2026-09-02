### Task 1: Axis A2 — calibration robustness script + cluster wrappers + plot

**Files:**
- Create: `light/noise/noise_calib_combo.py`
- Create: `light/noise/launch_noise_calib_combo.sh`
- Create: `light/noise/status_noise_calib_combo.sh`
- Create: `light/noise/plots_noise_calib_combo.py`

**Interfaces:**
- Consumes: `harness_noise` utilities and `optimal.FusedPreviewSensor/MPCPreviewController` (unchanged).
- Produces: `results/A2_calib.npz` with `point` records keyed `axis='A2'`, `arm in {'bias','gain'}`, `value` (bias as fraction of W0, or gain multiplier), `frac` (0.0 clean / 0.02 noisy), `sigma_del`, `bias_del`. Task 4 fills NOTES.md from these keys.

- [ ] **Step 1: Write `light/noise/noise_calib_combo.py`** with exactly this content:

```python
"""
Calibration robustness of the E2-combo pipeline: additive bias + gain error.

The sensor measures a mis-calibrated field Wt_meas = gain*Wt + bias
(bias = value*W0) with per-shot white noise sigma_fun(j). Inverse-variance
fusion removes VARIANCE, not systematic error: bias and gain pass through
untouched (bias_del = mean(Wc - Wnext) is logged to verify), so this axis
measures the MPC's own calibration tolerance -- the lidar bias/gain spec.

Clean sweep (sigma=1e-9, deterministic, 1 rollout/point):
  anchor bias=0 ; bias value in {+-0.02,+-0.05,+-0.10} ; gain in {0.8,0.9,1.1,1.2}
Noisy spot-checks (sigma=2%*W0 flat, 6 seeds rng 100+seed):
  bias +-0.05 ; gain 1.2

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, home cell W30/Tg0.4, DAMULT=3.
Output: results/A2_calib.npz
--smoke: clean {bias 0, bias +0.05} + noisy {bias +0.05} 2 seeds.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import harness_noise as H
from optimal import FusedPreviewSensor, MPCPreviewController

W0, Tg   = 30.0, 0.4
JMAX, N  = 50, 8
R        = 3e-4
R_DU     = 0.0
LAM      = 0.0
SMOKE    = '--smoke' in sys.argv
NSEED    = 2 if SMOKE else 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'A2_calib.npz')

if SMOKE:
    CLEAN_PTS = [('bias', 0.0), ('bias', 0.05)]
    NOISY_PTS = [('bias', 0.05)]
else:
    CLEAN_PTS = ([('bias', 0.0)]
                 + [('bias', b) for b in (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)]
                 + [('gain', g) for g in (0.8, 0.9, 1.1, 1.2)])
    NOISY_PTS = [('bias', -0.05), ('bias', 0.05), ('gain', 1.2)]


def _delivered(r):
    """(sigma_del, bias_del) of the Wc channel vs the true next-step gust."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw  = t <= (Tg + 0.5)
    err = Wc[mw] - Wnext[mw]
    return float(np.std(err)), float(np.mean(err))


# ---- harness adapter: wc_fun sets sensor.last; compute reads it ----------------
class _ComboCtrl:
    def __init__(self, sensor, mpc):
        self._sensor = sensor
        self._mpc    = mpc
        self._delta_prev = 0.0

    def reset(self):
        self._sensor.reset()
        self._mpc.reset()
        self._delta_prev = 0.0

    def compute(self, state, W_true, Wc):
        # sensor.last set by wc_fun before this call (harness protocol)
        return self._mpc.compute(state, self._sensor.last)


def make_combo(rng, frac, bias=0.0, gain=1.0):
    """Combo whose sensor measures the mis-calibrated field gain*Wt + bias."""
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor    = FusedPreviewSensor(rng, sigma_fun, JMAX, N, lam=LAM)
    mpc       = MPCPreviewController(
        H.aero, U=H.U, dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=H.CLTRIM, N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    cache = {}
    def wc(i, Wt, Nsteps):
        if 'Wm' not in cache:
            cache['Wm'] = gain * Wt + bias      # plant keeps the true Wt
        return sensor.wc_fun(i, cache['Wm'], Nsteps)
    return _ComboCtrl(sensor, mpc), wc


OL   = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# A2_calib | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='A2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]


def run_point(arm, val, frac, nseed):
    bias = val * W0 if arm == 'bias' else 0.0
    gain = val      if arm == 'gain' else 1.0
    ms, sds, bds = [], [], []
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        ctrl, wc = make_combo(rng, frac, bias=bias, gain=gain)
        r = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms.append(H.metrics(r, OL, Tg))
        sd, bd = _delivered(r)
        sds.append(sd); bds.append(bd)
    rec = H.point_record(ms, axis='A2', arm=arm, value=float(val),
                         W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM, R_du=R_DU,
                         frac=frac, sigma_del=float(np.mean(sds)),
                         bias_del=float(np.mean(bds)))
    print(f"  {arm}={val:+.2f} frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms))}  "
          f"sig_del={np.mean(sds):.3g}  bias_del={np.mean(bds):+.3g} m/s", flush=True)
    return rec


for arm, val in CLEAN_PTS:
    recs.append(run_point(arm, val, 0.0, 1))

for arm, val in NOISY_PTS:
    recs.append(run_point(arm, val, 0.02, NSEED))

H.save_records(OUT, recs)
print("# DONE", flush=True)
```

- [ ] **Step 2: Write `light/noise/launch_noise_calib_combo.sh`** (cluster-only wrapper, same shape as `launch_noise_white_combo.sh`):

```bash
#!/bin/bash
# Launch noise_calib_combo.py (cluster only).
#   ./launch_noise_calib_combo.sh smoke
#   ./launch_noise_calib_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_calib_combo.py --smoke" > A2.smoke.log 2>&1 &
    echo "launched noise_calib_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_calib_combo.py" > A2.log 2>&1 &
    echo "launched noise_calib_combo full pid $!"
fi
```

- [ ] **Step 3: Write `light/noise/status_noise_calib_combo.sh`** (same shape as `status_noise_white_combo.sh`):

```bash
#!/bin/bash
cd /work/u10677113/LDNet_GLA/light/noise 2>/dev/null || exit 1
for pair in "A2.smoke:noise_calib_combo.py --smoke" "A2:noise_calib_combo.py"; do
    L=${pair%%:*}; P=${pair##*:}
    LOG="${L}.log"
    [ -f "$LOG" ] || continue
    if grep -q "^# DONE" "$LOG"; then st="DONE"
    elif grep -qE "Traceback|Error|Killed" "$LOG"; then st="ERROR"
    elif pgrep -f "python3 -s -u $P" >/dev/null 2>&1; then st="RUNNING"
    else st="DEAD"; fi
    last=$(grep -E "^  |^#" "$LOG" 2>/dev/null | tail -1)
    echo "$L: $st | $last"
done
```

- [ ] **Step 4: Write `light/noise/plots_noise_calib_combo.py`** (local, post-scp):

```python
"""
Figure: CLred vs calibration error (bias, gain) for the E2-combo.
Reads results/A2_calib.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# our own npz (harness save_records object-array schema) -- trusted local file
recs = list(np.load(os.path.join(DIR, 'A2_calib.npz'), allow_pickle=True)['records'])
pts  = [r for r in recs if r.get('kind') == 'point']

ANCHOR = 80.51

def series(arm, frac):
    rows = sorted([r for r in pts if r['arm'] == arm
                   and abs(float(r['frac']) - frac) < 1e-9],
                  key=lambda r: float(r['value']))
    if arm == 'bias':
        x = [float(r['value']) * 100 for r in rows]          # %*W0
    else:
        x = [float(r['value']) for r in rows]
    return (x, [float(r['mean']) for r in rows], [float(r['lo']) for r in rows],
            [float(r['hi']) for r in rows], [int(r['nflag']) for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, arm, xl in [(axes[0], 'bias', 'bias [% of W0]'),
                    (axes[1], 'gain', 'gain [-]')]:
    x, m, lo, hi, nf = series(arm, 0.0)
    # anchor (bias=0) belongs to both panels
    if arm == 'gain' and 1.0 not in x:
        x = x + [1.0]; m = m + [ANCHOR]; lo = lo + [ANCHOR]
        hi = hi + [ANCHOR]; nf = nf + [0]
        order = np.argsort(x)
        x  = list(np.array(x)[order]);  m  = list(np.array(m)[order])
        lo = list(np.array(lo)[order]); hi = list(np.array(hi)[order])
        nf = list(np.array(nf)[order])
    ax.plot(x, m, 'o-', color='tab:blue', label='clean (deterministic)')
    for xi, mi, f in zip(x, m, nf):
        if f:
            ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
    xn, mn, lon, hin, nfn = series(arm, 0.02)
    if xn:
        yerr = [np.array(mn) - np.array(lon), np.array(hin) - np.array(mn)]
        ax.errorbar(xn, mn, yerr=yerr, fmt='s', color='tab:orange',
                    capsize=3, label='sigma=2% x 6 seeds')
        for xi, mi, f in zip(xn, mn, nfn):
            if f:
                ax.plot(xi, mi, 's', mfc='none', mec='red', ms=12, mew=1.5)
    ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_xlabel(xl); ax.set_ylabel('CLred [%]')
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
axes[0].set_title('A2 -- sensor bias (red ring = flagged)')
axes[1].set_title('A2 -- sensor gain')
fig.suptitle('E2-combo calibration robustness (W30/Tg0.4, DAMULT=3)', y=1.02)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_calib_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')
```

- [ ] **Step 5: Parse-test all four files**

Run from repo root:
```bash
python3 - <<'EOF'
import ast
for f in ('light/noise/noise_calib_combo.py', 'light/noise/plots_noise_calib_combo.py'):
    ast.parse(open(f).read()); print(f, 'parse OK')
EOF
bash -n light/noise/launch_noise_calib_combo.sh && echo launch OK
bash -n light/noise/status_noise_calib_combo.sh && echo status OK
```
Expected: two `parse OK` lines, `launch OK`, `status OK`. (No TF locally — do NOT try to import the script.)

- [ ] **Step 6: Commit (only the four new files)**

```bash
git add light/noise/noise_calib_combo.py light/noise/launch_noise_calib_combo.sh light/noise/status_noise_calib_combo.sh light/noise/plots_noise_calib_combo.py
git commit -m "feat(noise): A2 calibration robustness axis for E2-combo (bias+gain)"
```

---
## Global Constraints

- Do NOT modify `light/optimal.py`, `light/noise/harness_noise.py`, or any existing results (`light/noise/results/*.npz`, `light/results_cs25*/`).
- `PYTHONNOUSERSITE=1` and `DAMULT=3` mandatory on cluster (both already set inside `run_axis.sh` — new scripts run through it).
- Fixed config: W0=30, Tg=0.4 (home cell), Jmax=50, N=8, R=3e-4, R_du=0 (harmful — do not add), lam=0, G=161, seeds rng(100+seed), 6 seeds per noisy point, metrics window t<=Tg+0.5, explosion flags 3x open-loop.
- Anchors (dp45 tree, must reproduce or STOP): combo clean = **+80.51%**; combo flat sigma=2% per-seed clred (rng 100..105) = **[80.3860, 80.5235, 80.5510, 80.4897, 80.5364, 80.5024]** (from `results/W_combo.npz`).
- Local tree has uncommitted user changes (one-step prune + latex): `git add` ONLY the files this plan creates — never `git add -A` / `git add -u`.
- New scripts must import only symbols present in BOTH the local (pruned) and cluster (unpruned) trees: from `optimal` only `FusedPreviewSensor`, `MPCPreviewController`; from `harness_noise` only `rollout, metrics, point_record, seed_stats, fmt_stats, save_records, aero, U, DT, C, CLTRIM, DMAX, DDOT_MAX`. Only NEW files get copied to the cluster.
- Cluster quoting trap: remote commands with `$`/redirects do not survive Git Bash→wsl→ssh; all launch/status logic lives in `.sh` scripts ON the cluster, invoked by path. Cluster: `u10677113@10.78.18.100`, tree `/work/u10677113/LDNet_GLA`.
- Smoke before every full run. Smoke gates are the zero-error anchors only; nonzero-error smoke points are DATA (exploratory), not gates — do not "debug" a surprising bias/shift result.

---
