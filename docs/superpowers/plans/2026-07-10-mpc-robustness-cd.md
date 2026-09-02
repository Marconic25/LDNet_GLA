# MPC-Combo Robustness Axes (C2 Spatial Jitter + D2 Model Mismatch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the E2-combo robustness study with the two deferred axes: per-shot spatial jitter (the 1D surrogate of lidar coherence loss / range-gate error) and structural model mismatch (plant parameters differ from the MPC's internal model). Combo-only, no comparison arms.

**Architecture:** Same pattern as A2/B2 (plan 2026-07-09-mpc-robustness.md, commits beb29791/011c4f38): two axis scripts in `light/noise/` + launch/status cluster wrappers + local plot scripts. C2 uses a `JitterSensor` subclass (per-shot node error, bit-exact to the parent at k=0); D2 uses a controller adapter that toggles `structure` module globals around `mpc.compute` (nominal for the controller's horizon, perturbed for the plant), with the open-loop reference recomputed per perturbed plant.

**Tech Stack:** Python 3 / numpy / TF (LDNet via harness), Apptainer on cluster, matplotlib locally.

## Global Constraints

- Do NOT modify `light/optimal.py`, `light/noise/harness_noise.py`, `light/structure.py`, or any existing results (`light/noise/results/*.npz`, `light/results_cs25*/`).
- Fixed config: W0=30, Tg=0.4 (home cell), Jmax=50, N=8, R=3e-4, R_du=0, lam=0, G=161, seeds rng(100+seed), 6 seeds per stochastic point, metrics window t<=Tg+0.5, explosion flags 3x open-loop, DAMULT=3 (set by run_axis.sh).
- Anchor (dp45 tree, must reproduce or STOP): combo clean = **+80.51%** (zero-error point of each sweep).
- Local tree has uncommitted user changes: `git add` ONLY the files this plan creates — never `git add -A` / `git add -u` / `git add .`.
- New scripts import only: from `optimal` — `FusedPreviewSensor`, `MPCPreviewController`; from `harness_noise` — `rollout, metrics, point_record, seed_stats, fmt_stats, save_records, aero, U, DT, C, CLTRIM, DMAX, DDOT_MAX`; plus `import structure` (D2 only, for the toggle — read/setattr on D_ALPHA/K_ALPHA only, at runtime, never editing the file).
- Cluster quoting trap: launch/status logic lives in `.sh` scripts ON the cluster invoked by path (no `$`/redirects through ssh). Cluster: `u10677113@10.78.18.100`, tree `/work/u10677113/LDNet_GLA`.
- Smoke before full. Smoke gates are the zero-error anchors only; nonzero-error smoke rows are DATA.
- D2 metrics rule: each perturbed plant is compared against ITS OWN open-loop rollout (same perturbed globals), never against the nominal open loop.

---

### Task 1: Axis C2 — spatial per-shot jitter script + cluster wrappers + plot

**Files:**
- Create: `light/noise/noise_jitter_combo.py`
- Create: `light/noise/launch_noise_jitter_combo.sh`
- Create: `light/noise/status_noise_jitter_combo.sh`
- Create: `light/noise/plots_noise_jitter_combo.py`

**Interfaces:**
- Consumes: `harness_noise` utilities and `optimal.FusedPreviewSensor/MPCPreviewController` (unchanged).
- Produces: `results/C2_jitter.npz` with `point` records keyed `axis='C2'`, `arm='jitter'`, `value` (k, max node offset), `frac` (0.0 jitter-only / 0.02 compound), `sigma_del`, `bias_del`.

- [ ] **Step 1: Write `light/noise/noise_jitter_combo.py`** with exactly this content:

```python
"""
Spatial per-shot jitter robustness of the E2-combo pipeline.

Each individual shot samples the gust at the WRONG node: the sensor believes
it measured node i+j but the returned value is W at node i+j+eta, with
eta ~ U{-k..+k} drawn independently per shot (range-gate error / probe-volume
averaging / frozen-field violation -- the 1D surrogate of lidar coherence
loss, Schlipf/Guo WES 2023). Inverse-variance fusion cannot cancel it: node m
averages samples of its neighbours, so the fused profile converges to W
convolved with the jitter distribution -- a spatial low-pass that degrades
the PHASE content (gradients, zero crossings) the MPC lives on.

At the node spacing U*dt = 0.16 m, k in {1,2,5} = 0.16/0.32/0.80 m of range
error. k=0 skips the eta draws entirely and reproduces FusedPreviewSensor
bit-exactly (anchor gate +80.51%).

Points: anchor k=0 clean (deterministic); jitter-only k in {1,2,5}
(sigma=1e-9, 6 seeds -- eta is the only randomness); compound k=2 with
sigma=2%*W0 (6 seeds).

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, home cell W30/Tg0.4, DAMULT=3.
Output: results/C2_jitter.npz
--smoke: k=0 clean + k=2 jitter-only 2 seeds.
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'C2_jitter.npz')

if SMOKE:
    PTS = [(0, 0.0, 1), (2, 0.0, NSEED)]          # (k, frac, nseed)
else:
    PTS = ([(0, 0.0, 1)]
           + [(k, 0.0, NSEED) for k in (1, 2, 5)]
           + [(2, 0.02, NSEED)])


def _delivered(r):
    """(sigma_del, bias_del) of the Wc channel vs the true next-step gust."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw  = t <= (Tg + 0.5)
    err = Wc[mw] - Wnext[mw]
    return float(np.std(err)), float(np.mean(err))


class JitterSensor(FusedPreviewSensor):
    """
    FusedPreviewSensor whose every shot samples the field at a jittered node.

    The value registered at node i+j is W(clip(i+j+eta)) + noise with
    eta ~ U{-k..+k} per shot; the registration index is unchanged (the sensor
    does not know it sampled the wrong place). wc_fun body duplicated from
    optimal.FusedPreviewSensor (kept in sync by eye) with the sampled index
    perturbed. k=0 draws no eta and reproduces the parent bit-exactly (same
    rng stream).
    """

    def __init__(self, rng, sigma_fun, Jmax, N, lam=0.0, k=0):
        super().__init__(rng, sigma_fun, Jmax, N, lam=lam)
        self.k = int(k)

    def _eta(self, size):
        if self.k == 0:
            return np.zeros(size, dtype=int)
        return self.rng.integers(-self.k, self.k + 1, size=size)

    def wc_fun(self, i, Wt, Nsteps):
        js, sigs, inv2 = self.js, self.sigs, self.inv2
        if self.num is None:
            self.num = np.zeros(Nsteps)
            self.den = np.zeros(Nsteps)
            for ii in range(-(self.Jmax - 1), 0):
                mm = ii + js
                keep = mm >= 0
                mk = np.minimum(mm[keep], Nsteps - 1)
                mt = np.clip(mm[keep] + self._eta(int(keep.sum())), 0, Nsteps - 1)
                yk = Wt[mt] + self.rng.normal(0.0, sigs[keep])
                np.add.at(self.num, mk, yk * inv2[keep])
                np.add.at(self.den, mk, inv2[keep])

        m_reg  = np.minimum(i + js, Nsteps - 1)
        m_true = np.clip(i + js + self._eta(js.size), 0, Nsteps - 1)
        y = Wt[m_true] + self.rng.normal(0.0, sigs)
        np.add.at(self.num, m_reg, y * inv2)
        np.add.at(self.den, m_reg, inv2)

        lo = min(i + 1, Nsteps - 1)
        hi = min(i + self.Jmax, Nsteps - 1)
        w = self.den[lo:hi + 1].copy()
        n = w.size
        ybar = np.zeros(n)
        good = w > 0.0
        ybar[good] = self.num[lo:hi + 1][good] / w[good]

        if self.lam == 0.0 or n < 3:
            u = ybar
        else:
            lam_eff = self.lam * float(np.mean(w))
            D = (np.eye(n - 2, n, 0) - 2.0 * np.eye(n - 2, n, 1)
                 + np.eye(n - 2, n, 2))
            A = np.diag(w) + lam_eff * (D.T @ D)
            u = np.linalg.solve(A, w * ybar)

        idx = np.minimum(np.arange(self.N), n - 1)
        self.last = np.maximum(0.0, u[idx])
        return float(self.last[0])


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


def make_combo(rng, frac, k):
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor = JitterSensor(rng, sigma_fun, JMAX, N, lam=LAM, k=k)
    mpc    = MPCPreviewController(
        H.aero, U=H.U, dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=H.CLTRIM, N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    return _ComboCtrl(sensor, mpc), sensor.wc_fun


OL   = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# C2_jitter | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='C2', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]

for k, frac, nseed in PTS:
    ms, sds, bds = [], [], []
    for seed in range(nseed):
        rng = np.random.default_rng(100 + seed)
        ctrl, wc = make_combo(rng, frac, k)
        r = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms.append(H.metrics(r, OL, Tg))
        sd, bd = _delivered(r)
        sds.append(sd); bds.append(bd)
    rec = H.point_record(ms, axis='C2', arm='jitter', value=int(k),
                         W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM, R_du=R_DU,
                         frac=frac, sigma_del=float(np.mean(sds)),
                         bias_del=float(np.mean(bds)))
    recs.append(rec)
    print(f"  jitter k={k} frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms))}  "
          f"sig_del={np.mean(sds):.3g}  bias_del={np.mean(bds):+.3g} m/s", flush=True)

H.save_records(OUT, recs)
print("# DONE", flush=True)
```

- [ ] **Step 2: Write `light/noise/launch_noise_jitter_combo.sh`**:

```bash
#!/bin/bash
# Launch noise_jitter_combo.py (cluster only).
#   ./launch_noise_jitter_combo.sh smoke
#   ./launch_noise_jitter_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_jitter_combo.py --smoke" > C2.smoke.log 2>&1 &
    echo "launched noise_jitter_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_jitter_combo.py" > C2.log 2>&1 &
    echo "launched noise_jitter_combo full pid $!"
fi
```

- [ ] **Step 3: Write `light/noise/status_noise_jitter_combo.sh`**:

```bash
#!/bin/bash
cd /work/u10677113/LDNet_GLA/light/noise 2>/dev/null || exit 1
for pair in "C2.smoke:noise_jitter_combo.py --smoke" "C2:noise_jitter_combo.py"; do
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

- [ ] **Step 4: Write `light/noise/plots_noise_jitter_combo.py`**:

```python
"""
Figure: CLred vs per-shot spatial jitter for the E2-combo.
Reads results/C2_jitter.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# our own npz (harness save_records object-array schema) -- trusted local file
recs = list(np.load(os.path.join(DIR, 'C2_jitter.npz'), allow_pickle=True)['records'])
pts  = [r for r in recs if r.get('kind') == 'point']

ANCHOR = 80.51
DX_M   = 0.16   # node spacing U*dt [m]

def series(frac):
    rows = sorted([r for r in pts if abs(float(r['frac']) - frac) < 1e-9],
                  key=lambda r: float(r['value']))
    return ([float(r['value']) for r in rows],
            [float(r['mean']) for r in rows], [float(r['lo']) for r in rows],
            [float(r['hi']) for r in rows], [int(r['nflag']) for r in rows],
            [int(r['n']) if 'n' in r else len(r['clred']) for r in rows])

fig, ax = plt.subplots(figsize=(7, 4.2))
x, m, lo, hi, nf, nn = series(0.0)
yerr = [np.array(m) - np.array(lo), np.array(hi) - np.array(m)]
ax.errorbar(x, m, yerr=yerr, fmt='o-', color='tab:blue', capsize=3,
            label='jitter only (6 seeds; k=0 deterministic)')
for xi, mi, f in zip(x, m, nf):
    if f:
        ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
xn, mn, lon, hin, nfn, _ = series(0.02)
if xn:
    yerr = [np.array(mn) - np.array(lon), np.array(hin) - np.array(mn)]
    ax.errorbar(xn, mn, yerr=yerr, fmt='s', color='tab:orange', capsize=3,
                label='jitter + sigma=2% x 6 seeds')
    for xi, mi, f in zip(xn, mn, nfn):
        if f:
            ax.plot(xi, mi, 's', mfc='none', mec='red', ms=12, mew=1.5)
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ticks = sorted(set(x) | set(xn))
ax.set_xticks(ticks)
ax.set_xticklabels([f'{int(t)}\n({t*DX_M:.2f} m)' for t in ticks])
ax.set_xlabel('per-shot node jitter k  (range error)')
ax.set_ylabel('CLred [%]')
ax.set_title('C2 -- spatial per-shot jitter (red ring = flagged)\n'
             'E2-combo, W30/Tg0.4, DAMULT=3')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_jitter_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')
```

- [ ] **Step 5: Parse-test all four files**

```bash
python3 - <<'EOF'
import ast
for f in ('light/noise/noise_jitter_combo.py', 'light/noise/plots_noise_jitter_combo.py'):
    ast.parse(open(f).read()); print(f, 'parse OK')
EOF
bash -n light/noise/launch_noise_jitter_combo.sh && echo launch OK
bash -n light/noise/status_noise_jitter_combo.sh && echo status OK
```
Expected: two `parse OK`, `launch OK`, `status OK`. (No TF locally — do NOT import the script.)

- [ ] **Step 6: Commit (only the four new files)**

```bash
git add light/noise/noise_jitter_combo.py light/noise/launch_noise_jitter_combo.sh light/noise/status_noise_jitter_combo.sh light/noise/plots_noise_jitter_combo.py
git commit -m "feat(noise): C2 spatial per-shot jitter robustness axis for E2-combo"
```

---

### Task 2: Axis D2 — structural model mismatch script + cluster wrappers + plot

**Files:**
- Create: `light/noise/noise_mismatch_combo.py`
- Create: `light/noise/launch_noise_mismatch_combo.sh`
- Create: `light/noise/status_noise_mismatch_combo.sh`
- Create: `light/noise/plots_noise_mismatch_combo.py`

**Interfaces:**
- Consumes: `harness_noise` utilities, `optimal` classes, `structure` module globals (D_ALPHA, K_ALPHA — runtime setattr only).
- Produces: `results/D2_mismatch.npz` with `point` records keyed `axis='D2'`, `arm in {'dalpha','kalpha','uinf','cltrim'}`, `value` (multiplier, or U in m/s for 'uinf'), `frac=0.0`, `cex0` (open-loop excursion of the plant actually used), `sigma_del`, `bias_del`.

- [ ] **Step 1: Write `light/noise/noise_mismatch_combo.py`** with exactly this content:

```python
"""
Structural model-mismatch robustness of the E2-combo pipeline.

All previous axes corrupt what the controller SEES; here plant and internal
model stop being twins: the PLANT flies with perturbed parameters while the
MPC predicts its horizon with the nominal ones (Forte/NASA 2023: 2.5%
disturbance-frequency mismatch collapsed their GLA 69->39%; Fournier 2022
makes the synthesis robust to model uncertainty by construction).

Arms (deterministic, oracle-clean preview -- the axis isolates model error):
  dalpha  plant D_ALPHA x {0.67,0.83,1.17,1.33}  (plant DAMULT ~ 2,2.5,3.5,4
          vs the controller's 3; DAMULT=3 is applied at import by the
          harness, multipliers are relative to that nominal)
  kalpha  plant K_ALPHA x {0.90,0.95,1.05,1.10}  (pitch stiffness -> natural
          frequency, the Forte analogue)
  uinf    controller U in {76,84} m/s vs plant 80 (airspeed estimate error:
          the MPC's q AND its aero evaluations use the believed U)
  cltrim  controller C_L_trim x {0.95,1.05} (trim estimate error)
  anchor  all nominal (gate +80.51%)

Mechanics: structure.D_ALPHA / structure.K_ALPHA are module globals read at
call time by BOTH the plant step (structure.rhs via step_dp45) and the MPC
horizon (optimal.dp45_batch). The adapter sets them to NOMINAL for the
duration of mpc.compute and restores the PERTURBED values before returning,
so the plant integrates with the perturbed structure while the controller
predicts with the nominal one. Each perturbed plant is compared against ITS
OWN open-loop rollout (cex0 logged per point). Declared limit: the LDNet
aero model is shared plant/controller -- this axis tests STRUCTURAL
mismatch only. X0 is the nominal equilibrium; the residual initial
transient under perturbed stiffness is negligible vs the gust response and
cancels in the relative metric.

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, home cell W30/Tg0.4, DAMULT=3.
Output: results/D2_mismatch.npz
--smoke: anchor + dalpha x1.33.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import harness_noise as H
import structure as S
from optimal import FusedPreviewSensor, MPCPreviewController

W0, Tg   = 30.0, 0.4
JMAX, N  = 50, 8
R        = 3e-4
R_DU     = 0.0
LAM      = 0.0
SMOKE    = '--smoke' in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'D2_mismatch.npz')

DA_NOM = float(S.D_ALPHA)      # after the harness applied DAMULT=3
KA_NOM = float(S.K_ALPHA)

if SMOKE:
    PTS = [('anchor', 1.0), ('dalpha', 1.33)]
else:
    PTS = ([('anchor', 1.0)]
           + [('dalpha', m) for m in (0.67, 0.83, 1.17, 1.33)]
           + [('kalpha', m) for m in (0.90, 0.95, 1.05, 1.10)]
           + [('uinf', u) for u in (76.0, 84.0)]
           + [('cltrim', m) for m in (0.95, 1.05)])


def _delivered(r):
    """(sigma_del, bias_del) of the Wc channel vs the true next-step gust."""
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw  = t <= (Tg + 0.5)
    err = Wc[mw] - Wnext[mw]
    return float(np.std(err)), float(np.mean(err))


class _MismatchCtrl:
    """
    Combo adapter that runs mpc.compute under NOMINAL structure globals and
    restores the PERTURBED (plant) values before returning, so the plant
    step that follows in the harness loop integrates the perturbed system.
    pert: list of (attr, plant_value, nominal_value); empty for uinf/cltrim
    arms (those mis-set the controller's own constructor args instead).
    """

    def __init__(self, sensor, mpc, pert):
        self._sensor = sensor
        self._mpc    = mpc
        self._pert   = list(pert)

    def reset(self):
        self._sensor.reset()
        self._mpc.reset()

    def compute(self, state, W_true, Wc):
        for a, pv, nv in self._pert:
            setattr(S, a, nv)
        try:
            return self._mpc.compute(state, self._sensor.last)
        finally:
            for a, pv, nv in self._pert:
                setattr(S, a, pv)


def build(arm, val):
    """-> (pert list, mpc kwargs overrides)"""
    if arm == 'dalpha':
        return [('D_ALPHA', val * DA_NOM, DA_NOM)], {}
    if arm == 'kalpha':
        return [('K_ALPHA', val * KA_NOM, KA_NOM)], {}
    if arm == 'uinf':
        return [], dict(U=float(val))
    if arm == 'cltrim':
        return [], dict(C_L_trim=float(val) * H.CLTRIM)
    return [], {}                                    # anchor


def make_combo(rng, mpc_kw):
    sensor = FusedPreviewSensor(rng, lambda j: 1e-9, JMAX, N, lam=LAM)
    kw = dict(U=H.U, C_L_trim=H.CLTRIM)
    kw.update(mpc_kw)
    mpc = MPCPreviewController(
        H.aero, U=kw['U'], dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=kw['C_L_trim'], N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    return sensor, mpc


OL_NOM   = H.rollout(None, W0, Tg)
cex0_nom = H.metrics(OL_NOM, OL_NOM, Tg)['exo']
print(f"# D2_mismatch | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0_nom:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='D2', W0=W0, Tg=Tg, cex0=cex0_nom,
             t=OL_NOM['_t'], W=OL_NOM['_Wt'], CL=OL_NOM['CL'])]

for arm, val in PTS:
    pert, mpc_kw = build(arm, val)

    # plant state = perturbed for the whole point (OL + closed loop)
    for a, pv, nv in pert:
        setattr(S, a, pv)
    try:
        OLp = H.rollout(None, W0, Tg) if pert else OL_NOM
        cex0p = H.metrics(OLp, OLp, Tg)['exo']

        rng = np.random.default_rng(100)
        sensor, mpc = make_combo(rng, mpc_kw)
        ctrl = _MismatchCtrl(sensor, mpc, pert)
        r = H.rollout(ctrl, W0, Tg, wc_fun=sensor.wc_fun)
        m = H.metrics(r, OLp, Tg)
    finally:
        for a, pv, nv in pert:
            setattr(S, a, nv)

    sd, bd = _delivered(r)
    rec = H.point_record([m], axis='D2', arm=arm, value=float(val),
                         W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM, R_du=R_DU,
                         frac=0.0, cex0=float(cex0p),
                         sigma_del=sd, bias_del=bd)
    recs.append(rec)
    print(f"  {arm}={val:g}: {H.fmt_stats(H.seed_stats([m]))}  "
          f"cex0={cex0p:.4f}", flush=True)

H.save_records(OUT, recs)
print("# DONE", flush=True)
```

- [ ] **Step 2: Write `light/noise/launch_noise_mismatch_combo.sh`**:

```bash
#!/bin/bash
# Launch noise_mismatch_combo.py (cluster only).
#   ./launch_noise_mismatch_combo.sh smoke
#   ./launch_noise_mismatch_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_mismatch_combo.py --smoke" > D2.smoke.log 2>&1 &
    echo "launched noise_mismatch_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_mismatch_combo.py" > D2.log 2>&1 &
    echo "launched noise_mismatch_combo full pid $!"
fi
```

- [ ] **Step 3: Write `light/noise/status_noise_mismatch_combo.sh`**:

```bash
#!/bin/bash
cd /work/u10677113/LDNet_GLA/light/noise 2>/dev/null || exit 1
for pair in "D2.smoke:noise_mismatch_combo.py --smoke" "D2:noise_mismatch_combo.py"; do
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

- [ ] **Step 4: Write `light/noise/plots_noise_mismatch_combo.py`**:

```python
"""
Figure: CLred vs structural / controller-parameter mismatch for the E2-combo.
Reads results/D2_mismatch.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# our own npz (harness save_records object-array schema) -- trusted local file
recs = list(np.load(os.path.join(DIR, 'D2_mismatch.npz'), allow_pickle=True)['records'])
pts  = [r for r in recs if r.get('kind') == 'point']

ANCHOR = 80.51
U_NOM  = 80.0

def series(arm):
    rows = sorted([r for r in pts if r['arm'] == arm],
                  key=lambda r: float(r['value']))
    return ([float(r['value']) for r in rows],
            [float(r['mean']) for r in rows],
            [int(r['nflag']) for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# left: structural multipliers (plant vs nominal internal model)
ax = axes[0]
for arm, color, label in [('dalpha', 'tab:blue', 'plant D_ALPHA x (ctrl assumes x1)'),
                          ('kalpha', 'tab:green', 'plant K_ALPHA x (ctrl assumes x1)')]:
    x, m, nf = series(arm)
    x = x + [1.0]; m = m + [ANCHOR]; nf = nf + [0]
    order = np.argsort(x)
    x = list(np.array(x)[order]); m = list(np.array(m)[order])
    nf = list(np.array(nf)[order])
    ax.plot(x, m, 'o-', color=color, label=label)
    for xi, mi, f in zip(x, m, nf):
        if f:
            ax.plot(xi, mi, 'o', mfc='none', mec='red', ms=12, mew=1.5)
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('plant parameter multiplier vs internal model')
ax.set_ylabel('CLred [%]')
ax.set_title('D2 -- structural mismatch (red ring = flagged)')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)

# right: controller-side estimate errors (U, CLtrim), x = % error
ax = axes[1]
xu, mu, nfu = series('uinf')
xu_pct = [(u / U_NOM - 1.0) * 100 for u in xu]
xc, mc, nfc = series('cltrim')
xc_pct = [(c - 1.0) * 100 for c in xc]
for xs, ms_, nfs, color, mk, label in [
        (xu_pct, mu, nfu, 'tab:purple', 'o', 'controller U error'),
        (xc_pct, mc, nfc, 'tab:brown', 's', 'controller C_L_trim error')]:
    xs = xs + [0.0]; ms_ = ms_ + [ANCHOR]; nfs = nfs + [0]
    order = np.argsort(xs)
    xs = list(np.array(xs)[order]); ms_ = list(np.array(ms_)[order])
    nfs = list(np.array(nfs)[order])
    ax.plot(xs, ms_, mk + '-', color=color, label=label)
    for xi, mi, f in zip(xs, ms_, nfs):
        if f:
            ax.plot(xi, mi, mk, mfc='none', mec='red', ms=12, mew=1.5)
ax.axhline(ANCHOR, color='gray', lw=0.7, ls=':', label=f'clean anchor +{ANCHOR}%')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('controller estimate error [%]')
ax.set_ylabel('CLred [%]')
ax.set_title('D2 -- controller-side parameter errors')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)

fig.suptitle('E2-combo model-mismatch robustness (W30/Tg0.4, DAMULT=3)', y=1.02)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_mismatch_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')
```

- [ ] **Step 5: Parse-test all four files** (same commands as Task 1 Step 5 with the mismatch filenames). Expected: `parse OK` x2, `launch OK`, `status OK`.

- [ ] **Step 6: Commit (only the four new files)**

```bash
git add light/noise/noise_mismatch_combo.py light/noise/launch_noise_mismatch_combo.sh light/noise/status_noise_mismatch_combo.sh light/noise/plots_noise_mismatch_combo.py
git commit -m "feat(noise): D2 structural model-mismatch robustness axis for E2-combo"
```

---

### Task 3: Cluster deployment, smokes, full launches (controller-executed)

- [ ] Step 1: scp the 6 cluster-relevant files (2 axis .py, 4 .sh), one per command; chmod +x the .sh via ssh.
- [ ] Step 2: launch C2 smoke and D2 smoke concurrently.
- [ ] Step 3: gates —
  - C2: `jitter k=0 frac=0%` == +80.51% (±0.1; bit-exact path: k=0 draws no eta). `k=2` row is DATA.
  - D2: `anchor=1` == +80.51% (±0.1) with `cex0=0.4600`. `dalpha=1.33` row is DATA (its cex0 will differ — that is expected, the perturbed plant has its own open loop).
  - Any gate failure -> STOP, debug, do not launch full.
- [ ] Step 4: launch both fulls in parallel. C2 = 25 rollouts ~ 2 h; D2 = 13 closed + 8 open rollouts ~ 1.5 h (measured pace ~4.5 min/combo rollout).
- [ ] Step 5: watch until both logs show `# DONE` (background poll, count DONE per file — no uniq).

### Task 4: Results, figures, NOTES.md, final commit (controller-executed)

- [ ] Step 1: scp back `results/C2_jitter.npz` and `results/D2_mismatch.npz` (one scp per file).
- [ ] Step 2: run both plot scripts locally.
- [ ] Step 3: append to `light/noise/NOTES.md`:
  - `# C2 — spatial per-shot jitter` — table (k, meters, jitter-only mean [min,max] flags, compound row, sigma_del); mechanism note (fusion converges to W convolved with the jitter kernel — spatial low-pass, phase damage; relate to B2's early/late asymmetry if the data shows it).
  - `# D2 — structural model mismatch` — table per arm (multiplier / % error, CLred, flags, cex0 of the perturbed plant); mechanism notes (over-assumed damping = the dangerous direction hypothesis; K_ALPHA = frequency mismatch, the Forte analogue; declared limit: shared LDNet, structural mismatch only).
  - Update `# Robustness envelope of the E2-combo` — add rows: per-shot jitter tolerance, plant D_ALPHA range, plant K_ALPHA range, controller U error, controller C_L_trim error. Derive thresholds from actual numbers; unbracketed edges stated as ">last tested".
- [ ] Step 4: single commit: the two npz, the two figures, NOTES.md.

```bash
git add light/noise/results/C2_jitter.npz light/noise/results/D2_mismatch.npz light/noise/results/fig_noise_jitter_combo.png light/noise/results/fig_noise_mismatch_combo.png light/noise/NOTES.md
git commit -m "results(noise): C2 jitter + D2 model-mismatch robustness of the E2-combo"
```
