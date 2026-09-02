## Task 5 â€“ White-noise robustness of the combo (noise_white_combo.py)

**Files:**
- Create: `light/noise/noise_white_combo.py`
- Create: `light/noise/launch_noise_white_combo.sh`
- Create: `light/noise/status_noise_white_combo.sh`
- Create: `light/noise/plots_noise_white_combo.py`

**Context:** Axis A (`noise_white.py`) showed the one-step controller collapses at Ïƒâ‰ˆ1%Â·W0. This script sweeps Ïƒ for the combo (Jmax=50, N=8, R=3e-4, R_du=0, lam=0) to find where the combo degrades. Reference: E2-combo is perfect at Ïƒ=2% (+80.5%) and with DLR raw 1-3 m/s; we push to Ïƒ=20%.

- [ ] **Step 1: Write noise_white_combo.py**

Create `light/noise/noise_white_combo.py`:

```python
"""
White-noise robustness of the E2-combo pipeline vs the one-step argmin.

sigma/W0 in {0, 0.01, 0.02, 0.05, 0.10, 0.20} â€” white Gaussian noise on each
individual sensor measurement BEFORE fusion (i.e. the raw-shot sigma, not the
delivered sigma). With Jmax=50 the fusion already reduces the effective preview
noise dramatically; sigma_del = std(Wc - W_true_next) is printed and logged.

Config: Jmax=50, lam=0, N=8, R=3e-4, R_du=0, 6 seeds (rng 100+seed), home cell
W30/Tg0.4, DAMULT=3. Paired baseline: one-step none (wc_plain, same frac).
Metrics and t<=Tg+0.5 window identical to harness_noise axes.

Output: results/W_combo.npz
--smoke: sigma in {0, 0.02}, 2 seeds, same OUT schema.
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
FRACS    = [0.0, 0.02] if SMOKE else [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'W_combo.npz')


def _delivered_sigma(r):
    t, Wt, Wc = r['_t'], r['_Wt'], r['_Wc']
    n = len(t)
    Wnext = Wt[np.minimum(np.arange(n) + 1, n - 1)]
    mw    = t <= (Tg + 0.5)
    return float(np.std(Wc[mw] - Wnext[mw]))


# ---- harness adapter: wc_fun sets sensor.last; compute reads it ----------------
class _ComboCtrl:
    def __init__(self, sensor, mpc):
        self._sensor = sensor
        self._mpc    = mpc
        self._delta_prev = 0.0

    def reset(self):
        self._sensor.reset()
        self._mpc.reset()

    def compute(self, state, W_true, Wc):
        # sensor.last set by wc_fun before this call (harness protocol)
        return self._mpc.compute(state, self._sensor.last)


def make_combo(rng, frac):
    sigma_fun = (lambda j: frac * W0) if frac > 0.0 else (lambda j: 1e-9)
    sensor    = FusedPreviewSensor(rng, sigma_fun, JMAX, N, lam=LAM)
    mpc       = MPCPreviewController(
        H.aero, U=H.U, dt=H.DT, rho=1.225, S=0.05, C=H.C,
        C_L_trim=H.CLTRIM, N=N, R=R, R_du=R_DU,
        G=161, delta_max=H.DMAX, delta_dot_max=H.DDOT_MAX)
    return _ComboCtrl(sensor, mpc), sensor.wc_fun


def wc_plain(rng, frac):
    return lambda i, Wt, Nsteps: max(0.0, Wt[min(i+1, Nsteps-1)]
                                     + rng.normal(0.0, frac * W0))


OL   = H.rollout(None, W0, Tg)
cex0 = H.metrics(OL, OL, Tg)['exo']
print(f"# W_combo | W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get('DAMULT','1')} "
      f"N={N} Jmax={JMAX} R={R:g} R_du={R_DU:g} | open cex0={cex0:.4f}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

recs = [dict(kind='open', axis='Wco', W0=W0, Tg=Tg, cex0=cex0,
             t=OL['_t'], W=OL['_Wt'], CL=OL['CL'])]

for frac in FRACS:
    ms_c, rs_c, sds_c = [], [], []
    ms_n, rs_n, sds_n = [], [], []
    for seed in range(NSEED):
        rng_c = np.random.default_rng(100 + seed)
        rng_n = np.random.default_rng(100 + seed)

        # combo arm
        ctrl, wc = make_combo(rng_c, frac)
        rc = H.rollout(ctrl, W0, Tg, wc_fun=wc)
        ms_c.append(H.metrics(rc, OL, Tg)); rs_c.append(rc)
        sds_c.append(_delivered_sigma(rc))

        # one-step baseline (paired, same frac)
        if frac > 0.0:
            none_ctrl = H.make_optimal(R=R)
            rc_n = H.rollout(none_ctrl, W0, Tg, wc_fun=wc_plain(rng_n, frac))
            ms_n.append(H.metrics(rc_n, OL, Tg)); rs_n.append(rc_n)
            sds_n.append(_delivered_sigma(rc_n))

    sig_c = float(np.mean(sds_c))
    rec_c = H.point_record(ms_c, axis='Wco', arm='combo',
                           W0=W0, Tg=Tg, R=R, N=N, Jmax=JMAX, lam=LAM,
                           R_du=R_DU, frac=frac, sigma_del=sig_c)
    recs.append(rec_c)
    print(f"  combo frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms_c))}  "
          f"sig_del={sig_c:.3g} m/s", flush=True)

    if frac > 0.0 and ms_n:
        sig_n = float(np.mean(sds_n))
        rec_n = H.point_record(ms_n, axis='Wco', arm='none',
                               W0=W0, Tg=Tg, R=R, frac=frac, sigma_del=sig_n)
        recs.append(rec_n)
        print(f"  none  frac={frac:.0%}: {H.fmt_stats(H.seed_stats(ms_n))}  "
              f"sig_del={sig_n:.3g} m/s", flush=True)

H.save_records(OUT, recs)
print("# DONE", flush=True)
```

- [ ] **Step 2: Write launch and status scripts**

Create `light/noise/launch_noise_white_combo.sh`:
```bash
#!/bin/bash
# Launch noise_white_combo.py (cluster only).
#   ./launch_noise_white_combo.sh smoke
#   ./launch_noise_white_combo.sh full
cd /work/u10677113/LDNet_GLA/light/noise || exit 1
if [ "$1" = "smoke" ]; then
    nohup ./run_axis.sh "noise_white_combo.py --smoke" > Wco.smoke.log 2>&1 &
    echo "launched noise_white_combo smoke pid $!"
else
    nohup ./run_axis.sh "noise_white_combo.py" > Wco.log 2>&1 &
    echo "launched noise_white_combo full pid $!"
fi
```

Create `light/noise/status_noise_white_combo.sh`:
```bash
#!/bin/bash
cd /work/u10677113/LDNet_GLA/light/noise 2>/dev/null || exit 1
for pair in "Wco.smoke:noise_white_combo.py --smoke" "Wco:noise_white_combo.py"; do
    L=${pair%%:*}; P=${pair##*:}
    LOG="${L}.log"
    [ -f "$LOG" ] || continue
    if grep -q "^# DONE" "$LOG"; then st="DONE"
    elif grep -qE "Traceback|Error|Killed" "$LOG"; then st="ERROR"
    elif pgrep -f "python3 -s -u $P" >/dev/null 2>&1; then st="RUNNING"
    else st="DEAD"; fi
    last=$(grep -E "^  combo|^  none|^# DONE|^#" "$LOG" 2>/dev/null | tail -1)
    echo "$L: $st | $last"
done
```

- [ ] **Step 3: Write plots_noise_white_combo.py**

Create `light/noise/plots_noise_white_combo.py`:

```python
"""
Figure: CLred vs sigma/W0 for E2-combo vs one-step none.
Reads results/W_combo.npz. Run locally after scp.
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
recs = list(np.load(os.path.join(DIR, 'W_combo.npz'), allow_pickle=True)['records'])

combo = [r for r in recs if r.get('kind') == 'point' and r.get('arm') == 'combo']
none  = [r for r in recs if r.get('kind') == 'point' and r.get('arm') == 'none']

combo_f = sorted({float(r['frac']) for r in combo})
none_f  = sorted({float(r['frac']) for r in none})

def get_stats(rows, frac):
    pts = [r for r in rows if abs(float(r['frac']) - frac) < 1e-9]
    if not pts: return None
    r = pts[0]
    return float(r['mean']), float(r['lo']), float(r['hi'])

fig, ax = plt.subplots(figsize=(7, 4))
c_fracs = combo_f
n_fracs = none_f

c_means, c_lo, c_hi = [], [], []
for f in c_fracs:
    s = get_stats(combo, f)
    c_means.append(s[0]); c_lo.append(s[1]); c_hi.append(s[2])

n_means, n_lo, n_hi = [], [], []
for f in n_fracs:
    s = get_stats(none, f)
    if s: n_means.append(s[0]); n_lo.append(s[1]); n_hi.append(s[2])
    else: n_means.append(float('nan')); n_lo.append(float('nan')); n_hi.append(float('nan'))

pcts = [f * 100 for f in c_fracs]
ax.fill_between(pcts, c_lo, c_hi, alpha=0.2, color='tab:blue')
ax.plot(pcts, c_means, 'o-', color='tab:blue', label='E2-combo (Jmax=50, N=8)')

if n_fracs:
    npcts = [f * 100 for f in n_fracs]
    ax.fill_between(npcts, n_lo, n_hi, alpha=0.2, color='tab:red')
    ax.plot(npcts, n_means, 's--', color='tab:red', label='one-step optimal (no fusion)')

ax.axhline(0, color='gray', lw=0.7)
ax.axhline(32.0, color='gray', lw=0.7, ls=':', label='prop-W clean (+32%)')
ax.set_xlabel('Raw measurement noise Ïƒ / W0 [%]')
ax.set_ylabel('CLred [%]')
ax.set_title('White-noise robustness â€” E2-combo vs one-step (W30/Tg0.4, DAMULT=3)')
ax.legend(frameon=False); ax.grid(alpha=0.3)
plt.tight_layout()
fn = os.path.join(DIR, 'fig_noise_white_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}')
```

- [ ] **Step 4: Sync scripts and launch**

Sync:
```bash
scp light/noise/noise_white_combo.py \
    light/noise/launch_noise_white_combo.sh \
    light/noise/status_noise_white_combo.sh \
    light/noise/plots_noise_white_combo.py \
    u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/noise/

ssh -n u10677113@10.78.18.100 \
    'chmod +x /work/u10677113/LDNet_GLA/light/noise/launch_noise_white_combo.sh \
              /work/u10677113/LDNet_GLA/light/noise/status_noise_white_combo.sh'
```

Smoke first (2 seeds, Ïƒâˆˆ{0,2%}, ~20 min):
```bash
ssh -n u10677113@10.78.18.100 \
    '/work/u10677113/LDNet_GLA/light/noise/launch_noise_white_combo.sh smoke'
# Poll:
ssh -n u10677113@10.78.18.100 \
    '/work/u10677113/LDNet_GLA/light/noise/status_noise_white_combo.sh'
```

Smoke pass criterion: combo frac=0.0 CLred â‰ˆ dp45 combo clean anchor (Â±1 pt); combo frac=0.02 â‰ˆ same. Then full run:
```bash
ssh -n u10677113@10.78.18.100 \
    '/work/u10677113/LDNet_GLA/light/noise/launch_noise_white_combo.sh full'
```

Expected runtime: 6 Ïƒ-levels Ã— (6 combo + 5 none) seeds Ã— ~8 min combo / ~1 min none â‰ˆ **~5h** total (single job).

- [ ] **Step 5: Collect results and generate figure**

Scp results back:
```bash
scp u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/noise/results/W_combo.npz \
    /home/marco/LDNet_OF/light/noise/results/

python3 light/noise/plots_noise_white_combo.py
```

- [ ] **Step 6: Add W_combo results section to NOTES.md**

Append the following to `light/noise/NOTES.md` (fill in actual numbers from the run):

```markdown
---

# W_combo â€” white-noise robustness of the E2-combo (2026-07-08)

Axis: noise_white_combo.py â€” white Gaussian raw measurement noise Ïƒ applied BEFORE
fusion; FusedSensor(Jmax=50, lam=0, N=8) + MPCPreviewController(R=3e-4, R_du=0).
Home cell W30/Tg0.4, DAMULT=3, 6 seeds rng(100+seed). Baseline 'none' = one-step
argmin (same R, same raw frac), same seeds (paired).

| Ïƒ/W0 | combo CLred | combo [min,max] flags/6 | Ïƒ_del m/s | none CLred | none flags/6 |
|---|---|---|---|---|---|
| 0%   | TBD | TBD | TBD | â€” | â€” |
| 1%   | TBD | TBD | TBD | TBD | TBD |
| 2%   | TBD | TBD | TBD | TBD | TBD |
| 5%   | TBD | TBD | TBD | TBD | TBD |
| 10%  | TBD | TBD | TBD | TBD | TBD |
| 20%  | TBD | TBD | TBD | TBD | TBD |

**Break-even vs combo-clean:** Ïƒ/W0 = TBD%
**Break-even vs prop-W clean (+32%):** Ïƒ/W0 = TBD%

Figure: `results/fig_noise_white_combo.png`

Verdict: TBD (fill after cluster run).
```

- [ ] **Step 7: Commit noise robustness scripts**
```bash
git add light/noise/noise_white_combo.py \
        light/noise/launch_noise_white_combo.sh \
        light/noise/status_noise_white_combo.sh \
        light/noise/plots_noise_white_combo.py
git commit -m "feat(noise): white-noise robustness study of E2-combo pipeline"
```

---

