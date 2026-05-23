# LDNet Reduced-Input Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce LDNet input signals from 6 (`[h, ḣ, α, α̇, δ, W]`) to 2 (`[δ, W]`), making it a full aeroelastic ROM where the latent state implicitly captures structural dynamics.

**Architecture:** LDNet takes only `[δ, W]` as inputs and outputs `[C_L, C_M]`; the structural ODE (`smd.py`) is integrated explicitly at each timestep in the closed-loop simulation, closing the loop externally. The latent state `z` is propagated forward via Euler integration and must encode both aerodynamic memory and implicit structural response.

**Tech Stack:** TensorFlow 2.x (float64), NumPy, SciPy, h5py, scikit-learn (PCA), Matplotlib. Training runs on cluster; inference runs locally in WSL2.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `data/preprocess_GLA.py` | Modify | Extract only `[δ, W]` as input signals in HDF5 |
| `src/TestCase_OF.py` | Modify | Update `problem` dict and `normalization` to 2 input signals |
| `models_cluster/config.json` | Modify | Updated config (written by `TestCase_OF.py` at train time; also edit manually for inference) |
| `src/aerodynamics/model.py` | Modify | Remove structural state args from `step()`, `step_tf()`, `normalize_input()` |
| `src/aeroelastic/system.py` | Rewrite | Proper closed-loop: integrate structural ODE step-by-step coupled with LDNet |
| `src/pca_validation.py` | Create | PCA dimensionality check on raw CSV dataset |
| `src/sensitivity_latent.py` | Create | Train + compare NRMSE for `num_latent_states ∈ {3, 5, 8, 10, 15}` |

---

## Task 1: PCA Validation Script

Verify the 2D dimensionality hypothesis before touching the model.

**Files:**
- Create: `src/pca_validation.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""PCA dimensionality analysis on GLA dataset.
Loads all training CSVs and checks that [h, hd, a, ad, delta, W_gust]
variance is explained by ≤3 principal components.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).parent.parent / "data" / "GLA_data" / "timeseries"
COLUMNS = ["t", "h", "hd", "a", "ad", "delta", "W_gust", "C_L", "C_M"]
SIG_COLS = [1, 2, 3, 4, 5, 6]   # h, hd, a, ad, delta, W_gust
SIG_NAMES = ["h", "hd", "a", "ad", "delta", "W_gust"]

def load_all_csvs(data_dir):
    rows = []
    for f in sorted(data_dir.glob("sim_*_train.csv")):
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        rows.append(data[:, SIG_COLS])
    return np.vstack(rows)   # (N_total_timesteps, 6)

X = load_all_csvs(DATA_DIR)
print(f"Loaded {X.shape[0]} timesteps × {X.shape[1]} signals")

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_std)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("\nExplained variance per component:")
for i, (ev, cv) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {ev*100:.2f}%  (cumulative: {cv*100:.2f}%)")

n_99 = int(np.searchsorted(cumulative, 0.99)) + 1
print(f"\n→ {n_99} components needed to explain 99% of variance")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, len(explained)+1), explained*100, label="Individual")
ax.plot(range(1, len(explained)+1), cumulative*100, 'ro-', label="Cumulative")
ax.axhline(99, color='gray', linestyle='--', label="99% threshold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance [%]")
ax.set_title("PCA of GLA dataset: [h, ḣ, α, α̇, δ, W]")
ax.legend()
fig.tight_layout()
fig.savefig(Path(__file__).parent.parent / "results" / "pca_variance.png", dpi=150)
print("Saved results/pca_variance.png")
plt.show()
```

- [ ] **Step 2: Run it (from `src/` directory)**

```bash
cd /home/marco/LDNet_OF/src
mkdir -p ../results
python pca_validation.py
```

Expected output: `n_99` should be 2 or 3. If it's ≥4, stop and reassess before proceeding.

---

## Task 2: Update `preprocess_GLA.py` — extract only `[δ, W]` as input signals

**Files:**
- Modify: `data/preprocess_GLA.py`

The key change is in `write_h5`: currently `input_signals` is `s[:, 1:7]` (columns h, hd, a, ad, delta, W_gust). Change to `s[:, 5:7]` (columns delta, W_gust only). The structural columns `h, hd, a, ad` are kept in the CSV but not written as network inputs.

- [ ] **Step 1: Edit `write_h5` in `data/preprocess_GLA.py`**

Find the `write_h5` function (lines 251–276). Change the `input_signals` line:

```python
# Before (line ~260):
input_signals = np.stack([s[:, 1:7] for s in simulations])

# After:
input_signals = np.stack([s[:, 5:7] for s in simulations])   # delta, W_gust only
```

Also update the comment on line ~245:
```python
# Before:
input_signals = [] #N_sim, N_time, 6 (h, h_dot, a, ad, delta, W_gust)

# After:
input_signals = [] #N_sim, N_time, 2 (delta, W_gust)
```

- [ ] **Step 2: Run the preprocessor to regenerate HDF5 files**

```bash
cd /home/marco/LDNet_OF/data
python preprocess_GLA.py
```

Expected output:
```
Dataset salvato correttamente in: GLA_train.h5
Dataset salvato correttamente in: GLA_valid.h5
Dataset salvato correttamente in: GLA_test.h5
```

- [ ] **Step 3: Verify HDF5 shape**

```python
import h5py
with h5py.File('GLA_train.h5', 'r') as f:
    print(f['input_signals'].shape)   # expect (N, 1500, 2)
    print(f['output_signals'].shape)  # expect (N, 1500, 1, 2) — unchanged
```

- [ ] **Step 4: Commit**

```bash
git add data/preprocess_GLA.py
git commit -m "feat: reduce LDNet input signals to [delta, W_gust] in HDF5 preprocessor"
```

---

## Task 3: Update `TestCase_OF.py` — problem definition and normalization

**Files:**
- Modify: `src/TestCase_OF.py`

- [ ] **Step 1: Update `problem` dict (lines 19–46)**

```python
problem = {
    "space": {
        "dimension": 2
    },
    "input_parameters": [{"name": "U_inf"}],

    "input_signals": [
        {"name": "delta"},
        {"name": "W_gust"},
    ],

    "output_signals": [
        {"name": "C_L"},
        {"name": "C_M"},
    ],

    "output_fields": [
        {"name": "ux"},
        {"name": "uy"},
    ]
}
```

- [ ] **Step 2: Update `normalization` dict (lines 48–80)**

```python
normalization = {
    'space': {
        'min': [0, 0],
        'max': [1, 1],
    },
    'time': {
        'time_constant': dt_base
    },
    'input_parameters': {
        'U_inf': {'min': 0, 'max': 120}
    },
    'input_signals': {
        'delta':  {'min': -20.0, 'max': 20.0},
        'W_gust': {'min': 0.0,   'max': 70.0},
    },
    'output_signals': {
        'C_L': {'min': -0.01, 'max': 0.5},
        'C_M': {'min': -0.05, 'max': 0.05},
    },
    'output_fields': {
        'ux': {'min': -50, 'max': 150},
        'uy': {'min': -100, 'max': 100},
    }
}
```

Note: `W_gust` max increased from 50 to 70 m/s to cover the new dataset range (W_peak up to 60 m/s + margin).

- [ ] **Step 3: Update the `num_latent_states` line (line 16)**

```python
# Start with 5 (same as current); sensitivity_latent.py will sweep this value
num_latent_states = 5
```

No change needed here — leave at 5 for the baseline run.

- [ ] **Step 4: Remove the debug print lines (lines 112–115)**

```python
# Remove these 4 lines:
v = NNdyn.variables[0]
print(type(v))
print(type(v.value))
print(hasattr(v, '_variable'))
print(hasattr(v, 'handle'))
```

- [ ] **Step 5: Commit**

```bash
git add src/TestCase_OF.py
git commit -m "feat: update TestCase_OF problem definition to 2 input signals [delta, W_gust]"
```

---

## Task 4: Update `src/aerodynamics/model.py` — remove structural state arguments

This is the most critical change. The `step()` and `step_tf()` methods currently take `h, hd, a, ad` as arguments. These are removed entirely.

**Files:**
- Modify: `src/aerodynamics/model.py`

- [ ] **Step 1: Replace `normalize_input` (lines 63–71)**

```python
def normalize_input(self, delta, W_gust, U_inf):
    delta_n = (2.0*delta - self.normalization['input_signals']['delta']['min']
               - self.normalization['input_signals']['delta']['max']) / (
               self.normalization['input_signals']['delta']['max']
               - self.normalization['input_signals']['delta']['min'])
    W_gust_n = (2.0*W_gust - self.normalization['input_signals']['W_gust']['min']
                - self.normalization['input_signals']['W_gust']['max']) / (
                self.normalization['input_signals']['W_gust']['max']
                - self.normalization['input_signals']['W_gust']['min'])
    U_inf_n = (2.0*U_inf - self.normalization['input_parameters']['U_inf']['min']
               - self.normalization['input_parameters']['U_inf']['max']) / (
               self.normalization['input_parameters']['U_inf']['max']
               - self.normalization['input_parameters']['U_inf']['min'])
    return np.array([delta_n, W_gust_n]), np.array([U_inf_n])
```

- [ ] **Step 2: Replace `step_tf` (lines 78–110)**

```python
def step_tf(self, z, delta, W_gust, U_inf, dt):
    """TF-native step for use inside tf.GradientTape. All inputs are tf.Tensor float64."""
    norm = self.normalization
    def n(v, key):
        lo = norm['input_signals'][key]['min']
        hi = norm['input_signals'][key]['max']
        return (2.0 * v - lo - hi) / (hi - lo)
    U_lo = norm['input_parameters']['U_inf']['min']
    U_hi = norm['input_parameters']['U_inf']['max']
    U_n  = (2.0 * U_inf - U_lo - U_hi) / (U_hi - U_lo)

    inp = tf.stack([n(delta, 'delta'), n(W_gust, 'W_gust')], axis=0)
    inp_full = tf.reshape(tf.concat([z, [U_n], inp], axis=0), (1, -1))

    state = self.NNdyn(inp_full, training=False)
    dt_ref = norm['time']['time_constant']
    z_new = z + (dt / dt_ref) * tf.reshape(state, (-1,))

    pts = tf.zeros(2, dtype=tf.float64)
    rec_inp = tf.reshape(tf.concat([z_new, inp, pts], axis=0), (1, 1, 1, -1))
    out_n = self.NNrec(rec_inp, training=False)
    CL_n = out_n[0, 0, 0, 0]
    CM_n = out_n[0, 0, 0, 1]

    CL_lo = norm['output_signals']['C_L']['min']
    CL_hi = norm['output_signals']['C_L']['max']
    CM_lo = norm['output_signals']['C_M']['min']
    CM_hi = norm['output_signals']['C_M']['max']
    C_L = 0.5 * CL_n * (CL_hi - CL_lo) + 0.5 * (CL_hi + CL_lo)
    C_M = 0.5 * CM_n * (CM_hi - CM_lo) + 0.5 * (CM_hi + CM_lo)
    return z_new, C_L, C_M
```

- [ ] **Step 3: Replace `step` (lines 112–129)**

```python
def step(self, z, delta, W_gust, U_inf, dt):
    """NumPy step for closed-loop simulation."""
    input_signals_n, input_parameters_n = self.normalize_input(delta, W_gust, U_inf)

    state = self.NNdyn(np.reshape(
        np.concatenate([z, input_parameters_n, input_signals_n]),
        (1, len(input_signals_n) + len(input_parameters_n) + self.num_latent_states)
    ))
    dt_ref = self.normalization['time']['time_constant']
    z_new = (z + (dt / dt_ref) * state).numpy().flatten()

    points_full = np.array([0.0, 0.0])
    output_signals_n = self.NNrec(np.reshape(
        np.concatenate([z_new, input_signals_n, points_full]),
        (1, 1, 1, len(input_signals_n) + self.num_latent_states
                   + self.problem['space']['dimension'])
    ))
    C_L_n, C_M_n = output_signals_n[0, 0, 0, 0], output_signals_n[0, 0, 0, 1]
    C_L, C_M = self.denormalize_output(C_L_n, C_M_n)
    return z_new, C_L, C_M
```

- [ ] **Step 4: Commit**

```bash
git add src/aerodynamics/model.py
git commit -m "feat: remove structural state args from LDNetModel.step() and step_tf()"
```

---

## Task 5: Rewrite `src/aeroelastic/system.py` — proper coupled closed-loop

The current code has a bug (structural state frozen at `h0=0` during the loop). Rewrite it with the correct step-by-step coupling.

**Files:**
- Modify: `src/aeroelastic/system.py`

- [ ] **Step 1: Rewrite the entire file**

```python
#!/usr/bin/env python3
"""
Aeroelastic closed-loop simulation: LDNet (2-input) + structural ODE.

Loop at each timestep t_i:
  1. C_L, C_M = LDNet.step(z, delta, W)        # aerodynamic forces (ROM)
  2. Fy, Mz   = forces_from_coefficients(C_L, C_M)
  3. [h, ḣ, α, α̇] = structural_step(Fy, Mz, delta, dt)   # structural ODE
  4. delta = controller(h, alpha, ...)          # control law (optional)
  5. z propagated forward inside LDNet.step()

LDNet replaces pimpleFoam — same coupling structure as cosim_driver.py.
"""
import numpy as np
from scipy.integrate import solve_ivp
from aerodynamics.model import LDNetModel as AeroModel
from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA, _D_X
from pathlib import Path

# ── Physical constants ───────────────────────────────────────────────────────
rho_inf = 1.225   # [kg/m³]
S_ref   = 0.05    # [m²]
c_ref   = 1.0     # [m]
U_INF   = 80.0    # [m/s]

q_inf   = 0.5 * rho_inf * U_INF**2   # dynamic pressure [Pa]


def forces_from_coefficients(C_L, C_M):
    """Convert aerodynamic coefficients to dimensional forces."""
    Fy = q_inf * S_ref * C_L
    Mz = q_inf * S_ref * c_ref * C_M
    return Fy, Mz


def gust_velocity(t, W0=60.0, t_start=0.0, t_end=0.8):
    """1-cosine EASA CS-25 gust profile."""
    t_rel = t - t_start
    T_g   = t_end - t_start
    if 0.0 <= t_rel <= T_g:
        return (W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t_rel / T_g))
    return 0.0


def _delta_derivatives(delta_arr, t_arr):
    """Finite-difference delta_dot and delta_ddot from arrays (rad, rad/s, rad/s²)."""
    dt = t_arr[1] - t_arr[0]
    delta_rad = np.radians(delta_arr)
    delta_dot   = np.gradient(delta_rad, dt)
    delta_ddot  = np.gradient(delta_dot,  dt)
    return delta_dot, delta_ddot


def run_aeroelastic_simulation(
    delta_schedule,        # callable delta_schedule(t) → deg, or array of shape (N,)
    U_inf=U_INF,
    T_END=3.0,
    DT=0.01,
    gust_params=None,      # dict with keys: W0, t_start, t_end
    h0=None, hd0=0.0,      # initial structural state; h0=None → use trim from config
    a0=None, ad0=0.0,
    z0=None,               # initial latent state; None → calibrate from trim burn-in
    burnin_time=0.5,       # seconds of burn-in with delta=0, W=0 to find trim z
    aero_model=None,
    model_dir=None,
):
    """
    Run coupled aeroelastic simulation.

    Returns dict with keys:
        t, h, hd, a, ad, delta, W_gust, C_L, C_M, Fy, Mz, z_history
    """
    if aero_model is None:
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / 'models_cluster'
        aero_model = AeroModel(str(model_dir))

    if gust_params is None:
        gust_params = {'W0': 60.0, 't_start': 0.0, 't_end': 0.8}

    t_arr = np.arange(0.0, T_END + DT * 0.5, DT)
    N = len(t_arr)

    # Build delta array
    if callable(delta_schedule):
        delta_arr = np.array([delta_schedule(t) for t in t_arr])
    else:
        delta_arr = np.asarray(delta_schedule, dtype=float)
        assert len(delta_arr) == N, f"delta_schedule length {len(delta_arr)} != {N}"

    delta_dot_arr, delta_ddot_arr = _delta_derivatives(delta_arr, t_arr)

    # ── Trim calibration: find z for steady flight (delta=0, W=0) ──────────
    if z0 is None:
        z_trim = np.zeros(aero_model.num_latent_states)
        n_burnin = int(burnin_time / DT)
        for _ in range(n_burnin):
            z_trim, _, _ = aero_model.step(z_trim, 0.0, 0.0, U_inf, DT)
        z0 = z_trim

    # ── Initial structural state (from config or provided) ──────────────────
    if h0 is None:
        # Estimate trim from static equilibrium: K_h * h_eq = -Fy_trim
        # Use C_L_trim from LDNet at z0 with delta=0, W=0
        _, C_L_trim, C_M_trim = aero_model.step(z0.copy(), 0.0, 0.0, U_inf, DT)
        Fy_trim = float(C_L_trim) * q_inf * S_ref
        Mz_trim = float(C_M_trim) * q_inf * S_ref * c_ref
        from structural.smd import K_H, K_ALPHA
        h0  = -Fy_trim / K_H
        a0  = Mz_trim / K_ALPHA if a0 is None else a0
    elif a0 is None:
        a0 = 0.0

    # ── Storage ─────────────────────────────────────────────────────────────
    h_arr   = np.zeros(N);  h_arr[0]  = h0
    hd_arr  = np.zeros(N);  hd_arr[0] = hd0
    a_arr   = np.zeros(N);  a_arr[0]  = a0
    ad_arr  = np.zeros(N);  ad_arr[0] = ad0
    C_L_arr = np.zeros(N)
    C_M_arr = np.zeros(N)
    Fy_arr  = np.zeros(N)
    Mz_arr  = np.zeros(N)
    W_arr   = np.array([gust_velocity(t, **gust_params) for t in t_arr])
    z_hist  = np.zeros((N, aero_model.num_latent_states))
    z_hist[0] = z0
    z = z0.copy()

    # ── Main loop ────────────────────────────────────────────────────────────
    h, hd, a, ad = h0, hd0, a0, ad0

    for i in range(N):
        W = W_arr[i]
        delta = delta_arr[i]

        # 1. Aero step
        z, C_L, C_M = aero_model.step(z, delta, W, U_inf, DT)
        C_L, C_M = float(C_L), float(C_M)
        C_L_arr[i] = C_L
        C_M_arr[i] = C_M
        z_hist[i] = z

        # 2. Forces
        Fy = q_inf * S_ref * C_L
        Mz = q_inf * S_ref * c_ref * C_M
        Fy_arr[i] = Fy
        Mz_arr[i] = Mz

        # 3. Structural integration over [t_i, t_{i+1}]
        if i < N - 1:
            t_span = [t_arr[i], t_arr[i + 1]]
            sol = solve_ivp(
                structural_rhs,
                t_span,
                [h, hd, a, ad],
                args=(Fy, Mz, delta_dot_arr[i], delta_ddot_arr[i]),
                method='RK45',
                rtol=1e-6, atol=1e-8,
                max_step=DT,
            )
            h, hd, a, ad = sol.y[:, -1]
            h_arr[i + 1]  = h
            hd_arr[i + 1] = hd
            a_arr[i + 1]  = a
            ad_arr[i + 1] = ad

    return {
        't':       t_arr,
        'h':       h_arr,
        'hd':      hd_arr,
        'a':       a_arr,
        'ad':      ad_arr,
        'delta':   delta_arr,
        'W_gust':  W_arr,
        'C_L':     C_L_arr,
        'C_M':     C_M_arr,
        'Fy':      Fy_arr,
        'Mz':      Mz_arr,
        'z_history': z_hist,
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/aeroelastic/system.py
git commit -m "feat: rewrite aeroelastic system with correct coupled LDNet + structural ODE loop"
```

---

## Task 6: Quick smoke test — run open-loop simulation

Verify the model loads and steps without errors before doing training.

**Files:**
- Run in terminal (no file changes)

- [ ] **Step 1: Run a quick simulation**

```python
# run from /home/marco/LDNet_OF/src
import sys; sys.path.insert(0, '.')
from aeroelastic.system import run_aeroelastic_simulation
import numpy as np

# Constant delta=0, W=0 → should stay near trim
result = run_aeroelastic_simulation(
    delta_schedule=lambda t: 0.0,
    T_END=1.0, DT=0.01,
    gust_params={'W0': 0.0, 't_start': 0.0, 't_end': 0.8},
)
print("C_L range:", result['C_L'].min(), result['C_L'].max())
print("h range [mm]:", result['h'].min()*1000, result['h'].max()*1000)
print("No crash ✓")
```

Expected: no exceptions, `C_L` near trim value (~0.03–0.05), `h` near trim displacement.

**Note:** This test will use the OLD model weights (6-input). It will fail with a shape mismatch because the saved weights correspond to the old 6-input network. This is expected — the model needs to be retrained first (Task 7). The test is only for verifying the Python logic (imports, loop structure). To actually run it end-to-end, skip to after Task 7.

---

## Task 7: Retrain LDNet with 2-input dataset

**Files:**
- Run: `src/TestCase_OF.py` (on cluster or locally if GPU available)

- [ ] **Step 1: Verify HDF5 input shape matches the updated problem**

```python
import h5py
with h5py.File('../data/GLA_train.h5', 'r') as f:
    sig = f['input_signals'][:]
    print(sig.shape)   # must be (N, 1500, 2) — not (N, 1500, 6)
```

If shape is still `(N, 1500, 6)`, re-run Task 2 first.

- [ ] **Step 2: Run training**

```bash
cd /home/marco/LDNet_OF/src
python TestCase_OF.py
```

Training produces:
- `../models/NNdyn_weights.weights.h5`
- `../models/NNrec_weights.weights.h5`
- `../models/config.json`
- `TestCase2.png` (loss curves + prediction plot)

Expected: NRMSE < 10% after Adam+BFGS. If NRMSE > 10%, proceed to Task 8 (sensitivity) to find better `num_latent_states`.

- [ ] **Step 3: Copy weights to `models_cluster/` for inference**

```bash
cp ../models/NNdyn_weights.weights.h5 ../models_cluster/
cp ../models/NNrec_weights.weights.h5 ../models_cluster/
cp ../models/config.json ../models_cluster/
```

- [ ] **Step 4: Commit**

```bash
git add models_cluster/NNdyn_weights.weights.h5 models_cluster/NNrec_weights.weights.h5 models_cluster/config.json
git commit -m "feat: retrained LDNet with 2-input [delta, W_gust] — baseline num_latent_states=5"
```

---

## Task 8: Latent state sensitivity analysis

**Files:**
- Create: `src/sensitivity_latent.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""
Sweep num_latent_states ∈ {3, 5, 8, 10, 15} and report NRMSE on test set.
Saves each trained model to models/sensitivity/n{k}/.
"""
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import utils, optimization

# ── Shared config ─────────────────────────────────────────────────────────
dt       = 0.2
dt_base  = 5.4
SWEEP    = [3, 5, 8, 10, 15]

problem = {
    "space": {"dimension": 2},
    "input_parameters": [{"name": "U_inf"}],
    "input_signals":    [{"name": "delta"}, {"name": "W_gust"}],
    "output_signals":   [{"name": "C_L"}, {"name": "C_M"}],
    "output_fields":    [{"name": "ux"}, {"name": "uy"}],
}
normalization = {
    'space':  {'min': [0, 0], 'max': [1, 1]},
    'time':   {'time_constant': dt_base},
    'input_parameters': {'U_inf': {'min': 0, 'max': 120}},
    'input_signals': {
        'delta':  {'min': -20.0, 'max': 20.0},
        'W_gust': {'min': 0.0,   'max': 70.0},
    },
    'output_signals': {
        'C_L': {'min': -0.01, 'max': 0.5},
        'C_M': {'min': -0.05, 'max': 0.05},
    },
    'output_fields': {
        'ux': {'min': -50, 'max': 150},
        'uy': {'min': -100, 'max': 100},
    },
}

dataset_train = utils.load_gla_h5('../data/GLA_train.h5')
dataset_valid = utils.load_gla_h5('../data/GLA_valid.h5')
dataset_tests = utils.load_gla_h5('../data/GLA_test.h5')
utils.process_dataset(dataset_train, problem, normalization, dt=dt, num_points_subsample=1)
utils.process_dataset(dataset_valid, problem, normalization, dt=dt, num_points_subsample=1)
utils.process_dataset(dataset_tests, problem, normalization, dt=dt)

results = {}

for n_lat in SWEEP:
    print(f"\n{'='*50}\nTraining num_latent_states = {n_lat}\n{'='*50}")
    tf.random.set_seed(0); np.random.seed(0)

    NNdyn = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation=tf.nn.tanh,
            input_shape=(n_lat + len(problem['input_parameters'])
                         + len(problem['input_signals']),)),
        tf.keras.layers.Dense(7, activation=tf.nn.tanh),
        tf.keras.layers.Dense(n_lat),
    ])
    NNrec = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation=tf.nn.tanh,
            input_shape=(None, None,
                         n_lat + len(problem['input_signals'])
                         + problem['space']['dimension'])),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(len(problem['output_signals'])),
    ])

    dt_ref = normalization['time']['time_constant']

    def evolve_dynamics(dataset):
        num_samples = dataset['input_signals'].shape[0]
        num_times   = dataset['input_signals'].shape[1]
        state = tf.zeros((num_samples, n_lat), dtype=tf.float64)
        state_history = tf.TensorArray(tf.float64, size=num_times)
        state_history = state_history.write(0, state)
        for i in tf.range(num_times - 1):
            state = state + dt/dt_ref * NNdyn(tf.concat([
                state,
                tf.expand_dims(dataset['input_parameters'][:, 0], axis=-1),
                dataset['input_signals'][:, i, :]], axis=-1))
            state_history = state_history.write(i + 1, state)
        return tf.transpose(state_history.stack(), perm=(1, 0, 2))

    def reconstruct_output(dataset, states):
        states_exp = tf.broadcast_to(tf.expand_dims(states, 2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], n_lat])
        inp_exp = tf.broadcast_to(tf.expand_dims(dataset['input_signals'], 2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'],
             len(problem['input_signals'])])
        out = NNrec(tf.concat([states_exp, inp_exp, dataset['points_full']], axis=3))
        alpha = 0.05
        return (out**3 + alpha*out) / (1 + alpha)

    def LDNet(dataset):
        return reconstruct_output(dataset, evolve_dynamics(dataset))

    weight_direction = 0
    epsilon = 1e-4

    def get_direction(v):
        return tf.math.divide(v, (epsilon + tf.expand_dims(tf.norm(v, axis=3), axis=-1)))

    tgt_dir_train = get_direction(dataset_train['output_signals'])
    tgt_dir_valid = get_direction(dataset_valid['output_signals'])

    def loss(dataset, tgt_dir):
        vel = LDNet(dataset)
        mse_v = tf.reduce_mean(tf.square(vel - tf.cast(dataset['output_signals'], tf.float64)))
        mse_d = tf.reduce_mean(tf.square(get_direction(vel) - tf.cast(tgt_dir, tf.float64)))
        return mse_v + weight_direction * mse_d

    loss_train = lambda: loss(dataset_train, tgt_dir_train)
    loss_valid = lambda: loss(dataset_valid, tgt_dir_valid)

    trainable = NNdyn.variables + NNrec.variables
    opt = optimization.OptimizationProblem(trainable, loss_train, loss_valid)
    opt.optimize_keras(200, tf.keras.optimizers.Adam(learning_rate=1e-2))
    opt.optimize_BFGS(10000)

    # Evaluate on test set
    out_n   = LDNet(dataset_tests)
    fom     = utils.denormalize_output(dataset_tests['output_signals'], problem, normalization).numpy()
    rom     = utils.denormalize_output(out_n, problem, normalization).numpy()

    nrmse_CL = (np.sqrt(np.mean(np.square(rom[:,:,0,0] - fom[:,:,0,0])))
                / (np.max(fom[:,:,0,0]) - np.min(fom[:,:,0,0])))
    nrmse_CM = (np.sqrt(np.mean(np.square(rom[:,:,0,1] - fom[:,:,0,1])))
                / (np.max(fom[:,:,0,1]) - np.min(fom[:,:,0,1])))

    import scipy.stats
    r_CL = scipy.stats.pearsonr(rom[:,:,0,0].flatten(), fom[:,:,0,0].flatten())[0]
    r_CM = scipy.stats.pearsonr(rom[:,:,0,1].flatten(), fom[:,:,0,1].flatten())[0]

    results[n_lat] = {
        'nrmse_CL': float(nrmse_CL), 'nrmse_CM': float(nrmse_CM),
        'r_CL': float(r_CL), 'r_CM': float(r_CM),
    }
    print(f"n_lat={n_lat}: NRMSE C_L={nrmse_CL:.3e}  C_M={nrmse_CM:.3e}  "
          f"r_CL={r_CL:.4f}  r_CM={r_CM:.4f}")

    # Save model
    save_dir = Path(f'../models/sensitivity/n{n_lat}')
    save_dir.mkdir(parents=True, exist_ok=True)
    NNdyn.save_weights(str(save_dir / 'NNdyn_weights.weights.h5'))
    NNrec.save_weights(str(save_dir / 'NNrec_weights.weights.h5'))
    cfg = {'problem': problem, 'normalization': normalization, 'num_latent_states': n_lat}
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

print("\n\n===== SENSITIVITY SUMMARY =====")
print(f"{'n_lat':>6}  {'NRMSE C_L':>12}  {'NRMSE C_M':>12}  {'r C_L':>8}  {'r C_M':>8}")
for n_lat, r in sorted(results.items()):
    flag = " ← PASS" if r['nrmse_CL'] < 0.05 and r['nrmse_CM'] < 0.05 else ""
    print(f"{n_lat:>6}  {r['nrmse_CL']:>12.3e}  {r['nrmse_CM']:>12.3e}  "
          f"{r['r_CL']:>8.4f}  {r['r_CM']:>8.4f}{flag}")

with open('../results/sensitivity_latent.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results/sensitivity_latent.json")
```

- [ ] **Step 2: Run sensitivity sweep**

```bash
cd /home/marco/LDNet_OF/src
mkdir -p ../results
python sensitivity_latent.py 2>&1 | tee ../results/sensitivity_latent.log
```

Expected: at least one `n_lat` value shows ` ← PASS`. Use the **smallest passing `n_lat`** as the final model.

- [ ] **Step 3: Copy best model to `models_cluster/`**

```bash
# Example: if n_lat=8 is the smallest passing value
cp ../models/sensitivity/n8/NNdyn_weights.weights.h5 ../models_cluster/
cp ../models/sensitivity/n8/NNrec_weights.weights.h5 ../models_cluster/
cp ../models/sensitivity/n8/config.json ../models_cluster/
```

- [ ] **Step 4: Commit**

```bash
git add src/sensitivity_latent.py results/sensitivity_latent.json results/sensitivity_latent.log
git add models_cluster/
git commit -m "feat: latent state sensitivity sweep — selected n_lat=X with NRMSE<5%"
```

---

## Task 9: Closed-loop validation

Run the full coupled simulation and verify results are physically sensible.

**Files:**
- Run in terminal (no file changes unless bugs found)

- [ ] **Step 1: Run and plot**

```python
# from /home/marco/LDNet_OF/src
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from aeroelastic.system import run_aeroelastic_simulation

result = run_aeroelastic_simulation(
    delta_schedule=lambda t: 0.0,   # no control
    T_END=3.0, DT=0.01,
    gust_params={'W0': 60.0, 't_start': 0.0, 't_end': 0.8},
)

fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axs[0].plot(result['t'], result['h']*1000);   axs[0].set_ylabel('h [mm]')
axs[1].plot(result['t'], np.degrees(result['a'])); axs[1].set_ylabel('α [deg]')
axs[2].plot(result['t'], result['C_L']);       axs[2].set_ylabel('C_L')
axs[3].plot(result['t'], result['C_M']);       axs[3].set_ylabel('C_M')
axs[3].set_xlabel('t [s]')
fig.suptitle('LDNet 2-input: open-loop gust response')
fig.tight_layout()
fig.savefig('../results/closed_loop_validation.png', dpi=150)
plt.show()
```

Expected: `h` oscillates and decays after gust; `C_L` peaks during gust window (0–0.8 s); `α` shows coupled response. If `h` or `C_L` diverges, check `num_latent_states` (increase) or `dt` (decrease).

- [ ] **Step 2: Commit results**

```bash
git add results/closed_loop_validation.png
git commit -m "test: closed-loop validation plot for 2-input LDNet gust response"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: PCA (Task 1), preprocessor (Task 2), TestCase (Task 3), model.py (Task 4), system.py (Task 5), training (Task 7), sensitivity (Task 8), closed-loop validation (Task 9)
- [x] **No placeholders**: all steps have exact code or commands
- [x] **Type consistency**: `step(z, delta, W_gust, U_inf, dt)` used consistently in Tasks 4, 5; `normalize_input(delta, W_gust, U_inf)` matches Task 4 Step 1; `structural_rhs` imported from `smd.py` with correct signature `(t, state, Fy, Mz, delta_dot, delta_ddot)`
- [x] **MPC**: `src/control/mpc.py` calls `step_tf` — after Task 4, callers need updating. This is deferred to a separate plan (MPC integration) since the MPC is only needed after the model is validated.
