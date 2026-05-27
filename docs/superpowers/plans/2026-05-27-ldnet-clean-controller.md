# LDNet Clean Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the linear aerodynamic model in `clean/` with the trained LDNet neural network (weights at `results/sensitivity/latent_10/` on the cluster), keeping the controller, structure, and observer interfaces intact.

**Architecture:** Create `clean/ldnet_aero.py` as a standalone stateful LDNet wrapper with `predict()` (read-only z) and `advance()` (updates z once per timestep). Modify `observer.py` to accept any `aero_module`. Modify `run.py` to accept `--model-dir` CLI arg and select the model. Add `run_ldnet.pbs` for cluster submission.

**Tech Stack:** Python 3, TensorFlow 2.x (float64), NumPy, SciPy, h5py, PBS/Torque job scheduler.

---

## File Map

| File | Action |
|------|--------|
| `clean/ldnet_aero.py` | CREATE — stateful LDNet wrapper |
| `clean/observer.py` | MODIFY — accept `aero_module` param |
| `clean/run.py` | MODIFY — argparse `--model-dir`, call `advance()` |
| `clean/run_ldnet.pbs` | CREATE — PBS job script |

---

### Task 1: Create `clean/ldnet_aero.py`

**Files:**
- Create: `clean/ldnet_aero.py`

This file is standalone — it does NOT import from `src/`. It mirrors the logic of `src/aerodynamics/model.py` but exposes `predict()` (read-only z) and `advance()` (updates z) instead of a single `step()`.

- [ ] **Step 1: Create `clean/ldnet_aero.py`**

```python
"""
Stateful LDNet aerodynamic model wrapper for use in clean/.

Loads NNdyn and NNrec from a model directory (NNdyn_weights.weights.h5,
NNrec_weights.weights.h5, config.json) and exposes the same interface as
clean/aero.py:

    predict(state, delta_deg, W, U) -> (C_L, C_M)

The latent state z is maintained internally. predict() is read-only
(does not modify z) so the controller's scalar optimizer can call it
repeatedly without corrupting state. advance() steps z forward once
with the true delta and true gust — call it once per timestep in run.py.
"""
import json
import shutil
import tempfile
import numpy as np
import os
import tensorflow as tf
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float64')


class LDNetAero:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self._norm = config['normalization']
        self._problem = config['problem']
        self._num_z = config['num_latent_states']
        self._dt_ref = self._norm['time']['time_constant']

        n_signals = len(self._problem['input_signals'])      # 6
        n_params  = len(self._problem['input_parameters'])   # 1 (U_inf)
        n_space   = self._problem['space']['dimension']       # 2

        dyn_in = self._num_z + n_params + n_signals
        self.NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(7, activation='tanh', input_shape=(dyn_in,)),
            tf.keras.layers.Dense(7, activation='tanh'),
            tf.keras.layers.Dense(self._num_z),
        ])

        rec_in = self._num_z + n_signals + n_space
        self.NNrec = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='tanh', input_shape=(None, None, rec_in)),
            tf.keras.layers.Dense(24, activation='tanh'),
            tf.keras.layers.Dense(24, activation='tanh'),
            tf.keras.layers.Dense(24, activation='tanh'),
            tf.keras.layers.Dense(len(self._problem['output_signals'])),
        ])

        self._load_weights(model_dir)
        self._z = np.zeros(self._num_z)

    def _load_weights(self, model_dir):
        try:
            self.NNdyn.load_weights(model_dir / 'NNdyn_weights.weights.h5')
            self.NNrec.load_weights(model_dir / 'NNrec_weights.weights.h5')
        except OSError as e:
            if 'lock' in str(e).lower() or 'Unable to synchronously open' in str(e):
                print('  [WARNING] h5py lock detected, using tempdir workaround...')
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    shutil.copy(model_dir / 'NNdyn_weights.weights.h5',
                                tmp / 'NNdyn_weights.weights.h5')
                    shutil.copy(model_dir / 'NNrec_weights.weights.h5',
                                tmp / 'NNrec_weights.weights.h5')
                    self.NNdyn.load_weights(str(tmp / 'NNdyn_weights.weights.h5'))
                    self.NNrec.load_weights(str(tmp / 'NNrec_weights.weights.h5'))
                    print('  [OK] weights loaded from temp location')
            else:
                raise

    # ------------------------------------------------------------------
    def _normalize_signals(self, h, hd, a, ad, delta, W):
        """Normalize the 6 input signals to [-1, 1]."""
        s = self._norm['input_signals']
        def n(v, key):
            lo, hi = s[key]['min'], s[key]['max']
            return (2.0 * v - lo - hi) / (hi - lo)
        return np.array([n(h,'h'), n(hd,'hd'), n(a,'a'),
                         n(ad,'ad'), n(delta,'delta'), n(W,'W_gust')])

    def _normalize_U(self, U):
        p = self._norm['input_parameters']['U_inf']
        return (2.0 * U - p['min'] - p['max']) / (p['max'] - p['min'])

    def _denorm_CL_CM(self, CL_n, CM_n):
        o = self._norm['output_signals']
        CL = 0.5 * float(CL_n) * (o['C_L']['max'] - o['C_L']['min']) \
             + 0.5 * (o['C_L']['max'] + o['C_L']['min'])
        CM = 0.5 * float(CM_n) * (o['C_M']['max'] - o['C_M']['min']) \
             + 0.5 * (o['C_M']['max'] + o['C_M']['min'])
        return CL, CM

    def _forward(self, z, sigs_n, U_n):
        """Run NNdyn and NNrec, return (z_new, C_L, C_M). Does not mutate self._z."""
        dyn_inp = np.reshape(
            np.concatenate([z, [U_n], sigs_n]),
            (1, self._num_z + 1 + len(sigs_n))
        )
        dz = self.NNdyn(dyn_inp, training=False)
        z_new = z + (self._dt / self._dt_ref) * dz.numpy().flatten()

        rec_inp = np.reshape(
            np.concatenate([z_new, sigs_n, [0.0, 0.0]]),
            (1, 1, 1, self._num_z + len(sigs_n) + 2)
        )
        out_n = self.NNrec(rec_inp, training=False)
        C_L, C_M = self._denorm_CL_CM(out_n[0, 0, 0, 0], out_n[0, 0, 0, 1])
        return z_new, C_L, C_M

    # ------------------------------------------------------------------
    def predict(self, state, delta_deg, W, U):
        """
        Predict (C_L, C_M) using current z — read-only, does NOT update z.

        Same signature as clean/aero.predict:
          state     : (h, hd, alpha, alpha_dot)
          delta_deg : flap deflection [degrees]
          W         : gust velocity [m/s]
          U         : freestream velocity [m/s]
        """
        h, hd, a, ad = state
        sigs_n = self._normalize_signals(h, hd, a, ad, delta_deg, W)
        U_n    = self._normalize_U(U)
        _, C_L, C_M = self._forward(self._z, sigs_n, U_n)
        return float(C_L), float(C_M)

    def advance(self, state, delta_deg, W, U, dt):
        """
        Advance latent state z one step using true (state, delta, W, U, dt).
        Call once per timestep in run.py after true forces are computed.
        """
        self._dt = float(dt)
        h, hd, a, ad = state
        sigs_n = self._normalize_signals(h, hd, a, ad, delta_deg, W)
        U_n    = self._normalize_U(U)
        z_new, _, _ = self._forward(self._z, sigs_n, U_n)
        self._z = z_new

    def reset(self):
        """Reset latent state to zero (call before each simulation)."""
        self._z = np.zeros(self._num_z)
```

- [ ] **Step 2: Fix `_forward` to use `self._dt` correctly**

The `_dt` attribute is set in `advance()` but `predict()` also calls `_forward`. Set a default `_dt` in `__init__` to avoid AttributeError when `predict()` is called before `advance()`:

In `__init__`, after `self._z = np.zeros(self._num_z)`, add:
```python
        self._dt = 0.01  # default timestep; overridden by advance()
```

- [ ] **Step 3: Verify the file loads without error**

```bash
cd /home/marco/LDNet_OF
python -c "
import sys; sys.path.insert(0, 'clean')
from ldnet_aero import LDNetAero
m = LDNetAero('models/')
print('num_z:', m._num_z)
print('predict test:', m.predict([0,0,0,0], 0.0, 0.0, 80.0))
"
```

Expected output (approximate — depends on weights):
```
num_z: 1
predict test: (0.03..., 0.00...)
```
No errors, no NaN.

- [ ] **Step 4: Commit**

```bash
cd /home/marco/LDNet_OF
git add clean/ldnet_aero.py
git commit -m "feat: add LDNetAero wrapper for clean/ controller"
```

---

### Task 2: Modify `clean/observer.py` to accept `aero_module`

**Files:**
- Modify: `clean/observer.py`

Replace the hardcoded `import aero` with a parameter so the same Observer works with either the linear model or LDNet.

- [ ] **Step 1: Edit the import and `__init__` in `clean/observer.py`**

Replace the top of `observer.py` (lines 1–22, through the `TAU_LEAK` constant) — keep `TAU_LEAK`, change the import:

Old `__init__` signature and first line of class:
```python
import aero


TAU_LEAK = 5.0   # leaky integrator time constant [s]


class Observer:
    """
    ...
    """

    def __init__(self, dt, U):
        self.dt = float(dt)
        self.U  = float(U)
        self._x_hat = np.zeros(4)   # [h, ḣ, α, α̇]
        self._W_hat = 0.0
```

New version (add `aero_module=None` parameter):
```python
import aero as _default_aero


TAU_LEAK = 5.0   # leaky integrator time constant [s]


class Observer:
    """
    ...
    """

    def __init__(self, dt, U, aero_module=None):
        self.dt    = float(dt)
        self.U     = float(U)
        self._aero = aero_module if aero_module is not None else _default_aero
        self._x_hat = np.zeros(4)   # [h, ḣ, α, α̇]
        self._W_hat = 0.0
```

- [ ] **Step 2: Update `_estimate_gust` to use `self._aero`**

In `_estimate_gust`, replace:
```python
        def CL_at(W):
            return aero.predict(self._x_hat, delta, W, self.U)[0]
```
with:
```python
        def CL_at(W):
            return self._aero.predict(self._x_hat, delta, W, self.U)[0]
```

- [ ] **Step 3: Verify backward compatibility — linear model still works**

```bash
cd /home/marco/LDNet_OF/clean
python -c "
from observer import Observer
obs = Observer(dt=0.01, U=80.0)   # no aero_module -> uses aero (linear)
x_hat, W_hat = obs.update(0.1, 0.01, 0.001, 0.0, 0.05)
print('x_hat:', x_hat)
print('W_hat:', W_hat)
"
```

Expected: no errors, `x_hat` is a 4-element array, `W_hat` is a float.

- [ ] **Step 4: Verify with LDNetAero**

```bash
cd /home/marco/LDNet_OF
python -c "
import sys; sys.path.insert(0, 'clean')
from ldnet_aero import LDNetAero
from observer import Observer
m = LDNetAero('models/')
obs = Observer(dt=0.01, U=80.0, aero_module=m)
x_hat, W_hat = obs.update(0.1, 0.01, 0.001, 0.0, 0.05)
print('x_hat:', x_hat)
print('W_hat:', W_hat)
"
```

Expected: no errors, valid numbers.

- [ ] **Step 5: Commit**

```bash
cd /home/marco/LDNet_OF
git add clean/observer.py
git commit -m "feat: make Observer accept generic aero_module"
```

---

### Task 3: Modify `clean/run.py` — argparse + LDNet integration

**Files:**
- Modify: `clean/run.py`

Add `--model-dir` CLI argument. Select model. Call `advance()` once per timestep. Reset `aero_module` before each simulation.

- [ ] **Step 1: Add argparse at the top of `run.py`**

After the existing imports (after `from controller import Controller`), add:

```python
import argparse

_parser = argparse.ArgumentParser(description='GLA simulation — linear or LDNet aero')
_parser.add_argument('--model-dir', default=None,
                     help='Path to LDNet weights directory. '
                          'Omit to use the linear Theodorsen model.')
_args = _parser.parse_args()
```

- [ ] **Step 2: Add model selection block (replace the two existing `aero` references in run.py)**

Find the section where `aero.predict` is used. Before the `gust()` function definition, add:

```python
# ── Aerodynamic model selection ────────────────────────────────────────────────

if _args.model_dir:
    from ldnet_aero import LDNetAero
    _aero_module = LDNetAero(_args.model_dir)
    print(f"Using LDNet model from: {_args.model_dir}  (latent_dim={_aero_module._num_z})")
else:
    import aero as _aero_module
    print("Using linear Theodorsen model")
```

- [ ] **Step 3: Update `simulate()` to use `_aero_module`**

In `simulate()`, find:

```python
    obs = Observer(dt=DT, U=U_INF)
```

Replace with:

```python
    obs = Observer(dt=DT, U=U_INF, aero_module=_aero_module)
```

Find:

```python
        C_L, C_M = aero.predict(x, delta, W_true[i], U_INF)
```

Replace with:

```python
        C_L, C_M = _aero_module.predict(x, delta, W_true[i], U_INF)
        if hasattr(_aero_module, 'advance'):
            _aero_module.advance(x, delta, W_true[i], U_INF, DT)
```

Find the `ctrl.reset()` call inside `simulate()`:

```python
    if ctrl is not None:
        ctrl.reset()
```

Add `aero_module.reset()` right after:

```python
    if ctrl is not None:
        ctrl.reset()
    if hasattr(_aero_module, 'reset'):
        _aero_module.reset()
```

- [ ] **Step 4: Update Controller instantiation to use `_aero_module.predict`**

Find:

```python
ctrl = Controller(
    aero_predict  = aero.predict,
```

Replace with:

```python
ctrl = Controller(
    aero_predict  = _aero_module.predict,
```

- [ ] **Step 5: Verify linear model still works (no `--model-dir`)**

```bash
cd /home/marco/LDNet_OF/clean
python run.py
```

Expected: prints metrics table, saves `results/fig1_loads.png` etc. No errors.

- [ ] **Step 6: Verify LDNet run completes (local models/ weights)**

```bash
cd /home/marco/LDNet_OF/clean
python run.py --model-dir ../models/
```

Expected:
```
Using LDNet model from: ../models/  (latent_dim=1)
Running open loop...
Running closed loop (optimal controller)...
...metrics table...
Saved results/fig1_loads.png
```
No NaN in metrics. If C_L reduction is negative (controller making things worse), that is acceptable at this stage — it means the gains need tuning, not that the integration is broken.

- [ ] **Step 7: Commit**

```bash
cd /home/marco/LDNet_OF
git add clean/run.py
git commit -m "feat: add --model-dir argparse to run.py, integrate LDNet in simulation loop"
```

---

### Task 4: Create `clean/run_ldnet.pbs`

**Files:**
- Create: `clean/run_ldnet.pbs`

- [ ] **Step 1: Create the PBS script**

```bash
#!/bin/bash
#PBS -N ldnet_gla_clean
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=0
#PBS -l walltime=02:00:00
#PBS -o logs/ldnet_clean.out
#PBS -e logs/ldnet_clean.err

# Path to trained LDNet weights (latent_dim=10)
WEIGHTS=/work/u10677113/LDNet_GLA/results/sensitivity/latent_10

# Repo root — set to wherever you cloned LDNet_OF on the cluster
REPO=/work/u10677113/LDNet_OF

mkdir -p "$REPO/clean/logs"

cd "$REPO/clean"
module load singularity

singularity exec "$SIF" python run.py --model-dir "$WEIGHTS"
```

Save as `clean/run_ldnet.pbs`.

- [ ] **Step 2: Commit**

```bash
cd /home/marco/LDNet_OF
git add clean/run_ldnet.pbs
git commit -m "feat: add PBS job script for LDNet GLA clean run on cluster"
```

---

### Task 5: Push and cluster run

- [ ] **Step 1: Push to remote**

```bash
cd /home/marco/LDNet_OF
git push
```

- [ ] **Step 2: Pull on cluster**

```bash
# On cluster login node:
cd /work/u10677113/LDNet_OF
git pull
```

- [ ] **Step 3: Edit `REPO` path in `run_ldnet.pbs` if needed**

Check that `REPO` in `clean/run_ldnet.pbs` matches the actual cluster path. Edit if different:

```bash
sed -i 's|REPO=.*|REPO=/work/u10677113/LDNet_OF|' clean/run_ldnet.pbs
```

- [ ] **Step 4: Submit job**

```bash
cd /work/u10677113/LDNet_OF
qsub clean/run_ldnet.pbs
```

- [ ] **Step 5: Check output**

```bash
tail -f clean/logs/ldnet_clean.out
```

Expected: metrics table with `C_L`, `h_ddot`, `h`, `α` columns. No NaN or TF errors.

---

## Self-Review

**Spec coverage:**
- ✅ `clean/ldnet_aero.py` standalone wrapper with `predict()` (read-only z) and `advance()`
- ✅ `observer.py` accepts `aero_module` parameter, bisection uses it
- ✅ `run.py` has `--model-dir` argparse, calls `advance()` once per timestep, resets before each sim
- ✅ `run_ldnet.pbs` with correct cluster paths and no GPU requirement

**Placeholder scan:**
- `REPO` in `.pbs` is intentionally set to a real cluster path. No TBD/TODO.
- `$SIF` is an environment variable expected to be set on the cluster (same convention as existing `run_pipeline.pbs`).

**Type consistency:**
- `predict(state, delta_deg, W, U)` signature consistent across Task 1, 2, 3
- `advance(state, delta_deg, W, U, dt)` defined in Task 1, called in Task 3 with the same 5 args
- `_aero_module` variable name consistent throughout Task 3
- `aero_module` parameter name consistent in Observer (Task 2) and run.py (Task 3)
