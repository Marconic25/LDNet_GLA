# Design: LDNet-based controller in `clean/` — cluster deployment

**Date:** 2026-05-27  
**Status:** Approved

## Context

The `clean/` folder contains a production-grade, minimal GLA (Gust Load Alleviation) simulation pipeline. Currently it uses a quasi-steady linear Theodorsen aerodynamic model (`clean/aero.py`). The goal is to replace this with the trained LDNet neural-network model (weights at `/work/u10677113/LDNet_GLA/results/sensitivity/latent_10/` on the cluster) and run the closed-loop simulation on the HPC cluster.

The controller (`clean/controller.py`) is already model-agnostic — it accepts any callable `aero_predict(state, delta_deg, W, U) -> (C_L, C_M)`. The main work is:
1. Creating a stateful LDNet wrapper that exposes this same interface
2. Making the observer model-agnostic (it currently hardcodes `aero.predict`)
3. Adding CLI argument for model path to `run.py`
4. Writing a PBS job script for cluster submission

## Architecture

### Files changed / created

| File | Action | Description |
|------|--------|-------------|
| `clean/ldnet_aero.py` | **NEW** | Stateful LDNet wrapper — standalone copy of model loading logic |
| `clean/observer.py` | **MODIFIED** | Accept `aero_module` parameter instead of hardcoded `aero` import |
| `clean/run.py` | **MODIFIED** | `argparse --model-dir`, selects linear vs LDNet, calls `advance()` |
| `clean/run_ldnet.pbs` | **NEW** | PBS job script for cluster |
| `clean/aero.py` | unchanged | Linear baseline |
| `clean/controller.py` | unchanged | One-step optimal controller |
| `clean/structure.py` | unchanged | Structural dynamics |

---

## `clean/ldnet_aero.py`

Standalone class (does not import from `src/`) that:

- Loads `NNdyn_weights.weights.h5`, `NNrec_weights.weights.h5`, `config.json` from `model_dir`
- Builds TensorFlow networks matching `src/aerodynamics/model.py` architecture (same sizes, tanh activations)
- Maintains internal latent state `z` (shape `(num_latent_states,)`, initialized to zeros)
- Includes h5py file-locking workaround for WSL2/Windows (copy to tempdir)

### Key methods

```python
class LDNetAero:
    def __init__(self, model_dir): ...          # loads weights, builds NNdyn + NNrec
    def predict(state, delta_deg, W, U) -> (C_L, C_M)  # read-only z, same signature as aero.predict
    def advance(state, delta_deg, W, U) -> None         # updates self.z one step (call once per timestep)
    def reset() -> None                                  # zeros z
```

**Critical design decision:** `predict()` is read-only — it uses `self.z` but does NOT update it. The controller calls `predict()` multiple times during scalar optimization over candidate δ values; updating z on each call would corrupt the latent state. `advance()` is called once per timestep in `run.py` after the true δ and true W are known.

### Normalization

Inputs normalized to [-1, 1] using min/max from `config.json['normalization']`. Time constant for latent update: `dt_ref = config['normalization']['time']['time_constant']`. The latent update is:

```
z_new = z + (dt / dt_ref) * NNdyn([z, U_inf_n, signals_n])
```

### Input order to NNdyn

`[z_0, ..., z_{n-1}, U_inf_n, h_n, hd_n, a_n, ad_n, delta_n, W_gust_n]`

### Input order to NNrec

`[z_0, ..., z_{n-1}, h_n, hd_n, a_n, ad_n, delta_n, W_gust_n, 0.0, 0.0]`  
(spatial points fixed at 0,0 — we only need scalar C_L, C_M)

---

## `clean/observer.py` — modification

Change constructor signature:

```python
def __init__(self, dt, U, aero_module=None):
    import aero as _default_aero
    self._aero = aero_module if aero_module is not None else _default_aero
```

Replace `aero.predict(...)` in `_estimate_gust` with `self._aero.predict(...)`.

The bisection uses `predict()` (read-only) — correct because we are searching over W at fixed z.

---

## `clean/run.py` — modification

Add at top:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', default=None)
args = parser.parse_args()
```

Model selection:

```python
if args.model_dir:
    from ldnet_aero import LDNetAero
    aero_module = LDNetAero(args.model_dir)
    aero_predict = aero_module.predict
else:
    import aero as aero_module
    aero_predict = aero.predict
```

Observer instantiation:

```python
obs = Observer(dt=DT, U=U_INF, aero_module=aero_module)
```

Controller instantiation (unchanged, uses `aero_predict`):

```python
ctrl = Controller(aero_predict=aero_predict, ...)
```

**Latent state advance in simulation loop** — after forces are computed with true state/gust, before structural integration:

```python
C_L, C_M = aero_module.predict(x, delta, W_true[i], U_INF)  # true aero
if hasattr(aero_module, 'advance'):
    aero_module.advance(x, delta, W_true[i], U_INF)          # update z with true values
```

Reset before each run:

```python
if hasattr(aero_module, 'reset'):
    aero_module.reset()
```

---

## `clean/run_ldnet.pbs`

```bash
#!/bin/bash
#PBS -N ldnet_gla_clean
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=0
#PBS -l walltime=02:00:00
#PBS -o logs/ldnet_clean.out
#PBS -e logs/ldnet_clean.err

WEIGHTS=/work/u10677113/LDNet_GLA/results/sensitivity/latent_10
REPO=/path/to/LDNet_OF   # to be set

cd $REPO/clean
module load singularity

singularity exec $SIF python run.py --model-dir $WEIGHTS
```

Note: no GPU needed for inference-only controller (CPU TF is fine). `$SIF` should point to the existing Apptainer container used in `run_pipeline.pbs`.

---

## Data flow per timestep

```
t = k:
  ctrl.compute(x_hat, W_hat)
    └── optimizer calls ldnet_aero.predict(x_hat, δ_cand, W_hat, U)  [read-only z]
        └── returns (C_L_pred, C_M_pred) with z unchanged
  → δ_cmd

  aero_module.predict(x_true, δ_cmd, W_true[k], U)  → (C_L_true, C_M_true)
  aero_module.advance(x_true, δ_cmd, W_true[k], U)  → updates z_hat

  structure.step_rk4(x, Fy_true, Mz_true, dt)  → x_next

  observer.update(h_ddot, α, α̇, δ_cmd, C_L_true)
    └── bisection: ldnet_aero.predict(x_hat, δ_cmd, W_cand, U)  [read-only z]
        for W_cand ∈ [0, 80] until C_L_pred ≈ C_L_true
    → (x_hat_next, W_hat_next)
```

---

## Verification

1. **Local unit test:** `python clean/run.py --model-dir models/` (latent_1 local weights) — should produce valid (non-NaN) timeseries and metrics table.
2. **Linear baseline check:** `python clean/run.py` (no `--model-dir`) — must give identical results to before these changes.
3. **On cluster:** `qsub clean/run_ldnet.pbs` — check `logs/ldnet_clean.out` for metrics table and PNG outputs.
