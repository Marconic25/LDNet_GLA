# PID vs Greedy-LDNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PID and Greedy-N1 controllers and a comparison script (OL / PID / Greedy) for gust load alleviation benchmarking.

**Architecture:** Both controllers expose a `.solve(x_hat, z_hat, W_hat)` interface compatible with `run_simulation()` in `src/control/run_controller.py`. The Greedy uses a linear Theodorsen aero model (Phase 1) that will be swapped for LDNet after training. A new comparison script `src/PIDvsGreedy.py` drives the three simulations and produces plots + a metrics table.

**Tech Stack:** Python 3, NumPy, SciPy (`minimize_scalar`), Matplotlib, existing `run_simulation()` loop.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/control/linear_aero.py` | Create | Theodorsen linear aero: `predict(x, delta, W, U) -> (C_L, C_M)` |
| `src/control/pid.py` | Create | `PIDController` with `.solve(x_hat, z_hat, W_hat) -> delta` |
| `src/control/greedy.py` | Create | `GreedyN1Controller` with `.solve(x_hat, z_hat, W_hat) -> delta` |
| `src/PIDvsGreedy.py` | Create | Comparison script: 3 simulations + plots + metrics table |

---

## Task 1: Linear Theodorsen Aerodynamic Model

**Files:**
- Create: `src/control/linear_aero.py`

The thin-airfoil / Theodorsen steady + quasi-steady coefficients at U=80 m/s, chord c=1 m.
Formulas (quasi-steady, small angle):
```
C_L = 2π*(α + W/U + ḣ/U) + C_Lδ * δ
C_M = C_Mα * α + C_Mδ * δ
```
Where `C_Lδ ≈ -0.7` (flap effectiveness, ~30% chord flap, in rad⁻¹; negative because positive δ = trailing edge down increases C_L and our convention is sign-consistent with the LDNet dataset), `C_Mδ ≈ -0.35`, `C_Mα ≈ -0.1` (aerodynamic centre offset from mid-chord).

- [ ] **Step 1.1: Create `src/control/linear_aero.py`**

```python
"""
Quasi-steady Theodorsen linear aerodynamic model for Phase 1 Greedy controller.

C_L = 2π*(α + W/U + ḣ/U) + C_Lδ * δ_rad
C_M = C_Mα * α + C_Mδ * δ_rad

δ is in degrees (same convention as rest of codebase); converted internally.
"""
import numpy as np

# Flap effectiveness coefficients (30% chord trailing-edge flap, thin airfoil)
C_La    = 2.0 * np.pi    # lift-curve slope [rad⁻¹]
C_Ldelta = -0.7          # flap lift coefficient [rad⁻¹]  (negative: TE down → C_L up in LDNet sign)
C_Ma    = -0.1           # pitch moment vs alpha [rad⁻¹]
C_Mdelta = -0.35         # pitch moment vs flap [rad⁻¹]


def predict(x, delta_deg, W, U):
    """
    Quasi-steady linear aerodynamic prediction.

    Parameters
    ----------
    x : array-like, shape (4,)  — [h, ḣ, α, α̇]  (SI units: m, m/s, rad, rad/s)
    delta_deg : float            — flap deflection [degrees]
    W : float                   — gust vertical velocity [m/s]
    U : float                   — freestream velocity [m/s]

    Returns
    -------
    C_L : float
    C_M : float
    """
    h, hd, a, ad = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    delta_rad = np.deg2rad(float(delta_deg))
    W = float(W)
    U = max(float(U), 1.0)  # avoid division by zero

    C_L = C_La * (a + W / U + hd / U) + C_Ldelta * delta_rad
    C_M = C_Ma * a + C_Mdelta * delta_rad
    return float(C_L), float(C_M)
```

- [ ] **Step 1.2: Verify the model manually**

Run in Python (or a quick script):
```python
import sys; sys.path.insert(0, 'src')
from control.linear_aero import predict
# At trim (no gust, no flap, zero states): C_L should be ~0
CL, CM = predict([0,0,0,0], 0.0, 0.0, 80.0)
print(CL, CM)   # expected: 0.0, 0.0  ✓

# With alpha=0.01 rad, no gust, no flap
CL, CM = predict([0,0,0.01,0], 0.0, 0.0, 80.0)
print(CL, CM)   # expected: CL ≈ 0.0628, CM ≈ -0.001
```

- [ ] **Step 1.3: Commit**

```bash
git add src/control/linear_aero.py
git commit -m "feat: add quasi-steady Theodorsen linear aero model for Phase 1 Greedy"
```

---

## Task 2: PID Controller

**Files:**
- Create: `src/control/pid.py`

The PID acts on heave and pitch independently. It must expose `.solve(x_hat, z_hat, W_hat)` to match the interface used in `run_simulation()` (see `src/control/run_controller.py:155-158`).

- [ ] **Step 2.1: Create `src/control/pid.py`**

```python
"""
PID controller for aeroelastic gust load alleviation.

Two-channel PD (no integrator to avoid wind-up) on heave and pitch.
Interface: .solve(x_hat, z_hat, W_hat) -> delta [degrees]
Matches the LQR interface used in run_simulation().
"""
import numpy as np


class PIDController:
    """
    Proportional-derivative controller on heave and pitch.

    Parameters
    ----------
    Kp_h, Kd_h : float  — proportional and derivative gains on heave h [m]
    Kp_a, Kd_a : float  — proportional and derivative gains on pitch α [rad]
    delta_max   : float  — saturation limit [degrees]
    delta_dot_max : float — rate limit [degrees/s]
    DT          : float  — timestep [s], used for rate limiting
    """

    def __init__(self, Kp_h=0.0, Kd_h=0.0, Kp_a=0.0, Kd_a=0.0,
                 delta_max=20.0, delta_dot_max=100.0, DT=0.01):
        self.Kp_h = float(Kp_h)
        self.Kd_h = float(Kd_h)
        self.Kp_a = float(Kp_a)
        self.Kd_a = float(Kd_a)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self.DT = float(DT)
        self._delta_prev = 0.0

    def solve(self, x_hat, z_hat, W_hat=0.0):
        """
        Compute control command.

        Parameters
        ----------
        x_hat : array-like, shape (4,)  — [h, ḣ, α, α̇]
        z_hat : ignored (no aero model)
        W_hat : ignored (no aero model)

        Returns
        -------
        delta : float  — flap deflection [degrees]
        """
        h, hd, a, ad = float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3])
        delta_raw = (self.Kp_h * h + self.Kd_h * hd +
                     self.Kp_a * a + self.Kd_a * ad)

        # Rate limit
        delta_dot = (delta_raw - self._delta_prev) / self.DT
        if abs(delta_dot) > self.delta_dot_max:
            delta_raw = self._delta_prev + np.sign(delta_dot) * self.delta_dot_max * self.DT

        # Saturate
        delta = float(np.clip(delta_raw, -self.delta_max, self.delta_max))
        self._delta_prev = delta
        return delta

    def reset(self):
        self._delta_prev = 0.0
```

- [ ] **Step 2.2: Verify sign and saturation**

```python
import sys; sys.path.insert(0, 'src')
from control.pid import PIDController
pid = PIDController(Kp_h=500, Kd_h=50, Kp_a=200, Kd_a=20, DT=0.01)

# Positive h should produce a negative (or bounded) delta to push h back to 0
d = pid.solve([0.005, 0.0, 0.0, 0.0], None, 0.0)
print(d)   # expected: Kp_h * 0.005 = 2.5 degrees (within ±20°)

# Saturation check
pid.reset()
d = pid.solve([0.1, 0.0, 0.0, 0.0], None, 0.0)
print(d)   # expected: 20.0 (saturated)
```

- [ ] **Step 2.3: Commit**

```bash
git add src/control/pid.py
git commit -m "feat: add PID controller with PD gains, saturation, and rate limiting"
```

---

## Task 3: Greedy-N1 Controller

**Files:**
- Create: `src/control/greedy.py`

The Greedy controller:
1. At each step, calls `linear_aero.predict(x_hat, δ, W_hat, U)` to get `(C_L, C_M)` for a candidate `δ`.
2. Converts `C_L, C_M` → forces → integrates one structural RK4 step → gets `x_next`.
3. Minimizes `Q_h * h_next² + Q_a * a_next² + R * δ²` over `δ ∈ [-delta_max, delta_max]`.
4. Uses `scipy.optimize.minimize_scalar` (golden section) for the 1D minimization.

The structural step mirrors what `run_simulation()` does: RK4 with `structural_rhs`.

- [ ] **Step 3.1: Create `src/control/greedy.py`**

```python
"""
Greedy one-step-ahead controller (N=1 MPC) for aeroelastic GLA.

At each timestep solves:
    min_{δ}  Q_h * h(k+1)² + Q_a * a(k+1)² + R * δ²
subject to |δ| ≤ delta_max

where x(k+1) is predicted by:
    1. aero_model.predict(x_hat, δ, W_hat, U) → C_L, C_M
    2. one RK4 step of structural_rhs

Phase 1: aero_model is LinearAeroModel (Theodorsen).
Phase 2: swap for LDNetModel — only the predict() call changes.

Interface: .solve(x_hat, z_hat, W_hat) -> delta [degrees]
"""
import numpy as np
from scipy.optimize import minimize_scalar
from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA


# Aero → force conversion constants (same as run_simulation)
_RHO     = 1.225   # [kg/m³]
_S_REF   = 0.05    # [m²]
_C_REF   = 1.0     # [m]


class GreedyN1Controller:
    """
    One-step greedy optimal controller.

    Parameters
    ----------
    aero_predict : callable(x, delta_deg, W, U) -> (C_L, C_M)
        Aerodynamic prediction function. Use LinearAeroModel.predict for Phase 1,
        or wrap LDNetModel.step for Phase 2.
    U_INF   : float  — freestream velocity [m/s]
    DT      : float  — timestep [s]
    Q_h     : float  — cost weight on heave h [m]
    Q_a     : float  — cost weight on pitch α [rad]
    R       : float  — cost weight on control effort δ [deg]
    delta_max      : float  — saturation limit [degrees]
    delta_dot_max  : float  — rate limit [degrees/s]
    """

    def __init__(self, aero_predict, U_INF=80.0, DT=0.01,
                 Q_h=1e4, Q_a=1e4, R=1.0,
                 delta_max=20.0, delta_dot_max=100.0):
        self.aero_predict  = aero_predict
        self.U_INF         = float(U_INF)
        self.DT            = float(DT)
        self.Q_h           = float(Q_h)
        self.Q_a           = float(Q_a)
        self.R             = float(R)
        self.delta_max     = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self._delta_prev   = 0.0

        q_dyn = 0.5 * _RHO * U_INF**2 * _S_REF
        self._q_dyn = q_dyn

    def _predict_next_state(self, x_hat, delta_deg, W_hat):
        """One RK4 step of structural dynamics with linear aero prediction."""
        C_L, C_M = self.aero_predict(x_hat, delta_deg, W_hat, self.U_INF)
        Fy = self._q_dyn * C_L
        Mz = self._q_dyn * C_M * _C_REF

        t = 0.0  # time unused in structural_rhs (no time-varying params)
        def rhs(s):
            return np.array(structural_rhs(t, s, Fy, Mz, 0.0, 0.0))

        x = np.array(x_hat, dtype=float)
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * self.DT * k1)
        k3 = rhs(x + 0.5 * self.DT * k2)
        k4 = rhs(x + self.DT * k3)
        x_next = x + (self.DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def _cost(self, delta_deg, x_hat, W_hat):
        x_next = self._predict_next_state(x_hat, delta_deg, W_hat)
        h_next = x_next[0]
        a_next = x_next[2]
        return self.Q_h * h_next**2 + self.Q_a * a_next**2 + self.R * delta_deg**2

    def solve(self, x_hat, z_hat=None, W_hat=0.0):
        """
        Compute optimal one-step control.

        Parameters
        ----------
        x_hat : array-like, shape (4,)  — [h, ḣ, α, α̇]
        z_hat : ignored in Phase 1 (linear aero has no latent state)
        W_hat : float  — estimated gust velocity [m/s]

        Returns
        -------
        delta : float  — flap deflection [degrees]
        """
        # Apply rate limit to search bounds
        dot_limit = self.delta_dot_max * self.DT
        lb = max(-self.delta_max, self._delta_prev - dot_limit)
        ub = min( self.delta_max, self._delta_prev + dot_limit)

        if lb >= ub:
            return float(np.clip(self._delta_prev, -self.delta_max, self.delta_max))

        result = minimize_scalar(
            self._cost,
            bounds=(lb, ub),
            method='bounded',
            args=(x_hat, float(W_hat)),
            options={'xatol': 0.01}
        )
        delta = float(np.clip(result.x, -self.delta_max, self.delta_max))
        self._delta_prev = delta
        return delta

    def reset(self):
        self._delta_prev = 0.0
```

- [ ] **Step 3.2: Verify optimizer finds a non-trivial solution**

```python
import sys; sys.path.insert(0, 'src')
import numpy as np
from control.linear_aero import predict
from control.greedy import GreedyN1Controller

greedy = GreedyN1Controller(predict, U_INF=80.0, DT=0.01, Q_h=1e4, Q_a=1e4, R=1.0)

# With a positive gust (W=30 m/s), controller should generate a non-zero delta
x_hat = np.array([0.001, 0.01, 0.002, 0.05])
d = greedy.solve(x_hat, None, W_hat=30.0)
print(f"delta = {d:.3f} deg")   # expected: non-zero (negative to counteract lift increase)
assert abs(d) > 0.01, "Greedy returned near-zero delta — cost may be dominated by R"
assert abs(d) <= 20.0, "Greedy violated saturation"
print("OK")
```

- [ ] **Step 3.3: Commit**

```bash
git add src/control/greedy.py
git commit -m "feat: add Greedy-N1 one-step optimal controller with linear aero (Phase 1)"
```

---

## Task 4: Comparison Script

**Files:**
- Create: `src/PIDvsGreedy.py`

This script runs three simulations (OL / PID / Greedy) using the existing `run_simulation()` harness, then produces plots and a metrics table. It follows the same structure as `src/OLvsLQR.py`.

**Key design decisions:**
- Uses `observer='true_state'` for simplicity (avoids needing LDNet for the observer in Phase 1, since LDNet is not yet trained). This gives an "ideal sensor" scenario that isolates the controller effect.
- The existing `run_simulation()` needs `aero_model` — we still pass the real LDNet model for the **true system** (ground truth). The Greedy controller uses `linear_aero.predict` internally. This is intentional: the true physics use LDNet (or CFD), the controller only knows the linear model.

**Wait** — in Phase 1 LDNet is not trained yet. So we need a way to run the true system too. Two options:
  a) Use the existing (possibly untrained/random) LDNet as "true system" — only for testing the pipeline.
  b) Use a different structural-only simulation for the "true system".

Since the goal is to **test the pipeline**, option (a) is fine — we use whatever weights are in `models/` as the true system, even if they produce unrealistic aero forces. The comparison script will work correctly regardless of model quality.

- [ ] **Step 4.1: Create `src/PIDvsGreedy.py`**

```python
#!/usr/bin/env python3
"""
Comparison script: Open Loop / PID / Greedy-N1.

Phase 1: uses linear Theodorsen model inside Greedy controller.
         True system uses LDNet (or whatever weights are in models/).
         observer='true_state' (ideal sensors — isolates controller effect).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aerodynamics.model import LDNetModel as AeroModel
from structural.smd import get_space_state_matrices
from control.run_controller import run_simulation
from control.pid import PIDController
from control.greedy import GreedyN1Controller
from control.linear_aero import predict as linear_predict

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
U_INF = 80.0
T_END = 3.0
DT    = 0.01

GUST_W0  = 60.0   # peak gust [m/s]
GUST_DUR = 1.0    # gust duration [s]

PID_KP_H = 500.0   # proportional gain on h
PID_KD_H = 50.0    # derivative gain on ḣ
PID_KP_A = 200.0   # proportional gain on α
PID_KD_A = 20.0    # derivative gain on α̇

GREEDY_Q_H = 1.0 / 0.004**2   # weight on h  (normalised by expected peak)
GREEDY_Q_A = 1.0 / 0.008**2   # weight on α
GREEDY_R   = 1.0 / 5.0**2     # weight on δ

DELTA_MAX     = 20.0    # [°]
DELTA_DOT_MAX = 100.0   # [°/s]

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
print("Loading aerodynamic model (used as true system)...")
models_dir = Path(__file__).parent.parent / 'models_cluster'
if not models_dir.exists():
    models_dir = Path(__file__).parent.parent / 'models'
aero_model = AeroModel(str(models_dir))
print(f"  LDNet: {aero_model.num_latent_states} latent state(s)")

A_s, B_s, _, _ = get_space_state_matrices()

def gust_profile(t):
    if 0.0 <= t <= GUST_DUR:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / GUST_DUR))
    return 0.0

# ─────────────────────────────────────────────────────────────
# CONTROLLERS
# ─────────────────────────────────────────────────────────────
pid = PIDController(
    Kp_h=PID_KP_H, Kd_h=PID_KD_H,
    Kp_a=PID_KP_A, Kd_a=PID_KD_A,
    delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX, DT=DT
)

greedy = GreedyN1Controller(
    aero_predict=linear_predict,
    U_INF=U_INF, DT=DT,
    Q_h=GREEDY_Q_H, Q_a=GREEDY_Q_A, R=GREEDY_R,
    delta_max=DELTA_MAX, delta_dot_max=DELTA_DOT_MAX
)

# ─────────────────────────────────────────────────────────────
# SIMULATIONS
# ─────────────────────────────────────────────────────────────
print("\nRunning Open Loop...")
res_ol = run_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                        gust_profile=gust_profile, observer='true_state')

print("Running PID...")
pid.reset()
res_pid = run_simulation(U_INF, T_END, DT, aero_model, pid, A_s, B_s,
                         gust_profile=gust_profile, observer='true_state')

print("Running Greedy-N1...")
greedy.reset()
res_g = run_simulation(U_INF, T_END, DT, aero_model, greedy, A_s, B_s,
                       gust_profile=gust_profile, observer='true_state')

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def amplitude(arr):
    return (arr.max() - arr.min()) / 2.0

_W_END = GUST_DUR + 0.5

def amp_window(res, t_end):
    mask = (res['t'] >= 0.0) & (res['t'] <= t_end)
    return {k: amplitude(res[k][mask])
            for k in ('h', 'a', 'h_ddot', 'a_ddot', 'C_L', 'C_M')}

def actuation_energy(res):
    return float(np.sum(res['delta']**2) * DT)

g_ol  = amp_window(res_ol,  _W_END)
g_pid = amp_window(res_pid, _W_END)
g_g   = amp_window(res_g,   _W_END)

print(f"\n── Gust window [0 – {_W_END:.1f}s] ─────────────────────────────────────────")
print(f"{'':12s}  {'OL':>10s}  {'PID':>10s}  {'PID Red%':>9s}  {'Greedy':>10s}  {'Gr Red%':>8s}")
for name in ('h', 'a', 'h_ddot', 'a_ddot', 'C_L', 'C_M'):
    ol, p, gr = g_ol[name], g_pid[name], g_g[name]
    r_pid = (ol - p)  / ol * 100 if ol > 0 else 0.0
    r_gr  = (ol - gr) / ol * 100 if ol > 0 else 0.0
    print(f"  {name:<10s}  {ol:10.5f}  {p:10.5f}  {r_pid:+8.1f}%  {gr:10.5f}  {r_gr:+7.1f}%")

print(f"\n  Actuation energy (Σδ²·dt):")
print(f"    OL:     {actuation_energy(res_ol):.4f}")
print(f"    PID:    {actuation_energy(res_pid):.4f}")
print(f"    Greedy: {actuation_energy(res_g):.4f}")

# Sanity checks
for name, res in [('PID', res_pid), ('Greedy', res_g)]:
    assert np.all(np.abs(res['delta']) <= DELTA_MAX + 1e-6), f"{name}: delta exceeds {DELTA_MAX}°"
    print(f"  [{name}] max |δ| = {np.max(np.abs(res['delta'])):.2f}°  ✓")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / 'results' / 'pid_vs_greedy'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OL  = dict(color='steelblue',  lw=1.5, alpha=0.85, label='Open loop')
PID = dict(color='darkorange',  lw=1.5, alpha=0.85, label='PID')
GR  = dict(color='seagreen',   lw=1.5, alpha=0.85, label='Greedy-N1')

def fmt(ax, ylabel, title=None):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

t = res_ol['t']

# Fig 1 — Structural state
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig.suptitle('Structural State  —  OL / PID / Greedy-N1', fontsize=13)
axes[0,0].plot(t, res_ol['h'],  **OL)
axes[0,0].plot(t, res_pid['h'], **PID)
axes[0,0].plot(t, res_g['h'],   **GR)
fmt(axes[0,0], 'h [m]', 'Heave displacement')
axes[0,1].plot(t, res_ol['hd'],  **OL)
axes[0,1].plot(t, res_pid['hd'], **PID)
axes[0,1].plot(t, res_g['hd'],   **GR)
fmt(axes[0,1], 'ḣ [m/s]', 'Heave velocity')
axes[1,0].plot(t, np.rad2deg(res_ol['a']),  **OL)
axes[1,0].plot(t, np.rad2deg(res_pid['a']), **PID)
axes[1,0].plot(t, np.rad2deg(res_g['a']),   **GR)
fmt(axes[1,0], 'α [°]', 'Pitch angle')
axes[1,1].plot(t, np.rad2deg(res_ol['ad']),  **OL)
axes[1,1].plot(t, np.rad2deg(res_pid['ad']), **PID)
axes[1,1].plot(t, np.rad2deg(res_g['ad']),   **GR)
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig1_state.png', dpi=150)
print(f"\n[OK] {OUT_DIR / 'fig1_state.png'}")

# Fig 2 — Control input
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle('Control Input  —  OL / PID / Greedy-N1', fontsize=13)
axes[0].plot(t, res_ol['delta'],  **OL)
axes[0].plot(t, res_pid['delta'], **PID)
axes[0].plot(t, res_g['delta'],   **GR)
axes[0].axhline( DELTA_MAX, color='k', ls=':', lw=0.8, label=f'±{DELTA_MAX}°')
axes[0].axhline(-DELTA_MAX, color='k', ls=':', lw=0.8)
fmt(axes[0], 'δ [°]', 'Flap deflection')
delta_rate_pid = np.gradient(res_pid['delta'], t)
delta_rate_g   = np.gradient(res_g['delta'],   t)
axes[1].plot(t, np.zeros_like(t), **OL)
axes[1].plot(t, delta_rate_pid,   **PID)
axes[1].plot(t, delta_rate_g,     **GR)
fmt(axes[1], 'δ̇ [°/s]', 'Flap rate')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_control.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig2_control.png'}")

# Fig 3 — Aerodynamic coefficients
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle('Aerodynamic Coefficients  —  OL / PID / Greedy-N1', fontsize=13)
axes[0].plot(t, res_ol['C_L'],  **OL)
axes[0].plot(t, res_pid['C_L'], **PID)
axes[0].plot(t, res_g['C_L'],   **GR)
fmt(axes[0], '$C_L$', 'Lift coefficient')
axes[1].plot(t, res_ol['C_M'],  **OL)
axes[1].plot(t, res_pid['C_M'], **PID)
axes[1].plot(t, res_g['C_M'],   **GR)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_aero.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig3_aero.png'}")

# Fig 4 — Accelerations
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle('Structural Accelerations  —  OL / PID / Greedy-N1', fontsize=13)
axes[0].plot(t, res_ol['h_ddot'],  **OL)
axes[0].plot(t, res_pid['h_ddot'], **PID)
axes[0].plot(t, res_g['h_ddot'],   **GR)
fmt(axes[0], 'ḧ [m/s²]', 'Heave acceleration')
axes[1].plot(t, np.rad2deg(res_ol['a_ddot']),  **OL)
axes[1].plot(t, np.rad2deg(res_pid['a_ddot']), **PID)
axes[1].plot(t, np.rad2deg(res_g['a_ddot']),   **GR)
fmt(axes[1], 'α̈ [°/s²]', 'Pitch acceleration')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig4_accels.png', dpi=150)
print(f"[OK] {OUT_DIR / 'fig4_accels.png'}")

plt.show()
```

- [ ] **Step 4.2: Commit**

```bash
git add src/PIDvsGreedy.py
git commit -m "feat: add PID vs Greedy-N1 comparison script with plots and metrics table"
```

---

## Task 5: End-to-End Verification

- [ ] **Step 5.1: Run the comparison script**

From the `src/` directory (or project root with `src` in path):
```bash
cd /home/marco/LDNet_OF
python src/PIDvsGreedy.py
```

Expected output (no crash, no NaN):
```
Loading aerodynamic model (used as true system)...
  LDNet: 1 latent state(s)
Running Open Loop...
Running PID...
Running Greedy-N1...
── Gust window [0 – 1.5s] ...
  h             ...
  a             ...
  ...
  [PID]    max |δ| = ...°  ✓
  [Greedy] max |δ| = ...°  ✓
[OK] results/pid_vs_greedy/fig1_state.png
[OK] results/pid_vs_greedy/fig2_control.png
[OK] results/pid_vs_greedy/fig3_aero.png
[OK] results/pid_vs_greedy/fig4_accels.png
```

- [ ] **Step 5.2: Verify physical plausibility**

Check the following in the plots:
1. OL shows growing h and α during gust (t=0 to 1 s)
2. PID delta is non-zero and changes sign (proportional + derivative action)
3. Greedy delta is non-zero and differs from PID (different control law)
4. Neither controller's delta exceeds ±20°
5. No NaN in any time series

- [ ] **Step 5.3: Final commit**

```bash
git add -A
git commit -m "feat: complete PID vs Greedy-N1 GLA comparison pipeline (Phase 1, linear aero)"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] PID controller with Kp/Kd, saturation, rate limit → Task 2
- [x] Greedy-N1 with linear aero, one-step optimization, scipy minimize_scalar → Task 3
- [x] Linear Theodorsen aero model → Task 1
- [x] Comparison script OL/PID/Greedy with metrics table and 4 plots → Task 4
- [x] Verification: bounds check, physical plausibility, end-to-end run → Task 5

**Interface compatibility:**
- `pid.solve(x_hat, z_hat, W_hat)` → float: matches `run_simulation()` LQR dispatch at line 157
- `greedy.solve(x_hat, z_hat, W_hat)` → float: same interface
- Neither has `solve_tf` → `run_simulation()` will take the LQR path (not MPC path) ✓

**Type consistency:**
- `linear_aero.predict(x, delta_deg, W, U)` called in `GreedyN1Controller._cost()` ✓
- `structural_rhs(t, s, Fy, Mz, 0.0, 0.0)` — signature matches `smd.py:43` ✓
- `run_simulation(..., observer='true_state')` — valid observer string per `run_controller.py:79` ✓
