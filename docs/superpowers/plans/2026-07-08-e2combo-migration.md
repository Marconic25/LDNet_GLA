# E2-Combo Pipeline Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `light/` from one-step optimal to the E2-combo pipeline (FusedPreviewSensor + MPCPreviewController N=8), re-run the CS-25 study comparing both controllers, and characterise the combo's white-noise robustness curve.

**Architecture:** `optimal.py` gains three self-contained additions (`dp45_batch`, `FusedPreviewSensor`, `MPCPreviewController`) appended after `OptimalController` (which is never touched); `run.py` gains `mode='combo'`; two new study scripts import the classes from `optimal.py` without copying them. All cluster runs follow the existing `run_axis.sh` + `launch_*.sh` + `status_*.sh` pattern. Every cluster sync is scp local→cluster; never pull back.

**Tech Stack:** Python 3, NumPy, SciPy (`solve_ivp` in dp45_batch), TensorFlow / LDNetAero; cluster via `ssh u10677113@10.78.18.100` + Apptainer `tensorflow_gpu.sif`; PYTHONNOUSERSITE=1 mandatory.

## Global Constraints

- `PYTHONNOUSERSITE=1` in every apptainer invocation (numpy 2.x in ~/.local breaks TF)
- `DAMULT=3` in every simulation (`structure.D_ALPHA` scaled once at import)
- `OMP_NUM_THREADS=3 TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1` when >2 concurrent jobs
- Never modify: `OptimalController`, `results_cs25/`, `light/noise/results/E2_*.npz`, `76/`, `clean/`
- `harness_noise.py`: only the getattr integrator fix (no other edits)
- Cluster commands containing `$` or redirects go via shell scripts on the cluster, not inline ssh
- Prefix `MSYS_NO_PATHCONV=1` on wsl.exe calls with /work/… paths
- Smoke test BEFORE every full run; verify anchor numbers before proceeding
- If any non-chaotic clean anchor deviates >1 pt from expectation, stop and debug (systematic-debugging skill)
- WSL repo root: `/home/marco/LDNet_OF`; cluster root: `/work/u10677113/LDNet_GLA`

---

## File map

| Path | Action | Responsible task |
|---|---|---|
| `light/noise/harness_noise.py` | Fix getattr integrator fallback | Task 0 |
| `light/optimal.py` | Append dp45_batch + FusedPreviewSensor + MPCPreviewController | Task 1 |
| `light/run.py` | Add mode='combo' + NH param; add LAM module-level | Task 2 |
| `light/tests/cs25_combo_study.py` | New: MODE-parametrised CS-25 driver | Task 3 |
| `light/tests/launch_cs25_combo.sh` | New: cluster launch script for combo+optimal rows | Task 3 |
| `light/tests/status_cs25_combo.sh` | New: cluster status script | Task 3 |
| `light/tests/cs25_combo_plots.py` | New: side-by-side comparison plots | Task 4 |
| `light/noise/noise_white_combo.py` | New: white-noise robustness of combo | Task 5 |
| `light/noise/launch_noise_white_combo.sh` | New: cluster launch script | Task 5 |
| `light/noise/status_noise_white_combo.sh` | New: cluster status script | Task 5 |
| `light/noise/plots_noise_white_combo.py` | New: CLred-vs-sigma figure | Task 5 |
| `light/noise/NOTES.md` | Append dp45 baseline table + W_combo results section | Tasks 0+6 |
| `light/results_cs25_combo/summary.md` | New: one-step vs combo comparison table | Task 4 |

---

## Task 0 – Fix harness_noise.py + document dp45 anchor discrepancy in NOTES.md

**Files:**
- Modify: `light/noise/harness_noise.py:118` (step integrator line)
- Modify: `light/noise/NOTES.md` (append integrator-migration section)

**Context:** `harness_noise.py` line 118 calls `structure.step_dp45` directly. The spec requires a `getattr` guard so the file works on a cluster that still has the old `step_rk4`. After syncing the local dp45 tree to the cluster, all results will use dp45 — this is acceptable and must be documented.

- [ ] **Step 1: Apply getattr guard to harness_noise.py**

In `light/noise/harness_noise.py`, find line 118:
```python
        x = structure.step_dp45(x, Fy, Mz, DT)
```
Replace with:
```python
        # f92d8975: step_rk4 removed locally; guard for cluster trees that still have it
        _step = getattr(structure, 'step_rk4', structure.step_dp45)
        x = _step(x, Fy, Mz, DT)
```

Note: the `_step` lookup should be done ONCE outside the loop. Move it to just after `aero.reset(dt=DT)` in `rollout()` (line ~93), before the for-loop:
```python
    _step_struct = getattr(structure, 'step_rk4', structure.step_dp45)
```
Then in the loop: `x = _step_struct(x, Fy, Mz, DT)`

- [ ] **Step 2: Append integrator-migration note to NOTES.md**

Append the following section at the end of `light/noise/NOTES.md`:

```markdown
---

# Integrator migration note (f92d8975 → dp45, 2026-07-08)

`structure.py` at commit f92d8975 replaced `step_rk4` with `step_dp45`
(Dormand-Prince RK45 via scipy.integrate.solve_ivp).  After syncing the local
tree to the cluster the cluster also uses dp45; the recorded E2/E2CC/E2-combo
results (all in `results/E2_*.npz`) were computed with `rk4_batch` in the MPC
horizon and `step_rk4` in the plant.  The new dp45 tree will not reproduce
those numbers bit-exactly — this is expected and **not a bug**.

## dp45 baseline anchors (home cell W30/Tg0.4, DAMULT=3) — TO BE FILLED

After running the smoke regression on cluster post-sync, fill in the dp45 values:

| check | rk4 value | dp45 value | delta |
|---|---|---|---|
| open cex0 | 0.4600 | TBD | TBD |
| one-step optimal R=3e-4 | +76.58% | TBD | TBD |
| one-step optimal R=1e-4 | +80.67% | TBD | TBD |
| combo oracle clean R=3e-4 | +80.5% | TBD | TBD |

Use the dp45 values as the new anchors for all subsequent studies.
If any non-chaotic number (open cex0, combo clean) differs by >1 pt, debug.
```

- [ ] **Step 3: Commit harness fix + NOTES update**
```bash
git add light/noise/harness_noise.py light/noise/NOTES.md
git commit -m "fix(harness_noise): getattr guard step_rk4→step_dp45; note integrator migration"
```

---

## Task 1 – Add dp45_batch + FusedPreviewSensor + MPCPreviewController to optimal.py

**Files:**
- Modify: `light/optimal.py` (append after line 122 / after `OptimalController.reset`)

**Interfaces produced:**
- `dp45_batch(x_b, Fy_b, Mz_b, dt) -> x_b_next` — batched (G,4) structural step
- `FusedPreviewSensor(rng, sigma_fun, Jmax, N, lam=0.0)` — `.wc_fun(i, Wt, Nsteps)` returns scalar, caches `.last` (N-array)
- `MPCPreviewController(aero, U, dt, rho, S, C, C_L_trim, N, R, R_du, G, delta_max, delta_dot_max)` — `.compute(state, w_seq)` returns delta, `.reset()`, `._delta_prev` attribute

- [ ] **Step 1: Add dp45_batch to optimal.py**

Append to `light/optimal.py` after the `OptimalController` class (after line 122):

```python

# ---------------------------------------------------------------------------
# Batched Dormand-Prince RK45 structural step — for MPCPreviewController horizon
# (verbatim of controllers_ref.dp45_batch; kept here so optimal.py is self-contained)
# ---------------------------------------------------------------------------

def dp45_batch(x_b, Fy_b, Mz_b, dt):
    """
    Advance G structural states by one step using the DP45 5th-order coefficients.

    Parameters
    ----------
    x_b   : (G, 4) array  — [h, hd, alpha, ad] for each candidate
    Fy_b  : (G,) array    — aerodynamic heave force per candidate
    Mz_b  : (G,) array    — aerodynamic pitch moment per candidate
    dt    : float         — time step [s]

    Returns
    -------
    x_next : (G, 4) array
    """
    import structure as _S

    def _rhs(xx):
        hd = xx[:, 1]; ad = xx[:, 3]
        rh = -Fy_b - _S.D_H * hd - _S.K_H * xx[:, 0]
        ra = Mz_b - _S.D_ALPHA * ad - _S.K_ALPHA * xx[:, 2]
        hdd = (_S.M_AA * rh - _S.M_HA * ra) / _S.DET
        add = (_S.M_HH * ra - _S.M_HA * rh) / _S.DET
        return np.stack([hd, hdd, ad, add], axis=1)

    k1 = _rhs(x_b)
    k2 = _rhs(x_b + dt * (1.0 / 5) * k1)
    k3 = _rhs(x_b + dt * (3.0 / 40 * k1 + 9.0 / 40 * k2))
    k4 = _rhs(x_b + dt * (44.0 / 45 * k1 - 56.0 / 15 * k2 + 32.0 / 9 * k3))
    k5 = _rhs(x_b + dt * (19372.0 / 6561 * k1 - 25360.0 / 2187 * k2
                           + 64448.0 / 6561 * k3 - 212.0 / 729 * k4))
    k6 = _rhs(x_b + dt * (9017.0 / 3168 * k1 - 355.0 / 33 * k2
                           + 46732.0 / 5247 * k3 + 49.0 / 176 * k4
                           - 5103.0 / 18656 * k5))
    return x_b + dt * (35.0 / 384 * k1 + 500.0 / 1113 * k3
                       + 125.0 / 192 * k4 - 2187.0 / 6784 * k5 + 11.0 / 84 * k6)
```

- [ ] **Step 2: Add FusedPreviewSensor to optimal.py**

Append after `dp45_batch`:

```python

# ---------------------------------------------------------------------------
# T1 DLR massed-measurement fusion sensor — port of e2_combo.FusedSensor
# Reference: light/noise/NOTES.md §E2-combo, §E2CC
# ---------------------------------------------------------------------------

class FusedPreviewSensor:
    """
    Rolling inverse-variance fusion database that delivers a fused N-node
    preview VECTOR at each time step.

    Mechanics (verbatim of e2_combo.FusedSensor, kept in sync by eye):
    - At step i, measure node m = i+j for j in 1..Jmax with sigma_fun(j).
    - Accumulate (num, den) running sums per spatial node (inverse-variance).
    - Pre-warm with Jmax-1 virtual steps before t=0 so the database is full
      at the first real step.
    - Optional Tikhonov smoothness penalty (lam) via second-difference operator.
    - Raw samples NEVER clamped; output (self.last) clamped to >= 0.
    - self.last[k] = fused estimate of W at node i+(k+1), k=0..N-1.
    - wc_fun() returns float(self.last[0]) — the scalar the harness logs as Wc.

    Parameters
    ----------
    rng       : numpy Generator — fresh per rollout (rng 100+seed)
    sigma_fun : callable j -> sigma [m/s] — noise of a j-step-ahead measurement
    Jmax      : int — lookahead window (50 for the E2 winner)
    N         : int — preview horizon length (8 for the E2 winner)
    lam       : float — Tikhonov regularisation (0=none, 1=DLR-realistic arm)
    """

    def __init__(self, rng, sigma_fun, Jmax, N, lam=0.0):
        self.rng = rng
        self.js = np.arange(1, int(Jmax) + 1)
        self.sigs = np.array([float(sigma_fun(j)) for j in self.js])
        self.inv2 = 1.0 / self.sigs ** 2
        self.Jmax = int(Jmax)
        self.N = int(N)
        self.lam = float(lam)
        self.num = None
        self.den = None
        self.last = None

    def wc_fun(self, i, Wt, Nsteps):
        """
        Harness-compatible wc_fun signature: called at step i, returns scalar.
        Side-effect: sets self.last to the N-node fused preview vector.
        """
        js, sigs, inv2 = self.js, self.sigs, self.inv2
        if self.num is None:
            self.num = np.zeros(Nsteps)
            self.den = np.zeros(Nsteps)
            for ii in range(-(self.Jmax - 1), 0):
                mm = ii + js
                keep = mm >= 0
                mk = np.minimum(mm[keep], Nsteps - 1)
                yk = Wt[mk] + self.rng.normal(0.0, sigs[keep])
                np.add.at(self.num, mk, yk * inv2[keep])
                np.add.at(self.den, mk, inv2[keep])

        m_idx = np.minimum(i + js, Nsteps - 1)
        y = Wt[m_idx] + self.rng.normal(0.0, sigs)
        np.add.at(self.num, m_idx, y * inv2)
        np.add.at(self.den, m_idx, inv2)

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

    def reset(self):
        """Clear database state — call before reuse on a new gust (or create fresh)."""
        self.num = None
        self.den = None
        self.last = None
```

- [ ] **Step 3: Add MPCPreviewController to optimal.py**

Append after `FusedPreviewSensor`:

```python

# ---------------------------------------------------------------------------
# T7 N-step constant-flap MPC with fused preview — port of e2_combo.MPCPrevRdu
# Reference: light/noise/NOTES.md §E2-combo verdict
#
# Recorded results (cluster rk4 tree, W30/Tg0.4, DAMULT=3, N=8, R=3e-4, R_du=0):
#   clean oracle       +80.5%
#   flat sigma=2%*W0   +80.5 [80.4,80.6] 0/6 flags
#   DLR raw 1-3 m/s    +81.1 [80.3,83.7] 0/6 flags  (lam=1)
#   W10/Tg0.7 clean    +93.5%,  W30/Tg0.7 clean  +91.5%
# dp45 anchors: TBD (fill after Task 0 regression run).
# ---------------------------------------------------------------------------

class MPCPreviewController:
    """
    N-step constant-flap MPC that consumes a fused N-node gust preview vector.

    Cost over horizon: J = R*delta^2 + R_du*(delta-delta_prev)^2
                           + sum_{k=0}^{N-1} (C_L_k - trim)^2
    wnext convention: horizon step k is evaluated at w_seq[k] = W(t+(k+1)*dt).
    No causal sign gate (unlike OptimalController).
    Rate limit enforced; argmin from a G-point grid (no refine needed with N>1).

    Parameters
    ----------
    aero          : LDNetAero — shared plant model
    U             : float  freestream velocity [m/s]  (default 80)
    dt            : float  time step [s]  (default 0.002)
    rho           : float  air density [kg/m^3]  (default 1.225)
    S             : float  reference area [m^2]  (default 0.05)
    C             : float  chord [m]  (default 1.0)
    C_L_trim      : float  trim C_L  (default 0.0; pass CLTRIM from run.py)
    N             : int    preview horizon length  (default 8)
    R             : float  flap-effort weight  (default 3e-4)
    R_du          : float  move-suppression weight  (default 0 — keep at 0)
    G             : int    flap grid points  (default 161)
    delta_max     : float  flap limit [deg]  (default 14)
    delta_dot_max : float  rate limit [deg/s]  (default 300)

    API
    ---
    compute(state, w_seq) -> delta [deg]
        state  : (h, hd, alpha, ad)
        w_seq  : array-like length >= N, w_seq[k] = W(t+(k+1)*dt)
    reset()  -> clears _delta_prev
    _delta_prev  : float — last commanded delta (read/written by run.py LPF chain)
    """

    def __init__(self, aero, U=80.0, dt=0.002, rho=1.225, S=0.05, C=1.0,
                 C_L_trim=0.0, N=8, R=3e-4, R_du=0.0, G=161,
                 delta_max=14.0, delta_dot_max=300.0):
        self.aero = aero
        self.U = float(U)
        self.dt = float(dt)
        self.q = 0.5 * float(rho) * float(U) ** 2 * float(S)
        self.C = float(C)
        self.lam = float(aero._z_leak)
        self.C_L_trim = float(C_L_trim)
        self.N = int(N)
        self.R = float(R)
        self.R_du = float(R_du)
        self.G = int(G)
        self.delta_max = float(delta_max)
        self.delta_dot_max = float(delta_dot_max)
        self._dg = np.linspace(-self.delta_max, self.delta_max, self.G)
        self._delta_prev = 0.0

    def compute(self, state, w_seq):
        """
        Return optimal constant-flap deflection [deg].

        w_seq : array-like of length N — fused/oracle gust preview
                w_seq[k] = W(t + (k+1)*dt)  (wnext convention)
        """
        aero = self.aero
        dg = self._dg
        G = self.G
        reach = self.delta_dot_max * self.dt
        ratem = np.abs(dg - self._delta_prev) <= reach + 1e-9

        z_b = np.tile(np.asarray(aero._z, float).reshape(1, -1), (G, 1))
        x_b = np.tile(np.asarray(state, float).reshape(1, -1), (G, 1))
        J = self.R * dg ** 2 + self.R_du * (dg - self._delta_prev) ** 2

        for k in range(self.N):
            Wk = float(w_seq[k]) if k < len(w_seq) else 0.0
            CL, CM, z_new = aero.batch_step(z_b, x_b, dg, Wk, self.U, self.dt)
            z_b = z_new - self.lam * z_b
            Fy = self.q * CL
            Mz = self.q * CM * self.C
            x_b = dp45_batch(x_b, Fy, Mz, self.dt)
            J = J + (CL - self.C_L_trim) ** 2

        J = np.where(ratem, J, np.inf)
        d = float(dg[int(np.argmin(J))])
        d = float(np.clip(d, self._delta_prev - reach, self._delta_prev + reach))
        d = float(np.clip(d, -self.delta_max, self.delta_max))
        self._delta_prev = d
        return d

    def reset(self):
        """Reset flap state — call before each new simulation run."""
        self._delta_prev = 0.0
```

- [ ] **Step 4: Verify optimal.py is importable and OptimalController is untouched**

From `/home/marco/LDNet_OF/` (WSL) or via cluster smoke:
```bash
# Quick local import test (no TF needed — just check parse)
python3 -c "import ast; ast.parse(open('light/optimal.py').read()); print('parse OK')"
# Verify OptimalController is still present and unchanged
python3 -c "from light.optimal import OptimalController, FusedPreviewSensor, MPCPreviewController, dp45_batch; print('imports OK')"
```
Expected: `parse OK` then `imports OK`.

- [ ] **Step 5: Commit optimal.py additions**
```bash
git add light/optimal.py
git commit -m "feat(optimal): add dp45_batch, FusedPreviewSensor, MPCPreviewController — E2-combo pipeline"
```

---

## Task 2 – Add mode='combo' to run.py

**Files:**
- Modify: `light/run.py`

**Changes needed:**
1. Add `LAM = float(aero._z_leak)` after `CLTRIM` definition (module level)
2. Add `from optimal import OptimalController, MPCPreviewController` (already imports OptimalController; add MPCPreviewController)
3. Add `NH=8, R_du=0.0` parameters to `simulate()`
4. Rename `N = int(round(TEND/DT)) + 1` → `Nsteps = ...` inside `simulate()` (needed because `NH` is now a parameter named N in the original)
5. Add `elif mode == 'combo':` branches in controller construction and the loop

**CRITICAL:** `mode='open'` and `mode='optimal'` must remain bit-identical. Only ADD new branches, never touch existing ones.

- [ ] **Step 1: Add LAM and import MPCPreviewController**

In `light/run.py`, find:
```python
from optimal import OptimalController
```
Replace with:
```python
from optimal import OptimalController, MPCPreviewController
```

Then find:
```python
CLTRIM = float(aero.predict(X0, 0., 0., U)[0])
```
Replace with:
```python
CLTRIM = float(aero.predict(X0, 0., 0., U)[0])
LAM    = float(aero._z_leak)
```

- [ ] **Step 2: Add NH, R_du params to simulate(); rename internal N → Nsteps**

Find the function signature:
```python
def simulate(mode, W0, Tg, TEND=3.0, R=3e-4, DLPF=0.0, DMAX=14., NGRID=161):
```
Replace with:
```python
def simulate(mode, W0, Tg, TEND=3.0, R=3e-4, DLPF=0.0, DMAX=14., NGRID=161, NH=8, R_du=0.0):
```

Inside `simulate()`, find the first line:
```python
    N  = int(round(TEND/DT)) + 1
    ts = np.arange(N)*DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])
```
Replace with:
```python
    Nsteps = int(round(TEND/DT)) + 1
    ts = np.arange(Nsteps)*DT
    Wt = np.array([gust(t, W0, Tg) for t in ts])
```

Then find every remaining use of bare `N` inside `simulate()` and replace with `Nsteps`:
- `ctrl = None` block: unchanged
- `if mode == 'optimal':` block: unchanged (uses local `ctrl`, not `N`)
- `for i in range(N):` → `for i in range(Nsteps):`
- `Wn = float(Wt[i+1]) if i + 1 < N else 0.0` → `Wn = float(Wt[i+1]) if i + 1 < Nsteps else 0.0`

- [ ] **Step 3: Add combo controller construction**

Find:
```python
    aero.reset(dt=DT)
    ctrl = None
    if mode == 'optimal':
        ctrl = OptimalController(
            aero, U=U, dt=DT, R=R, n_grid=NGRID,
            C_L_trim=CLTRIM, delta_max=DMAX, delta_dot_max=300.)
        ctrl.reset()
```
Replace with:
```python
    aero.reset(dt=DT)
    ctrl = None
    if mode == 'optimal':
        ctrl = OptimalController(
            aero, U=U, dt=DT, R=R, n_grid=NGRID,
            C_L_trim=CLTRIM, delta_max=DMAX, delta_dot_max=300.)
        ctrl.reset()
    elif mode == 'combo':
        ctrl = MPCPreviewController(
            aero, U=U, dt=DT, rho=RHO, S=S, C=C,
            C_L_trim=CLTRIM, N=NH, R=R, R_du=R_du,
            G=NGRID, delta_max=DMAX, delta_dot_max=300.)
        ctrl.reset()
```

- [ ] **Step 4: Add combo branch in the simulation loop**

Find inside the loop:
```python
        if mode == 'optimal':
            t0 = time.perf_counter()
            de_raw = ctrl.compute(x, Wi, Wn)
            comp_t += time.perf_counter()-t0; comp_n += 1
        else:
            de_raw = 0.0
```
Replace with:
```python
        if mode == 'optimal':
            t0 = time.perf_counter()
            de_raw = ctrl.compute(x, Wi, Wn)
            comp_t += time.perf_counter()-t0; comp_n += 1
        elif mode == 'combo':
            t0 = time.perf_counter()
            lo = i + 1; hi = min(i + 1 + NH, Nsteps)
            w_seq = np.zeros(NH)
            w_seq[:hi - lo] = Wt[lo:hi]
            de_raw = ctrl.compute(x, w_seq)
            comp_t += time.perf_counter()-t0; comp_n += 1
        else:
            de_raw = 0.0
```

Also update the `if mode != 'open':` block to handle combo:
```python
        # 2nd-order LPF smoothing (only when DLPF > 0)
        if mode != 'open':
```
This block is already correct — `combo` is not `'open'` so LPF and `ctrl._delta_prev` update apply. Verify the block is intact:
```python
        if mode != 'open':
            if DLPF > 0.0:
                de_f  = DLPF*de_f  + (1.0-DLPF)*de_raw
                de_f2 = DLPF*de_f2 + (1.0-DLPF)*de_f
                de = de_f2
            else:
                de = de_raw
            ctrl._delta_prev = de
        else:
            de = 0.0
```
This is unchanged. `MPCPreviewController._delta_prev` is the attribute that holds the rate-limiter state.

- [ ] **Step 5: Update __main__ block to allow COMBO mode**

Find:
```python
    OL  = simulate('open',    W0, TG, **kw)
    OPT = simulate('optimal', W0, TG, **kw)
    mo  = metrics(OPT, OL, TG)
```
Replace with:
```python
    MODE  = os.environ.get('MODE', 'optimal')
    NH    = int(os.environ.get('NH',   '8'))
    R_DU  = float(os.environ.get('R_DU', '0.0'))

    OL  = simulate('open', W0, TG, **kw)
    RES = simulate(MODE,   W0, TG, NH=NH, R_du=R_DU, **kw)
    mo  = metrics(RES, OL, TG)
```
Also update the print to show mode:
```python
    print(f'W0={W0:.1f} Tg={TG:.2f} mode={MODE} | R={cfg["R"]:g} DLPF={cfg["DLPF"]:g}', flush=True)
    print(f'  {MODE:7s}: CLexc {mo["exo"]:.3f}->{mo["clexc"]:.3f} ({mo["clred"]:+.0f}%)'
          f'  flap_max={mo["flap_max"]:.1f}  adot_RMS={mo["adrms"]:.3f}'
          f'  {"EXPLODE: "+mo["flag"] if mo["flag"] else "stable"}', flush=True)
```

- [ ] **Step 6: Verify parse and import**
```bash
python3 -c "import ast; ast.parse(open('light/run.py').read()); print('parse OK')"
```
Expected: `parse OK`

- [ ] **Step 7: Commit run.py**
```bash
git add light/run.py
git commit -m "feat(run): add mode='combo' (MPCPreviewController oracle, NH=8) to simulate()"
```

---

## Task 3 – dp45 baseline regression on cluster + create CS-25 combo study

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

After smoke passes (cex0 matches dp45 anchor ±0.001, CLred plausible), launch full:
```bash
ssh -n u10677113@10.78.18.100 '/work/u10677113/LDNet_GLA/light/tests/launch_cs25_combo.sh full'
```

Expected run time: combo rows ~4h each (6 cells × 5 R values × ~8 min), optimal rows ~30 min each. All 6 jobs run in parallel. Total wall time ≈ 4h.

- [ ] **Step 9: Commit new scripts**
```bash
git add light/tests/cs25_combo_study.py light/tests/launch_cs25_combo.sh \
        light/tests/status_cs25_combo.sh light/smoke_dp45_baseline.sh
git commit -m "feat(cs25): MODE-parametrised cs25_combo_study.py + cluster scripts"
```

---

## Task 4 – CS-25 comparison plots and summary

**Files:**
- Create: `light/tests/cs25_combo_plots.py`
- Create (generated): `light/results_cs25_combo/summary.md`

**Prerequisite:** Task 3 cluster jobs DONE — `results_cs25_combo/traces_W{10,20,30}.npz` and (optionally) the dp45 re-run of optimal in `results_cs25/traces_W{10,20,30}.npz`.

- [ ] **Step 1: Write cs25_combo_plots.py**

Create `light/tests/cs25_combo_plots.py`:

```python
"""
CS-25 comparison plots: one-step optimal vs E2-combo.

Reads:
  results_cs25/traces_W{10,20,30}.npz       (one-step optimal)
  results_cs25_combo/traces_W{10,20,30}.npz (combo)

Generates in results_cs25_combo/:
  heatmap_clred_combo.png   — side-by-side CLred heatmap (optimal | combo)
  summary_lines_combo.png   — CLred vs Tg, two lines per W0
  summary.md                — per-cell comparison table

Run after both studies are complete:
  python3 -s -u cs25_combo_plots.py
"""
import csv, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

U = 80.0
W_LIST  = [10, 20, 30]
TG_LIST = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]
_THIS   = os.path.dirname(os.path.abspath(__file__))
DIR_OPT   = os.path.join(_THIS, '..', 'results_cs25')
DIR_COMBO = os.path.join(_THIS, '..', 'results_cs25_combo')
os.makedirs(DIR_COMBO, exist_ok=True)


def H_ft(Tg): return U * Tg / 2.0 / 0.3048
def kred(Tg): return np.pi / (U * Tg)


def load_row(d, w):
    path = os.path.join(d, f'traces_W{w}.npz')
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


rows = []
for w in W_LIST:
    d_opt   = load_row(DIR_OPT,   w)
    d_combo = load_row(DIR_COMBO, w)
    for Tg in TG_LIST:
        tag = f'Tg{Tg:.2f}'
        row = dict(W0=w, Tg=Tg, H_ft=round(H_ft(Tg)), k=round(kred(Tg), 3))
        for label, d in [('opt', d_opt), ('combo', d_combo)]:
            if d is None:
                row[f'{label}_clred'] = float('nan')
                row[f'{label}_Rstar'] = float('nan')
                row[f'{label}_fmax']  = float('nan')
                row[f'{label}_pitch'] = float('nan')
                row[f'{label}_flag']  = '?'
            else:
                jb   = int(d[f'{tag}_jb'])
                row[f'{label}_clred'] = round(float(d[f'{tag}_clred']), 1)
                row[f'{label}_Rstar'] = float(d[f'{tag}_Rstar'])
                row[f'{label}_fmax']  = round(float(d[f'{tag}_fmax'][jb]), 1)
                row[f'{label}_pitch'] = round(float(d[f'{tag}_pitch']), 2)
                row[f'{label}_flag']  = str(d[f'{tag}_flags'][jb])
        rows.append(row)

# --- heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), sharey=True)
for ax, label, title in [
    (axes[0], 'opt',   'One-step optimal (wnext+refine)'),
    (axes[1], 'combo', 'E2-combo (FusedSensor + MPC N=8)'),
]:
    M  = np.array([[r[f'{label}_clred'] for r in rows if r['W0'] == w] for w in W_LIST])
    RS = np.array([[r[f'{label}_Rstar'] for r in rows if r['W0'] == w] for w in W_LIST])
    FL = np.array([[r[f'{label}_flag']  for r in rows if r['W0'] == w] for w in W_LIST])
    vmin = min(0., float(np.nanmin(M))); vmax = max(10., float(np.nanmax(M)))
    pc = ax.pcolormesh(np.arange(len(TG_LIST)+1), np.arange(len(W_LIST)+1), M,
                       cmap='RdYlGn', vmin=vmin, vmax=vmax,
                       edgecolors='white', linewidth=2)
    for i, ww in enumerate(W_LIST):
        for j, Tg in enumerate(TG_LIST):
            warn = ' !' if FL[i, j] else ''
            ax.text(j+0.5, i+0.5, f'{M[i,j]:+.0f}%{warn}\nR={RS[i,j]:g}',
                    ha='center', va='center', fontsize=8,
                    color=('darkred' if FL[i, j] else 'black'))
    ax.set_xticks(np.arange(len(TG_LIST))+0.5)
    ax.set_xticklabels([f'Tg={Tg:g}s\nk={kred(Tg):.3f}\nH={H_ft(Tg):.0f}ft'
                        for Tg in TG_LIST], fontsize=8)
    ax.set_yticks(np.arange(len(W_LIST))+0.5)
    ax.set_yticklabels([f'W0={ww}' for ww in W_LIST], fontsize=9)
    ax.set_title(title, fontsize=9)
    fig.colorbar(pc, ax=ax).set_label('CLred [%]', fontsize=9)

fig.suptitle('CS-25.341 — CLred comparison (DAMULT=3, R* per cell, no-explosion pick)', fontsize=9)
plt.tight_layout()
fn = os.path.join(DIR_COMBO, 'heatmap_clred_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}', flush=True)

# --- line plot ---
fig, ax = plt.subplots(figsize=(7, 4))
colors = ['tab:blue', 'tab:orange', 'tab:red']
for w, col in zip(W_LIST, colors):
    cr_opt   = [r['opt_clred']   for r in rows if r['W0'] == w]
    cr_combo = [r['combo_clred'] for r in rows if r['W0'] == w]
    ax.plot(TG_LIST, cr_opt,   'o--', color=col, alpha=0.6, label=f'optimal W0={w}')
    ax.plot(TG_LIST, cr_combo, 's-',  color=col,            label=f'combo   W0={w}')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('Tg [s]'); ax.set_ylabel('CLred [%]')
ax.set_title('CS-25 — one-step optimal (dashed) vs E2-combo (solid)')
ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
fn = os.path.join(DIR_COMBO, 'summary_lines_combo.png')
fig.savefig(fn, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {fn}', flush=True)

# --- summary.md ---
md = [
    '# CS-25.341 — one-step optimal vs E2-combo (light/)\n\n',
    'Both controllers: DAMULT=3, TEND=3 s, dmax=14 deg, rate 300 deg/s.\n'
    'R* per cell = MAX CLred with no explosion flag; fallback: min pitch.\n'
    'Optimal: wnext+refine, dp45 integrator. Combo: FusedSensor Jmax=50, MPC N=8,\n'
    'R=R*, R_du=0, oracle preview w_seq=Wt[i+1:i+N+1], dp45 horizon.\n\n',
    '| W0 | Tg | H [ft] | k | '
    'opt CLred | opt R* | opt flap | opt pitch | opt flag | '
    'combo CLred | combo R* | combo flap | combo pitch | combo flag |\n',
    '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n',
]
for r in rows:
    md.append(
        f"| {r['W0']} | {r['Tg']:.2f} | {r['H_ft']} | {r['k']} "
        f"| {r['opt_clred']:+.1f} | {r['opt_Rstar']:g} | {r['opt_fmax']} "
        f"| {r['opt_pitch']} | {r['opt_flag']} "
        f"| {r['combo_clred']:+.1f} | {r['combo_Rstar']:g} | {r['combo_fmax']} "
        f"| {r['combo_pitch']} | {r['combo_flag']} |\n"
    )
with open(os.path.join(DIR_COMBO, 'summary.md'), 'w') as f:
    f.writelines(md)
print(f'saved {DIR_COMBO}/summary.md', flush=True)
print('# PLOTS DONE', flush=True)
```

- [ ] **Step 2: Run plots after cluster results arrive**

Scp results back:
```bash
scp -r u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/results_cs25_combo/ \
    /home/marco/LDNet_OF/light/

# Also get updated optimal if re-run:
scp -r u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/results_cs25/ \
    /home/marco/LDNet_OF/light/
```

Then from WSL (needs matplotlib, numpy only — no TF):
```bash
cd /home/marco/LDNet_OF
python3 light/tests/cs25_combo_plots.py
```

- [ ] **Step 3: Commit plots script + results**
```bash
git add light/tests/cs25_combo_plots.py \
        light/results_cs25_combo/summary.md \
        light/results_cs25_combo/*.png \
        light/results_cs25_combo/*.npz
git commit -m "feat(cs25): E2-combo CS-25 study results + comparison plots and summary"
```

---

## Task 5 – White-noise robustness of the combo (noise_white_combo.py)

**Files:**
- Create: `light/noise/noise_white_combo.py`
- Create: `light/noise/launch_noise_white_combo.sh`
- Create: `light/noise/status_noise_white_combo.sh`
- Create: `light/noise/plots_noise_white_combo.py`

**Context:** Axis A (`noise_white.py`) showed the one-step controller collapses at σ≈1%·W0. This script sweeps σ for the combo (Jmax=50, N=8, R=3e-4, R_du=0, lam=0) to find where the combo degrades. Reference: E2-combo is perfect at σ=2% (+80.5%) and with DLR raw 1-3 m/s; we push to σ=20%.

- [ ] **Step 1: Write noise_white_combo.py**

Create `light/noise/noise_white_combo.py`:

```python
"""
White-noise robustness of the E2-combo pipeline vs the one-step argmin.

sigma/W0 in {0, 0.01, 0.02, 0.05, 0.10, 0.20} — white Gaussian noise on each
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
ax.set_xlabel('Raw measurement noise σ / W0 [%]')
ax.set_ylabel('CLred [%]')
ax.set_title('White-noise robustness — E2-combo vs one-step (W30/Tg0.4, DAMULT=3)')
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

Smoke first (2 seeds, σ∈{0,2%}, ~20 min):
```bash
ssh -n u10677113@10.78.18.100 \
    '/work/u10677113/LDNet_GLA/light/noise/launch_noise_white_combo.sh smoke'
# Poll:
ssh -n u10677113@10.78.18.100 \
    '/work/u10677113/LDNet_GLA/light/noise/status_noise_white_combo.sh'
```

Smoke pass criterion: combo frac=0.0 CLred ≈ dp45 combo clean anchor (±1 pt); combo frac=0.02 ≈ same. Then full run:
```bash
ssh -n u10677113@10.78.18.100 \
    '/work/u10677113/LDNet_GLA/light/noise/launch_noise_white_combo.sh full'
```

Expected runtime: 6 σ-levels × (6 combo + 5 none) seeds × ~8 min combo / ~1 min none ≈ **~5h** total (single job).

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

# W_combo — white-noise robustness of the E2-combo (2026-07-08)

Axis: noise_white_combo.py — white Gaussian raw measurement noise σ applied BEFORE
fusion; FusedSensor(Jmax=50, lam=0, N=8) + MPCPreviewController(R=3e-4, R_du=0).
Home cell W30/Tg0.4, DAMULT=3, 6 seeds rng(100+seed). Baseline 'none' = one-step
argmin (same R, same raw frac), same seeds (paired).

| σ/W0 | combo CLred | combo [min,max] flags/6 | σ_del m/s | none CLred | none flags/6 |
|---|---|---|---|---|---|
| 0%   | TBD | TBD | TBD | — | — |
| 1%   | TBD | TBD | TBD | TBD | TBD |
| 2%   | TBD | TBD | TBD | TBD | TBD |
| 5%   | TBD | TBD | TBD | TBD | TBD |
| 10%  | TBD | TBD | TBD | TBD | TBD |
| 20%  | TBD | TBD | TBD | TBD | TBD |

**Break-even vs combo-clean:** σ/W0 = TBD%
**Break-even vs prop-W clean (+32%):** σ/W0 = TBD%

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

## Task 6 – Collect all results, fill tables, final documentation

**Prerequisite:** All cluster jobs DONE.

- [ ] **Step 1: Scp all results**
```bash
scp -r u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/results_cs25_combo/ \
    /home/marco/LDNet_OF/light/
scp -r u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/results_cs25/ \
    /home/marco/LDNet_OF/light/
scp u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/noise/results/W_combo.npz \
    /home/marco/LDNet_OF/light/noise/results/
```

- [ ] **Step 2: Generate CS-25 comparison plots**
```bash
cd /home/marco/LDNet_OF
python3 light/tests/cs25_combo_plots.py
```

- [ ] **Step 3: Generate noise robustness figure**
```bash
python3 light/noise/plots_noise_white_combo.py
```

- [ ] **Step 4: Fill in NOTES.md tables**

Replace all "TBD" entries with actual numbers from the cluster output logs and npz files.

- [ ] **Step 5: Verify the key deliverable numbers**

From the cluster output or via local analysis:
1. dp45 baseline table (NOTES.md §Integrator migration) — 4 numbers
2. CS-25 18-cell table — combo CLred, R*, flap_max, pitch, flag per cell
3. W_combo robustness table — 6 σ levels × combo+none stats

---

## Task 7 – Final commit of all results and documentation

- [ ] **Step 1: Stage all generated files**
```bash
git add light/optimal.py \
        light/run.py \
        light/noise/harness_noise.py \
        light/noise/NOTES.md \
        light/noise/noise_white_combo.py \
        light/noise/launch_noise_white_combo.sh \
        light/noise/status_noise_white_combo.sh \
        light/noise/plots_noise_white_combo.py \
        light/noise/results/W_combo.npz \
        light/noise/results/fig_noise_white_combo.png \
        light/tests/cs25_combo_study.py \
        light/tests/cs25_combo_plots.py \
        light/tests/launch_cs25_combo.sh \
        light/tests/status_cs25_combo.sh \
        light/results_cs25_combo/ \
        light/smoke_dp45_baseline.sh
```

- [ ] **Step 2: Final commit**
```bash
git commit -m "$(cat <<'EOF'
feat: E2-combo pipeline migration + CS-25 study + noise robustness

Task 1: optimal.py — append dp45_batch, FusedPreviewSensor, MPCPreviewController
  (T1 DLR fusion + T7 N-step MPC, N=8, R=3e-4, R_du=0, wnext convention, no gate)
Task 2: run.py — mode='combo' (oracle preview, NH=8); open/optimal bit-identical
Task 3: cs25_combo_study.py — MODE-parametrised CS-25 driver; re-runs both
  controllers on dp45 tree; results in results_cs25_combo/ and results_cs25/
Task 4: cs25_combo_plots.py — side-by-side comparison heatmap/lines/summary.md
Task 5: noise_white_combo.py — σ sweep {0,1,2,5,10,20}%·W0; finds combo
  break-even; NOTES.md §W_combo verdict appended

Integrator: cluster synced to dp45 (f92d8975); rk4 anchors in NOTES.md §migration;
dp45 anchors verified and documented. 0/6 flags expected for combo at σ≤2%·W0.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

### Spec coverage
- [x] Task 1: optimal.py — dp45_batch, FusedPreviewSensor, MPCPreviewController added; OptimalController untouched
- [x] Task 1: run.py — mode='combo', oracle w_seq, NH/R_du params; open/optimal unchanged
- [x] Task 1: Regression gate — combo oracle clean ≈ dp45 anchor (fill after run)
- [x] Task 2: CS-25 — MODE env, results_cs25_combo/ for combo, no overwrite of existing
- [x] Task 2: Both modes re-run with dp45 integrator (same-integrator comparison)
- [x] Task 2: cs25_combo_plots.py generates summary.md with both controllers
- [x] Task 3: noise_white_combo.py — σ grid {0,1,2,5,10,20}%, 6 seeds, paired none baseline
- [x] Task 3: harness coupling pattern: wc_fun fills sensor.last, ctrl.compute reads it
- [x] Task 3: NOTES.md §W_combo section + figure
- [x] Cluster workflow: smoke before full, launch scripts avoid quoting trap, OMP/TF caps set
- [x] harness_noise.py: ONLY the getattr fix, nothing else
- [x] Never copy FusedPreviewSensor/MPCPreviewController — import from optimal.py

### Potential issues
1. **N variable collision in run.py**: Renamed `N` → `Nsteps` inside `simulate()`. Verify grep for bare `N` after edit: `grep -n '\bN\b' light/run.py | grep -v Nsteps`.
2. **aero singleton**: `light/run.py` uses a module-level `aero`; `simulate()` calls `aero.reset(dt=DT)` at the top. `MPCPreviewController.__init__` stores `self.aero = aero` — it is the same singleton. The latent `_z` is reset by `aero.reset()` before each rollout. This is correct.
3. **C_L_trim in MPCPreviewController**: Pass `C_L_trim=CLTRIM` from `run.py`'s module-level `CLTRIM`. In harness scripts pass `C_L_trim=H.CLTRIM`.
4. **Cluster quoting trap**: `launch_cs25_combo.sh` runs ON the cluster (scp'd as a script), so internal `$` variables expand at cluster runtime — correct. The ssh call only calls the script by path.
5. **dp45_batch uses structure.***: `import structure as _S` inside function avoids module-level dependency in optimal.py; `_S.D_ALPHA` is already scaled when the function runs (DAMULT set before any import chain).
