# Design: PID vs Greedy-LDNet — Gust Load Alleviation Comparison

**Date:** 2026-05-25  
**Status:** Approved  
**Scope:** Two-phase comparison study — linear surrogate now, LDNet after training

---

## Context

The LDNet_OF project builds a reduced-order model (ROM) of aeroelastic aerodynamics using a neural network (LDNet) trained on OpenFOAM CFD data. The system is a 2-DOF wing (heave `h`, pitch `α`) with a trailing-edge control flap (deflection `δ`), subjected to a 1-cosine EASA gust `W(t)`.

The goal is to compare:
1. A **classical controller** (PID) with no aerodynamic model — only structural state feedback
2. A **model-based greedy controller** (Greedy-N1) that uses a prediction model to optimize `δ` at each timestep

This comparison demonstrates the value of having a predictive aerodynamic model even at minimal online complexity (one-step lookahead, no horizon rollout).

### Two-phase plan

- **Phase 1 (now):** Use a linear aerodynamic model in the Greedy controller. Validates the control pipeline before LDNet is trained.
- **Phase 2 (after LDNet training):** Swap the linear model for LDNet in the Greedy controller. Demonstrates the additional benefit of the neural ROM.

---

## System Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              PHYSICAL SYSTEM                     │
  GUST W(t) ───────►│  Structure (RK45): h, ḣ, α, α̇                  │
                    │                          ▲                       │
       δ(k) ───────►│  Aero model: C_L, C_M    │ F_y, M_z            │
                    │  (CFD / LDNet / linear)   │                      │
                    └──────────────────────────────────────────────────┘
                                                       │
                                               SENSORS: h(k), α(k)
                                                       │
                                            ┌──────────┴──────────┐
                                            │      OBSERVER        │
                                            │ x_hat = [h,ḣ,α,α̇]  │
                                            │ W_hat via C_L inv.  │
                                            │ z_hat evolved       │
                                            └──────────┬──────────┘
                                                       │
                                            ┌──────────┴──────────┐
                                            │     CONTROLLER       │
                                            │  PID or Greedy-N1   │
                                            └──────────┬──────────┘
                                                       │ δ(k)
```

Both controllers receive the same sensor measurements: `h(k), α(k)` (and derivatives `ḣ, α̇`).

---

## Controller 1: PID

A two-channel SISO PID acting on heave and pitch independently.

```
δ(t) = Kp_h * h + Kd_h * ḣ + Kp_α * α + Kd_α * α̇
```

- **Inputs:** `[h, ḣ, α, α̇]` — structural states only, no aerodynamic model
- **No observer:** does not estimate W or z
- **Tuning:** gain optimization offline (scipy.minimize on gust simulation), or manual
- **Constraints:** `|δ| ≤ 20°`, rate limit `|Δδ/dt| ≤ 100°/s`
- **File:** `src/control/pid.py`

---

## Controller 2: Greedy-N1

One-step-ahead optimization at each timestep `k`:

```
min_{δ}  Q_h * h(k+1)² + Q_α * α(k+1)² + R * δ²

subject to:
  C_L, C_M = AeroModel(x_hat_k, δ, W_hat_k)   ← one forward pass
  x(k+1)   = structural_step(x_hat_k, C_L, C_M, δ)
  |δ| ≤ 20°
```

- **AeroModel (Phase 1):** Linear Theodorsen approximation:
  ```
  C_L = C_Lα*α + C_Lh*(h/c) + C_Lḣ*(ḣ/U) + C_Lδ*δ + C_LW*(W/U)
  C_M = C_Mα*α + C_Mδ*δ
  ```
  Coefficients from Theodorsen theory at U=80 m/s.

- **AeroModel (Phase 2):** `LDNetModel.step_tf()` — TF-native, differentiable
- **Optimizer:** `scipy.minimize_scalar` with bounds `[-20, 20]` (Phase 1); TF gradient on scalar `δ` (Phase 2)
- **Observer:** same C_L inversion + leaky integrator as existing MPC pipeline (`src/control/cl_inversion.py`, `src/control/run_controller.py`)
- **File:** `src/control/greedy.py`

### Observer detail (Greedy-N1)

At each timestep `k`:
1. `x_hat_k = [h, ḣ, α, α̇]` from sensors (direct or leaky integrator from accelerometers)
2. `C_L_meas = F_y / (0.5 * ρ * U² * S_ref)` — from force sensor or accelerometer
3. `W_hat_k` ← bisection on `AeroModel(x_hat, δ_{k-1}, W) = C_L_meas` (monotone in W≥0)
4. `z_hat_k` ← advanced from previous step (only in Phase 2 with LDNet)

---

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/control/pid.py` | Create | `PIDController` class |
| `src/control/greedy.py` | Create | `GreedyN1Controller` class (linear model Phase 1, LDNet Phase 2) |
| `src/PIDvsGreedy.py` | Create | Comparison script: OL / PID / Greedy, plots, metrics table |
| `src/control/linear_aero.py` | Create | Theodorsen linear aero model (Phase 1 surrogate) |

## Files to Reuse (no changes)

| File | What to reuse |
|------|--------------|
| `src/aeroelastic/system.py` | `run_simulation()` loop — pass different controller objects |
| `src/aerodynamics/model.py` | `LDNetModel.step_tf()` — used in Phase 2 Greedy |
| `src/control/cl_inversion.py` | `estimate_W_from_CL()` bisection — W_hat observer |
| `src/control/run_controller.py` | `heuristic_observer()` — structural state estimation |
| `src/structural/smd.py` | `structural_rhs()` — for one-step structural prediction in Greedy |

---

## Comparison Script: PIDvsGreedy.py

```
1. Load model (LDNet or linear, depending on phase)
2. Gust profile: EASA 1-cosine, W_0=60 m/s, T_g=0.8s, U=80 m/s
3. Run 3 simulations: OL, PID, Greedy-N1
4. Plot (6 subplots, 2 columns):
   - h(t), α(t)
   - δ(t), Δδ/dt (rate)
   - C_L(t), C_M(t)
5. Metrics table:
   - Peak |h|, Peak |α|
   - RMS ḧ, RMS α̈
   - Peak |C_L|, Peak |C_M|
   - Actuation energy: Σ R*δ²
   - Reduction % vs OL
```

---

## Metrics and Success Criteria

| Metric | Target (Greedy vs PID) |
|--------|----------------------|
| Peak \|h\| reduction | Greedy ≥ PID |
| Peak \|α\| reduction | Greedy > PID (main differentiator) |
| Actuation energy | Greedy within 2× of PID |
| Numerical stability | No divergence, δ always within bounds |

Phase 1 success = pipeline runs end-to-end without errors, PID and Greedy produce different δ(t) trajectories, Greedy reduces peak h or α relative to PID.

Phase 2 success = LDNet-based Greedy outperforms linear Greedy on at least one metric.

---

## Verification

1. **Unit test PID:** Apply step inputs to h and α → verify δ response has correct sign
2. **Unit test Greedy:** With known x_hat, W_hat → verify optimizer finds δ that reduces predicted h(k+1)
3. **Integration test:** Run full simulation with OL → verify h and α match existing OLvsLQR.py results
4. **Comparison run:** Execute PIDvsGreedy.py, inspect plots for physical plausibility (δ opposes gust-induced h growth)
5. **Bound check:** Assert `max(|δ|) ≤ 20°` and `max(|Δδ/dt|) ≤ 100°/s` in all runs
