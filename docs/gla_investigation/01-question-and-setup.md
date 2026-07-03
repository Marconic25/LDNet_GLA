# 01 — Research question and setup

[← Back to index](README.md)

## Research question

> **Does model-based optimal control beat classical proportional control for this
> GLA problem, and how do we compare the two fairly — same sensors, same
> information?**

The "fairness" clause is the crux. A model-based controller that is secretly fed
the true gust `W(t)` while the proportional only sees `C_L` is not a controller
comparison, it is an *information* comparison. Much of this investigation is about
peeling apart **controller sophistication** from **the information each controller
is given**.

## The plant

A **2-DOF aeroelastic wing** with degrees of freedom:

- heave `h` (vertical bending displacement),
- pitch `alpha` (torsion).

State vector `x = [h, h_dot, alpha, alpha_dot]`.

- **Structure**: linear. Mass/stiffness/damping matrices (`clean/structure.py`);
  integrated one step at a time with RK4. An artificial pitch-damping multiplier
  `DAMULT = 3` is applied (`structure.D_ALPHA *= DAMULT` in `clean/mpc_gust.py`)
  because the LDNet aero under-represents aerodynamic pitch damping and the
  lightly-damped pitch mode otherwise rings.
- **Aerodynamics**: the trained **LDNet** neural surrogate — nonlinear and
  **unsteady** (has aero memory). See below.

Operating constants (`clean/mpc_gust.py`):

| Symbol | Value | Meaning |
|--------|-------|---------|
| `U` | 80 m/s | freestream velocity |
| `DT` | 0.002 s | simulation timestep |
| `RHO` | 1.225 kg/m³ | air density |
| `S` | 0.05 m² | reference area |
| `C` | 1.0 m | reference chord |
| `DMAX` | 14 deg | flap deflection limit |
| rate limit | 300 deg/s | flap slew limit |
| `DAMULT` | 3 | artificial pitch-damping multiplier |

Trim state (regime-fixed, shared by every controller arm):
`X0 = [-6.49179e-3, 0, -8.76338e-4, 0]`, with `C_L_trim = predict(X0, 0, 0, U)`.
Using this fixed trim (consistent with `z = 0`, the regime the rollout model was
trained in) rather than an `fsolve` trim avoids a spurious startup transient.

## The LDNet surrogate

LDNet is a **Latent Dynamics Network** trained as an aerodynamic surrogate. It
predicts `(C_L, C_M)` and maintains an internal **latent state `z`** representing
aerodynamic memory (the unsteady wake). Wrapper: `clean/ldnet_aero.py`,
class `LDNetAero`. Model: `models_rollout/latent_10` (10 latent states).

Interface:

- `predict(state, delta_deg, W, U) -> (C_L, C_M)` — **read-only**; reconstructs the
  output from the *current* `z` without stepping it, so the controller's optimizer
  can call it repeatedly without corrupting state.
- `advance(state, delta_deg, W, U, dt)` — steps `z` forward one step with the *true*
  inputs; call once per timestep.
- `batch_step(z_b, x_b, delta_b, W, U, dt)` — vectorized one-step forward for a
  batch of candidate deltas (used by the MPC grid and the batched analysis sweeps).

Two structural facts about LDNet that drive the whole investigation:

1. **It is unsteady** — the latent `z` is an aerodynamic memory. At low reduced
   frequency it behaves quasi-steadily; at high `k` the memory matters (the lift
   response lags and is attenuated, à la Theodorsen `C(k)`). This is where a model
   *could* beat a static proportional law.
2. **`W` and `z` co-vary in training.** The surrogate never saw `W` swept
   independently of the latent state it induces, so probing `C_L` vs `W` at a
   *frozen* `z` walks off the training manifold — with pathological consequences
   for gust observability (see [03](03-gust-oracle-and-observability.md)).

> Related known LDNet issues (from project memory): the free-running latent drifts
> unbounded (no equilibrium), so a plain LDNet is not a reliable long-horizon
> forward sim. The **rollout-trained** model (`models_rollout/latent_10`) is the
> one that is both stable and flap-aware, and is the plant/model used here.

## The gust

Discrete CS-25 **"1-cosine"** gust (`clean/mpc_gust.py::gust`):

```
W(t) = (W0 / 2) * (1 - cos(2*pi*t / Tg))     for 0 <= t <= Tg
     = 0                                      otherwise
```

- `W0` — peak gust velocity [m/s].
- `Tg` — gust period [s].

## Reduced frequency `k`

The single most important physical knob:

```
k = pi / (80 * Tg)
```

- **Low `k`** (large `Tg`) → **quasi-steady** aerodynamics; Theodorsen
  `C(k) -> 1`; the lift responds almost instantaneously. A static gain is nearly
  optimal here.
- **High `k`** (small `Tg`) → **strongly unsteady**; lift lags and is attenuated by
  the unsteady deficit `1 - C(k)`. This is the regime where the model's knowledge
  of the aero dynamics could, in principle, pay off.

The investigation repeatedly asks: *does the model's advantage grow with `k`?* (See
[04](04-equal-information-and-chatter.md) and
[05](05-envelope-edge-and-open-questions.md).)

## The two controllers (`clean/controller.py`)

### `ProportionalController`

```
delta = clip( gain * (C_L_meas - C_L_trim) )      [magnitude- and rate-limited]
```

Needs only the **measured lift coefficient** — no gust estimate, no state estimate.
The gain is **signed**: in this regime `dC_L/ddelta > 0`, so a **negative** gain
reduces a **positive** `C_L` excursion.

### `Controller` (model-based)

At each step it solves the scalar optimization

```
min_delta   Q_h*h(k+1)^2 + Q_alpha*alpha(k+1)^2 + Q_alpha_dot*(alpha_dot(k+1)-alpha_dot(k))^2
          + Q_CL*(C_L(k+1) - C_L_trim)^2 + R*delta^2 + R_du*(delta - delta_prev)^2
subject to  |delta| <= DMAX,  |delta - delta_prev|/dt <= rate limit
```

The next state and `C_L` are predicted by calling LDNet with the candidate `delta`
and the current gust estimate `W_hat`, then advancing the structure one RK4 step.
Two modes:

- `mpc_horizon = 1` — **single-step optimal** (the workhorse of this study).
- `mpc_horizon > 1` — **constant-delta receding-horizon MPC** (holds `delta` over
  the horizon, rolling `z` and the structure forward).

Because the one-step cost is non-convex in `delta`, the controller does a **global
grid search** (`n_grid` points) plus a local `minimize_scalar` refine, then
**rate-limits** the move toward the target. Optional flags: `causal_basin`
(restrict search to the lift-reducing flap sign), `target_lpf` / `lpf_max`
(low-pass the target), `e_ref` / `R_sched_gain` (gust gain-scheduling).

## Metrics

Computed over the window `t <= Tg + 0.5` (gust + ring-down), in
`clean/mpc_gust.py::metrics`:

| Metric | Definition |
|--------|------------|
| `CLexc` | `max |C_L - C_L_trim|` over the window — the **primary** GLA metric |
| `clred` | `(CLexc_open - CLexc_closed) / CLexc_open * 100` — reduction % |
| pitch ratio | closed-loop peak `|alpha|` / open-loop peak `|alpha|` — a **constraint** |
| `adrms` | RMS of `alpha_dot` [deg/s] — torsional activity |
| TV(delta) | total variation of the flap command — **chatter** proxy |
| `flap_max` | peak flap deflection [deg] |

The pitch ratio matters: any lift alleviation that comes at the cost of a large
torsional excursion is not acceptable, so comparisons are made **at iso-pitch**
where possible.

## Training envelope

LDNet was trained on:

- gust period `Tg in [0.30, 1.20] s` → `k in [0.033, 0.131]`,
- gust amplitude `W0 in [8, 48] m/s`.

Cells **inside** this envelope are where the surrogate is trustworthy; cells at the
edge (`W0 = 48`, `Tg = 0.30`) or **outside** (e.g. `Tg = 2.0`) are flagged, because
model error there confounds any controller comparison. This distinction is decisive
at the envelope edge (see [05](05-envelope-edge-and-open-questions.md)).

---

Next: [02 — the single-step optimum is a proportional law →](02-single-step-equals-proportional.md)
