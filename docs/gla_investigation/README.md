# GLA Investigation: does model-based optimal control beat proportional?

Master's-thesis investigation into **Gust Load Alleviation (GLA)** for a 2-DOF
aeroelastic wing, using a trained neural aerodynamic surrogate (**LDNet**) as the
plant model. The central question:

> **Does model-based optimal control beat a classical proportional controller for
> this GLA problem — and how do we compare them fairly (same sensors, same
> information)?**

This folder captures the full reasoning chain, the numeric findings, the
dead-ends, and the bugs, so the line of argument can be reconstructed later (by a
future reader or the advisor) without the original conversation.

## The one-line answer

For this **2-DOF SISO** GLA problem, model-based optimal control does **not**
robustly beat a well-tuned, **smoothed**, **feedforward** proportional controller.
The model's apparent edge turned out to be **(a) contingent on gust knowledge that
is not freely observable**, and **(b) confined to narrow high-`k`/strong-gust
regimes** — and at the training-envelope edge the optimal even **loses**. The
single-step optimal is essentially an *adaptive proportional* law.

## How to read this folder

| File | Topic |
|------|-------|
| [01-question-and-setup.md](01-question-and-setup.md) | Research question, plant, LDNet surrogate, the two controllers, metrics, reduced frequency `k`, training envelope |
| [02-single-step-equals-proportional.md](02-single-step-equals-proportional.md) | Derivation: the one-step optimum **is** a proportional law on a locally-linear plant; where the model can differ |
| [03-gust-oracle-and-observability.md](03-gust-oracle-and-observability.md) | Gust-oracle win; `C_L(W)` non-invertibility; the honest (no-oracle) result that ties/loses |
| [04-equal-information-and-chatter.md](04-equal-information-and-chatter.md) | Equal-information test (both know `W`); the chatter/smoothing reversal |
| [05-envelope-edge-and-open-questions.md](05-envelope-edge-and-open-questions.md) | The `W0=48` envelope-edge loss; open questions |
| [06-bugs-and-pitfalls.md](06-bugs-and-pitfalls.md) | `batch_step` latent-leak omission; `DLPF` smoothing artifact; `causal_basin` sign; the pitch-pick artifact |
| [07-literature.md](07-literature.md) | LMPC-vs-PID (~10%); preview/feedforward as the real lever; references |

## Setup at a glance

- **Plant**: 2-DOF aeroelastic wing — heave `h` + pitch `alpha`. Structure is
  **linear**; aerodynamics are the **nonlinear, unsteady LDNet** surrogate
  (predicts `C_L`, `C_M` with an internal latent aero-memory state `z`).
- **Objective**: minimize the peak lift-coefficient excursion
  `CLexc = max |C_L - C_L_trim|` during a discrete CS-25 "1-cosine" gust, by
  deflecting a trailing-edge flap.
- **Operating point**: `U = 80 m/s`, `DAMULT = 3` (artificial pitch damping ×3),
  `DT = 0.002 s`, flap `DMAX = 14 deg`, rate limit `300 deg/s`.
- **Gust**: `W(t) = (W0/2)(1 - cos(2*pi*t/Tg))` for `0 <= t <= Tg`.
- **Reduced frequency**: `k = pi / (80 * Tg)` — the knob that controls how
  *unsteady* the aerodynamics are (low `k` = quasi-steady).
- **Two controllers** (`clean/controller.py`):
  - `ProportionalController` — reactive `C_L` feedback, no model.
  - `Controller` — model-based; single-step optimal at `mpc_horizon=1`, or
    constant-delta receding-horizon MPC for `mpc_horizon>1`.

See [01-question-and-setup.md](01-question-and-setup.md) for the full setup.

## Results at a glance

All numbers are **`CLexc` reduction %** (higher = better GLA), on the LDNet plant.
"Pitch" is the ratio of closed-loop peak pitch to open-loop peak pitch (a
constraint: we do not want to trade lift alleviation for torsional excursion).

### 1. Single-step optimal (gust oracle) vs *unsmoothed* proportional — `clean/onestep.py`

The oracle optimal wins clearly on sharp/strong gusts, but the proportional here is
raw (unsmoothed, high-gain).

| Cell | `k` | Regime | Prop (`C_L` red) | 1-step optimal (`C_L` red) |
|------|-----|--------|-----------------|----------------------------|
| W10/Tg1.0 | 0.039 | ~linear | tie | tie |
| W20/Tg1.0 | 0.039 | ~linear | tie | tie |
| W30/Tg1.0 | 0.039 | strong | ~ | modest win |
| **W30/Tg0.5** | **0.079** | strong/sharp | **~4%** | **~36%** |

> Caveat later found decisive: this used an **unsmoothed** proportional and a
> **gust oracle**. Both advantages evaporate under fair conditions (below).

### 2. Honest optimal (NO gust oracle, anchored to measured `C_L`) — `clean/honest_grid.py`

Removing the unobservable gust oracle makes the optimal **tie or lose**.

| Cell | Honest optimal (`C_L` red) | Proportional (`C_L` red) | Winner |
|------|----------------------------|--------------------------|--------|
| W30/Tg0.5 | ~2.5% | ~3.7% | proportional |

### 3. Equal-information (both know `W`): prop + feedforward vs opt-with-`W` — `clean/propw.py`

With gust feedforward added to the proportional (`delta = g1*(C_L-trim) + gw*W`),
the two are **within ~±5 points** across most of the in-envelope grid
(`W in {10,20,30} × Tg in {0.30..1.20}`); the optimal wins clearly only in a few
noisy high-`k`/strong cells — **no clean `k`-trend**.

### 4. Chatter / smoothing reversal — `clean/smoothprop.py`

| Arm | `C_L` red | TV(delta) [deg] |
|-----|-----------|-----------------|
| Proportional, raw (high gain) | baseline | ~200–500 (chatter) |
| Proportional + 2nd-order LPF (`DLPF≈0.7`) | **higher** | low |
| Optimal | comparable | ~15–65 (smooth) |

Adding a simple low-pass to the proportional **removes the chatter AND increases
its `C_L` reduction** — the chatter had been wasting control authority. The
smoothed proportional then **matches or beats** the optimal at equal (low) chatter.

### 5. Envelope edge (`W0=48`, high `k`) — `clean/smoothprop.py`

| Cell | `k` | Optimal (even *with* oracle) | Proportional |
|------|-----|------------------------------|--------------|
| W48/Tg0.30 | 0.131 | ~20–26% | ~30–50% |
| W48/Tg0.40 | 0.098 | ~20–26% | ~30–50% |
| W48/Tg0.50 | 0.079 | ~20–26% | ~30–50% |

At the training-envelope edge the optimal **loses** — attributed to model
inaccuracy at the edge plus one-step greedy myopia.

## Honest overall conclusion

For this 2-DOF SISO GLA problem, **model-based optimal control does not robustly
beat a well-tuned, smoothed, feedforward proportional controller.**

1. The single-step optimal is, on a locally-linear plant, **exactly an adaptive
   proportional law** (see [02](02-single-step-equals-proportional.md)). Its only
   structural edge is where the sensitivity `s = dC_L/ddelta` varies
   (nonlinearity / unsteadiness).
2. Its measured wins were **contingent on a gust oracle** — knowing the true
   `W(t)`. But `C_L(W)` is **non-invertible** on the LDNet manifold, so `W` cannot
   be recovered from the sensor; the oracle is not realizable
   (see [03](03-gust-oracle-and-observability.md)).
3. At **equal information** (both fed `W`), a proportional-plus-feedforward matches
   the optimal across most of the envelope
   (see [04](04-equal-information-and-chatter.md)).
4. The optimal's "smoothness + extra reduction" advantage was largely an artifact
   of comparing against an **unsmoothed** proportional; smoothing the proportional
   erases it (see [04](04-equal-information-and-chatter.md)).
5. At the **strong / high-`k` envelope edge**, the optimal even **loses**
   (see [05](05-envelope-edge-and-open-questions.md)).
6. This matches the **literature**: LMPC-vs-PID GLA studies find only ~10% edge for
   the model-based controller; the real lever in GLA is **gust preview /
   feedforward** — the *information*, not controller sophistication
   (see [07](07-literature.md)).

## Open questions

1. **Is the envelope-edge loss fixable or fundamental?** Would a genuine receding
   horizon (`> 1` step), a terminal cost, or move-suppression recover the optimal
   at `W48`/high-`k` — or is it a hard limit of LDNet's accuracy at the training
   edge plus one-step myopia?
2. **Does the model advantage genuinely grow with `k`?** A first "margin grows with
   `k`" reading was **walked back** as a pitch-constraint pick artifact. A clean
   iso-pitch comparison (`clean/ksweep2.py`) is needed to settle whether there is a
   real `k`-trend.
3. **Would a MIMO / multi-surface / hard-constrained problem favor model-based
   control** where a SISO proportional cannot scale? The SISO negative result does
   not necessarily transfer to multi-actuator GLA.

## Source scripts (in `clean/`)

- `controller.py` — `Controller` (model-based) and `ProportionalController`.
- `mpc_gust.py` — plant/gust harness, trim, `simulate()`, metrics, scheduling.
- `ldnet_aero.py` — LDNet surrogate wrapper (`predict`, `advance`, `batch_step`).
- `onestep.py` — single-step optimal vs proportional (oracle).
- `propw.py` — equal-information test (prop + feedforward vs opt-with-`W`).
- `smoothprop.py` — chatter/smoothing reversal + `W48` envelope edge.
- `honest_grid.py` — honest (no-oracle) optimal, anchored to measured `C_L`.
- `clw_sweep.py` — `C_L(W)` monotonicity / invertibility probe.
- `ksweep2.py` — reduced-frequency sweep with the MPC formulation fixed.
- `gridplots2.py` — full-grid time-history plots (open / prop / one-step optimal).
