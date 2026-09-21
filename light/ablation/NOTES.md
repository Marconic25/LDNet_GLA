# Internal-model ablation — LDNet vs linear unsteady aerodynamics

Answers the open question in `Conclusion.tex`:

> The most direct test of the premise on which the architecture rests is an
> ablation of the internal model: substituting a linear unsteady aerodynamic
> model for the LDNet inside the same loop, at equal preview and equal cost
> function, would measure how much of the alleviation the data-driven
> surrogate is responsible for, and would locate the region of the envelope
> — high $r_g$, large deflections — where its nonlinearity is required.

## Where this is written up

Appendix C of the thesis (`app:modelcomp` in `light/latex/appendixA.tex`),
titled "Comparison of internal models", one printed page. Structure (fixed
2026-09-18, third pass per user comments):
  1. the linear unsteady model's equations (downwash, lag states, loads),
     every symbol defined at first appearance per AGENTS.md ("un simbolo,
     una grafia" + "definire tutto, anche l'ovvio")
  2. open-loop load prediction, ONE figure `fig:modelcomp_cfd` — the
     trajectory panel only: true CFD C_L(t) vs the LDNet's and the linear
     model's rolling 8-step forecast, on the held-out trajectory with the
     largest peak deflection (14.8 deg). The stratified forecast-error bars
     that used to be panel (b) were DROPPED as a figure in this pass — the
     accuracy numbers they showed (9.8x overall, 15.1x/20.7x in the
     high-deflection/high-gust bins) are now stated in the text instead
  3. closed-loop control, no figure of its own — folded into one paragraph
     that reports the 18-cell CLred numbers and the self-consistency/model-
     effect decomposition (53.8pp/15.0pp) in prose, pointing back to the
     forecast accuracy gap rather than to a separate image
The heatmap image (`fig_ch3_modelcomp_envelope.png`) and the stratified-bars
image (`fig_ch3_modelcomp_cfd.png`) are therefore BOTH now unreferenced, same
status as the older flap-authority mechanism figure. All three still exist
on disk under `light/latex/Images/` but nothing in the .tex build points to
them. The appendix now carries exactly one figure.

Referenced from the Introduction structure paragraph, the end of chapter 3
section 4.4, and the Conclusion (Conclusion.tex needed a new sentence added
in the previous pass — its working copy had been reset to a pre-integration
draft that never mentioned the comparison at all; check this didn't happen
again before trusting old descriptions of Conclusion.tex's state).

`fig_trajectory.py` (in this directory) reuses the exact horizon-rolling
logic of `run_cfd_horizon.py` (same dt_model/sub handling, same
resynchronisation between probes) but keeps the per-probe N-step-ahead
forecast value instead of collapsing it to an RMS, so it can be plotted
against the true trace. On this one trajectory (the same used for the
14.8-deg |delta|max bin) the ratio comes out 13.5x, consistent with the
>11-deg stratified bin's 15.1x in `cfd_horizon.json`.

Terminology: the thesis does not use the word "ablation" — it is on the
forbidden list in `light/latex/AGENTS.md`. The study is an "internal-model
comparison", and the text has to say explicitly that the substitute model
runs inside the same loop, because "comparison" alone does not imply it.
This file keeps the word for brevity in the lab-notebook sense.

## What was built

`linear_aero.py` — a finite-state **linear unsteady** model: Wagner
(R.T. Jones two-pole) wake memory for airfoil motion, Küssner (two-pole) for
gust penetration, plus added mass. Four lag states. It exposes the same
interface as `LDNetAero` (`predict`, `advance`, `advance_z`, `batch_step`,
`_z_leak`, `_num_z`), so it drops into `MPCPreviewController` **without any
change to the loop**: same preview, same cost function, same flap grid, same
rate limit and saturation, same `dp45` structural integrator.

`fit_linear.py` / `fit_linear2.py` — gains fitted by least squares on the
**same CFD campaign the LDNet was trained on** (`data/GLA_train.h5`).

This fitting is deliberate and matters for the validity of the whole
exercise. A baseline carrying textbook thin-airfoil coefficients would lose
because 2π is the wrong lift-curve slope for this viscous airfoil, i.e. for
a gain error rather than for missing nonlinearity, and the ablation would
measure nothing of interest. Fitting gives the linear model the best
parameters available *within its own structure*, so the residual gap is
attributable to what that structure cannot represent.

## Fit diagnostics (why the coefficients look the way they do)

Three things were checked before trusting any fit, because the first
unconstrained fit returned `CL_a = -38` (a negative lift-curve slope):

1. **Collinearity — ruled out.** All VIFs ≈ 1; condition number 1.6e3. The
   negative slope was not a cancellation artefact.
2. **Identifiability — this was the cause.** `w_m_eff` carries only 2.8 % of
   the C_L standard deviation in this campaign, so `CL_a` is barely
   determined. Pinning it to 2π costs 0.0015 NRMSE (0.03671 → 0.03817 train,
   0.04502 → 0.04690 valid). The data is indifferent, so the physical value
   is used rather than a noise-fitted one.
3. **Sign of the gust gain — real, not a bug.** `CL_g < 0` is correct in this
   codebase's convention: the LDNet itself gives `dC_L/dW < 0` at trim, and
   `structure.rhs` applies the load as `rhs_h = -Fy`, so a positive C_L
   pushes the section down in `h`. Both models share the convention.

Final pinned fit: `CL_a = 2π`, `CL_g = −10.05`, `CL_d = 1.450 /rad`,
`CL_0 = 0.856` (against the LDNet's trim C_L of 0.868).

## Harness validation

The `ldnet` arm reproduces the published chapter-3 table to
**max |deviation| = 0.05 pp** (mean 0.03 pp) across the completed cells —
81.2 / 89.1 / 91.5 / 93.5 / 95.2 / 93.4 — with matching peak flap angles.
The harness is faithful; the LDNet column is not re-derived, it is recovered.

## Result 1 — closed loop, all 18 cells (LDNet is the controlled system)

Each arm is tuned independently with chapter 3's own selection rule
(max CLred among tunings passing the stability check), so neither is judged
at the other's operating point. **Read Result 3b before quoting the gap.**

| cell | LDNet | ch3 | linear | gap |
|---|---|---|---|---|
| 10:0.3 | 81.2 | 81.2 | −266.5 | +347.6 |
| 10:0.4 | 89.1 | 89.1 | −45.6 | +134.7 |
| 10:0.5 | 91.5 | 91.5 | −33.9 | +125.4 |
| 10:0.7 | 93.5 | 93.5 | −22.4 | +116.0 |
| 10:1.0 | 95.2 | 95.2 | +7.7 | +87.5 |
| 10:1.2 | 93.4 | 93.4 | −13.9 | +107.4 |
| 20:0.3 | 86.0 | 86.0 | +25.2 | +60.9 |
| 20:0.4 | 87.8 | 87.8 | +6.4 | +81.4 |
| 20:0.5 | 88.8 | 88.8 | +25.2 | +63.6 |
| 20:0.7 | 91.9 | 91.9 | +9.9 | +82.1 |
| 20:1.0 | 81.2 | 81.2 | +30.1 | +51.1 |
| 20:1.2 | 58.2 | 58.2 | +26.9 | +31.3 |
| 30:0.3 | 39.4 | 39.4 | +34.4 | **+5.0** |
| 30:0.4 | 80.5 | 80.5 | −3.7 | +84.2 |
| 30:0.5 | 81.0 | 81.0 | +66.2 | +14.8 |
| 30:0.7 | 91.8 | 91.8 | +46.9 | +44.9 |
| 30:1.0 | 57.3 | 57.3 | +14.8 | +42.5 |
| 30:1.2 | 41.7 | 41.7 | +2.6 | +39.1 |

LDNet mean +79.4 %, linear mean −5.0 %, LDNet ahead on 18/18; the linear arm
violates the stability check on 3 cells, the LDNet arm on none.

**Harness validation: all eighteen LDNet values reproduce the published
chapter-3 table to within 0.05 pp**, including the hard corners (39.4, 57.3,
41.7).

The smallest gap, +5.0 pp, is at W30/Tg0.3 — the sharp-severe corner where
chapter 3 already reports that the actuator rate limit, not the model, is
the binding constraint. When no controller can do much, the internal model
matters least. That is physically coherent and worth stating.

## Result 2 — the mechanism (this is what makes the result defensible)

A number alone would not be persuasive; the failure mode was isolated.
`diag_mpc_internals.py` dumps, at instants of a real gust, the flap each
controller picks against the flap that is actually optimal on the plant:

| t | plant-optimal δ | linear picks | LDNet picks |
|---|---|---|---|
| 0.050 | +0.00° | **+5.07°** | +0.35° |
| 0.100 | +1.40° | **+8.92°** | +2.10° |
| 0.150 | +4.20° | **+8.75°** | +4.90° |
| 0.300 | −1.40° | **−7.18°** | −2.10° |

The linear controller **systematically over-commands by 2–5×**. Cause: its
`dC_L/dδ` is a constant 0.0253 /deg, while the plant's varies over
0.018–0.047 /deg (≈0.0383 near trim). Underestimating flap effectiveness, it
asks for too much flap, saturates at ±14°, overshoots and rings — pitch ratio
4–5. Its "belief error" (predicted minus actual C_L at its own choice) reaches
0.23, larger than the gust excursion it is trying to cancel.

## Result 2b — no effort weight rescues the linear model

The collapse is not a tuning artefact. Sweeping R over six orders of
magnitude on W10/Tg0.4 (`diag_linear_fair.py`):

| R | CLred | flap | pitch ratio | stability |
|---|---|---|---|---|
| 1e-4 | −150.8 | 9.8° | 4.47 | violated |
| 1e-3 | −163.0 | 8.4° | 4.39 | violated |
| 1e-2 | −45.6 | 3.3° | 1.56 | ok |
| 1e-1 | −1.6 | 0.5° | 1.03 | ok |
| 1e0 | **+0.00** | 0.0° | 1.00 | ok |
| 1e2 | +0.00 | 0.0° | 1.00 | ok |

There is no weight at which the linear controller both **acts** and **helps**.
Its best stable tuning (R = 1) achieves exactly 0.00 % — the controller has
been penalised into doing nothing. The LDNet reaches +89.1 % on the same cell.

## Result 2c — correcting the flap gain does not rescue it either

Since the failure is an over-command driven by a gain error, the obvious
objection is that the baseline's `CL_d` was simply mis-fitted. It was not:

| variant | CL_d [/deg] | W10/Tg0.4 | W30/Tg0.4 | mean |
|---|---|---|---|---|
| global lsq | 0.0253 | −45.6 | −3.7 * | −24.6 |
| **plant-matched** | **0.0383** | **−44.6** | **−2.1 \*** | **−23.3** |
| secant ±14° | 0.0225 | −46.9 | −4.3 * | −25.6 |
| \|δ\|>2° refit | 0.0251 | −195.9 * | −6.4 * | −101.1 |

(* violates the stability check at its selected tuning)

Matching `CL_d` *exactly* to the plant's near-trim sensitivity — a 51 %
increase — moves the result by about one percentage point. The four variants
span 0.0225–0.0383 /deg, a 70 % range that brackets the plant's true
sensitivity, and **every one of them fails**. A **constant** gain cannot work
whatever its value, because the plant's sensitivity varies by ~2.6× over the
deflection range. This closes the "you under-tuned the baseline" objection:
the deficiency is structural, not parametric.

The strongest linear variant is `plant-matched` at −23.3 % mean, and that is
the number the thesis should quote — not the −24.6 % of the first fit.

## Result 3 — held-out CFD, which removes the plant asymmetry

The closed-loop test uses the LDNet as the plant, which gives its controller
a perfect internal model. The honest check is prediction accuracy against
**held-out CFD**, ground truth for both models. Measuring the 8-step C_L
forecast — exactly what the MPC minimises:

**overall RMS: LDNet 0.0128 vs linear 0.1254 → 9.8× better**

Stratified, this locates the envelope region the thesis asks about:

| \|δ\| [deg] | linear | LDNet | ratio |
|---|---|---|---|
| 0–2 | 0.068 | 0.0073 | 9.3 |
| 2–5 | 0.128 | 0.0084 | 15.2 |
| 5–8 | 0.167 | 0.0212 | 7.9 |
| 8–11 | 0.162 | 0.0148 | 10.9 |
| >11 | 0.191 | 0.0128 | 15.0 |

| gust W [m/s] | linear | LDNet | ratio |
|---|---|---|---|
| 0–5 | 0.109 | 0.0110 | 9.9 |
| 5–15 | 0.180 | 0.0425 | 4.3 |
| 15–25 | 0.175 | 0.0266 | 6.6 |
| 25–35 | 0.279 | 0.0063 | 44.2 |
| >35 | 0.300 | 0.0206 | 14.6 |

The linear model's error grows by 2.8× with deflection and 2.8× with gust
severity; the LDNet's stays flat. **This is the "high $r_g$, large
deflections" region named in the conclusion, confirmed independently of the
closed loop.**

## Fairness measures taken

The linear collapse is severe enough that it was tested for
mis-specification rather than reported at face value:

* **Extended effort-weight search** (`diag_linear_fair.py`) — R swept far
  beyond chapter 3's ladder, up to 1e2, to check the collapse is not just an
  over-aggressive tuning.
* **Corrected flap gain** (`fit_linear3.py`) — three fairer variants, since
  the failure is a gain error and it must be shown that fixing the gain does
  not rescue the model: `ctrlgain` (CL_d = 0.0383 /deg, matched to the
  plant's near-trim sensitivity), `secant` (0.0225 /deg over ±14°),
  `lsqflap` (0.0251 /deg, refit on |δ|>2° samples only). **The variant that
  performs best is the one that should be reported.**
* **Symmetric falsification** (`run_symmetric.py`) — the loop re-run with the
  *linear* model as the plant, so the asymmetry favours the baseline instead.

## Result 3b — MOST OF THE CLOSED-LOOP MARGIN IS PRIVILEGE (read this first)

The symmetric falsification test changed the conclusion, and it must not be
buried. Re-running the loop with the **linear model as the controlled
system**, the **linear** controller wins on every cell tried, by 25-59 pp.
Whichever model is the controlled system, **its own** controller wins: the
signature of self-consistency, not of model quality.

The 2x2 decomposition (`analyse_symmetry.py`) separates the two effects:

|  | controller = LDNet | controller = linear |
|---|---|---|
| **system = LDNet** | A | B |
| **system = linear** | C | D |

* self-consistency effect = ((A-C) + (D-B))/2
* model effect = ((A-B) + (C-D))/2

Over **9 cells** (6 in `symmetric_linear.json` plus the 3 of the first run):

| cell | self | model |
|---|---|---|
| 10:0.4 | +84.3 | +50.4 |
| 10:0.7 | +71.4 | +44.6 |
| 30:0.4 | +61.7 | +22.4 |
| 20:0.7 | +66.5 | +15.6 |
| 20:0.4 | +70.1 | +11.2 |
| 30:1.2 | +34.6 | +4.6 |
| 20:1.2 | +28.3 | +3.1 |
| 30:0.7 | +50.3 | -5.4 |
| 30:0.3 | +16.7 | -11.7 |

* **self-consistency = +53.8 pp**
* **model effect = +15.0 pp**, 95 % bootstrap CI **[+2.7, +28.6]**,
  positive on **7/9** cells
* **78 % of the raw closed-loop margin** comes from matching the controlled
  system

The CI excludes zero, so a real model advantage does survive in closed loop,
but it is roughly a fifth of the raw +84 pp mean gap.

Note on an earlier reading: with only the first 3 cells the model effect came
out at +8.8 pp with a flipping sign, and was reported here as within noise.
Extending to 9 cells corrects that — the effect is real, just much smaller
than the raw margin suggests. The earlier figures (+48.3 / +8.8 pp, 85 %)
are superseded.

**Consequence for the thesis: the closed-loop table must not be quoted on its
own as evidence that the LDNet is the better model.** It measures a loop in
which one arm has an exact internal model. The accuracy claim rests on
Result 4 below, which has no such confound. The closed-loop table is still
worth reporting as the demonstration that the linear internal model cannot be
substituted in and left to run.

This is also why the mechanism evidence matters: over-command by 2-5x and a
flap authority varying by a factor 2.6 are properties of the models, not of
which one is the controlled system.

## Result 4 — the plant-asymmetry caveat, resolved

The main ablation's weakness is that the LDNet is the plant, so its
controller has a perfect internal model. `run_cfd_horizon.py` removes that
privilege entirely: **CFD is the plant**, both models see the true state and
the true flap history, and each rolls its own MPC horizon (N = 8, dt = 2e-3).
Neither model is the reference.

**linear RMS 0.12702 vs LDNet RMS 0.01294 → 9.81×, 95 % CI [8.83, 10.95]**
(paired bootstrap, 1320 probes over 10 held-out trajectories).
The LDNet is better at **96.7 %** of individual probes.

The interval excludes 1 by a wide margin, so this is not a seed or tuning
artefact. Stratified on CFD ground truth:

| \|δ\|max [deg] | linear | LDNet | ratio |
|---|---|---|---|
| 0–2 | 0.0823 | 0.0089 | 9.2 |
| 2–5 | 0.1748 | 0.0130 | 13.5 |
| 5–8 | 0.1638 | 0.0223 | 7.4 |
| 8–11 | 0.1633 | 0.0150 | 10.9 |
| **>11** | **0.1905** | **0.0127** | **15.1** |

| gust Wmax [m/s] | linear | LDNet | ratio |
|---|---|---|---|
| 0–5 | 0.0997 | 0.0113 | 8.8 |
| 5–15 | 0.2373 | 0.0292 | 8.1 |
| 15–25 | 0.2425 | 0.0201 | 12.1 |
| **25–35** | **0.2352** | **0.0114** | **20.7** |
| >35 | 0.2600 | 0.0202 | 12.9 |

This is the thesis claim, on CFD ground truth, with the plant privilege gone:
the advantage is largest exactly at **high $r_g$ and large deflections**.

### A method that did NOT work, and why

`run_cfd_plant.py` attempted to recover the empirical `dC_L/dδ` from the
campaign by partial regression, to score flap choices directly against CFD.
It failed: the returned slopes (~0.0005 /deg) are ~50× too small and the
lowest bin comes out negative. The reason is intrinsic to the data — in the
campaign the flap is commanded **in response** to the gust, so δ is
correlated with α, W and α̇, and the partial slope cannot be separated from
them. Trajectory data with a closed-loop flap history cannot yield a clean
flap-authority estimate by regression; that would need a δ sweep at fixed
state, which the campaign does not contain. Recorded so the attempt is not
repeated.

## Caveats that remain

1. **Result 4 is a forecast comparison, not a closed loop against CFD.** It
   measures the quantity the MPC consumes (the horizon C_L prediction) on the
   real system, but it does not simulate the controlled trajectory with CFD in
   the loop. That would need a live CFD plant (~370× per run — see
   `recon/analysis/COST_VS_ACCURACY.md`) or a δ sweep at fixed state, neither
   of which exists here.
2. The linear model is a *2D thin-airfoil* theory applied to a viscous RANS
   case with a hinged flap; that is precisely the comparison the thesis
   introduction sets up, but it is not the strongest conceivable linear
   model (no separation-onset correction, no Goman-Khrabrov state). A
   reviewer could fairly ask for one of those as a harder baseline.
3. The closed-loop numbers (Results 1–2) retain the LDNet-as-plant asymmetry
   and should be presented together with Result 4, not on their own.

## Files

| file | purpose |
|---|---|
| `linear_aero.py` | the linear unsteady model, LDNetAero-compatible |
| `fit_linear.py` | regressor construction + first (unconstrained) fit |
| `fit_linear2.py` | pinned and free fits → `linear_coeffs*.json` |
| `fit_linear3.py` | fairer flap-gain variants |
| `run_ablation.py` | one cell, both arms, chapter-3 metrics |
| `sweep.py` | CS-25.341 grid with per-cell R* selection |
| `launch.sh` | parallel per-cell driver → `shards/` |
| `collect.py` | merge, chapter-3 regression check, envelope map, LaTeX |
| `replay_cfd.py` | 8-step forecast error vs held-out CFD |
| `run_symmetric.py` | falsification test with the linear model as plant |
| `diag_*.py` | the diagnostics behind every claim above |
