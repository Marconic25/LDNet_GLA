# 76/ — robustly reaching the +76% GLA branch at W30/Tg0.4, DAMULT=3

**Bottom line.** The +76.1% "good branch" is a **real, robust attractor**, not just a
lucky rounding. It is reached **deterministically by a single-step controller** whose only
change from the honest one-step optimal is evaluating flap candidates against the *known
next-step gust* **W(t+dt)** instead of W(t). This is a legal single-step move (W is an
oracle; its next value is known). At W30/Tg0.4 it gives **+76.5%**, and a perturbation
ensemble (X0 ± 1e-6, W0 ± 0.1%) holds it to **0.4 CLred points of spread** — versus a
**59.7-point** spread for the plain W(t) controller, which lives on a chaotic knife-edge
between the +16.6% and +76.1% branches.

Reference frame (all at W30/Tg0.4, DAMULT=3, gust window t ≤ Tg+0.5):
`open cex0 = 0.4600`, `CLTRIM = 0.8683`, `CLred = (cex0 − cex)/cex0`.
Success bar = beat prop-W (+39.1%) robustly; target ≈ +76%.

Everything here is an **honest B=1 scalar rollout** (deterministic, batch-position
independent). Physics (`ldnet_aero.py`, `structure.py`) is byte-identical to `light/` =
`clean/`. Because `dt == dt_ref = 0.002` (n_sub=1), the scalar `advance()` and the
leak-corrected `batch_step()` are the same map up to TF batch-position rounding — so the
scalar rollout and the batched oracle differ *only* by that rounding.

---

## H0 — Characterize the good vs bad branch (`characterize.py`, `analyze_char.py`)

Batched oracle (= `clean/propw.py batch_optW`, leak-corrected), **identical config in every
row** `R_GRID=[3e-4]×5` to isolate pure batch-position rounding:

| rows | cex | CLred | flap_max | pitch pk |
|---|---|---|---|---|
| 0–3 (good) | 0.1100 | **+76.1%** | 7.5° | 0.153° |
| 4 (bad, last batch position) | 0.3836 | +16.6% | 6.1° | 0.156° |

Reference R-sweep `[1e-2,3e-3,1e-3,3e-4,1e-4]` (rows 0–3 are "good" batch positions):
`−19.8, +35.9, +66.4, +76.1, +5.3%`. Inside the good basin the R-response is **smooth and
monotone** (more authority → better) up to R=3e-4; only the last batch position collapses.
→ **The good branch is the attractor; the bad branch is the rounding artifact.**

**Where the branches fork.**
- **Seed at t=0:** state = trim, W=0 ⇒ `cl0 == CLTRIM` *exactly*. The causal gate
  `cl0 ≥ CLTRIM` is an exact tie decided by last-bit rounding: good → gate "below trim" →
  strictly-positive flap half → forced nudge **δ=+0.175°**; bad → gate "above trim" → **δ=0°**.
- **Decision that matters (t≈0.14–0.24, gust rise→peak):** natural CL drops (gust lowers
  effective AoA); both add positive flap to hold CL up. The **good** branch uses slightly
  *less* positive flap (+5.25 vs +5.6°) and **reverses early** (dumps +5.4→+0.2° by t=0.24,
  then drives −7.5° to arrest the rebound) — CL stays within ±0.11. The **bad** branch
  *over-commits* positive flap (+5.6…+5.95°), reverses ~68 ms late (gate flips t=0.31 vs
  0.24), and CL **crashes to 0.48** (pushed into the non-monotone wrong-sign region).
- **Every step is a near-tie** (argmin margins 1e-5…1e-4). Genuine knife-edge ⇒ the fix must
  *remove* the sensitivity, not just fix t=0.

---

## H1–H6 — Single-step modifications (`controllers.py:OptGrid`, `variants.py`, `variants2.py`)

Honest B=1 scalar, W30/Tg0.4. Baseline = the honest one-step optimal on the frozen-z
reconstruction, 161-pt flap grid, causal gate.

| # | hypothesis | config | CLred | verdict |
|---|---|---|---|---|
| — | baseline scalar | hard gate, G161, R=3e-4 | **+16.6%** | lands on the **bad** branch |
| — | baseline scalar | hard gate, G161, R=1e-3 | +31.5% | best plain-scalar R |
| — | light replica | G15, R=1e-3, +minimize_scalar refine | −0.2% | refine *destabilizes* |
| H1 | causal-gate **deadband** | db ∈ {2e-3,5e-3,1e-2,2e-2}, R=3e-4 | **+16.6% (all)** | **inert** — fork is during rise (cl0 far from trim), not at the near-trim gate |
| H2 | **R_du** move-suppression | R_du ∈ {1e-4…3e-3}, R=3e-4 | +16.8 → +19.8% | plateaus; R_du=1e-2 over-damps → −31% |
| H3 | **no gate** + R_du | gate=none, R_du∈{1e-3,3e-3} | +17.8 / +7.6% | no |
| H4 | **z-aware** candidate eval (`predict_step`) | R=3e-4 / 1e-3 | +25.7 / +32.1% | helps a little; not enough |
| H5 | **W(t+dt)** phase lead | hard gate, G161, **R=3e-4** | **+76.5%** | ✅ **WINNER** (flap 7.9°, pitch 0.151°, α̇rms 0.86, no explosion) |
| H5 | W(t+dt), R=1e-3 | | +66.6% | too hot (α̇/α̈ explode) |
| H6 | z-aware **+** W(t+dt) | R=3e-4 | +76.9% | W(t+dt) is the driver; z-aware adds a hair |

**Interpretation of H5.** The one-step cost with W(t) chooses δ to null the *current*-step CL,
but the plant then advances into W(t+dt); the frozen-z prediction is a half-step stale, so the
optimizer over-commits positive flap during the rise (→ the bad branch). Feeding **W(t+dt)**
gives exactly the one-step phase lead the good branch had — it reverses the flap earlier — so
the deterministic scalar controller lands in the good basin *by construction*.

Deadband/hysteresis (H1) is inert precisely because the characterization showed the
branch-deciding divergence happens mid-rise where `cl0` is 0.6–0.7 (far outside any sane
deadband), not at the t=0 trim tie. Cost-shaping on the frozen-z prediction (H2/H3) cannot
tunnel the scalar rollout out of the bad basin; only fixing the *information* (the gust phase)
does.

---

## H7 — Short-horizon MPC fallback (`controllers.py:MPCConst`)

Constant-δ receding horizon, batched `batch_step` + leak correction, **no DLPF smoothing**
(the smoothing that cripples fast gusts — `clean/CONTROLLER_NOTES.md`). Gust held at W(t)
over the horizon.

| config | CLred |
|---|---|
| N=2 / 3 / 4 / 6, causal gate, R=3e-4 | +24.2 / +17.1 / +20.7 / +26.9% |
| N=4, causal gate, R=1e-3 | +30.1% |
| N=4, causal gate, R=3e-4, +Q_α̇=50 | +19.9% |
| **N=4, gate = none, R=3e-4** | **+79.2%** ✅ |

**Horizon alone (with the gate) does not help** — it stays on the bad branch. What unlocks the
good basin is **removing the causal gate**: over N=4 steps the accumulated cost already
penalizes the wrong-sign flap (holding it for 4 steps produces a large CL excursion), so the
horizon *replaces* the gate's job while avoiding its chattering discontinuity. This is a
multi-step result and is secondary to the single-step H5 winner.

---

## Robustness of the winner (`robustness_ensemble.py`)

Perturbation ensemble at W30/Tg0.4 — 17 members: X0 ± 1e-6 per component (≈1e10× the rounding
scale), W0 ± 0.1%, and random combinations. Metric vs the matching perturbed open loop.

| controller | CLred range | spread | verdict |
|---|---|---|---|
| **SS wnext R=3e-4** | +76.2% … +76.6% | **0.4 pts** (std 0.09) | **robust attractor** ✅ |
| plain W(t) R=3e-4 | +16.4% … +76.1% | 59.7 pts (std 28.3) | chaotic — a 1e-6 kick flips the branch |

The contrast is the whole story: the plain optimizer is a coin-flip between branches at the
rounding scale; **wnext removes the sensitivity and pins the good branch.**

The multi-step alternative **MPC N4 gate=none R=3e-4** is equally robust
(`robustness_ensemble_mpc.py`, 9 members): **+79.1% … +79.4%, spread 0.3 pts** (std 0.08).
Two independent controllers pinning the good branch ⇒ it is genuinely optimal, not a fluke.

---

## Generalization to neighbour cells (`generalize.py`)

R-sweep only, **no per-cell hand-tuning** (same controller, sweep R ∈ {3e-4,1e-3,3e-3}).
Baselines: open loop, best prop-W over a small gain grid.

| cell | open cex0 | prop-W best | **wnext R=3e-4** | MPC4-none R=3e-4 |
|---|---|---|---|---|
| W30/T0.4 (home) | 0.460 | −84.7% (flap sat, α̇/α̈ explode) | **+76.5%** clean | +79.2% clean |
| W30/T0.3 (sharpest, k highest) | 0.800 | +39.9% (flap sat, explode) | +35.5% (α̇ high, no flag) | +38.4% |
| W30/T0.5 | 0.496 | −61.9% (explode) | **+75.0%** clean (α̇rms 0.80) | +81.2% clean |
| W20/T0.4 | 0.301 | +77.7% clean | **+77.3%** clean | +87.0% clean |

- **wnext R=3e-4 transfers with a single R:** +76.5 / +35.5 / +75.0 / +77.3%, flap ≤ 12.2°,
  no explosion at 3/4 cells. The R-sweep confirms R=3e-4 is the universal choice (R=1e-3/3e-3
  give less reduction and ring at the strong cells) — no per-cell tuning needed.
- **prop-W is gain-fragile:** with gains not hand-tuned per cell it goes catastrophic
  (−84.7%, −61.9%, saturating and tripping the α̇/α̈ explosion flags). Its reference "+39.1%"
  requires a per-cell-tuned gain; wnext beats it robustly *without* that tuning.
- The sharpest gust (Tg=0.3, reduced freq k=π/(80·0.3)=0.131 — envelope edge) is hard for
  everyone: wnext and MPC ≈ prop-W (+35–40%), but only wnext/MPC stay off the explosion flag.
- **MPC4-none** transfers slightly better still (+79/+38/+81/+87%) but is the multi-step path.

Money plot (`moneyplot.py` → `money_W30_Tg04.png`): open cex 0.460, prop-W best in that grid
+32.2% (g=−60,−0.5; prop-W's optimum jumps around with the grid — fragile), **wnext +76.5%**.

---

## Is W(t+dt) legal? Preview realizability & LIDAR (`preview_study.py`, `mpc_noise.py`)

The wnext winner feeds the optimizer the **one-step-ahead gust W(t+dt)** = a preview of
dt = 2 ms = **0.16 m ahead** at U=80 m/s. Two-level justification:
1. The study already assumes W is known (prop-W and opt-W are gust-oracle controllers; C_L(W)
   is non-invertible ⇒ W is not observable from C_L ⇒ a gust sensor is required either way).
   W(t+dt) is the same class of assumption, one 2 ms step further.
2. The physical sensor is **forward-looking Doppler LIDAR**, the standard GLA gust sensor,
   which is *intrinsically* a preview sensor (measures the wind field 30–200 m ahead ≈
   190–1250 steps at 80 m/s). A 1-step preview is the mildest possible preview assumption.
   Established field: LIDAR-based feedforward/H∞-preview GLA on aircraft (>39% wing-root
   bending, >65% tip-accel reductions reported) and LIDAR-assisted feedforward pitch control on
   wind turbines (commercially deployed, Schlipf et al.).

**A) Preview horizon** — CLred for W(t+k·dt): k=0 +16.6% (no preview, bad branch), k=1 +76.5%,
k=2 +76.8%, k=3 +77.0% (clean), k=5 +80.2% (rings), k≥10 degrades (+31–35%, rings). The +76%
is not knife-edge on k=1 (k=1–3 all ≈+77%); the controller wants a *short* preview because it
uses only the k-ahead sample — real LIDAR far exceeds this.

**B) Causal extrapolation fails** — replacing the future sample by a causal predictor of the
known W (2·W(t)−W(t−dt) or quadratic, present+past only) gives **+17.3%**, no better than W(t).
Only the *true* future sample recovers +76.5%. ⇒ The result genuinely needs a preview sensor;
it cannot be faked with a gust-rate estimator, and it is not a hidden causal trick.

**C) Preview noise (white, worst-case) — the honest limitation.** White Gaussian error on the
gust the controller sees (plant uses the true gust), σ = frac·W0, 6 seeds:

| σ [m/s] | SS wnext (preview) | MPC N4-none (no preview) |
|---|---|---|
| 0 | +76.5% | +79.2% |
| 0.6 (2%) | +5.2% (−25…+29) | +33.1% (−4…+79, 4/6 flag) |
| 1.5 (5%) | −17.4% | +21.8% |
| 3.0 (10%) | −20.4% | +20.4% |

Both are noise-sensitive; the **single-step greedy argmin is a knife-edge** (needs a
near-exact preview — consistent with B). The **MPC horizon integrates** and degrades far less
(mean +33% vs +5% at 2%, best seeds still ≈+79%) — and, crucially, **MPC needs NO preview**
(uses only W(t)), sidestepping the preview question entirely. But white noise makes both chatter
(explosion flags), because it is full-bandwidth (500 Hz) — the worst case. Real LIDAR noise is
band-limited/temporally-smoothed, so this is pessimistic. Band-limited test: `mpc_noise2.py`.

**Band-limited noise (`mpc_noise2.py`, 5% white + 1st-order LPF, τ sweep), mean CLred:**

| τ [ms] | MPC N4-none (no preview) | SS wnext (preview) |
|---|---|---|
| 0 | +48.1% (std 38, 2/6 flag) | −14.2% (3/6) |
| 5 | +50.9% (std 29) | −1.0% |
| 10 | +51.0% (std 26, 3/6) | +19.8% (std 21, 1/6) |
| 20 | +41.4% (6/6 flag) | +12.5% |
| 40 | −15.9% (6/6 flag) | −9.3% |

There is a **τ sweet spot (~5–10 ms), then collapse**: too much filtering *lags* the gust (which
rises over ~100 ms → a 40 ms filter delays the estimate → wrong-phase flap → pitch excitation,
6/6 explosion flags). So at this sharpest cell there is a genuine tension — filter enough to kill
white-noise chatter but not so much you lag the gust — and the window is too narrow to fully
recover. Even at the best τ, MPC ≈ +51% (high variance, individual seeds −6…+67%) and wnext ≈
+20%. Band-limiting *partially* helps but does **not** make either controller cleanly robust to
5% gust-estimate noise here; that needs a noise-robust preview-control reformulation (as the
LIDAR-GLA literature does) or explicit flap-rate regularization, not a band-aid filter.

**Honest framing.** All of prop-W, wnext and MPC are gust-*oracle* controllers (they assume
accurate W — the study's consistent premise). The +76%/+79% headline figures are legitimate at
that oracle level (fair vs prop-W, same oracle). Noise sensitivity is a *shared* caveat, most
severe for the argmin-based single-step, less for the horizon MPC, and it is the natural
open problem for a physical LIDAR implementation.

**Takeaway.** W(t+dt) is legal and physically realizable (LIDAR), but the single-step +76%
requires an *accurate* preview; the multi-step MPC reaches +79% **without any preview** and is
the more noise-tolerant, practically-robust controller.

---

## Final answer

- **Is +76% robustly reachable single-step?** **Yes.** The one-step optimal evaluated against
  the known next-step gust **W(t+dt)** reaches **+76.5%** deterministically and holds it to
  **±0.2 pts** under X0/W0 perturbations. It stays single-step, uses ≤ 14° flap, keeps pitch ≈
  open-loop, and does not explode.
- **Only via MPC?** No — but N=4 MPC with the causal gate *removed* independently reaches
  **+79.2%**, confirming the good branch is genuinely optimal (two different controllers find
  it) rather than a fluke.
- **Not at all?** The plain frozen-z W(t) one-step optimizer cannot reach it robustly (chaotic,
  59.7-pt ensemble spread). Cost-shaping (deadband, R_du, gate removal, z-aware) does not fix
  it; only the correct gust *phase* (W(t+dt)) or a gate-free horizon does.

Deliverable controller: **`OptGrid(R=3e-4, G=161, gate='hard', use_wnext=True)`**.

Files: `characterize.py`/`analyze_char.py` (H0), `controllers.py` (all laws),
`variants.py`+`variants2.py` (H1–H7), `robustness_ensemble.py`(+`_mpc.py`), `generalize.py`,
`moneyplot.py` → `money_W30_Tg04.png`.

---

## Honest equal-information comparison (`honest_home.py` → `results_honest/`)

Direct answer to "is the wnext win just the preview?" — home cell W30/Tg0.4, DAMULT=3, same
scalar B=1 harness, and now the **same information set for both laws**: {x(t), C_L measurement,
W(t), **W(t+dt)**}. The prop exploits the preview the only way a static linear law can: the
feedforward term becomes `g_W·W(t+dt)` (`PropW(use_wnext=True)`), i.e. the identical 2 ms phase
lead the wnext optimal gets. Same tuning discipline for both: full sweep + **no-flag pick**
(max CLred s.t. no α̇/α̈/ḧ explosion flag). Prop sweeps the full 5×6 gain grid; optimal sweeps
the standard 5-value R grid.

| arm | best (no-flag pick) | config | flap | pitch pk |
|---|---|---|---|---|
| open | cex0 = 0.4600 | — | — | — |
| prop-W(t+dt) | **+32.0%** | g_CL=−60, g_W=−0.5 | 6.3° | 0.154° |
| optimal wnext | **+80.7%** | R\*=1e-4 (R=3e-4 → +76.6%) | 8.0° | 0.156° |

- **The preview is worth nothing to the prop**: +32.0% vs +32.2% with W(t) (moneyplot, same
  grid) — best at the *same* gains (−60, −0.5). A smooth static law has no argmin knife-edge to
  stabilize, and a 2 ms shift of a 400 ms gust barely changes the feedforward signal. The wnext
  gain is *not* "preview beats no-preview"; it is the model-based optimizer needing the correct
  gust *phase* to pick the good branch.
- **Prop gain fragility, quantified**: 24/30 grid combos explode (flap saturates at 14°,
  `ad!add!`, pitch ~0.5°) — including ALL pure-feedback columns (g_W=0). Only a narrow diagonal
  band (g_CL·g_W trade-off) is stable; the working range is ~6 clean combos.
- **What the optimal does that the prop cannot** (money plot `results_honest/honest_W30_Tg04.png`):
  same actuator budget (7.8° vs 6.3° flap), but the optimal *leads* the gust — flap peaks +7.7°
  during the rise and reverses ~50 ms earlier — holding C_L within ±0.09; the prop, reacting
  through the C_L error + a fixed-gain W term, lets C_L dip to 0.56 and rebound to 0.96.
- Regressions: open cex0 = 0.4600 exact; R=3e-4 → +76.6% (ref +76.5%); R\*=1e-4 → +80.7%,
  matching the cs25_wnext2 W30/T0.4 cell (the no-flag pick unlocks R=1e-4 here, clean).

**Verdict.** At equal information and equal tuning/pick rules, the one-step optimal beats the
proportional law +80.7% vs +32.0% (≈49 pts). The advantage is the **LDNet model in the loop**
(nonlinear inversion of C_L(δ, W, x) each step), not the preview sample.

---

## CS-25.341 gust study (`cs25_wnext.py` → `cs25_wnext2.py`)

Full 3×6 grid (W0∈{10,20,30} × Tg∈{0.30,0.40,0.50,0.70,1.00,1.20}), R swept per cell,
CS-25.341 framing H=U·Tg/2, k=π/(U·Tg); 4-panel plots per cell (C_L, δ, W, α̇) in the
`light/tests/test_optimal.py` style. Two defects of the first sweep and their fixes — **cost
J = (C_L−C_L*)² + R·δ² unchanged**:

1. **Staircase δ** = 161-pt flap-grid quantization (argmin snaps to grid points → flat-then-step).
   **Fix: `refine=True`** — parabolic sub-cell interpolation of the argmin on the SAME grid →
   continuous δ (roughness Σ|Δ²δ| drops ~8×, 28→3.4) with the SAME argmin ⇒ SAME good branch
   (CLred essentially unchanged). A **finer grid is the wrong fix**: G=481 changes the near-tie
   discretization and jumps to a worse branch (W30/T0.4 +82→+45%). Refine, not resolution.
2. **Inconsistent CLred** (naive sweep crashed at W30/T1.0 +0.2%, T1.2 −1%, W20/T0.5 +11.5%).
   Cause = the `pitch ≤ 1.10×open` pick rule, whose ratio is inflated where the open-loop pitch
   is small; the aggressive R that gives +64…+88% is rejected for pitch ~1.1–1.5. **Fix: pick R\*
   by MAX CLred subject to NO explosion flag** (α̇/α̈/ḧ < 3×open — the physical instability gate);
   pitch reported for transparency. This is an *evaluation-criterion* change, not an objective one.

Result (fixed sweep, refine + no-flag pick), CLred per cell:

| W0 \\ Tg | 0.30 | 0.40 | 0.50 | 0.70 | 1.00 | 1.20 |
|---|---|---|---|---|---|---|
| 10 | +93.1 | +93.2 | +94.3 | +95.8 | +96.4 | +95.8 |
| 20 | +69.5 | +87.7 | +88.1 | +92.3 | +81.0 | +63.1 |
| 30 | +37.2 | +80.7 | +82.2 | +90.6 | +64.2 | +40.9 |

W20/W30 are now consistent (no crashes; smooth arc peaking at mid-Tg, rolling off at the sharp
and long extremes). Recovered cells pay pitch up to 1.47×open but carry no instability flag. The
only genuinely hard cell is **W30/T0.30 (+37%)** — the sharpest gust (k=0.131, envelope edge), a
physical limit not removable without changing the objective. **Deliverable controller:**
`OptGrid(R*, G=161, gate='hard', use_wnext=True, refine=True)`. Plots in `results_cs25_wnext2/`.
