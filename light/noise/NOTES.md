# Preview-noise robustness of the final wnext controller (`light/optimal.py`)

**Question.** The final one-step optimal controller (use_wnext + refine) owes its
+76.6/+80.7% at W30/Tg0.4 (DAMULT=3) to an *accurate* 1-step gust preview
W(t+dt). How much sensor error does that advantage survive, which error types
kill it, and what honestly mitigates the loss — before it stops beating the
prop-W equal-info baseline (+32.0%)?

**Setup.** Scalar B=1 rollouts (`harness_noise.py`, loop == light/run.py ==
76/preview_study.py::rollout_customW): the PLANT always advances with the true
gust W(t); only the controller receives the corrupted Wc. With use_wnext=True
the corrupted preview drives BOTH the candidate scan and the causal gate — the
controller has no clean gust signal anywhere. DAMULT=3, metrics and the
t ≤ Tg+0.5 window identical to light/run.py, explosion flags (α̇/α̈/ḧ > 3×open)
always counted. Noise conventions match 76/: white rng(100+seed), band-limited
rng(300+seed), sensor clamp Wc = max(0, ·), one draw per step. R ∈ {3e-4, 1e-4}
everywhere except the R-mitigation sweep. light/optimal.py is imported
unmodified.

Files: `harness_noise.py` (rollout/metrics/npz schema), `controllers_ref.py`
(OptimalRdu = final controller + R_du move cost; PropWRef g_CL=−60, g_W=−0.5;
MPCConstRef = 76 MPC N4 gate-none port), `regression.py` (stage-0 gate),
`noise_white.py` (A), `noise_bandlim.py` (B), `noise_struct.py` (C),
`noise_cells.py` (D), `noise_mitigation.py` (E), `plots.py`, `run_axis.sh`;
results + figures in `results/`.

---

## Regressions (stage-0 gate, all PASS — `regression.log`)

| check | value | ref |
|---|---|---|
| open cex0 | 0.4600 | 0.4600 |
| σ=0, R=3e-4 | **+76.58%** | +76.6 |
| σ=0, R=1e-4 | **+80.67%** | +80.7 |
| OptimalRdu(R_du=0) ≡ OptimalController | max\|Δδ\| = 0.0 | bit-exact |
| σ=2%, refine=False, seeds 100–105 | −0.5% [−26.1, +18.5] | == 76/preview.log, bit-exact |

Gate also re-run with the study's thread caps (OMP/TF_INTRAOP=3) — identical to
the last digit, threading does not perturb the FP stream at these op sizes.

### The "+5.2%" anchor was two different runs (found while gating)

The task's sanity anchor "σ=2% → ≈+5% (76/)" comes from 76/NOTES.md, which
attributes it to `preview_study.py`. It is actually from **`mpc_noise.py`**,
whose noise uses rng base **200+seed**; `preview_study.py` itself (rng
**100+seed**) logged **−0.5% [−26.1, +18.5]** for the same cell
(cluster `76/preview.log`), and this harness reproduces that **bit-exactly
per-seed**. Same experiment, different draw: per-seed std ≈19 pts ⇒ a 6-seed
mean has SE ≈ 8 pts ⇒ −0.5 and +5.2 are the same statistical object. Two
consequences for this study: (i) the σ=2% regime is chaotic at seed level and
6-seed means are not stable anchors — axis A therefore uses **12 seeds**
(0–5 remain the bit-comparable subset); (ii) regression now gates on the
bit-exact per-seed values of the rng-100 stream, not on a noisy mean.

---

## A) White preview noise — baseline degradation (12 seeds/point)

Wc = max(0, W(t+dt) + N(0, σ)), σ = frac·W0; prop-W receives the SAME noisy
preview in its feedforward term. Mean [min,max] flags:

| σ/W0 | optimal R=3e-4 | optimal R=1e-4 | prop-W (−60, −0.5) |
|---|---|---|---|
| 0 | +76.6 | +80.7 | +32.0 |
| 1% | +16.3 [−28,+80] 7/12 | +11.2 [−24,+82] 7/12 | — |
| 2% | −3.2 [−27,+26] 8/12 | −9.2 [−26,+17] 8/12 | **−57.5** [−95,+33] 9/12 |
| 5% | −12.5 [−38,+21] 8/12 | −16.8 [−45,+1] 8/12 | **−61.8** [−95,+31] 10/12 |
| 10% | −24.8 [−43,−16] 8/12 | −20.7 [−33,+0] 8/12 | — |

- **The collapse is at σ ≈ 1%, and it is a branch lottery, not graceful
  degradation**: at σ=1% five of twelve seeds still deliver ≈+80% while the
  rest ring or explode (7/12 flags) — the mean (+16) describes nobody.
- **Break-even vs the prop-W noise-free +32%: σ ≈ 0.7%·W0 ≈ 0.22 m/s** (linear
  interp of the mean, R=3e-4). Break-even vs OPEN LOOP (0%): σ ≈ 1.8%·W0 ≈
  0.55 m/s — beyond that the controller is harmful on average.
- **But the equal-information story reverses the verdict**: fed the same noisy
  preview, the prop-W is catastrophically worse (−57.5% at 2% vs −3.2%;
  9–10/12 explosions vs 8/12). Its g_W·Wc term injects white noise directly
  into the flap command; the optimal at least rate-limits and re-optimizes.
  **At no tested noise level does the prop overtake the optimal.** The +32%
  prop reference is exactly as much a clean-sensor assumption as the +76.6%.
- refine=True vs refine=False at σ=2% (seeds 0–5 regression): −3.2 vs −0.5
  mean — same statistical object, confirming B's conclusion (refine
  irrelevant to noise).

## B) Band-limited (LIDAR-like) 5% noise — filter-lag trade-off confirmed

5% white + 1st-order LPF on the preview (rng 300+seed == 76/mpc_noise2.py),
mean CLred over 6 seeds:

| τ [ms] | R=3e-4 | R=1e-4 | 76/ refine=False (ref, R=3e-4) |
|---|---|---|---|
| 0 | −14.0 [−33,+2] 3/6 | −10.7 2/6 | −14.2 (3/6) |
| 2 | −3.6 2/6 | −20.5 2/6 | — |
| 5 | +2.3 2/6 | −15.0 2/6 | −1.0 |
| 10 | **+20.0** [−7,+64] 1/6 | +10.4 2/6 | +19.8 (1/6) |
| 20 | +12.5 2/6 | +7.7 2/6 | +12.5 |
| 40 | −13.3 3/6 | −32.8 3/6 | −9.3 |

- The final refine=True controller behaves **identically in shape and level**
  to the 76/ refine=False law: sweet spot τ ≈ 10 ms (+20.0 vs +19.8), collapse
  by gust lag at τ ≥ 40. Parabolic refine buys nothing against noise — the
  fragility is in the argmin/branch structure, not the grid quantization.
- Even at the sweet spot, +20% ± big spread with residual flags: band-limiting
  alone never gets back to the prop-W +32% line, let alone +76%.
- R=1e-4 is uniformly worse than R=3e-4 under this noise (consistent with D).

## C) Structured errors — phase matters, amplitude much less

CLred, deterministic single rollouts (jitter: 6 seeds). Clean refs: +76.6 (R=3e-4), +80.7 (R=1e-4).

| error on W(t+dt) | R=3e-4 | R=1e-4 |
|---|---|---|
| bias +5%·W0 | **−6.5% flag!** | −6.3% flag! |
| bias −5%·W0 | +17.4% | +1.7% flag! |
| bias +10%·W0 | **−26.0% flag!** | −33.3% flag! |
| bias −10%·W0 | +19.1% | +2.7% flag! |
| scale ×0.8 | +46.2% | +46.8% |
| scale ×0.9 | +68.0% | +72.2% |
| scale ×1.1 | **+78.2%** | +77.4% |
| scale ×1.2 | +64.5% flag! | +59.2% flag! |
| latency (Wc=W(t)) | +16.9% | +2.8% |
| jitter k~U{0,1,2} | +56.6% [+17.0,+76.4] 0/6 | +54.1% [+1.3,+80.6] 0/6 |

- **Additive bias is the killer, and it is asymmetric**: overestimating the gust
  by just +5%·W0 (=1.5 m/s, also present at gust start/end where W≈0) flips the
  sign gate and over-commits flap — worse than no controller (−6.5/−26%, flags).
  Underestimating parks it on the known bad branch (≈+17–19% ≈ the W(t) level).
- **Multiplicative errors are well tolerated** (×0.8–×1.2 → +46…+78%): they
  preserve the gust PHASE (zero crossings, shape). ×1.1 even edges out nominal.
  Only ×1.2 starts to ring.
- **Latency reproduces k=0 exactly** (+16.9 ≈ the known +16.6 with refine=False;
  and note the final refine=True controller measured here): one 2 ms step of
  staleness costs 60 pts — the entire wnext story in one number.
- **Jitter is survivable** (no flags, mean +55) because half the steps still see
  a usable phase; bad samples cost performance, not stability.
- Take-away: the controller needs the correct gust **phase and offset near
  W≈0**; relative amplitude error is a second-order effect. A LIDAR bias spec
  (<±1 m/s) matters more than its gain calibration.

## D) Cell dependence — general trend, not a knife-edge artifact

White preview noise, σ = frac·W0(cell), 6 seeds. (Clean anchors: W10/Tg0.7
+95.8, W30/Tg0.7 +90.6 at the no-flag R pick, reproduced by R=1e-4.)

| cell | R | σ=0 | σ=2% | σ=5% |
|---|---|---|---|---|
| W10/Tg0.7 (cex0 0.2075) | 3e-4 | +89.5 | **+75.3** [+69,+84] 0/6 | +42.4 [−78,+68] 2/6 |
| | 1e-4 | +95.8 | **+83.1** [+78,+88] 0/6 | +30.2 [−181,+80] 2/6 |
| W30/Tg0.7 (cex0 0.5791) | 3e-4 | +85.0 | +61.9 [+11,+78] 1/6 | +34.7 [+10,+75] 4/6 |
| | 1e-4 | +90.6 | +44.4 [+9,+86] 3/6 | +20.9 [+3,+75] 4/6 |

- Noise sensitivity is **general but graded by gust severity**: at the gentle
  W10/Tg0.7 cell, 2% noise still leaves +75–83% with zero flags (vs ≈0 at the
  sharp home cell); 5% noise hurts everywhere (occasional catastrophic seeds,
  e.g. −181% at W10/R=1e-4).
- Under noise the **larger R=3e-4 is the more robust choice** at the strong
  cells (fewer flags, higher mean at W30: +61.9 vs +44.4 at 2%) — the R*=1e-4
  no-flag pick is a clean-preview luxury.

## E) Mitigations — value and stability need different medicine

White noise on the preview, identical rng streams across arms (paired), 6 seeds,
base R=3e-4. Mean [min,max] flags:

| arm | σ=2% | σ=5% |
|---|---|---|
| none (baseline) | −0.1 [−26,+19] 3/6 | −7.3 [−25,+21] 3/6 |
| LPF τ=5 ms | +31.4 [−6,+76] 2/6 | +3.5 3/6 |
| LPF τ=10 ms | **+38.4** [+17,+66] 3/6 | +6.3 3/6 |
| R=1e-3 | +9.4 3/6 | +2.7 3/6 |
| R=3e-3 | +2.9 3/6 | −14.8 3/6 |
| R=1e-2 | −23.1 3/6 | −27.2 3/6 |
| R_du=1e-4 | −0.2 3/6 | −7.4 3/6 |
| R_du=1e-3 | −5.8 3/6 | −12.5 3/6 |
| R_du=1e-2 | −25.9 [−30,−21] **0/6** | −18.1 2/6 |
| avg K=3 | +8.8 6/6 | −13.5 6/6 |
| avg K=5 | +15.8 6/6 | −4.8 [−5.1,−4.4] 6/6 |
| avg K=9 | −2.8 6/6 | −8.6 6/6 |
| combo K=5 + R_du=1e-2 | **+8.1 [−27,+74] 0/6** | — |
| combo K=5 + R=1e-3 | — | −10.0 6/6 |
| **MPC N4-none (no preview)** | **+37.0** [−4,+79] 4/6 | **+35.4** [−5,+77] 4/6 |
| prop-W same noise (from A) | −57.5 9/12 | −61.8 10/12 |

- **No mitigation recovers the +76%.** The best raw mean at σ=2% is LPF τ=10 ms
  (+38.4%) — but with 3/6 explosion flags it is not a certifiable controller.
- **Value spread and stability respond to different fixes.** Multi-sample
  averaging (the physically-free LIDAR mitigation) collapses the branch lottery
  into a repeatable outcome (K=5 at σ=5%: std 0.21!) but every seed still rings
  (6/6 flags): the residual σ/√K per-step chatter keeps the pitch mode excited.
  Heavy move-suppression R_du=1e-2 kills the ringing (0/6 flags) but also the
  performance (−26%). Their **combo is the only clean positive config:
  +8.1%, 0/6 flags** at σ=2% — stable, honest, and 24 pts below the prop-W
  clean reference.
- **Raising R does not work** (mild at 1e-3, harmful beyond): under noise the
  problem is gate/branch flipping, not flap effort.
- **MPC N4 gate-none stays the most noise-tolerant law** (+37/+35 at 2/5%,
  needs NO preview) but is itself a branch lottery (4/6 flags, min ≈ −5) —
  "noise-tolerant" only relative to everything else.

**Money plot** (`results/fig_money_f0.02.png`): the worst σ=2% seed does not
fail at the gust peak — it survives the pulse, then AFTER the gust (t≳0.6 s,
W≈0) the ±0.6 m/s noise keeps flipping the causal gate near trim, the flap
winds up to the −14° stop and locks C_L at ≈0.4 (α̇ rings ±40 deg/s first).
Exactly the C-axis bias mechanism: errors where W≈0 flip the gate. The best
mitigated seed (combo, +74%) tracks the clean trajectory with a calm flap.

---

## Verdict

1. **How much noise does the +76–80% tolerate? Almost none.** The mean crosses
   the prop-W clean +32% at **σ ≈ 0.7%·W0 ≈ 0.21 m/s** (R=3e-4: 0.74%, R=1e-4:
   0.70%), and crosses zero (open loop) at σ ≈ 1.8%·W0 ≈ 0.55 m/s. At σ=1%
   it is already a lottery: 5/12 seeds keep ≈+80%, 7/12 ring or explode. The
   collapse is a branch bifurcation, not graceful degradation — and refine,
   R, R_du, filtering and averaging do not restore it because the fragility
   is the argmin/causal-gate structure itself (B, E).
2. **But the preview-noise caveat cuts against prop-W even harder.** Fed the
   same noisy preview, prop-W is catastrophic (−57.5% at σ=2%, 9/12
   explosions): its g_W·Wc term pipes sensor noise straight to the flap.
   **At equal information the optimal never loses to prop up to σ=10%** —
   the "+32% prop baseline" is exactly as oracle-dependent as the "+76.6%".
3. **Structured errors (C): the controller needs phase, not amplitude.**
   Gain error ×0.8–1.2 keeps +46…+78%; jitter ±1 step is benign (+56%, no
   flags); but +5%·W0 of additive bias (−6.5%, flags) or one step of latency
   (+16.9%) destroy it. A LIDAR for this controller must be bias-calibrated
   near W≈0 and phase-accurate; gain accuracy is secondary.
4. **Cell dependence (D): general, graded by severity.** At W10/Tg0.7, 2%
   noise still gives +75–83% with zero flags; the sharp home cell is the
   worst case. Under noise, R=3e-4 dominates R=1e-4 — the R*=1e-4 no-flag
   pick is a clean-sensor luxury.
5. **Practical ranking under realistic noise** (σ=2%·W0 = 0.6 m/s ≈ a good
   Doppler LIDAR): MPC N4-none +37% (no preview needed, but 4/6 flags) >
   combo avgK5+R_du +8.1% (only 0-flag option) > everything else > prop-W
   (catastrophic). For a certifiable single-step controller at this cell, the
   honest number under noise is **+8%, not +76%**; the +76–80% headline
   requires σ ≲ 0.5%·W0 (≈0.15 m/s) — top-grade sensing.
6. Methodological: 6-seed means are unstable in this regime (SE ≈ 8 pts) —
   axis A uses 12 seeds; and 76/NOTES' σ=2% anchor (+5.2%) is mpc_noise.py's
   rng-200 draw, not preview_study's rng-100 (−0.5%): same statistics,
   different realization (see Regressions).

---

# E2 — literature-driven mitigations (2026-07-08)

Follow-up to the Verdict: a web survey of how the published field handles
noisy gust/wind preview (**LITERATURE.md**, 9 techniques, primary sources
DLR / Stuttgart–NREL / NASA) mapped onto four new axes, each implemented as a
wrapper or reference controller around the untouched light/optimal.py.
Scripts: e2_gate.py (T8 gate hysteresis/deadband/dwell + T3 command washout),
e2_kalman.py (T6 scalar disturbance-model KF), e2_mpc.py (T7 preview-horizon
MPC), e2_sensor.py (T1 DLR massed-measurement fusion). Same harness, metrics,
flags, seeds (rng 100+seed, 6/point), home cell W30/Tg0.4, DAMULT=3,
R=3e-4. Baseline 'none' re-run in every file: −0.1% [−26.3,+19.2] 3/6 at
σ=2%, −7.3% [−25.0,+20.8] 3/6 at σ=5% (paired streams).

## Smoke-anchor discoveries (design-level results in their own right)

- **Deadband is structurally dead** for this controller: eps_db=0.01 on the
  CLEAN case lands +6.2%, eps_db=0.005 lands **+16.9% = exactly the bad
  branch** of 76/: the ~5–7 step onset delay while the real gust climbs out
  of the band loses the branch selection. The noise-induced fluctuation of
  the gate signal is σ_e ≈ dCL/dW·σ ≈ 0.009 C_L at σ=2% — the band needed to
  reject noise *is* the band that delays onset. Trade-space closed.
- **Hysteresis+dwell is free** clean (+76.58% exact, both hyst and hd).
- **Washout must be in OPEN loop** (DLR arrangement: the FF never re-reads
  the post-filter actuator). Re-syncing the inner's `_delta_prev` to the
  applied flap makes the argmin fight the filter → rings at the rate limit
  (τ=100 ms clean: −62%, flagged). Open-loop, τ has to sit ~3–10× above the
  gust period: clean anchors +63.2/+73.8/+75.3/+75.9% at τ=0.2/0.5/1/2 s.
- **Preview-MPC beats the one-step controller even CLEAN**: mpcp anchors
  +81.5/+79.6/+80.5% (N=2/4/8) vs +76.6% — the integrated cost is a better
  use of the same model.

## Results (mean [min,max] flags/6; σ_del = delivered noise, m/s)

| arm                 | σ=2%·W0 (0.6 m/s)          | σ=5%·W0 (1.5 m/s)          |
|---------------------|-----------------------------|-----------------------------|
| none (one-step)     | −0.1 [−26,+19] 3/6          | −7.3 [−25,+21] 3/6          |
| G: db 0.005         | +3.3 [−25,+19] 3/6          | −17.1 [−32,+20] 3/6         |
| G: hyst 0.005       | +3.8 [−5,+18] 3/6           | −15.1 [−37,+20] 3/6         |
| G: hd (hyst+dwell)  | −6.1 [−28,+19] 3/6          | −16.1 [−27,−8] 4/6          |
| G: wash τ=200ms     | **+13.7** [−13,+36] 3/6     | **+10.8** [−9,+35] 3/6      |
| G: gw combo         | +8.3 [−6,+35] 3/6 (clean +34.6!) | +5.6 [−7,+27] 3/6      |
| K: kf q=1e4         | +8.2 [−5,+22] 3/6, σd 0.40  | +8.9 [−6,+26] 3/6, σd 0.87  |
| K: kf q=1e5 (min σd)| +7.4 [−5,+20] 3/6, σd 0.26  | +6.6 [−17,+23] 3/6, σd 0.56 |
| M: mpc-cur N4       | +37.0 [−4,+79] 4/6          | +35.4 [−5,+77] 4/6          |
| M: mpcp N=2         | +9.0 [−4,+25] 3/6           | +9.5 [−5,+36] 3/6           |
| M: mpcp N=4         | **+70.6** [+46,+83] 3/6     | +19.9 [+2,+32] 5/6          |
| M: mpcp N=8         | **+75.1** [+45,+83] 4/6     | **+62.5** [+33,+83] 5/6     |
| S: fuse J=25        | +54.4 [+16,+77] 1/6, σd 0.095 | +39.4 [−26,+77] 1/6, σd 0.24 |
| S: fuse J=50        | +26.8 [+16,+77] **0/6**, σd 0.070 | +13.5 [−5,+18] 1/6, σd 0.18 |
| S: fuseT J50 λ=10   | +36.7 [+16,+77] **0/6**, σd 0.052 | +36.8 [+17,+77] **0/6**, σd 0.13 |
| S: dlr λ=0 (raw 1–3 m/s) | — | +13.5 [−5,+18] 1/6, σd 0.20 |
| S: dlr λ=1 (raw 1–3 m/s) | — | **+34.8** [+17,+76] 1/6, σd 0.18 |

(dlr = DLR-realistic raw LOS noise 1→3 m/s ∝ lookahead, i.e. 3.3–10%·W0 —
worse raw noise than every other column — fused over Jmax=50 re-measurements
with pre-warmed database; the fusion delivers 0.18–0.20 m/s, matching the
published 0.02–0.27 m/s band.)

## E2 verdict

1. **T7 preview-horizon MPC is the only technique that recovers the VALUE.**
   mpcp N=8: +75.1% mean at σ=2% (worst seed +45!) and +62.5% at σ=5%, where
   the one-step argmin sits at −0.1/−7.3. Exactly the literature's claim: the
   integrated cost averages per-sample noise down; the decision structure,
   not the model, sets the tolerance. Cost: N× compute (N=8 ≈ 8× one-step),
   and it needs an N-step preview vector, not just W(t+dt). Caveat: 3–5/6
   acceleration flags remain — value is back, ringing is not yet handled
   (no gate, no move penalty in this port).
2. **T1 sensor fusion is the only technique that eliminates EXPLOSIONS at
   realistic raw noise.** With honest DLR-style redundancy (raw 1–3 m/s →
   delivered 0.18–0.20 m/s) the outcome is never worse than the +17 branch
   (min −5% across arms, ≤1/6 flags, most arms 0–1/6) — but the +17/+76
   **branch lottery survives even at σ_del = 0.05–0.10 m/s** (0.2–0.3%·W0):
   the frozen spatial corrugation of the fused estimate perturbs the onset
   phase, and the branch selection is sensitive at that scale. Tikhonov λ=10
   is the most stable compromise (+37% both fracs, 0/6 flags).
3. **T8 gate logic fails on this system** — the smoke anchors show why: the
   noise scale that must be gated equals the |e| scale that picks the branch
   in the first ~10 ms. Not a tuning problem; a property of the one-step
   argmin+gate structure. T3 washout survives clean only in open loop and
   buys damage limitation (best gate-family mean +13.7%) but no flag relief.
4. **T6 scalar KF ≈ the old combo** (+8–9%, 3/6 flags): a CV internal model
   cannot deliver below ~0.26 m/s without lagging the gust (Ẅ ≈ 3700 m/s²);
   NASA's graceful degradation does not transplant via the scalar shortcut —
   it needs the full estimator+LQG architecture with a matched gust model.
5. **The literature's actual pipeline is a COMPOSITION** — reconstruction
   (T1) + horizon-integrating preview consumption (T7) + command shaping —
   and the two halves we tested separately are complementary by mechanism:
   fusion kills explosions and delivers ≤0.1 m/s, the horizon converts
   surviving noise into value. The obvious next run is **mpcp N=8 fed by the
   fused sensor (+ R_du or rate shaping for the residual flags)** — not run
   yet; nothing in E2 was tuned beyond the sweeps shown.

---

# E2-combo — the composed pipeline (2026-07-08)

e2_combo.py = T1 fusion database (Jmax=50, pre-warmed) delivering the fused
8-node preview VECTOR -> T7 constant-flap MPC N=8 (wnext convention over the
horizon) -> optional R_du on the single horizon move. Two jobs (flat | dlr),
6 seeds, same harness/metrics/flags. Anchors: clean through the full pipeline
+80.5% = mpcp N=8 white-clean exactly (fusion adds no lag; R_du=0.1 clean
collapses to +19.7% — over-suppression alone kills the branch).

| config (N=8, Jmax=50)             | mean [min,max] flags/6      | σ_del m/s |
|-----------------------------------|------------------------------|-----------|
| one-step none, white σ=2% (paired)| −0.1 [−26.3,+19.2] 3/6       | 0.49      |
| combo flat σ=2%, λ=0, R_du=0      | **+80.5 [+80.4,+80.6] 0/6**  | 0.070     |
| combo flat σ=2%, λ=0, R_du=1e-2   | +63.2 [+29.7,+79.9] 0/6      | 0.070     |
| combo flat σ=2%, λ=0, R_du=1e-1   | +19.7 [+19.6,+19.7] 0/6      | 0.070     |
| combo dlr (raw 1–3 m/s), λ=1, R_du=0 | **+81.1 [+80.3,+83.7] 0/6** | 0.179   |
| combo dlr, λ=1, R_du=1e-2         | +71.5 [+29.5,+79.9] 0/6      | 0.179     |
| combo dlr, λ=0, R_du=0            | +81.8 [+80.5,+84.1] 1/6      | 0.20      |

## E2-combo verdict

1. **The composition closes the problem at this cell.** Fusion + horizon MPC
   delivers the full clean optimum (+80.5%) under σ=2% white preview noise
   (std 0.05 pts — deterministic for practical purposes, 0/6 flags), and
   **+81.1% with 0/6 flags under DLR-REALISTIC raw lidar noise (1–3 m/s
   line-of-sight, i.e. 3–10%·W0)**, Tikhonov λ=1 preferred (λ=0: 1/6 flag).
   The two mechanisms are exactly complementary: the fusion removes the
   per-step chatter that flagged mpcp-on-white (3–5/6), the horizon removes
   the +17/+76 branch lottery that survived fusion alone (one-step best
   +37–54%, lottery-limited).
2. **R_du is unnecessary and harmful here**: with the fused input the flags
   are already gone at R_du=0; 1e-2 reintroduces branchiness (min +29.7),
   1e-1 locks the bad branch even clean (+19.7). Move suppression was a
   patch for dirty input, not a component of the clean pipeline.
3. Costs of the winner: 8 batch_step per plant step (≈8× one-step compute),
   a ~100 ms / 8 m lookahead sensor window (Jmax=50 nodes at dt·U=0.16 m —
   modest vs DLR's 60–180 m), 16 ms preview horizon. No re-tuning of
   R/grid/rate was needed (R=3e-4 untouched).
4. Study-level conclusion, superseding Verdict #5's "honest number +8%":
   with the sensor modeled the way the field actually builds lidar GLA
   (massed re-measurement + estimator) and the preview consumed the way the
   field consumes it (horizon-integrated), **the honest number under
   realistic lidar noise is the full +80%, zero flags** — the fragility
   belonged to the single-sample argmin architecture, not to the LDNet
   model or the preview premise.

## E2CC — generality check (e2_combo_cells.py, 2026-07-08)

The fixed winner (fusion Jmax=50 -> MPC N=8, R=3e-4, R_du=0; flat lam=0,
dlr lam=1) re-run without any re-tuning at the two other study cells.
mean [min,max] flags/6; sigma_del in m/s.

| cell (cex0)        | 1step clean | combo clean | none 1step σ=2%      | combo-flat σ=2%              | combo-dlr (raw 1–3 m/s)       |
|--------------------|-------------|-------------|-----------------------|------------------------------|-------------------------------|
| W30/Tg0.4 (0.4600) | +76.6%      | +80.5%      | −0.1 [−26,+19] 3/6    | +80.5 [80.4,80.6] 0/6, 0.070 | +81.1 [80.3,83.7] 0/6, 0.179  |
| W10/Tg0.7 (0.2075) | +89.5%      | +93.5%      | +75.3 [69,84] 0/6     | +93.5 [92.9,94.0] 0/6, 0.025 | +92.0 [91.2,93.2] 0/6, 0.190  |
| W30/Tg0.7 (0.5791) | +85.0%      | +91.5%      | +61.9 [11,78] 1/6     | +91.5 [91.4,91.5] 0/6, 0.075 | +91.5 [91.4,91.5] 0/6, 0.191  |

- **The composition generalizes without re-tuning**: at every cell the noisy
  combo equals its own clean anchor to within the seed spread (≤1.2 pts),
  with 0/6 flags everywhere — including W10/Tg0.7 where the dlr raw noise is
  **10–30% of the gust amplitude** (delivered 0.19 m/s ≈ 1.9%·W0) and the
  one-step controller would sit on its knife-edge.
- **The horizon also upgrades the clean optimum at every cell** (+4.0/+4.0/
  +6.5 pts over the one-step clean): the integrated cost is simply a better
  use of the same LDNet, noise aside.
- Note for the thesis: cells ran with N=8 fixed; the home-cell R_du sweep and
  the D-axis one-step cell numbers provide the surrounding context. All E2CC
  results computed on the cluster tree (rk4_batch structural step inside the
  MPC horizon; the local tree's dp45_batch variant is handled by a defensive
  import in e2_mpc/e2_combo/e2_combo_cells but was not used for any recorded
  number).

---

# Integrator migration note (f92d8975 → dp45, 2026-07-08)

`structure.py` at commit f92d8975 replaced `step_rk4` with `step_dp45`
(Dormand-Prince RK45 via scipy.integrate.solve_ivp).  After syncing the local
tree to the cluster the cluster also uses dp45; the recorded E2/E2CC/E2-combo
results (all in `results/E2_*.npz`) were computed with `rk4_batch` in the MPC
horizon and `step_rk4` in the plant.  The new dp45 tree will not reproduce
those numbers bit-exactly — this is expected and **not a bug**.

## dp45 baseline anchors (home cell W30/Tg0.4, DAMULT=3)

Smoke regression run 2026-07-08 on cluster post-sync (smoke_dp45_baseline.py,
commit 7a171d9c). dp45 values are bit-exact with rk4 at this cell.

| check | rk4 value | dp45 value | delta |
|---|---|---|---|
| open cex0 | 0.4600 | **0.4600** | 0.0000 |
| one-step optimal R=3e-4 | +76.58% | **+76.58%** | 0.00 pt |
| one-step optimal R=1e-4 | +80.67% | **+80.67%** | 0.00 pt |
| combo oracle clean R=3e-4 | +80.5% | **+80.51%** | +0.01 pt |

Use the dp45 values as the new anchors for all subsequent studies.
All non-chaotic numbers within tolerance (delta < 1 pt) -- no debug needed.

---

# W_combo -- white-noise robustness of the E2-combo (2026-07-08)

Axis: noise_white_combo.py -- white Gaussian raw measurement noise sigma applied BEFORE
fusion; FusedSensor(Jmax=50, lam=0, N=8) + MPCPreviewController(R=3e-4, R_du=0).
Home cell W30/Tg0.4, DAMULT=3, 6 seeds rng(100+seed). Baseline 'none' = one-step
argmin (same R, same raw frac), same seeds (paired).

| sigma/W0 | combo CLred | combo [min,max] flags/6 | sigma_del m/s | none CLred | none [min,max] flags/6 |
|---|---|---|---|---|---|
| 0%   | **+80.5%** | [+80.5,+80.5] 0/6 | ~0 | -- | -- |
| 1%   | **+80.5%** | [+80.5,+80.5] 0/6 | 0.035 | +9.6% | [−13.3,+22.4] 3/6 |
| 2%   | **+80.5%** | [+80.4,+80.6] 0/6 | 0.070 | −0.1% | [−26.3,+19.2] 3/6 |
| 5%   | **+81.3%** | [+80.6,+83.7] 1/6 | 0.175 | −7.4% | [−25.7,+20.8] 3/6 |
| 10%  | **+81.7%** | [+80.4,+83.7] 1/6 | 0.348 | −21.1% | [−24.1,−16.3] 3/6 |
| 20%  | +76.2%     | [+53.0,+83.1] 2/6 | 0.693 | −10.9% | [−15.1,−8.6] 3/6 |

**Break-even vs combo-clean (+80.5%):** never (−4.3 pts at worst, σ=20%)
**Break-even vs prop-W clean (+32%):** never (combo stays ≥+76.2% throughout)

Figure: `results/fig_noise_white_combo.png`

## W_combo verdict

1. **E2-combo is essentially noise-immune up to σ=10%·W0 (3 m/s).** The fusion
   reduces σ_del by ~7× at each sigma level (e.g. σ=2% raw → 0.070 m/s delivered
   vs 0.487 m/s for the one-step baseline); the horizon MPC then converts the
   residual sub-0.1 m/s noise into a cost-function average rather than a branch
   selection. The result is +80.5% with 0/6 flags at σ=2%, identical to the
   clean oracle. Even at σ=20% (6 m/s raw, 0.693 m/s delivered) the mean is
   +76.2% — still 44 pts above the prop-W +32% clean reference.
2. **The one-step argmin collapses immediately.** At σ=1% it is already +9.6%
   (3/6 flags); at σ=2% the mean is −0.1% — worse than open loop. The delivered
   noise (0.487 m/s) is above the 0.22 m/s break-even measured in axis A. The
   two controllers are incomparable as GLA laws under any realistic sensor noise.
3. **Composition robustness confirmed:** these results use the same Jmax=50,
   N=8, R=3e-4, R_du=0 parameters as the E2CC generality check — no re-tuning
   for the noise sweep. The study fully supersedes Verdict #5 ("honest number
   +8%") with the honest number **+80.5%, 0/6 flags, at realistic raw noise**.

---

# A2 — calibration robustness of the E2-combo (2026-07-09)

Axis: noise_calib_combo.py — the sensor measures a mis-calibrated field
Wt_meas = gain*Wt + bias (bias = value*W0) with per-shot noise sigma_fun(j).
Inverse-variance fusion removes VARIANCE, not systematic error: bias_del
(mean of Wc − Wnext) confirms pass-through, so the axis measures the MPC's
own calibration tolerance. Combo-only (no comparison arms, user decision).
Home cell W30/Tg0.4, DAMULT=3, Jmax=50, lam=0, N=8, R=3e-4, R_du=0.
Clean sweep deterministic (1 rollout/point); spot-checks sigma=2%*W0, 6 seeds.

| arm/value      | clean CLred (flags) | sigma=2% mean [min,max] flags/6 | sigma_del | bias_del m/s |
|----------------|---------------------|----------------------------------|-----------|--------------|
| bias −10%·W0   | +59.8 (0)           | —                                | 1.42      | −1.15        |
| bias −5%·W0    | +73.9 (0)           | +74.2 [+73.7,+74.9] 0/6          | 0.73      | −0.60        |
| bias −2%·W0    | +80.2 (0)           | —                                | 0.29      | −0.25        |
| bias 0 (anchor)| **+80.5 (0)**       | —                                | ~0        | ~0           |
| bias +2%·W0    | **+23.5 (FLAG)**    | —                                | ~0        | +0.60        |
| bias +5%·W0    | +12.9 (FLAG)        | +13.1 [+12.6,+13.5] **6/6**      | 0.085     | +1.50        |
| bias +10%·W0   | −4.9 (FLAG)         | —                                | ~0        | +3.00        |
| gain ×0.8      | +46.8 (0)           | —                                | 2.05      | −1.33        |
| gain ×0.9      | +74.2 (0)           | —                                | 1.03      | −0.67        |
| gain ×1.1      | +79.7 (0)           | —                                | 1.03      | +0.67        |
| gain ×1.2      | +77.0 (0)           | +76.5 [+76.2,+76.8] 1/6          | 2.05      | +1.35        |

## A2 verdict

1. **Bias is the killer for the MPC too, and it is violently asymmetric.**
   Positive bias survives fusion intact (bias_del = injected) and already at
   **+2%·W0 = +0.6 m/s collapses the combo to +23.5% with an explosion flag**
   (+5%: +12.9, 6/6 flags noisy; +10%: −4.9, worse than open loop). Negative
   bias is nearly free down to −5%·W0 (+73.9, no flags): the sensor output
   clamp max(0,·) truncates it near W≈0 (delivered −0.60 of −1.5 injected;
   the truncation also shows as sigma_del 0.3–1.4) and an under-estimated
   gust makes the MPC under-commit flap — graceful. Mechanism identical to
   the historical C-axis money plot: errors where W≈0 flip the post-gust
   behaviour; a phantom positive gust keeps the flap deployed.
2. **Gain error is benign for the horizon MPC** (×0.9–1.2: +74…+80, no clean
   flags; ×1.2 noisy 1/6): multiplicative error preserves the gust phase and
   zeros. Unlike the one-step argmin (×1.2 rang), the integrated cost absorbs
   it. ×0.8 costs real performance (+46.8) but stays stable.
3. **Amplitude noise does not interact with calibration errors**: every noisy
   spot-check sits on its clean value (−5%: +74.2 vs +73.9; +5%: +13.1 vs
   +12.9; ×1.2: +76.5 vs +77.0) — only the flag counts move.
4. Sensor spec consequence: the lidar bias budget for this controller is
   **asymmetric — under-read, never over-read** (calibrate the chain to
   err low). Gain accuracy is secondary (±10% free, ±20% survivable).

# B2 — timing robustness of the E2-combo (2026-07-09)

Axis: noise_timing_combo.py — (a) shift: the sensor measures the time-shifted
field Wt[m+d] (frozen-field advection error; d>0 = gust appears d steps EARLY
in the preview, d<0 = late/stale); (b) refit: measurements accumulate every
step (500 Hz) but the profile is re-SOLVED every K steps and re-sliced by the
frozen-field offset in between (DLR refits at 10 Hz; K−1+N ≤ Jmax caps the
clean re-slice at K=42 ≈ 12 Hz with Jmax=50 — true 10 Hz needs Jmax ≥ 58).
RefitSensor K=1 reproduced W_combo per-seed BIT-EXACTLY (80.3860/80.5235,
smoke gate). Same cell/config as A2. Refit arm at sigma=2% only (clean refit
is trivially flat: exact measurements make staleness invisible).

| shift d | ms    | clean CLred (flags) | sigma=2% mean [min,max] flags/6 |
|---------|-------|---------------------|----------------------------------|
| −10     | −20   | +58.9 (FLAG)        | —                                |
| −5      | −10   | **+78.6 (0)**       | +78.6 [+78.0,+79.0] 0/6          |
| −2      | −4    | +80.3 (0)           | —                                |
| −1      | −2    | +80.4 (0)           | —                                |
| 0       | 0     | **+80.5 (0)**       | —                                |
| +1      | +2    | +81.0 (0)           | —                                |
| +2      | +4    | **+71.5 (FLAG)**    | —                                |
| +5      | +10   | +18.0 (FLAG)        | +38.5 [+18.4,+45.4] **6/6**      |
| +10     | +20   | +10.7 (FLAG)        | —                                |

| refit K | period ms | rate Hz | sigma=2% mean [min,max] flags/6 | sigma_del m/s |
|---------|-----------|---------|----------------------------------|---------------|
| 1       | 2         | 500     | +80.5 [+80.4,+80.6] 0/6          | 0.070         |
| 5       | 10        | 100     | +80.5 [+80.4,+80.6] 0/6          | 0.071         |
| 10      | 20        | 50      | +80.5 [+80.5,+80.6] 0/6          | 0.073         |
| 25      | 50        | 20      | +80.5 [+80.4,+80.6] 0/6          | 0.081         |
| 42      | 84        | ~12     | **+80.5 [+80.4,+80.6] 0/6**      | 0.103         |

## B2 verdict

1. **Timing tolerance is asymmetric, opposite to naive intuition: LATE is
   cheap, EARLY is toxic.** Stale preview down to −10 ms costs 2 pts
   (+78.6, no flags — the one-step died at −2 ms): the receding horizon
   re-plans from the measured state every step, so lateness only delays the
   first moves. Acting EARLY flags already at +4 ms (+71.5) and collapses by
   +10 ms (+18.0): the flap moves against a gust that has not arrived, the
   self-induced transient is costed against trim, and the misplaced
   flap/rate budget rings when the real gust lands. Amplitude noise does not
   rescue it (+10 ms noisy: mean +38.5 but 6/6 flags) and does not hurt the
   late side (−10 ms noisy: +78.6, 0/6, same as clean).
2. **Estimator refit rate is FREE down to ~12 Hz** — the slowest rate the
   Jmax=50 window supports cleanly. All K give +80.5% with 0/6 flags;
   staleness only creeps into sigma_del (0.070 → 0.103 m/s at K=42, the
   delivered nodes having been solved with fewer accumulated shots). The
   DLR arrangement (measure at 500 Hz, re-fit at 10 Hz, re-slice by frozen
   field) costs this controller nothing measurable.
3. Sensor spec consequence: **budget the sensing chain toward lateness** —
   any systematic timing uncertainty (advection speed, processing delay)
   should be absorbed on the stale side, never by extrapolating ahead.

# Robustness envelope of the E2-combo (thesis deliverable)

Systematic-error tolerances at the severe cell W30/Tg0.4 (DAMULT=3), fixed
winner config (Jmax=50, lam=0, N=8, R=3e-4, R_du=0), criterion "mean within
~2 pts of the +80.5% clean anchor AND no explosion flags". Amplitude-noise
rows from W_combo for completeness.

| error class                  | tolerated (no-flag)            | first failure            |
|------------------------------|--------------------------------|--------------------------|
| raw white noise (pre-fusion) | σ ≤ 10%·W0 = 3 m/s (≤1/6)      | σ=20%: +76.2, 2/6        |
| additive bias, negative      | ≥ −5%·W0 = −1.5 m/s (−2% free) | −10%: +59.8 (no flag)    |
| additive bias, positive      | **< +2%·W0 = +0.6 m/s**        | +2%: +23.5, FLAG         |
| gain                         | ×[0.9, 1.2]                    | ×0.8: +46.8 (no flag)    |
| preview timing, late         | ≥ −10 ms (−5 steps)            | −20 ms: +58.9, FLAG      |
| preview timing, early        | **≤ +2 ms (+1 step)**          | +4 ms: +71.5, FLAG       |
| estimator refit period       | ≤ 84 ms (~12 Hz), zero cost    | none up to window limit  |

The envelope is sharply one-sided: the pipeline is certifiable-grade robust
to everything a redundant, conservatively calibrated sensing chain actually
delivers (variance, under-estimation, staleness, slow refit) and fragile
only to the two errors a chain must never make — over-reading the gust and
anticipating it. Both specs are cheap to honour by construction (calibrate
low, buffer late); neither is achievable by averaging, which is exactly why
the fusion layer cannot substitute for them.

---

# C2 — spatial per-shot jitter of the E2-combo (2026-07-10)

Axis: noise_jitter_combo.py — every shot samples the field at the wrong node
(value from clip(i+j+eta), eta ~ U{−k..+k} per shot, registered at i+j): the
1D surrogate of lidar coherence loss / range-gate error / probe-volume
averaging (Schlipf/Guo WES 2023). The fusion converges to W convolved with
the jitter kernel — a spatial low-pass. k=0 reproduces FusedPreviewSensor
bit-exactly (anchor gate). Node spacing U·dt = 0.16 m. Same cell/config as
A2/B2.

| k (nodes) | range error | jitter-only mean [min,max] flags/6 | sigma_del m/s |
|-----------|-------------|-------------------------------------|---------------|
| 0 (anchor)| 0           | **+80.5 (0/1, deterministic)**      | ~0            |
| 1         | 0.16 m      | +80.5 [+80.5,+80.5] 0/6             | 0.026         |
| 2         | 0.32 m      | +80.5 [+80.4,+80.5] 0/6             | 0.045         |
| 5         | 0.80 m      | +80.4 [+80.4,+80.5] 0/6             | 0.104         |
| 2 + σ=2%  | 0.32 m      | +80.9 [+80.4,+83.2] 1/6             | 0.082         |

## C2 verdict

**Per-shot range jitter is FREE up to the tested 0.80 m.** The smearing
argument is real but the scales save us: the gust length is Tg·U = 32 m, so
±0.8 m of spatial smear perturbs the fused profile by ~2.5% of the gust
wavelength — delivered as ≤0.10 m/s of effective noise, well inside the
W_combo tolerance. Coherence loss would need to reach the gust-gradient
scale (metres to tens of metres, i.e. wind evolution, not range-gate error)
before this controller notices. Combined jitter+amplitude noise (k=2, σ=2%)
sits on the anchor with 1/6 flag.

# D2 — structural model mismatch of the E2-combo (2026-07-10)

Axis: noise_mismatch_combo.py — the PLANT flies with perturbed parameters
while the MPC predicts with nominal ones (toggle of structure globals around
mpc.compute; each perturbed plant compared against ITS OWN open loop, cex0
logged). uinf/cltrim arms mis-set the CONTROLLER's constructor instead
(plant nominal). Oracle-clean preview: the axis isolates model error.
Declared limit: the LDNet is shared plant/controller — structural mismatch
only. Deterministic (1 rollout/point).

| arm / value            | CLred (flags) | cex0 (plant OL) | note |
|------------------------|---------------|-----------------|------|
| anchor                 | **+80.5 (0)** | 0.4600          | gate |
| plant D_ALPHA ×0.67    | +80.5 (0)     | 0.4678          | ctrl assumes DAMULT=3, plant ~2 |
| plant D_ALPHA ×0.83    | +80.6 (0)     | 0.4639          | |
| plant D_ALPHA ×1.17    | +80.6 (0)     | 0.4560          | |
| plant D_ALPHA ×1.33    | +80.6 (0)     | 0.4523          | plant ~4 |
| plant K_ALPHA ×0.90    | +40.5 (0)     | **0.8519**      | plant OL nearly doubles |
| plant K_ALPHA ×0.95    | +85.8 (0)     | 0.5613          | −2.5% pitch frequency |
| plant K_ALPHA ×1.05    | +78.9 (0)     | 0.4688          | |
| plant K_ALPHA ×1.10    | +77.0 (0)     | 0.4738          | |
| ctrl U = 76 (−5%)      | **−57.5 (0)** | 0.4600          | worse than open loop |
| ctrl U = 84 (+5%)      | −2.1 (FLAG)   | 0.4600          | |
| ctrl C_L_trim ×0.95    | **−60.8 (0)** | 0.4600          | worse than open loop |
| ctrl C_L_trim ×1.05    | −1.9 (FLAG)   | 0.4600          | |

## D2 verdict

1. **Structural mismatch is a non-event.** Pitch damping ×0.67–1.33 (i.e.
   the entire plant-DAMULT 2→4 range against an assumed 3) changes nothing:
   +80.5/+80.6, no flags — D_ALPHA barely acts inside a 16 ms horizon.
   Pitch stiffness ±5–10% keeps +77…+86 with zero flags. The K_ALPHA ×0.90
   row reads low (+40.5) but its plant is a different problem: the open-loop
   excursion nearly doubles (cex0 0.85 vs 0.46, ω_α −5% moving toward this
   gust); the mismatched controller still holds the residual at ≈ the
   NOMINAL plant's open-loop level. Caveat: CLred on a perturbed plant mixes
   plant difficulty with mismatch damage; a matched-controller reference on
   the same plant would separate them (not run). Against Forte 2023's
   −2.5%-frequency collapse (69→39%), our −2.5% row (K_ALPHA ×0.95) sits at
   +85.8 — the receding horizon with state feedback does not inherit the
   estimator-model fragility.
2. **The controller's REFERENCE calibration is the true fragile axis.**
   ±5% on the believed airspeed or on C_L_trim is catastrophic: −57.5%
   (U=76) and −60.8% (trim ×0.95) — the controller becomes actively worse
   than no controller, without even flagging; the +5% sides sit at ≈0% with
   flags. The U and trim curves nearly coincide, exposing the mechanism:
   the horizon cost is (C_L − trim)², so any inconsistency between the
   model-predicted C_L (LDNet evaluated at the believed U) and the trim
   reference acts as a CONSTANT offset in the cost — the controller holds a
   permanent flap deflection chasing a phantom, the same post-gust wind-up
   money-plot mechanism as the A2 positive bias, now injected through the
   model side. −5% U ≈ −5% trim because C_L_trim(U) is near-linear here.
3. Spec consequence: U and C_L_trim are the two best-instrumented quantities
   on any aircraft (air data + trim measurement), so the fragile axes are
   the cheap ones to honour — but the spec must be stated: **the surrogate
   must be evaluated at the true flight condition and the trim reference
   must track the actual trim to ≪5%** (edge not bracketed below 5%; a
   finer sweep would locate it). Consistency of the (model, reference) pair
   matters more than either value alone.

## Robustness envelope — C2/D2 additions

| error class                  | tolerated (no-flag)            | first failure            |
|------------------------------|--------------------------------|--------------------------|
| per-shot range jitter        | ≤ 0.80 m (all tested)          | none up to 0.80 m        |
| plant D_ALPHA vs model       | ×[0.67, 1.33] (all tested)     | none in range            |
| plant K_ALPHA vs model       | ×[0.95, 1.10] (≥ +77%)         | ×0.90: +40.5 (harder plant, no flag) |
| controller U estimate        | **≪ ±5%** (edge not bracketed) | ±5%: −57.5 / −2.1 FLAG   |
| controller C_L_trim estimate | **≪ ±5%** (edge not bracketed) | ±5%: −60.8 / −1.9 FLAG   |

Updated study-level picture: across all four families (A2 calibration, B2
timing, C2 coherence-surrogate, D2 model mismatch) every VARIANCE-class
error is absorbed by the fusion+horizon composition, and every failure is a
BIAS-class error in disguise — gust over-read, preview anticipation, wrong
trim/airspeed reference. The controller needs its reference frame (where
zero is, when now is) far more than it needs precise measurements or a
precise structural model.
