# Latent-ODE (NNdyn) literature findings — research run 2026-08-19

Scope: NNdyn only (the latent-ODE/dynamics MLP + its explicit-Euler integration loop +
its training procedure), targeting the flap-driven dynamic residual (~2.4-2.9e-2 NRMSE)
that survived every decoder-side (NNrec) intervention. No code changed, nothing launched.

Verification method: each claim below was checked by directly fetching the arXiv
abstract/HTML/PDF (not just a search snippet) unless marked otherwise. This is
single-pass primary-source verification (I read the source), not the NEARWALL notes'
multi-agent adversarial voting — so confidence tags here are `[FULL-TEXT]` (fetched and
read the actual mechanism/section/numbers), `[ABSTRACT-ONLY]` (fetch returned only
abstract-level claims, mechanism not confirmed), or `[SEARCH-ONLY]` (could not get a
usable fetch, relying on search-engine summaries — treat as unverified).

---

## 0. Framing correction (code-verified, not a literature claim — read first)

I read `recon/train_fields.py` (`build_networks`, `make_ldnet`, `evolve_dynamics`,
lines ~121-314) before searching, to ground the recommendations in what's actually
there. Two things in the task framing don't match the code as it stands today:

- **Training is NOT one-step teacher-forced.** `evolve_dynamics()` runs a `tf.range`
  loop over the *entire* trajectory (`nt` steps), each step computing
  `state = state + dt/dt_base * NNdyn(inp)` from the network's **own previous state**
  (not ground truth — there is no ground-truth latent to teacher-force against; only
  the exogenous `input_signals` come from data). `loss_fn` is MSE between the decoded
  field at *every* timestep of this free-running rollout and the true field snapshot.
  This is already full-trajectory single-shooting BPTT through the forward-Euler loop,
  end to end. It is the same *kind* of fix that `light/`'s
  `ldnet-rollout-solution.md` found necessary for the loads-LDNet (closed-loop rollout
  vs 1-step teacher forcing) — except here it's already the status quo, not a pending
  fix. (That memory's "1-step teacher-forced" referred to feeding *true structural
  states* as inputs while only integrating the aero loads; a different loop than
  recon's, which has no structural feedback at all — flap/gust are prescribed
  exogenous forcing here, not something the model predicts and re-feeds.)
- `--dyn-layers` (default 2) and `--dyn-width` (default 7) **already exist as CLI
  flags** in `train_fields.py` (lines 486-489) — they've just never been swept, per the
  task description. No new flag needed to test width/depth.

This matters for ranking below: it rules out "switch to multi-step/rollout training"
as a fresh lever (already true), but it *elevates* anything in the literature that is
specifically about pathologies of **long single-shooting BPTT** (multiple shooting,
noise-regularized rollouts), because that is exactly today's training regime.

---

## 1. VERIFIED — activation saturation kills oscillatory dynamics in Neural-ODE vector fields

1. `[FULL-TEXT]` A tanh MLP (3 hidden layers × 128) used as the RHS of a Neural ODE
   (`ḣ = f_θ(h)`) **fails to reproduce the oscillatory/limit-cycle behavior** of the
   Morris-Lecar neuron model across Hopf, SNLC and homoclinic bifurcation regimes;
   swapping the activation to **SiLU** "leads to visibly improved dynamics,
   particularly in the reproduction of the orbit structure" (quantified via
   RMSE/R² tables, e.g. homoclinic regime Tanh RMSE_V=4.82, R²_V=0.053 after 20k
   epochs where the network fails to reproduce oscillation at all).
   [Comparing Physics-Informed and Neural ODE Approaches for Modeling Nonlinear
   Biological Systems: A Case Study Based on the Morris-Lecar Model, arXiv:2603.26921]
2. `[FULL-TEXT]` The theoretical mechanism (companion paper, verified but scoped
   correctly — see caveat): saturating activations (tanh, sigmoid) attenuate the
   vector field's input Jacobian (‖Df_θ(x)‖ ≤ C(U), C shrinking as saturation depth
   δ→0); along a periodic orbit this **drives every Floquet exponent toward zero**,
   which kills both the contraction needed for a stable limit cycle and the
   instability needed for other bifurcation types. Non-saturating activations
   (derivative bounded away from 0, e.g. SiLU on its positive branch) avoid this.
   [Activation Saturation and Floquet Spectrum Collapse in Neural ODEs,
   arXiv:2604.00543]
   **Caveat (checked directly, do not overstate):** I fetched the full PDF and
   confirmed this paper's theory and its only two numerical validations
   (Stuart-Landau oscillator, Morris-Lecar) are for **autonomous** systems
   (`ḣ=f_θ(h)`, no external time-varying forcing). It does *not* contain a forced/driven
   experiment, and does *not* itself recommend a replacement activation (that
   recommendation, and the SiLU-vs-tanh numbers, come only from the companion paper
   #1). Applying this mechanism to NNdyn — which *is* forced (concatenates gust W and
   flap δ into the same vector field) — is a plausible analogy, not a demonstrated
   result: NNdyn's preactivations can still be pushed into tanh's flat region by
   large/fast exogenous inputs (this is exactly the flap-driven regime), so the same
   Jacobian-attenuation argument should still apply locally, but nobody has verified
   this for a forced case.

3. `[FULL-TEXT]` Independent of the above two, an unsteady-airfoil-specific paper
   (see §3 below) also uses concatenation of exogenous controls straight into a
   Neural-ODE vector field and reports that a plain architecture can't resolve
   trajectory-crossing (hysteresis) in the observed state — a different symptom but
   same family of "small vector-field MLP under-fits fast/oscillatory
   control-driven content."

**Read on our project's own evidence:** this is the same *class* of insight that made
the CORAL SIREN decoder win (periodic/non-saturating activations beat tanh for
oscillatory content) — but the mechanism is not identical. The decoder win was about
fitting *spatial* high-frequency content (sin(ω₀·x) in the coordinate inputs); the
NNdyn mechanism above is about *not saturating* the vector field under large driven
inputs so the latent can still trace fast/oscillatory *temporal* trajectories. Two
different failure modes that happen to share "swap tanh for something else" as the
fix — worth testing both a non-saturating swap (SiLU/GELU) and a periodic swap
(sine), see ranked list.

---

## 2. VERIFIED — training-time noise injection for rollout-trained dynamics models

4. `[FULL-TEXT]` Learned particle/mesh simulators trained with full free-running
   rollouts (like ours) are made more accurate over long horizons by **corrupting the
   model's own state input with random-walk Gaussian noise during training**
   (σ=3×10⁻⁴ on normalized velocities here) — confirmed exact mechanism via Section
   4.3/5.4 of the paper. Ablation (their Fig. 4g-h) shows rollout accuracy is best at
   an **intermediate** noise scale — too little leaves the model brittle to its own
   small errors, too much degrades one-step accuracy. Loss itself stays one-step MSE;
   only the *input* is corrupted.
   [Sanchez-Gonzalez et al., Learning to Simulate Complex Physics with Graph
   Networks, ICML 2020, arXiv:2002.09405]
   **Caveat:** their motivation is closing a *train/rollout mismatch* (clean
   teacher-forced training vs noisy self-generated rollout at inference). Per §0,
   `train_fields.py` has no such mismatch — training already free-runs. So the
   *original* motivation doesn't transfer cleanly; if this helps here it would be
   acting as a generic flatness/regularizer on the learned vector field, not fixing
   an exposure-bias gap. Flag this as the weakest-motivated of the "verified"
   findings even though the mechanism itself is solid and cheap to test.

## 3. VERIFIED — single-shooting BPTT over long/oscillatory trajectories is a known hard optimization regime, and the fix used elsewhere is domain-matched

5. `[ABSTRACT-ONLY, but direct quote obtained]` "If the data contains oscillations,
   then standard [single-shooting, whole-trajectory BPTT] fitting of a neural
   differential equation may result in a **flattened out trajectory that fails to
   describe the data**" — demonstrated on datasets the standard approach cannot fit
   at all, fixed by splitting the trajectory into shorter segments (multiple
   shooting) with their own initial-state variables, continuity enforced by a
   penalty or augmented-Lagrangian term.
   [Turan & Jäschke, Multiple shooting for training neural differential equations on
   time series, arXiv:2109.06786]
6. `[ABSTRACT-ONLY]` A 2023 follow-up formalizes this with a Bayesian prior instead of
   a hard penalty for the continuity constraint between segments, using a
   transformer-based recognition network to amortize the per-segment initial-state
   inference; reports SOTA on multiple long-trajectory benchmarks.
   [Iakovlev et al., Latent Neural ODEs with Sparse Bayesian Multiple Shooting,
   ICLR 2023, arXiv:2210.03466]
7. `[ABSTRACT-ONLY]` A very recent (2026) paper combines curriculum learning with
   multiple shooting specifically to stabilize NODE/UDE training on noisy, sparse,
   partially-observed series, reporting it "accelerates and stabilises training
   convergence, outperforms state-of-the-art training strategies" across twelve
   benchmarks vs. both single-shooting and plain multiple shooting.
   [Curriculum Multiple Shooting for Robust Training of Neural and Universal
   Differential Equations, arXiv:2608.05777]

**Why this is elevated, not just a generic lead (see §0):** `train_fields.py`'s
`evolve_dynamics` is exactly the single-shooting-over-a-whole-trajectory setup #5
identifies as the failure case, and the residual profile matches the qualitative
symptom description almost exactly — it is not the static/mean field that's wrong
(fixed already by mean-split), it's specifically the **time-fluctuating, most
oscillatory/fastest-changing (flap) content** that's worst, which is the "flattened
trajectory" signature #5 describes. This also offers a candidate explanation for an
already-observed dead end: tripling the L-BFGS budget made things *worse* via
overfitting — consistent with a single long non-convex BPTT landscape where more
optimizer effort digs into a bad-but-deep basin rather than finding a better one,
which is precisely the pathology multiple shooting is designed to avoid.

---

## 4. Domain-matched precedent: RK4 (not Euler) for unsteady-airfoil NODEs with exogenous control

8. `[FULL-TEXT]` A Neural-ODE model of **unsteady pitching-airfoil aerodynamics with
   exogenous control inputs** (very close to our setting) uses concatenation of
   controls into node features (`x_input = concat(y, u)` — i.e. the *same* plain
   concatenation NNdyn already uses, this is not evidence against concatenation) and
   integrates with **RK4** ("GNODE employs a fourth-order Runge-Kutta scheme,
   evaluating the internal network four times per integration step" — their Table 3),
   not Euler. Against an autoregressive (single-pass, effectively Euler-like)
   baseline (GNS) they report surface MAE 19.75→5.28 (≈4×), pitching-moment MAE
   21.81→2.61 (≈8×), phase error 22.2°→7.9°, and note GNS predictions "diverge from
   the CFD reference already during the first oscillation period" (negative R² on
   moment coefficient) while GNODE does not.
   [Spatio-Temporal Prediction of Unsteady Airfoil Aerodynamics Using Augmented Graph
   Neural ODEs with Exogenous Controls, arXiv:2607.18309]
   **Caveat (important — read the numbers correctly):** this comparison confounds
   *integrator order* with *architecture* (continuous-time NODE-with-RK4 vs.
   discrete-step autoregressive GNN) — it is not a clean Euler-vs-RK4 ablation *within*
   the same NODE, so the 4-8× numbers should not be read as "RK4 alone buys 4-8×."
   What *is* clean evidence from this paper: the authors' own architecture choice,
   for the closest published analogue to our problem (unsteady airfoil + exogenous
   control + latent ODE), is RK4, not forward Euler.
9. `[SEARCH-ONLY, weaker]` A broader benchmark of dynamics-learning priors found that
   "the use of continuous and time-reversible dynamics benefits models of all
   classes" — indirect support for higher-order/symmetric integration mattering
   generically, but I could not fetch full text to confirm what integrators were
   compared or by how much.
   [Botev et al., Which priors matter? Benchmarking models for learning latent
   dynamics, NeurIPS 2021 D&B, arXiv:2111.05458]

**Independent (non-cited) numerical-analysis argument**, safe to state without a
source: forward Euler has global error O(dt) vs RK4's O(dt⁴), and forward Euler's
stability region does not cover the imaginary axis — it can spuriously
amplify energy for the oscillatory error modes an oscillatory forcing (gust, flap)
excites, whereas RK4's region extends further along it. Whether `dt/dt_base` here is
small enough for this to matter is exactly what an ablation would answer.

---

## 5. Verification of the two previously-flagged (never-checked) leads

### arXiv:2209.10684 — "Attention Beats Concatenation for Conditioning Neural Fields"
`[FULL-TEXT via fetch]` **Real paper, correctly titled**, confirmed: Rebain, Matthews,
Yi, Sharma, Lagun, Tagliasacchi (2022). It compares concatenation vs. hypernetworks vs.
attention for conditioning **neural fields** (coordinate networks mapping (x,y[,z,t])
→ signal value) on 2D/3D/4D signal-modeling tasks, and finds attention wins,
**"particularly as the dimensionality of conditioning variables increases."**

**Verdict: real and correctly summarized by the earlier note, but the relevance to
NNdyn is weaker than the title suggests.** Two mismatches: (a) it's a neural-*field*
(decoder-shaped) paper, not a dynamics/ODE/control paper — nothing in it is about
vector fields, integration, or forced systems; (b) its own stated advantage kicks in
at high conditioning dimensionality (their experiments condition on shape/scene
latents that are much higher-dimensional than what's proposed here). NNdyn's
conditioning vector is tiny — `d_s(=1) + 1 input_parameter + 6 input_signals = 8`
total. That is far outside the regime where this paper's own evidence shows attention
winning. This doesn't mean attention conditioning can't help at low dimensionality,
just that **this specific paper provides no evidence either way at our scale** — rank
it accordingly (see §7).

### arXiv:2308.05732 — "PDE-Refiner"
`[FULL-TEXT via fetch]` **Real paper, correctly identified**: Lippe et al., NeurIPS
2023. Confirmed mechanism: base one-step-MSE prediction, then K≈3-4 **denoising
refinement** passes at both train and inference time with a decreasing noise
schedule (σₖ = σ_min^(k/K)), specifically because "the MSE model is only accurate for
a small, high-amplitude frequency band, while PDE-Refiner supports a much larger
frequency band" — one-step MSE training starves gradient signal to low-amplitude
high-frequency content that still matters for long-term nonlinear dynamics. Inference
cost is ≈4× (one extra forward pass per refinement step). Architectures tested: U-Net,
FNO, dilated ResNet — all large spatial-field networks, not small latent MLPs.

**Verdict: real, correctly summarized, but two domain gaps the earlier note didn't
flag.** (a) I confirmed by reading Eq. 6 in the paper that their own driving/forcing
term is **static** (`f = sin(4y)x̂ - 0.1u`, time-invariant) — there is no experiment
in the paper with a *time-varying* external control signal, so applying the "recovers
neglected high-frequency content" story to a gust/flap-*driven* residual is an
extrapolation, not a demonstrated result. (b) Per §0, `train_fields.py`'s training
already backprops through the *entire* trajectory — the specific failure mode
PDE-Refiner targets (one-step training starves rare/small spectral content because
gradient is dominated by common/large content) could still apply within a single
full-trajectory MSE (it's an amplitude-weighting argument, not strictly a "one-step
vs. multi-step" one), but it's a different, weaker match than the framing in the
original prompt implied. Rank accordingly — real mechanism, plausible, but the
biggest domain mismatch (1-D latent scalar vs. 2-D spatial field "frequency band")
of anything in this document.

---

## 6. Related but lower-priority / not independently chased down

- **Neural CDEs** (Kidger, Morrill, Foster, Lyons, NeurIPS 2020, arXiv:2005.08926)
  `[FULL-TEXT via ar5iv]`: `dz/dt = f_θ(z_t)·dX_t/dt`, where `X_t` is a continuous
  interpolant (natural cubic spline) of the observed/exogenous channels — the vector
  field is a function of the latent state *alone*, and the exogenous signal enters
  **multiplicatively through its own rate of change**, not by being concatenated and
  nonlinearly mixed inside the same MLP. The paper's stated reason cubic-spline (not
  linear) interpolation matters: linear interpolation's second-derivative
  discontinuities make the *adjoint* backward pass (needed for memory-efficient
  training) numerically slow/unstable (their Appendix A.2) — this specific
  justification is about the adjoint method, which `train_fields.py` doesn't use
  (it differentiates through the unrolled loop directly, not via adjoint sensitivity),
  so that particular argument for cubic splines doesn't transfer, though the
  structural argument (rate-of-change-driven vs. value-concatenated conditioning)
  still might.
- **Koopman-with-control / bilinear latent dynamics** `[SEARCH-ONLY]`: a family of
  methods (e.g. deep bilinear Koopman, generalized Koopman with control) replace the
  unstructured `f(z,u)` MLP with a **control-affine or bilinear** form,
  `dz/dt ≈ A z + B(u) z + C u`, i.e. an explicit structural prior that control enters
  partly *multiplicatively* with the state. Conceptually adjacent to Neural CDEs
  (both make the control's effect state-dependent rather than an independent additive
  input) but I did not verify a specific paper's numbers against our problem class —
  flagging as a structural idea worth a literature pass of its own before
  implementing, not as a verified recommendation.
- **Deep Latent Force Models** (arXiv:2311.14828) `[SEARCH-ONLY]`: hybridizes a
  mechanistic ODE structure with a GP-based forcing function; reported to beat
  latent-ODE and RNN baselines on interpolation/extrapolation. GP-based, not directly
  portable to this TF/Keras MLP pipeline, but the underlying idea (give the forcing
  channels their own structural role instead of raw concatenation) is the same
  family as the CDE/Koopman ideas above.
- **Jacobian/kinetic-energy regularization for Neural ODEs** (Finlay et al., ICML
  2020, arXiv:2002.02798) `[SEARCH-ONLY]`: regularizes the vector field to be
  "simple" so *adaptive* solvers take fewer steps — a training-speed fix for
  adaptive-step generative NODEs, not obviously relevant to a fixed-step
  forward-Euler forced ODE at this scale. Noted for completeness, not ranked.

---

## 7. Ranked recommendations for `recon/train_fields.py` (my synthesis, not a source)

Ranked by (expected impact on the flap-driven dynamic residual) × (implementation
cost), cheapest/best-evidenced first. Each entry: CLI-flag-style change, cost tier,
falsifiable prediction.

1. **`--dyn-activation silu`** (swap NNdyn's tanh for SiLU). Cost: **CHEAP** — one
   line in `build_networks` (train_fields.py:175-178), no shape/loop changes.
   Best-evidenced single change (§1, items 1-2, one full-text-verified direct
   tanh-vs-SiLU-in-a-NODE result). *Prediction:* dynamic-fluctuation NRMSE drops
   specifically in flap-driven segments, static/mean-split bias unaffected (it's
   already zeroed by mean-split and doesn't touch NNdyn). *Null result:* no change,
   or uniform improvement unrelated to flap timing — would argue the residual isn't
   an activation-saturation effect.
   - Cheap paired variant worth running alongside: **`--dyn-activation sine`**
     (SIREN-style, reusing the ω₀ hyperparameter already in the codebase for CORAL) —
     same cost, tests the *periodicity-matching* hypothesis instead of the
     *non-saturation* hypothesis. Flag explicitly: the CORAL SIREN dead end (ω₀=30/60
     overshoot) was for the *decoder's spatial* frequency; NNdyn's natural frequency
     is set by the *gust/flap timescale*, a different quantity, so that dead end
     does not automatically predict failure here — but the same overshoot risk
     pattern (need to sweep ω₀, don't assume ω₀=10 transfers) should be expected.

2. **`--integrator rk4`** (replace the forward-Euler step in `evolve_dynamics` with
   4-stage RK4, holding/interpolating `input_signals` at the intermediate half-step —
   e.g. linear interpolation between `input_signals[:,i,:]` and `[:,i+1,:]`). Cost:
   **MODERATE** — real but contained new code (~30-50 lines), no architecture change,
   decoder untouched. Evidence: closest domain analogue (unsteady airfoil + exogenous
   control NODE, §4 item 8) uses RK4, not Euler, plus a solid textbook argument about
   stability/order that specifically matters more for oscillatory forcing. *Caveat
   already flagged above:* the 4-8× numbers in the cited paper are confounded with
   architecture, not a clean ablation — treat this as "the closest published analogue
   chose RK4," not "RK4 buys Nx." *Prediction:* residual reduction concentrated where
   the exogenous signals change fastest relative to `dt` (flap commands, sharp gust
   fronts), roughly flat elsewhere. *Null result:* uniform or no improvement — would
   argue the current `dt/dt_base` is already fine relative to the fastest timescale
   in the signals, and the residual is not an integration-order artifact.

3. **`--dyn-train-noise <sigma>`** (inject `N(0, sigma)` into `state` each step of
   `evolve_dynamics`, before feeding it to `NNdyn`, sigma as a fraction of the
   latent's typical range). Cost: **CHEAP** (~5 lines). Evidence: solid mechanism,
   full-text verified (§2 item 4), but weakest-motivated of the cheap options because
   the specific train/rollout mismatch it was designed for doesn't exist here (§0) —
   test it as a generic regularizer on the single-shooting BPTT landscape, in the
   same spirit as the project's existing Tikhonov win. *Prediction:* validation
   dynamic residual improves at some intermediate sigma with training loss slightly
   higher (regularization signature, like the existing Tikhonov result). *Null
   result:* monotonic degradation at all tested sigma (matches the "more L-BFGS
   budget = overfitting" pattern already seen) — would argue noise isn't the lever.

4. **`--dyn-add-signal-rates`** (finite-difference `Ẇ_gust` and `δ̇` and concatenate
   as two extra NNdyn inputs; `ḣ, α̇` are already given). Cost: **CHEAP** (~10 lines,
   data-prep only). Cheap, low-risk probe of the Neural-CDE insight (§6) — cheap
   variant of "make rate-of-change of the forcing load-bearing" before committing to
   the full CDE restructure. *Prediction:* helps specifically in fast-transient flap
   segments if NNdyn is currently starved of clean rate information (it would
   otherwise have to infer rates from finite differences of its own noisy/smoothed
   latent state). *Null result:* no change — network already reconstructs what it
   needs from consecutive z's implicitly.

5. **`--dyn-training curriculum-shooting`** (split each training trajectory into K
   segments with their own learnable initial latent states, continuity penalty
   between segments, anneal K→1 or penalty weight up over training). Cost:
   **MODERATE-MAJOR** — new trainable variables per trajectory segment, a continuity
   loss term, and probably a change to how `RecordingOptimizationProblem`/BFGS
   stitches variables (segment-count now varies the parameter vector size across a
   training schedule). Evidence: §3, the *exact* pathology (long single-shooting BPTT
   through an oscillatory/forced trajectory "flattening out" the fit) is both
   textually and structurally matched to `evolve_dynamics`'s current loop (§0), and
   offers a candidate explanation for the already-observed L-BFGS-budget dead end.
   Ranked below the cheap items purely on cost, not confidence. *Prediction:* at
   equal or less total optimizer effort, achieves lower training loss AND lower
   dynamic-residual validation than plain single-shooting, most visibly in gust+flap
   trajectories vs. gust-only. *Null result:* same or worse loss at equal budget —
   would argue the ceiling is representational (architecture/activation/integrator),
   not an optimization-landscape artifact, and would deprioritize this whole avenue.

6. **`--dyn-width {16,32}`** (paired with item 1, not standalone). Cost: **CHEAP**
   (flag already exists). Not independently literature-backed — flagged by the task
   as untested and distinct from the ruled-out `d_s` axis. Worth sweeping jointly
   with the activation change since width-7 leaves very little redundancy to escape
   tanh saturation collectively (§1). *Prediction:* SiLU + modest width increase
   compounds; tanh + width increase alone does not (would isolate whether the
   problem is capacity-in-NNdyn generically, which the task says is unlikely, vs.
   specifically activation saturation).

7. **`--dyn-cond cde`** (full Neural-CDE restructure: NNdyn outputs a
   `(d_s × n_signals)` matrix contracted with a spline-derivative of the exogenous
   signals, replacing concatenation). Cost: **MAJOR** — new NNdyn output shape,
   spline fitting/derivative of `input_signals`, rewritten integration loop. Do this
   only if item 4 (the cheap rate-features probe) shows a real effect; if it doesn't,
   the full structural mechanism is unlikely to pay for its cost. *Prediction:*
   stronger version of item 4's prediction — larger effect specifically because
   forcing is now structurally gated through the state rather than concatenated.
   *Null result:* no better than item 4 — mechanism doesn't matter beyond having the
   rate information at all.

8. **`--dyn-refine-steps K`** (PDE-Refiner-style denoising refinement, §5). Cost:
   **MAJOR**, and the weakest domain match in this document (built and validated for
   large spatial-field networks with static forcing; here the "spectral band" concept
   is being applied to a single scalar latent channel with time-varying forcing,
   neither of which the source paper tested). Only worth pursuing after cheaper items
   are exhausted. *Prediction:* residual shrinks roughly monotonically with K if the
   "MSE starves rare/small-amplitude content" story holds for a 1-D latent.
   *Null result:* flat/no change across K — mechanism doesn't transfer to this scale.

9. **`--dyn-cond attention`** (self-attention conditioning of NNdyn on
   `[z, params, signals]` tokens, arXiv:2209.10684). Cost: **MODERATE-MAJOR**. Ranked
   last of the concrete items: the source paper's own evidence for attention beating
   concatenation is conditional on high conditioning dimensionality, and NNdyn's
   conditioning vector (8 scalars) is far below that regime (§5) — a positive result
   here would be genuinely new evidence beyond the source paper's scope, but there is
   no a-priori reason from the literature to expect it at this scale. *Prediction:*
   given the dimensionality mismatch, a-priori expected result is null (no
   improvement over concatenation); a real gain would be a surprising, separately
   interesting finding, not a "the literature already showed this" confirmation.

---

## 8. Not resolved / would need more digging

- No source found that runs any of the mechanisms above (activation choice,
  integrator order, shooting strategy) specifically on a **forced/driven** ODE with
  both a stochastic-ish (gust) and a commanded (flap) exogenous input simultaneously
  — every well-matched paper found is either autonomous (Floquet/Morris-Lecar work)
  or single-input-driven (GNODE airfoil paper: pitch only, no combined gust+flap).
  The gust+flap-combo-is-harder-than-gust-alone symptom this project already
  documented has no direct literature analogue I could find; it's consistent with
  (but not proven by) the single-shooting "flattens oscillatory content" story if the
  combined signal has more/faster spectral content than either alone.
- Did not verify whether `torchdiffeq`/`diffrax`-style adaptive-step solvers (as
  opposed to fixed-step RK4) would help more than fixed RK4 for this problem —
  out of scope of what was fetched; flagging as a follow-up question, not a
  recommendation (adaptive stepping is a bigger implementation change and interacts
  awkwardly with a loss that's evaluated at fixed CFD-sample timestamps).
- Could not obtain quantitative numbers for the multiple-shooting papers (§3, items
  5-7) beyond the qualitative claims quoted — abstract-only fetches. If item 5 in the
  ranked list is pursued, read the full PDFs of arXiv:2109.06786 and/or
  arXiv:2608.05777 first to get actual segment-count/penalty-weight guidance before
  implementing.
