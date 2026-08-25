# Stall/separation literature findings — research run 2026-08-22

Scope: the D-RES residual, newly confirmed (`recon/analysis/MEANSPLIT_NOTES.md`
"STALL/SEPARATION HYPOTHESIS" section, `recon/analysis/decomp_stall.py`) to be a real,
localized, transient, sign-flipping flow-reversal event near the flap — not a diffuse
smooth-fit problem. Targets literature on regime-switching/MoE dynamics, semi-empirical
dynamic-stall models, flap-gap/cove separation physics, separation-aware losses,
local/discontinuity-aware decoders, GNN decoders, and calibration numbers for
post-stall surrogate error. Five prior D-RES levers (decoder depth, near-wall sampling,
optimizer budget, multiple shooting, rate-only CDE conditioning) are excluded from
consideration per task instructions — they targeted the whole field/whole trajectory,
not this localized event.

Verification method, same convention as `LATENTODE_LITERATURE_NOTES.md`: `[FULL-TEXT]`
= I fetched and read the actual paper content (HTML/ar5iv rendering with quotable
text, not just an abstract page). `[ABSTRACT-ONLY]` = fetch returned only abstract-level
content (mechanism plausible but not confirmed in depth) — this includes several cases
where WebFetch on an arXiv PDF returned an unparseable binary stream and only the
`/abs/` page or a prior search snippet gave usable text; I did not silently upgrade
these to FULL-TEXT. `[SEARCH-ONLY]` = no direct fetch succeeded (403/paywall/bot-check),
relying on search-engine result snippets — treat as unverified pointers, not confirmed
claims.

---

## 0. Framing check (code/data-verified, not a literature claim — read first)

Before searching, I checked whether this project's flap geometry is a plain
continuously-hinged single-surface flap (which would make classical AoA-driven dynamic
stall the natural literature match) or a discrete multi-element flap with a physical
gap (which would make flap-cove/gap literature the natural match). It is the latter,
confirmed two ways:
- `recon/analysis/decomp_stall.py` line 34 and `MEANSPLIT_NOTES.md` line 493 both treat
  "main" and "flap" as **separate perimeter-ordered node sets** (`main[0:292)` +
  `flap[292:387)`), i.e. two distinct closed contours, not one continuous surface.
- `MEANSPLIT_NOTES.md` line 148 gives the actual geometry: "x>~0.735 (gap 0.72→0.75);
  hinge ≈ flap LE (min-x flap node), chord 0.9776" — a real ≈0.03-chord physical gap
  between the main-element trailing edge and the flap leading edge.

This matters directly for item 3 below: this is a **two-element airfoil-with-gap**
configuration, structurally the flap-cove/flap-gap family (main-element wake + gap
shear layer interacting with a downstream flap), not a single continuous surface
undergoing classical leading-edge angle-of-attack stall. Combined with the diagnostic's
own finding that the reversal sits "right on and immediately behind the flap upper
surface/trailing edge" (`MEANSPLIT_NOTES.md` line 505-508) — i.e., at/downstream of the
gap, not at a leading-edge suction peak — this is the deciding evidence for ranking
flap-gap/cove literature above classical dynamic-stall literature as the physical
match, addressing the task's item 3 explicitly.

---

## 1. Flap-gap/cove separation physics — the family verdict (item 3)

1. `[SEARCH-ONLY]` The unsteady flow in the gap region of a high-lift multi-element
   airfoil is dominated by a shear layer that separates from the upstream element's
   cove/gap lip and rolls up via Kelvin-Helmholtz instability into discrete coherent
   vortices that convect downstream and interact with the next element's surface.
   This mechanism is documented for both slat coves (leading-edge gap) and flap coves
   (main-element/flap gap) as structurally the same phenomenon applied to different
   gap locations.
   [Unsteady aerodynamics of flap cove flow in a high-lift device configuration,
   AIAA 2001-707; companion slat-cove paper, ResearchGate 238364895]
   **Caveat:** I could not get a clean fetch of either paper (AIAA paywall, ResearchGate
   403) — the search-engine summary blended language from both the flap-cove and the
   slat-cove abstract, so I am not citing a specific impingement direction (e.g.
   "strikes the flap leading edge") as confirmed; only the general mechanism
   (cove-lip shear-layer roll-up into vortices) is trustworthy here.
2. `[SEARCH-ONLY]` "Study of Gap Physics of Airfoils with Unsteady Flaps" reports that
   for an **oscillating/deflecting flap** specifically (not a static gap), "the lag
   between aerodynamic response and deflection input increases approximately linearly
   with airfoil-flap gap size," and that "strong vorticity within the gap between the
   airfoil and the flap must be reproduced if boundary layer behavior, separation, and
   drag are to be properly reproduced."
   [Journal of Aircraft, arc.aiaa.org/doi/10.2514/1.C032026 — fetch blocked, 403]
   This is the single most directly relevant hit in the whole search: it is
   specifically about **unsteady flap motion** (our exact forcing — flap-deflection
   δ(t)) generating gap vorticity with a response lag proportional to gap size,
   independent of AoA/leading-edge stall physics entirely. It also gives a candidate
   mechanistic explanation for why the event is tied to the *combination* of gust and
   flap command rather than either alone (previously an unexplained symptom, see
   `LATENTODE_LITERATURE_NOTES.md` §8) — a fast flap deflection during a gust could be
   exactly the transient that excites this gap-vorticity lag response.
3. `[SEARCH-ONLY]` A companion paper, "The physics of modeling unsteady flaps with
   gaps," addresses the same phenomenon (title/topic only, could not fetch abstract
   past a 403).
   [Journal of Fluids and Structures, ScienceDirect S0889974612002332]

**Verdict on item 3 (my synthesis): flap-gap/cove separation is the better-matched
family, not classical AoA-driven leading-edge dynamic stall.** Three independent
reasons: (a) the geometry confirmed in §0 is a discrete two-element gap, not a
continuous surface; (b) the diagnostic's own spatial localization (flap upper
surface/trailing edge, i.e. at/downstream of the gap) matches gap-shear-layer physics,
not a leading-edge suction-peak stall location; (c) item 2's most relevant hit is
specifically about *unsteady flap deflection* driving gap vorticity with a
gap-size-scaled lag, which is a mechanistically closer match to "gust+flap-driven,
transient, localized at the flap" than any leading-edge AoA-stall model reviewed in
§2 below, none of which have a gap/second-element concept at all.

---

## 2. Semi-empirical dynamic-stall models and neural hybrids (item 2)

4. `[SEARCH-ONLY, equations from secondary/aggregated sources, not an independently
   fetched primary text]` The classical **Goman-Khrabrov model** represents the
   attached/separated flow state with a single scalar internal variable X(t) ∈ [0,1]
   (1 = fully attached, 0 = fully separated), governed by a first-order lag ODE:
   `τ₁ (dX/dt) + X = X₀(α(t) − τ₂ α̇(t))`, where X₀(·) is the static separation-point
   curve, τ₁ a relaxation time constant, τ₂ a stall-delay time constant tied to pitch
   rate. Lift then follows via a Kirchhoff-flow closure,
   `C_l(α) = (dC_l/dα)|₀ · sinα · ((1+√X)/2)²`.
   **This is the single most directly transferable structural idea found in the whole
   review**: a scalar "attachment state" with its own lag dynamics, driven by a
   *rate-augmented* version of the forcing (α and α̇, not α alone) — i.e., exactly the
   "state-augmented dynamics with a separation lag state" the task explicitly asked
   about. It is a genuinely different structural pattern from NNdyn's current
   undifferentiated concatenation of z with raw signal values.
5. `[ABSTRACT-ONLY]` A 2026 paper replaces the Goman-Khrabrov model's *empirical*
   time constants τ₁, τ₂ with **physics-derived** ones: stall delay is "a function of a
   normalised instantaneous pitch rate" and shown largely independent of airfoil
   geometry/Re/motion type; post-stall decay rate is "directly related to the Strouhal
   number of the post-stall vortex shedding." Validated across multiple airfoils,
   Re 75k–1M, sinusoidal and ramp motions. No accuracy numbers vs. the empirical
   version were available from the abstract.
   [Ayancik & Mulleners, "All you need is time to generalise the Goman-Khrabrov
   dynamic stall model," JFM / arXiv:2110.08516]
6. `[SEARCH-ONLY]` A 2025 Physics of Fluids paper integrates Leishman-Beddoes force
   components directly into a "data fusion neural network" (DFNN) as physical
   structure/knowledge, reporting extrapolation to new reduced-frequency/AoA
   conditions using only **1/5 the training samples** a pure data-driven model needs.
   [Data-knowledge-driven dynamic stall modeling guided by stall patterns and
   semi-empirical model, Physics of Fluids 37(4):045106, 2025 — fetch blocked, 403]
7. `[SEARCH-ONLY]` A very recent (AIAA SciTech 2026) paper learns dynamic-stall models
   directly from **surface pressure data** via a "conditional autoencoder structure to
   compress the high-dimensional surface pressure data to a latent space that is
   conditioned on the angle of attack," trained across "pre-stall, stall and
   post-stall flow regimes" from ramp-hold-return pitching motions.
   [Discovering Interpretable Dynamic Stall Models from Surface Pressure Data,
   AIAA 2026-0293, doi:10.2514/6.2026-0293]
   **Relevance:** this is structurally the closest published analogue to this
   project's own (latent z, exogenous signal)→field setup, but purpose-built for a
   stall/separation transition specifically — worth chasing the full text if this
   avenue is pursued further (see handoff prompt).
8. `[FULL-TEXT]` A WaveNet-based (dilated causal convolutions, mixture-density output,
   autoregressive) stochastic dynamic-stall model, trained on raw experimental
   AoA→C_l/C_d/C_m data, explicitly targets what a semi-empirical model misses: "the
   model can only do what it is designed to do, which is to predict the mean lift
   values. Strong fluctuations in the curves are thus extremely smoothed" (their
   characterization of Beddoes-Leishman in QBlade). The WaveNet model reproduces the
   stochastic leading-edge-vortex shedding fluctuations instead. It conditions
   directly on AoA and Re (not rate, not an engineered stall-onset indicator) and
   lets the autoregressive/dilated-conv architecture infer transition timing from
   its own receptive-field history (~128 timesteps). No quantitative accuracy number
   vs. Beddoes-Leishman is given (evaluated via a DTW+EMD distributional score, not
   pointwise error).
   [A WaveNet-based fully stochastic dynamic stall model, Wind Energy Science
   7:1889-1906, 2022, wes.copernicus.org/articles/7/1889/2022]
   **Relevance/caveat:** the "semi-empirical models over-smooth the transient
   fluctuation, need a model expressive enough to reproduce it" framing matches this
   project's D-RES symptom description well (smooth mean-fit vs. missed transient),
   but the WaveNet fix is an autoregressive-in-time, large-receptive-field
   architecture change — a materially bigger lift than NNdyn's current single-MLP
   forward-Euler step, and 1-D scalar C_l output, not a spatial field.
9. `[SEARCH-ONLY]` "State-Space Neural Networks" (discrete-time nonlinear state-space
   models, NOT the SSM/Mamba family — a system-identification usage of that name)
   are reported to accurately predict pitching-airfoil loads in **both pre-stall and
   post-stall** regimes, "capturing highly nonlinear flow features such as the delay
   in flow separation and the formation and shedding of dynamic stall vortex."
   [Modeling airfoil dynamic stall using State-Space Neural Networks, AIAA 2023-1945;
   companion JFM paper on NACA 0018 — both behind paywalls, not fetched]
10. `[SEARCH-ONLY, title/topic only, not chased down]` Several 2025-2026 papers extend
    similar directions: "Predicting airfoil dynamic stall loads using neural
    networks" (ScienceDirect S1270963825005371); "A Dual-Step Deep Learning-Based
    Surrogate Model for Dynamic Stall Predictions" (VTOL Forum 2024); "Prediction of
    airfoil dynamic stall unsteady loads using dynamic neural networks architectures"
    (Meccanica, 2026). None fetched — flagged as follow-up reading if this avenue is
    pursued.
11. `[SEARCH-ONLY]` The Boeing-Vertol model is confirmed as a real third member of the
    semi-empirical family (alongside Beddoes-Leishman and ONERA), described only at
    the level of "semi-empirical aerodynamic theory... stall hysteresis for lift,
    drag, and pitching moment" — no distinguishing structural detail obtained beyond
    that it exists and is commonly grouped with Beddoes-Leishman/ONERA/RISO/Oye in
    survey articles. Not independently differentiated from Leishman-Beddoes in this
    pass.

**Reading:** every semi-empirical model in this family (Goman-Khrabrov, Leishman-
Beddoes, ONERA, Boeing-Vertol) is built around the same core idea — a **low-order lag
state tracking the current degree of attachment**, driven by the forcing *and its
rate*, closing the loop with a static nonlinear map back to loads. This is a
genuinely different structural pattern from NNdyn's current "concatenate raw signals,
let the MLP figure it out" design, and it is a *much* smaller structural change than
the already-tried CDE restructure (item 5 of the "already closed" list) — it adds one
new *scalar* state with an interpretable, literature-prescribed lag-ODE form, rather
than restructuring the entire vector field's conditioning mechanism.

---

## 3. Regime-switching / mixture-of-experts dynamics (item 1)

12. `[ABSTRACT-ONLY]` A 2026 paper (external automotive aerodynamics, DrivAerML
    dataset) trains a mixture-of-experts **gating network over three heterogeneous
    surrogate architectures** (DoMINO, X-MeshGraphNet, FigConvNet — different backbone
    families, not just different seeds of the same net). The gate "learns a
    spatially-variant weighting strategy, assigning credibility to each expert based
    on its localized performance." Result: "the MoE model achieves a significant
    reduction in L-2 prediction error, outperforming not only the ensemble average but
    also the most accurate individual expert model across all evaluated physical
    quantities." An entropy-regularization term on the gate prevents expert collapse
    (all weight going to one expert). No exact percentage was obtainable from the
    fetch (abstract-level only).
    [A Mixture of Experts Gating Network for Enhanced Surrogate Modeling in External
    Aerodynamics, arXiv:2508.21249]
    **Relevance:** not stall-specific, but real evidence — in an actual CFD-surrogate
    pipeline, not a toy problem — that a *spatially-varying* gate beating both the
    best single expert AND the ensemble average is achievable, i.e. the gate is doing
    real localized work, not just averaging. The caveat is the "experts" here are
    three different architecture families (expensive to reproduce this specific
    recipe), not evidence that a *cheap* per-region gate over otherwise-identical
    small heads would work as well.
13. `[ABSTRACT-ONLY]` MODE (Mixtures Of Dynamical Experts) is a **mixture-of-latent-ODE-
    experts** framework: a soft gating network assigns (normalized, not hard-switched)
    weights across multiple ODE-vector-field experts, each specializing in a different
    dynamical regime, tested on a synthetic bifurcating oscillator (Goldbeter mitotic
    model) and biological cell-cycle data. This is structurally the closest match
    found to "give NNdyn multiple regime-specific vector fields with a learned gate,"
    applied to exactly the kind of ODE-with-latent-state setting NNdyn already is —
    but validated only on low-dimensional biological/synthetic systems, not
    aerodynamics, and not on anything forced by fast exogenous signals.
    [MODE: Learning compositional representations of complex systems with Mixtures
    Of Dynamical Experts, arXiv:2510.09594]
14. `[SEARCH-ONLY]` General background on switching state-space models (Ghahramani &
    Hinton-style SLDS: a pool of linear/stationary local dynamics models selected by a
    probabilistic switching process, with variational inference for the intractable
    exact posterior) — this is the classical statistical antecedent of the MoE-dynamics
    idea above, confirmed to exist as a well-established modeling family, but the
    concrete hits found were all in neuroscience/signal-processing applications, none
    in aerodynamics or CFD.
    [Switching state-space modeling of neural signal dynamics, PLOS Comp Bio 2023;
    general SLDS literature]

**Reading:** no paper was found that applies mixture-of-experts or regime-switching
dynamics specifically to an aerodynamic/flow-separation latent-ODE — this remains a
genuine literature gap, not a well-trodden path (flag explicitly, see §9 "not
resolved"). The two closest matches (items 12, 13) each confirm half of the idea
independently — spatially-varying gating helps in a real aero-surrogate pipeline
(item 12), and mixture-of-ODE-experts with soft gating is a working mechanism for
regime-switching latent dynamics (item 13) — but nobody has combined them for this
exact problem class. A from-scratch implementation here would be genuinely novel, not
"literature already showed this."

---

## 4. Local/patch decoders and discontinuity-aware neural fields (item 5)

15. `[SEARCH-ONLY]` Discontinuity-Aware 2D Neural Fields (DANF) represents an image/2D
    field on a curved triangular mesh, storing features on vertices and on a subset of
    edges explicitly marked discontinuous, decoded by a shallow MLP — producing a field
    that is discontinuous exactly across the marked edges and smooth everywhere else.
    **Critically, it requires the discontinuity locations as input** (given as
    Bézier curves) — it does not discover them.
    [Belhe et al., ACM TOG 2023, doi:10.1145/3618379]
16. `[FULL-TEXT]` The direct follow-up removes that requirement: "2D Neural Fields
    with Learned Discontinuities" treats **every mesh edge as a potential
    discontinuity**, introducing a continuous per-edge jump-magnitude variable
    optimized jointly with the feature field by ordinary gradient descent — "these
    variables are continuous and happily optimized along with feature vectors and
    mesh vertex positions." When the learned weight →0 the edge is effectively
    continuous; when large, it represents a real jump. Reported gains vs. a plain
    continuous neural field (InstantNGP) baseline: **+5.5dB PSNR on denoising**
    (44.486 vs 39.016), **+11.2dB PSNR on 2× super-resolution** (43.913 vs 32.715),
    and **3.5× smaller Chamfer distance** to the true discontinuity curve vs. a
    classical Mumford-Shah-based mesh method (0.165 vs 0.580).
    [Liu et al., "2D Neural Fields with Learned Discontinuities," Computer Graphics
    Forum 2025, arXiv:2408.00771]
    **Caveat, explicitly confirmed by fetching the full text (not an inferred gap):**
    this method is validated only on **static** 2D signals (images, depth/normal maps)
    — the paper contains no time-varying/transient discontinuity experiment, and the
    per-edge jump variable is a *fixed* learned quantity, not a function of time or of
    an exogenous forcing signal. Applying this to a discontinuity that appears,
    grows, and disappears within a ≈0.3s window as a function of (z, gust, flap) would
    require making the jump-magnitude itself a small network's output rather than a
    free per-edge scalar — a real extension, not a drop-in reuse.
17. `[SEARCH-ONLY, general/textbook-level]` The underlying reason plain coordinate-MLP
    decoders (ReLU or SIREN alike) cannot represent a true discontinuity is structural:
    ReLU nets are piecewise-linear with discontinuous *derivatives* but continuous
    *values*; SIREN's sinusoidal activations are infinitely differentiable everywhere,
    so both the field and all its derivatives are smooth by construction — a sign flip
    at a moving boundary can only be approximated by a steep-but-continuous transition,
    which is exactly the qualitative failure this project's diagnostic already
    observed (ROM smooths through the reversal rather than crossing zero sharply).
    This is a structural/textbook point about SIREN's mathematical properties, not a
    single paper's claim, so it is not independently attributed to one source.

**Reading:** item 16 is the best-evidenced concrete mechanism in this whole document
for the *decoder-side* half of the problem (representing a spatially localized,
possibly sign-changing feature without corrupting the surrounding smooth field) — but
it is validated only for static fields, so porting it here means treating "is this
edge/patch currently in a separation state" as **itself a small learned function of
(z, exogenous signals)**, i.e. structurally similar to a local, spatially-gated version
of the mixture-of-experts idea in §3, not a literal reuse of their per-edge scalars.

---

## 5. Loss reweighting, hard-example mining, curriculum weighting (items 4 and 6)

18. `[SEARCH-ONLY, well-established/landmark methods, not independently re-verified
    this session]` **Online Hard Example Mining (OHEM)** keeps only the
    highest-loss examples/regions per batch for the gradient update, discarding easy
    ones entirely. **Focal Loss** instead keeps all examples but multiplies each one's
    loss by a smooth modulating factor that down-weights already-well-fit ("easy")
    examples and up-weights poorly-fit ("hard") ones — the two are usually
    contrasted as hard-discard vs. soft-reweight variants of the same idea.
    [Shrivastava et al., OHEM, CVPR 2016; Lin et al., Focal Loss, ICCV 2017]
19. `[SEARCH-ONLY]` "Not All Pixels Are Equal: Learning Pixel Hardness for Semantic
    Segmentation" replaces a hand-tuned weighting rule with a **learned** per-pixel
    hardness map used to reweight the segmentation loss — closer in spirit to a
    model-driven than a hand-engineered weighting scheme.
    [arXiv:2305.08462]
20. `[SEARCH-ONLY]` Two 2026 curriculum-PINN papers directly target *spatially
    localized, hard-to-fit regions* inside an otherwise-smooth-field training loop —
    exactly this project's confirmed-open loss-weighting situation:
    - "Curriculum Learning of Physics-Informed Neural Networks based on Spatial
      Correlation" partitions the domain into subregions and does "region-adaptive
      local reweighting... based on regional PDE residuals and gradient contributions"
      to "reduce local residuals and improve the recovery of high-frequency details."
      [arXiv:2605.15254]
    - CGMPINN periodically fits a **Gaussian mixture model to the current residual
      distribution** to quantify spatially varying difficulty, then smoothly shifts
      training focus toward the harder regions over training.
      [Curriculum-Guided Gaussian Mixture PINN, arXiv:2605.19263]

**Reading:** this is the cheapest, most directly implementable family in the whole
review, because the project already confirmed the exact hook is open and unused
(`train_fields.py` ~line 432-433, plain unweighted `tf.reduce_mean(tf.square(...))`,
no per-point/per-region weight anywhere). All four items above are variations on one
mechanism — multiply the per-point squared error by a weight before reducing — that
requires zero new trainable parameters, zero architecture change, and is distinct
from the already-failed *sampling* reweight (item 2 of the closed list) because it
does not sacrifice far-field point budget to see more near-flap points; it changes how
much each already-sampled point's residual counts. The curriculum-PINN framing (items
20) is the best domain match: build the weight from the *local residual magnitude*
itself (data-driven, no separate "ground truth separation label" needed), optionally
restricted temporally to windows where the residual is large (naturally concentrating
on the gust-peak transient without hand-labeling "when is separation happening").

---

## 6. GNN / message-passing decoders (item 7)

21. `[SEARCH-ONLY]` A message-passing GNN reconstructs pressure and velocity fields
    around airfoils from sparse surface-pressure sensing.
    [Graph Neural Networks for Aerodynamic Flow Reconstruction from Sparse Sensing,
    OpenReview — fetch blocked by bot-check, no content obtained]
22. `[SEARCH-ONLY]` A "flow-field-message-informed" GNN aggregates edge-weighted
    attributes across multiple hop layers specifically for **unsteady compressible**
    flow prediction.
    [Physics-constrained and flow-field-message-informed graph neural network for
    solving unsteady compressible flows, Physics of Fluids 36(4):046123, 2024 —
    abstract-level snippet only]
23. `[SEARCH-ONLY]` A mesh-adaptive hypergraph neural network is specifically built for
    "Unsteady Flow Around Oscillating and Rotating Structures" — i.e., a moving-
    geometry setting structurally analogous to this project's moving flap.
    [arXiv:2503.22252 — title/topic only, not fetched]
24. `[SEARCH-ONLY]` A multiscale GNN autoencoder does mesh-based super-resolution of
    fluid flow fields.
    [arXiv:2409.07769 — WebFetch failed, file exceeded the tool's 10MB size limit;
    no content obtained beyond the title]

**Reading:** message-passing GNN decoders are local by construction — each layer only
mixes information within a graph neighborhood, so a k-layer GNN has a receptive field
of k hops, in sharp contrast to a coordinate-MLP/SIREN decoder where every output
point has *global* receptive field over the full (z, x, y) input every single forward
pass. That structural property is exactly what would let a decoder represent a
localized feature without the smoothing-through-the-whole-field failure mode described
in §4. However: (a) none of the 4 items above were obtained past a search-engine
snippet, so no quantitative evidence was verified this session for how much this
locality actually helps on a *separation* feature specifically (as opposed to sparse
reconstruction or super-resolution, different tasks); (b) per the project context, no
graph/adjacency structure exists anywhere in this codebase's decoder today — this
would be genuinely new territory, the most expensive option reviewed here, more so
than the local-discontinuity idea in §4 which can be built as a small modification to
the *existing* coordinate-conditioned SIREN rather than a wholesale architecture swap.

---

## 7. Calibration: how bad is "normal" for surrogates at post-stall/separated flow (item 8)

25. `[FULL-TEXT]` A deep-learning inference model for compressible turbulent flow over
    airfoils reports L1 loss broken out by regime: attached-flow cases achieve
    **≈4.90×10⁻⁴**; the worst separated case (aerofoil goe398, α=−22°, strongly
    separated) achieves **≈13.30×10⁻⁴ (±0.26×10⁻⁴)** — about **2.7× worse** than
    attached; the worst shock-induced-separation transonic case (aerofoil e221)
    achieves **≈11.52×10⁻⁴ (±0.15×10⁻⁴)** vs. **≈2.67×10⁻⁴ (±0.12×10⁻⁴)** for a
    shock-free transonic case at the same Mach — about **4.3× worse**.
    [Towards high-accuracy deep learning inference of compressible turbulent flows
    over aerofoils, arXiv:2109.02183]
    **This is the most load-bearing calibration number found.** It establishes that a
    **~2.5–4× error inflation at separated/shock-separated conditions relative to
    attached conditions is itself the published norm** for airfoil-flow DL surrogates,
    not a symptom unique to this project's approach. Read against this project's own
    numbers (ROM misses 21/23 percentage points of the reversed-flow fraction, i.e.
    a near-total qualitative miss rather than a 2.5-4x quantitative inflation), the
    D-RES residual is *worse in kind*, not just worse in degree, than what this
    reference paper calls a normal-hard case — consistent with §4's structural
    argument that this is a sign-flip decoder-expressiveness problem, not merely an
    under-trained-harder-region problem that more capacity/data would gradually close.
26. `[SEARCH-ONLY]` A PINN paper specifically targets "the separated Reynolds-averaged
    turbulent flow field around an airfoil under variable angles of attack" — title and
    topic are a strong match, but the full text was behind a Springer login wall
    (redirect to `idp.springer.com`) and no quantitative numbers were obtained.
    [Data-assisted training of a physics-informed neural network to predict the
    separated Reynolds-averaged turbulent flow field around an airfoil under variable
    angles of attack, Neural Computing and Applications, 2024,
    doi:10.1007/s00521-024-09883-9]
27. `[SEARCH-ONLY]` Broader search corroborates the same qualitative pattern without
    giving further numbers I could verify: several surrogate papers explicitly note
    good overall/global accuracy but flag "room for improvement... particularly where
    flow separation occurs" or in "adverse pressure gradient regions."
28. Could not find a verified AirfRANS-benchmark-specific number for post-stall/
    high-AoA error (AirfRANS itself is mostly attached/near-stall RANS cases; no
    leaderboard entry broken out by attached-vs-separated subsets was found this
    session) — flagged as unresolved, not fabricated (see §9).

---

## 8. Not resolved / would need more digging

- No paper was found that combines regime-switching/MoE dynamics with an
  aerodynamic flow-separation latent-ODE specifically (§3) — the two closest matches
  are each in a different domain (automotive steady-state field gating; biological
  ODE switching). A genuinely novel combination if pursued.
- Could not obtain the full text of the semi-empirical/neural hybrid papers with the
  most direct relevance (Data-knowledge-driven LB hybrid, AIP PoF 2025; AIAA SciTech
  2026-0293's conditional-autoencoder stall model) — both blocked by paywalls/403.
  If item 2 (separation-lag state) in the ranked recommendations is pursued, fetching
  these two full texts first (via institutional access, not open web) would firm up
  the design before implementation.
- Could not independently verify the Goman-Khrabrov model equations against the
  original primary source (PDF fetch failed to parse) — the equations quoted in §2
  item 4 came from search-engine-aggregated secondary sources. They are consistent
  with how this model is near-universally described in the wind-turbine/rotorcraft
  aeroelasticity literature, so confidence is reasonably high, but flagging per this
  document's own verification standard.
- Did not find a clean AirfRANS-leaderboard-style number specifically isolating
  post-stall/separated-flow error (item 8's open half, §7 item 28) — a follow-up
  search specifically on the AirfRANS paper's own per-regime error breakdown (not
  just leaderboard aggregate NRMSE) would be the next step if tighter calibration is
  needed.
- Could not confirm whether the flap-cove-flow paper's shear-layer-impingement
  direction (§1 item 1) applies to a moving/oscillating flap the way it's described
  for a static-geometry high-lift configuration — item 2 in §1 (unsteady flap gap
  paper) is the better match for *unsteady* flap motion specifically but its own
  abstract text beyond the two quoted sentences was not obtained (403).

---

## 9. Ranked recommendations for `recon/train_fields.py` (my synthesis, not a source)

Ranked by (how specifically the mechanism targets a LOCALIZED, TRANSIENT, sign-flipping
event, not generic capacity) × (implementation cost, using the project's own confirmed-
open extension points). Each entry: CLI-flag-style change, cost tier, falsifiable
prediction.

1. **`--loss-weight residual-curriculum`** (multiply each sampled point's squared
   error by a weight derived from its own recent residual magnitude — e.g. an
   exponential moving average of `|pred-target|` per spatial region/time-window,
   normalized so the mean weight stays ≈1 to avoid silently changing the effective
   learning rate). Cost: **CHEAP** — a few lines at `train_fields.py`'s `loss_fn`
   (~line 432-433), no new trainable variables, no shape changes; the exact hook the
   task confirmed is open and never used. Evidence: §5 (items 18-20), especially the
   two 2026 curriculum-PINN papers whose stated mechanism (region-adaptive reweighting
   by local residual magnitude, "reduces local residuals and improves recovery of
   high-frequency details") is close to a direct match. *Prediction:* the sign-flip
   rate at the diagnostic's peak time (currently 21.7% of near-flap points, ROM
   reversed-flow fraction 2.1% vs FOM 23.3%) drops measurably, with far-field/attached
   NRMSE flat or only mildly worse (the expected bias-variance signature of
   reweighting, not a free lunch). *Null result:* the sign-flip rate is unchanged
   regardless of how strongly the near-flap/peak-time region is upweighted — would
   argue the residual is a representational ceiling (the decoder/dynamics literally
   cannot express the reversal, however much gradient signal is aimed at it), directly
   motivating item 2 below.

2. **`--dyn-sep-state`** (add one scalar "attachment state" to NNdyn's own latent,
   governed by a lag-ODE of the Goman-Khrabrov form — `τ₁ Ẋ + X = X₀(gust_rate,
   flap_rate)`, with τ₁, X₀'s parameters trainable — concatenated as an extra input
   alongside z, matching the existing concat pattern NNdyn already uses for every
   other signal). Cost: **CHEAP-MODERATE** — a handful of new trainable scalars/small
   MLP for X₀(·), one extra state channel integrated in the same forward-Euler loop
   already there; `src/optimization.py`'s confirmed-generic `VariablesStitcher` picks
   up new trainable tensors with no framework change (exactly how the multiple-
   shooting experiment added its own free variables). Evidence: §2 (items 4-5, 8),
   the best-matched structural idea in the whole review — a low-order lag state
   tracking attachment degree, driven by *rate*-augmented forcing, is the shared
   architecture across every semi-empirical dynamic-stall model found, and directly
   answers the task's own suggestion ("state-augmented dynamics with a separation lag
   state"). Also consistent with the project's own already-learned CDE lesson (adding
   information helped where removing it hurt) — this *adds* a state without removing
   anything NNdyn currently has. *Prediction:* the reversed-flow fraction gap (ROM 2.1%
   vs FOM 23.3% at peak) closes substantially because the dynamics now carries an
   explicit memory of "an attachment-loss event is in progress," which the current
   design has no mechanism to represent at all. *Null result:* no change — would argue
   the missing piece isn't a *dynamics*-side memory/indicator but a *decoder*-side
   inability to spatially express the reversal even given a perfect indicator,
   motivating item 3.

3. **A spatially/temporally local decoder head for the near-flap region** — either
   (a) a small MoE-style gate over 2-3 decoder heads active only near the flap
   (informed by §3 item 12's spatially-variant-gate result and §4's discontinuity-
   aware neural field mechanism, made time-varying by making the local "jump/gate"
   strength itself a function of (z, exogenous signals) rather than a static per-edge
   scalar), or (b) a genuinely new local/message-passing decoder family for the
   near-flap subset of points only. Cost: **MAJOR** — new architecture, no existing
   scaffolding in this codebase (confirmed: no gating/MoE/graph structure exists
   anywhere today), and `reconstruct_fields.py` would need to be kept in sync (a past
   bug already happened here once per the project context). Evidence: §4 (item 16,
   full-text-verified quantitative gains for a *static* field — 3.5× better
   discontinuity localization, +5.5-11dB PSNR vs. a plain continuous coordinate field)
   and §3 (item 12, real spatially-gated aero-surrogate evidence), both real but
   neither validated on a moving/transient discontinuity the way this problem needs —
   this would be a genuine extension of both, not a drop-in. Rank third specifically
   because items 1-2 are cheaper, comparably well-evidenced, and their null results
   would sharpen which of "needs more gradient emphasis," "needs a dynamics-side
   memory," or "needs decoder-side locality" is actually true before paying this
   architecture's cost. *Prediction:* if items 1-2 both null out, a local/gated
   decoder should recover the sign at the worst near-flap points specifically (a
   clean localized gain, unlike the multiple-shooting dead end where a global-metric
   win came with the actual target residual getting worse). *Null result:* the same
   global/local mismatch as multiple shooting — would argue the ceiling isn't
   spatial-locality of the decoder either, and the bottleneck is further upstream
   (e.g. the FOM/training-data resolution of the event itself, or the latent
   dimensionality d_s=1 not having room to encode both the smooth field and a
   transient regime bit simultaneously — a question not chased down this session).

4. **Full regime-switching/MoE latent-ODE for NNdyn** (§3, item 13's mechanism: 2
   full NNdyn expert networks — "attached" and "separated" — with a learned soft gate,
   rather than the single scalar state in item 2). Cost: **MAJOR**, and ranked below
   item 2 specifically because it is the same underlying hypothesis (give the dynamics
   explicit regime information) at higher cost and with weaker domain-specific
   evidence (§3's own reading: no paper found doing this for aerodynamic separation
   specifically). Only worth it if item 2's cheaper lag-state succeeds partially but
   not fully — i.e. as an escalation, not a first move.

Items are additive/diagnostic, not mutually exclusive — the recommended order (1→2→3,
skipping 4 unless 2 partially succeeds) is designed so that each null result narrows
down *which* of "training emphasis," "dynamics memory," or "decoder locality" is the
true bottleneck, rather than guessing at the most expensive fix first.
