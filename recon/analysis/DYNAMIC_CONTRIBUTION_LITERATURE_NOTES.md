# Dynamic-contribution-as-a-problem-class literature findings — research run 2026-08-26

Scope: per task instructions, this review deliberately does NOT re-tread flap/stall-
specific literature (`STALL_LITERATURE_NOTES.md` already covers that ground well and
is read/cross-referenced throughout below). Instead it searches for how neural networks
represent the DYNAMIC/time-varying contribution of a field as a problem class distinct
from static/mean reconstruction, across four angles: (1) DMD/Koopman-operator networks
and a "residual/high-frequency mode" concept, (2) multi-scale/multi-resolution temporal
architectures, (3) GNN/message-passing decoders for unsteady CFD specifically (flagged
in the prior review as never actually tried here), and (4) any other genuinely
different way of modeling a dynamic residual on top of a mean/steady baseline
(generative/stochastic, operator learning, field-level regime separation).

Baseline context this review was written against (from `MEANSPLIT_NOTES.md`'s final
"CUMULATIVE SUMMARY UPDATE" section, read in full before starting): **TEN** independent
levers have been tried against the champion (`coral_o10_s0`: mean-split + CORAL
shift-modulated SIREN decoder, ω0=10, d_s=1, uniform sampling) and lost —
(1) decoder depth, (2) near-wall sampling, (3) optimizer/BFGS budget, (4) multiple
shooting, (5) Neural-CDE rate-only dynamics conditioning, (6) residual-curriculum loss
weighting, (7) added signal-rate input channels, (8) static geometric flap-proximity
loss weighting, (9) a Goman-Khrabrov separation-lag scalar state, (10) a spatially/
regime-gated additive local decoder head. Every entry below is explicitly checked
against this list — new ideas are flagged as genuinely distinct or as a re-tread/
escalation of a specific numbered item above.

Verification method, same convention as `STALL_LITERATURE_NOTES.md`: `[FULL-TEXT]` = I
fetched and read real paper content (arXiv HTML/PDF or publisher HTML with quotable
text, mechanism and/or numbers extracted from the actual document, not a search
snippet). `[ABSTRACT-ONLY]` = a fetch succeeded but only returned abstract-level
content, or a full-text fetch succeeded but did not yield the specific quantitative
numbers I was looking for (mechanism confirmed, numbers not). `[SEARCH-ONLY]` = no
direct fetch succeeded (403/paywall/redirect-to-login/bot-check) or I relied on
search-engine snippets only — treat as an unverified pointer, not a confirmed claim.

---

## 1. DMD / Koopman-operator neural networks — is there a "residual/high-frequency
   mode" concept? (task item 1)

1. `[FULL-TEXT]` **Multi-resolution DMD (mrDMD)** (Kutz, Fu, Brunton, arXiv:1506.00564,
   SIAM MMS 2016) is the single most direct structural match found in this entire
   review to the task's own framing. Its mechanism, confirmed from the fetched paper
   content: apply standard DMD to the full snapshot sequence, reconstruct from the
   *dominant* (slowest/near-zero-frequency) modes only, **subtract** that
   reconstruction from the data to form a residual, then recursively re-apply DMD to
   that residual at a finer time window — repeating until a desired depth. "Thus at
   each level, the slow dynamics are separated from the fast dynamics, giving a
   recursive scheme for multi-scale, equation-free modeling." The paper's own worked
   examples are exactly this project's shape of problem: separating a slowly-varying
   background from a **short-lived, spatially-localized, faster-timescale foreground**
   (video foreground/background separation; the El Niño signal against the ocean
   temperature background). Modes from the fast/foreground levels are reported to be
   more spatially localized than the slow/background modes — i.e., "faster timescale
   → more localized in space" is an actual finding of this method, not an assumption.
   **Important caveat, explicitly confirmed (not inferred):** the paper's own
   treatment is for **autonomous/unforced systems** — mrDMD's recursive procedure
   decomposes a system evolving under its own internal dynamics, with no built-in
   mechanism for continuous exogenous forcing (this project's gust velocity and flap
   deflection commands). Porting the idea requires either (a) running mrDMD purely as
   an offline diagnostic on FOM snapshot data (forcing held fixed per-trajectory,
   analysis performed after the fact) rather than folding it into the online forced
   model, or (b) combining it with DMD-with-control (next item).
2. `[SEARCH-ONLY]` **Dynamic Mode Decomposition with Control (DMDc)** (Proctor,
   Brunton, Kutz, SIAM J. Appl. Dyn. Syst. 2016, arXiv:1409.6358) is the standard
   extension of DMD to systems with a known exogenous/control input, explicitly built
   to "disambiguate between the underlying dynamics and the effects of actuation" — the
   exact gap flagged in item 1's caveat. Not fetched past search-engine summary this
   session, so the precise mechanism (SVD-based dimension reduction of the augmented
   [state; input] snapshot pairs) is reported at the level the literature nearly
   universally describes it, not independently re-derived from the primary text here.
   **Relevance:** mrDMD + DMDc together are the two halves of a data-driven diagnostic
   that could be run **entirely offline, on existing FOM dumps, with zero training** —
   the cheapest and most different-in-kind item in this whole document (see §5
   ranked recommendations, item A).
3. `[FULL-TEXT]` **NDKoop** ("End-to-End Neural Decomposition with Koopman Operators
   for Time-Series Forecasting," arXiv:2608.08788) is a genuine neural realization of
   "give the smooth and the fast/oscillatory components their own operators." A
   learnable signal-decomposition module splits a time series into a trend
   `x_trend(t)` and a residual/seasonal `x_season(t)`; the trend is propagated by a
   single **frequency-independent, time-invariant, shared Koopman operator K**; the
   seasonal part is first decomposed further (via neural variational mode
   decomposition) into N frequency-specific sub-signals, **each with its own
   frequency-dependent Koopman operator K_n** (`h_{k+1,n} = K_n h_{k,n}`); the two
   branches' predictions are summed at the end. Reported gain: MSE 0.121 vs. Koopa's
   0.130 and DLinear's 0.158 on the ECL 48-step benchmark (7–31% improvement over
   single-operator baselines). **Caveat:** this is 1-D scalar/multivariate time-series
   forecasting (electricity load, not a spatial PDE field), and the "residual" being
   modeled is ordinary periodic seasonality, not a rare, localized, sign-changing
   transient event — the architecture pattern (two branches, two operators, summed) is
   the transferable idea, not the specific numbers.
4. `[FULL-TEXT]` **Residual DMD (ResDMD)** (Colbrook & Townsend, JFM 2024) is
   confirmed to be a **false friend for this project's naming**: despite the name
   "residual," it does NOT separate a transient/localized dynamical mode from smooth
   background dynamics. It is a numerical-analysis tool for computing verified
   Koopman spectra with error control — its "residual" is the eigenvalue-equation
   residual `‖(A − λ)v‖` used to reject spurious/spectrally-polluted eigenvalues from
   a finite-dimensional EDMD discretization, unrelated to a "residual field" or
   "residual dynamics" in this project's sense. Flagging explicitly so this name is
   not mistakenly cited as support for a residual-mode hypothesis.
5. Cross-reference to `STALL_LITERATURE_NOTES.md` §3 (already covered, not
   re-verified here): **MODE** (Mixtures of Dynamical Experts, arXiv:2510.09594) is
   the closest match found anywhere (either review) to "give the latent-ODE itself
   multiple regime-specific vector fields with a learned gate" — still, per that
   document's own reading, never applied to an aerodynamic/flow-separation setting.

**Reading:** yes, a "residual/high-frequency mode" concept genuinely exists in this
literature family, and mrDMD is a clean, well-established, directly-quotable precedent
for exactly the framing the task asked about (task item 1's "residual mode" question).
Its own field of use (foreground/background *video* separation, background = slow
mean-like structure, foreground = brief, spatially-compact, faster-moving content) is
about as close an analogy to "smooth mean-flow dynamics + a localized, transient
separation burst" as this whole review found. The catch is that it is a **linear,
non-neural, offline SVD-based decomposition of a snapshot matrix**, not a component of
the online forced-dynamics network this project trains — its most honest and cheapest
use here is as a **diagnostic**, not a drop-in architecture change (see §5 item A).
NDKoop shows the "two operators, one per timescale/frequency band, summed" pattern
does work as an actual trainable neural mechanism, but only demonstrated on scalar
time series, and its "residual" (periodic seasonality) is a much gentler regime than a
rare 0.3s-wide sign-flipping burst.

---

## 2. Multi-scale / multi-resolution temporal architectures (task item 2)

6. `[FULL-TEXT, mechanism confirmed; quantitative tables not reached in the fetched
   excerpt]` **LoGlo-FNO** ("Efficient Learning of Local and Global Features in
   Fourier Neural Operators," arXiv:2504.04260) couples a **global Fourier branch**
   (long-range/smooth structure) to a **local patchwise branch** that retains all
   Fourier modes *within* each spatial patch, plus a dedicated "high-frequency
   propagation module" and a "frequency-sensitive loss" that explicitly up-weights
   high-frequency error components during training. The paper positions this
   explicitly against FNO's known spectral bias (global Fourier operators
   systematically under-resolve sharp/localized/discontinuous features — the same
   qualitative failure mode this project's own Fourier-features arm (`ff10_40`, in
   `MEANSPLIT_NOTES.md`) already confirmed empirically: higher spatial frequency
   monotonically hurt rather than helped, ruling OUT spatial spectral bias as this
   project's specific cause). LoGlo-FNO's target use case (localized/transient/
   discontinuous features in time-varying PDEs) is a structural description of this
   project's exact symptom, but the fetched excerpt did not surface the paper's actual
   comparison numbers against plain FNO, so its practical payoff size is unconfirmed.
7. `[SEARCH-ONLY]` A general **multi-timescale RNN** literature (Fast-Slow RNN,
   arXiv:1705.08639; "Multiple-Timescale Neural Networks," PMC8702558) establishes
   that giving different sub-networks different effective time constants, with slow
   units stabilizing/gating the fast units, is a well-studied mechanism outside
   aerodynamics (originally neuroscience/sequence-modeling motivated). No aerodynamic
   or CFD-surrogate application was found in this pass; this is background evidence
   the general mechanism is sound, not a domain-specific result.
8. `[SEARCH-ONLY]` A composite **CNN-GRU-PINN (CGPINN)** architecture splits spatial
   feature extraction (CNN) from temporal feature extraction (GRU) explicitly, with a
   PINN constraint layer on top, for unsteady flow prediction — a coarse-grained
   spatial/temporal separation of labor, but not a residual/multi-frequency
   decomposition in the mrDMD/NDKoop sense; not independently verified past a search
   snippet.
9. `[SEARCH-ONLY]` A **wavelet-enhanced attention fusion network (WAFNet)** combines
   wavelet + convolutional encoding with an FNO-based decoder for airfoil flow-field
   prediction. Search-snippet level only; appears to target steady-state pressure
   fields, not confirmed unsteady/transient.
10. mrDMD (item 1) is the cleanest multi-resolution-*temporal* precedent found and is
    cross-listed here: its recursive slow-then-fast decomposition is literally a
    multi-resolution analysis in time, just not neural.

**Reading:** the multi-scale/multi-resolution family is real and reasonably populated,
but every concretely-verified item that targets PDE/flow fields specifically
(LoGlo-FNO) does so at the level of *spatial* frequency (global-vs-local Fourier
content), which this project's own Fourier-features arm already ruled OUT as the
cause of its residual — a caveat worth stating plainly: don't over-read LoGlo-FNO as
support without checking that its actual lever (frequency-sensitive *loss* + a local
patch branch) isn't just a fancier version of the already-failed FF/near-wall-sampling
family. The temporal-multi-resolution items (mrDMD, NDKoop, multi-timescale RNNs) are
better matched in *kind* to "smooth slow dynamics + a rare fast transient" but none
have been demonstrated on a spatial CFD field with exogenous forcing.

---

## 3. GNN / message-passing decoders for unsteady CFD (task item 3 — flagged as
   never actually tried, only costed, in the prior review)

11. `[FULL-TEXT]` **"Read, Write, Relax: Why Neural PDE Surrogates Need Both Global
    and Local Processing"** (arXiv:2608.21677) is the best-evidenced, most directly
    relevant find in this entire document for task item 3, and it answers the
    question with an actual mechanism plus real numbers on an industrial CFD
    benchmark (branched-pipe flow, 60k cells). Its central claim, confirmed from the
    fetched text: **global and local processing fail from opposite ends of the error
    spectrum.** Message-passing GNNs (MeshGraphNet-style) resolve fine/local
    structure well but "information travels only one edge per layer... the graph
    diameter far exceeds the depth of a typical [message-passing] processor, so the
    domain-scale content is out of reach" — in spectral terms, local processing gains
    early on fine scales then **plateaus across the entire spectrum** because it never
    accumulates long-range coupling. Global latent-attention models (their own "RWP"
    baseline, and Transolver-style physics-attention) do the opposite: they compress
    through a bottleneck of `L≪N` latent tokens, acting as **"a spatial low-pass
    filter"** — "the low band contracts by more than an order of magnitude... while
    the mid- and high-band errors stall early." **This is precisely this project's own
    CORAL ω0 sweep finding, independently reproduced in a different codebase on a
    different problem**: `MEANSPLIT_NOTES.md`'s CORAL ω0=30/60 result ("OVERSHOOT,
    worse than plain-ms, like high FF scale") is the signature of a global decoder
    whose frequency content was pushed too high; RWR's spectral argument gives a
    structural reason *why* a purely global decoder has a hard mid/high-frequency
    ceiling regardless of which frequency you tune it to, rather than "ω0=10 happens
    to be right and ω0=30 happens to be wrong."
    The paper's proposed fix, **Read-Write-Relax (RWR)**, interleaves the two: a
    "read" (restriction of the mesh onto an `L`-dimensional latent, i.e. the global
    step) and "write" (prolongation back to the mesh) bracket a small number `m` of
    **local message-passing relaxation sweeps** directly on the mesh — explicitly
    described as analogous to a multigrid V-cycle, with `r=0` recovering plain
    message-passing and `m₁=m₂=0` recovering the plain global/latent-attention model.
    Quantitative result (Table 2, branched-pipe benchmark): wall shear stress
    `τ_w,x` error 0.131 (RWR) vs. 0.157 (global-only RWP, 1.20× worse) vs. 0.246
    (GeoTransolver, 1.88× worse); vorticity `ω_y` error 0.117 (RWR) vs. 0.138
    (RWP, 1.19×) vs. 0.212 (GeoTransolver, 1.82×) — RWR's biggest margins are
    specifically on **near-wall gradient/vorticity quantities**, the same *class* of
    quantity (near-wall velocity-direction sign) this project's own residual lives in,
    though not the same benchmark and not a sign-changing/reversal feature
    specifically (not independently confirmed for that exact failure mode).
12. `[FULL-TEXT]` **Courant** ("A State-Adaptive Perceiver-Based Neural Surrogate with
    Local Support and Interpretable Field Decomposition," arXiv:2605.25115) is a
    third, structurally distinct decoder family — neither the project's own global
    coordinate-conditioned SIREN, nor GNN message-passing, nor a graph at all. It
    encodes the domain onto a moderate number of spatial "anchor" latent tokens (via
    Perceiver-style cross-attention with shared random-Fourier-feature coordinate
    embeddings), evolves those tokens through a latent Neural ODE
    (`dZ/dτ = SelfAttn(Z(τ))`), and decodes with a **partition-of-unity attention**
    over the anchors: `û(x) = u_0 + Σ_k δ_k(x)`, where each `δ_k(x)` is a per-anchor
    basis function whose spatial support is *learned* (softmax attention weight,
    naturally decaying away from its anchor) rather than fixed or hand-placed. On a
    genuinely transient, spatially-localized benchmark — **2D cylinder-obstructed
    (Kármán vortex-shedding) flow, Re≈90** — the paper reports that distinct latent
    anchors **spontaneously specialize**: "some latent anchors track transient
    features (dynamic modes) while others remain anchored to the geometry (stationary
    modes)," with the latent power spectrum sharply peaked at the true shedding
    frequency. This is an *emergent*, architecture-driven separation of a transient
    dynamical event from the static/geometric background — conceptually close to what
    this project's D-RES investigation has been trying to engineer by hand (sep-state
    scalar, local/gated decoder), but arising from the decoder's own attention
    structure rather than a hand-specified lag-ODE or a hand-placed spatial gate.
    **Quantitative results are mixed, not a clean win:** on that same cylinder case
    Courant's NMAE (1.7×10⁻¹) is *worse* than Transolver's physics-attention baseline
    (1.3×10⁻¹), though it beats UPT (2.8×10⁻¹) and, on two other (steady) industrial
    benchmarks, beats both Transolver and a mesh-GNN baseline (MGN) by a wide margin
    (e.g. centrifugal pump: Courant 1.7×10⁻¹ vs. Transolver 2.9×10⁻¹, MGN
    out-of-memory). No comparison specifically targets a sign-changing/flow-reversal
    metric. The paper's own stated limitation: the interpretability claims are
    "primarily qualitative and visual," not backed by a formal decomposition metric.
13. `[FULL-TEXT, no separation-specific breakdown obtained]` A **mesh-adaptive
    hypergraph neural network for oscillating/rotating structures** (arXiv:2503.22252)
    confirms message-passing-only locality is workable for a moving-geometry problem
    structurally analogous to this project's moving flap (co-rotating + static
    subdomains with mesh adaptation at the interface), reporting R²>0.95 for
    velocity/pressure in stable regimes but "accumulated phase and amplitude error"
    for low-amplitude oscillations. No baseline comparison against a coordinate-MLP/
    global decoder, and no explicit separation/reversal breakdown, was obtainable.
14. `[FULL-TEXT, aggregate numbers only]` The original **PointNet-based point-cloud
    flow-field framework** (Kashefi, Rempe, Guibas, arXiv:2010.09469, Physics of
    Fluids 2021) reports that its network "successfully predicts the flow separation
    phenomenon" qualitatively and that maximum pointwise errors occur "on the edges of
    objects, where the no-slip condition has to be applied" — consistent with this
    project's own finding that the hardest region is right at the body surface, but
    with only aggregate L2 errors reported (u: 4.50e-2, v: 3.71e-2, p: 2.72e-2), no
    region-decomposed or separation-specific numbers, and explicitly **no comparison
    against a grid-based CNN or coordinate-MLP baseline** — the paper argues against
    interpolation-based CNN methods conceptually but does not benchmark against one.
15. `[SEARCH-ONLY]` **AMGNET** (multi-scale GNN, Connection Science 2022) reports
    "significantly lower prediction errors than [a plain] GCN baseline" on airfoil and
    cylinder flow via message passing at multiple mesh-coarsening scales — a
    multi-scale-in-*space* GNN, not decomposed for separation specifically, and full
    text was blocked (403) this session.
16. `[SEARCH-ONLY]` **Transolver**'s physics-attention (softly assigning mesh points to
    a fixed number of learned "slice" tokens, cross- then self- then cross-attention)
    is confirmed to exist as a real, competitively-benchmarked third family (used as
    Courant's own strongest baseline above), but was not independently fetched this
    session — noted because Courant's comparison establishes it is currently the
    strongest of the non-GNN, non-this-project's-decoder alternatives on multiple
    industrial CFD benchmarks.

**Reading, directly addressing the task's item 3 question:** yes — real, if
imperfect, quantitative evidence now exists (item 11, full-text, real CFD numbers;
item 12, full-text, an actual transient/vortex-shedding benchmark) that a decoder with
*some* form of locality outperforms a purely global one specifically on near-wall
gradient/vorticity-class quantities, and that a locally-supported attention decoder
can spontaneously discover a transient dynamical feature as a separate "mode" from
the static geometry. **Neither result is the same mechanism as this project's already-
tried local/gated decoder (lever 10)**: lever 10 was a *fixed, hand-placed* spatial
gate (flap-proximity distance) multiplying a *second, independent, additively-summed*
SIREN head; RWR interleaves genuine local *message-passing* (mesh-graph-based, needs
real adjacency structure, mixes into the SAME representation the global step uses,
multigrid-cycle-style) while Courant's locality is *learned attention support over a
moderate number of free-floating spatial anchors* (no graph at all, no fixed
proximity heuristic, no single hard-wired "near the flap" location — many anchors can
specialize anywhere in the domain). Both are structurally closer to "give the decoder
an actual mechanism for locality" than lever 10's "bolt a second global-ish head onto
the existing one and gate it by a hand-picked distance," which is worth stating
plainly as the reason lever 10's null result does not fully close off this whole
family — it tested one specific (and, per RWR's spectral argument, possibly the
weakest) way of adding locality, not the concept itself.

---

## 4. Other genuinely different dynamic-residual mechanisms (task item 4)

17. `[FULL-TEXT]` **CorrDiff** ("Residual Corrective Diffusion Modeling for Km-scale
    Atmospheric Downscaling," Mardani et al., arXiv:2309.15214) is the clearest
    concrete instance found of a **generative/stochastic residual-correction**
    paradigm, and it is explicitly framed by its own authors as "inspired by
    Reynolds decomposition in fluid dynamics" — i.e., the same mean/fluctuation split
    this project's `--mean-split` already performs, but taken one level further.
    Mechanism, confirmed from the fetched text: a deterministic UNet regression
    predicts the conditional mean `μ = E[x|y]` (MSE loss) capturing "many of the
    physics... some of which are deterministic"; a **diffusion model then learns
    `p(r|y)` on the zero-mean residual `r = x − μ`** left over. The residual
    formulation is reported to make the generative model's job strictly easier
    ("learning the distribution p(r) can be much easier than learning p(x)... allowing
    for smaller noise levels") — again structurally identical in spirit to this
    project's own confirmed win that mean-splitting reduces what the network has to
    learn. Quantitative results: CorrDiff's CRPS beats plain-UNet's MAE-equivalent
    skill "in 205/205 validation times," and per-scale spectral analysis shows the
    diffusion step specifically **"restores variance missing from UNet... especially
    at all length scales" for radar reflectivity** — i.e., the regression step
    systematically under-represents small-scale/sharp structure (exactly this
    project's smooth-mean-fit-misses-the-transient symptom), and the generative
    residual step is what recovers it.
    **The load-bearing caveat, and the reason this is ranked low despite being the
    most novel paradigm found:** CorrDiff's residual is genuinely **stochastic/
    aleatoric** — it corrects for *unresolved, non-repeatable* small-scale weather
    variability across an ensemble of physically-plausible realizations consistent
    with the same coarse conditioning. This project's target is **fully deterministic
    given the inputs**: the same (gust, flap) trajectory always produces the exact
    same separation event in the FOM. There is no ensemble-of-realizations structure
    to sample over here — a diffusion model trained on this project's residual would
    either (a) learn a degenerate, near-delta-function conditional distribution
    (in which case it reduces to an expensively-trained deterministic corrector,
    conceptually not different in kind from the already-tried local/gated decoder,
    just parameterized as a denoiser) or (b) genuinely need some source of
    randomness to justify the generative framing, which the problem doesn't have.
    Flagging this exactly as the task instructions anticipated ("pure generative
    modeling may not transfer directly... note anything relevant").
18. `[FULL-TEXT]` **FluidFlow** (flow-matching generative model, arXiv:2604.08586) is
    confirmed to be steady-state only — the authors themselves state "the present
    work has focused on steady aerodynamic quantities, so extending FluidFlow to
    non-stationary three dimensional flows would be a natural next step." No
    transient/separation evaluation exists in this paper. Included only to record
    that it was checked and ruled out, not as a positive lever.
19. `[ABSTRACT-ONLY]` A **Fourier Neural Operator for airfoil dynamic stall** (Physics
    of Fluids 35(11):115126, 2023) predicts the full unsteady flow field
    (`vx, vy, p`, vorticity) through the dynamic-stall cycle via an iterative
    next-step operator, reported to be fast and accurate in aggregate, but full text
    was blocked (403) and no separated-vs-attached error breakdown or comparison
    against a coordinate-MLP/GNN baseline could be obtained. This is the direct
    operator-learning analogue the task asked about (item 4's "FNO/DeepONet applied
    specifically to unsteady/separated flow") but its evidentiary value here is weak
    — it demonstrates FNO *can* run on a dynamic-stall dataset at all, not that it
    handles the localized reversal better than a coordinate-conditioned decoder.
20. `[SEARCH-ONLY]` Multiple **PINN papers explicitly targeting field-level attached/
    separated flow** were found (AIAA 2022-0187 "Physics-Informed Neural Networks for
    Flow Around Airfoil," reported to predict "the existence of the stagnation point,
    an attached or separated flow, boundary layers... high- and low-pressure
    regions"; the already-flagged-in-`STALL_LITERATURE_NOTES.md` Neural Computing &
    Applications 2024 paper, still blocked by a Springer login wall this session
    too). These are genuinely "field-level regime separation" attempts (not a scalar
    indicator, which is what this project's already-tried sep-state lever was) but
    none could be independently verified past search-engine summaries, and none is
    confirmed to condition on a rate-driven exogenous forcing signal the way this
    project's flap/gust setup does — the PINN framing typically uses the physics
    residual (a term this project's pure data-fit setup does not have) as the
    mechanism that lets the network discover the attached/separated transition,
    which does not directly transfer without a governing-equation residual term.

**Reading:** the generative/diffusion family (CorrDiff) is real, well-evidenced, and
conceptually elegant (mean/residual decomposition mirroring this project's own
mean-split, taken one level deeper) — but its actual value proposition (recovering
genuinely *stochastic*, ensemble-scale variability) is a structural mismatch with a
deterministic single-trajectory reconstruction target, which should be stated as a
hard caveat rather than glossed over. Operator-learning (FNO/DeepONet) applied to
separated flow exists but the evidence obtained this session is weak (blocked
full texts); no quantitative claim from that family should be relied on here without
further chasing. Field-level (not scalar) attached/separated PINN work exists and is
conceptually the right shape, but its mechanism (physics-residual-driven regime
discovery) does not obviously port to this project's pure-data-fit, no-PDE-residual
training loop without adding a governing-equation term this project has never used.

---

## 5. Not resolved / would need more digging

- DMDc's precise mechanism was not independently re-derived from its primary source
  this session (item 2, §1) — if the mrDMD/DMDc offline-diagnostic recommendation
  (§6 item A) is pursued, fetching the actual SIAM paper (not just search-engine
  summaries) would be the first step.
- LoGlo-FNO's quantitative tables (item 6, §2) were not reached in the fetched
  excerpt — the mechanism (global Fourier + local patch branch + frequency-sensitive
  loss) is confirmed, but whether it actually beats plain FNO by a meaningful margin,
  and on what kind of feature, is not verified.
- The FNO-dynamic-stall paper (item 19, §4) and both field-level attached/separated
  PINN papers (item 20, §4) remain blocked by paywalls/login walls in this session as
  in the prior one — institutional access would be needed to firm up item 4's
  operator-learning and PINN-regime-separation angles beyond what is reported here.
- No paper was found that runs mrDMD/DMDc-style modal analysis, or any of this
  section's neural analogues (NDKoop, LoGlo-FNO, RWR, Courant), on a genuinely
  moving-geometry, externally-forced (gust + actuator), two-element airfoil
  configuration — every positive result in §3 and §4 above is either on a
  fixed-geometry benchmark (Kármán shedding cylinder, branched pipe) or a different
  physical domain (weather downscaling) than this project's own case. This is a
  genuine gap in the literature, not just an unfetched paper — flag explicitly rather
  than paper over it.
- No quantitative comparison of a GNN/message-passing decoder against a coordinate-
  MLP/implicit-neural-representation decoder, specifically isolating a sign-changing/
  flow-reversal feature, was found or verified in either this review or the prior
  stall-literature review — RWR (item 11) is the closest available substitute
  (near-wall gradient/vorticity error, not sign/reversal specifically), and this
  absence should be stated as a persistent, still-open gap rather than implied to be
  resolved by RWR's numbers.

---

## 6. Ranked recommendations (my synthesis, not a source)

Ranked by (how genuinely distinct the mechanism is from all 10 already-tried levers)
× (implementation cost using this project's own confirmed infrastructure) ×
(strength of the literature evidence obtained this session). Each entry states which
of the 10 closed levers it is/isn't a re-tread of, a cost tier, and a falsifiable
predicted outcome.

**A. mrDMD/DMDc-style offline modal diagnostic on the existing FOM dumps** (§1, items
1–2). Cost: **CHEAP** — pure post-hoc analysis code (numpy/scipy SVD, no TensorFlow,
no training, no cluster time), runnable on data this project already has
(`recon/analysis/decomp_stall.py`'s own FOM/ROM dumps or the raw FOM snapshot files).
**Not a re-tread of any of the 10** — none of the ten training-time levers ever asked
"does the D-RES residual actually correspond to a distinct, quantifiable dynamical
mode (its own growth/decay rate, its own dominant frequency) separable via linear
modal analysis from the smooth background modes?" This is a pure diagnostic, exactly
analogous in spirit (and cost) to how `decomp_stall.py` itself was the highest-value
single step in the whole D-RES/stall investigation — it reframed the problem before
any architecture was touched. *Concrete recipe:* run mrDMD (or plain windowed DMD with
DMDc's control-augmented formulation, since the flap/gust forcing is known) on the
FOM `sim_Cc_060` near-flap patch across the separation-event window; check whether a
small number of modes with (i) a *non-trivial oscillation/growth rate* distinct from
the near-zero background modes and (ii) spatial support localized to the same
near-flap band `decomp_stall.py` already isolated, capture most of the event's
variance. *Falsifiable prediction:* if such a mode exists and is low-rank (say,
rank ≤2–3), it gives a concrete, DATA-DERIVED target shape for a "fast branch" —
directly informing whether recommendation B below (or a revisit of the already-tried
sep-state lever with a data-derived rather than hand-assumed ODE form) has any chance.
*Null result:* if the event's variance is spread across many modes with no clean
scale separation from the background (i.e., mrDMD's own core assumption — that slow
and fast dynamics are separable — does not hold for this specific residual), this is
itself a valuable, cheap negative result: it would argue against ANY two-timescale
architectural split (this recommendation's own escalation, B below, AND a revisit of
regime-switching MoE dynamics) before spending real training compute on either.

**B. RWR-style local-global hybrid decoder (interleaved mesh-graph relaxation sweeps
bracketed by the existing global read/write step)** (§3, item 11). Cost: **MAJOR** —
requires building an actual mesh/graph adjacency structure, which per this project's
own confirmed state does not exist anywhere in the codebase today (same cost class as
lever 10). **Distinct from lever 10, not just a repeat at higher cost:** lever 10 was
a *fixed-location, hand-gated, additively-summed second global-ish SIREN head* — it
never actually built a local (graph-neighborhood) operator; RWR's mechanism is real
message-passing (information genuinely restricted to mesh neighbors per sweep),
interleaved into the SAME latent representation the global step reads from and writes
to, multigrid-cycle style, not a parallel/gated correction branch. It is also
independently, quantitatively motivated by this project's OWN CORAL ω0 finding: RWR's
spectral argument (global processing is a low-pass filter that plateaus on mid/high
frequency content regardless of tuning) gives a structural reason for the ω0=30/60
overshoot this project already observed, rather than treating ω0=10 as a lucky
hyperparameter. *Falsifiable prediction:* a small number of local relaxation sweeps
interleaved with the existing CORAL global step should measurably improve near-wall
gradient-type error (this project's own analogue: sign-flip rate / reversed-flow
fraction at the gust peak) beyond what raising or lowering ω0 alone could reach,
because the mechanism adding locality is structurally different (real local mixing,
not a global function tuned to a different frequency). *Null result:* if this ALSO
fails to move the sign-flip metric, it would be considerably stronger evidence than
lever 10's null that decoder locality per se (not just "the one way we tried adding
it") is not the bottleneck — because this is a mechanistically cleaner test of the
locality hypothesis than lever 10 was.

**C. Courant-style local-support partition-of-unity attention decoder** (§3, item
12). Cost: **MAJOR**, and higher than B — a full Perceiver-style multi-anchor
encoder/processor/decoder with random-Fourier-feature anchor embeddings and a latent
Neural ODE processor is substantially more new machinery than either lever 10 or
recommendation B. **Distinct from lever 10:** no fixed spatial-proximity heuristic
anywhere (lever 10's gate was literally "distance to the flap, hand-tau'd"); Courant's
locality is learned attention support over a number of *free-floating* anchors that
can specialize anywhere the training signal rewards it, and its own paper reports
this specialization actually happening (dynamic vs. stationary anchor modes) on a
genuinely transient benchmark. Ranked below B specifically because its own reported
numbers are a mixed bag (loses to Transolver on the one transient benchmark checked,
wins on two steady ones) — the evidence that this exact mechanism helps on a
*transient* feature is real but weaker than RWR's. *Falsifiable prediction:* if
implemented, distinct anchors should be visibly assignable to "near-flap, active only
during the gust-peak window" vs. "everywhere else, all times" without hand-specifying
that split anywhere (the sep-state lever's failure mode was that hand-specifying the
lag-ODE FORM didn't help — Courant's mechanism doesn't require assuming a form).
*Null result:* if no anchor specialization emerges and accuracy doesn't improve, this
would suggest d_s=1's severe capacity limit (already established,
`recon-intrinsic-latent-dim-1`) is choking any decoder-side locality mechanism
regardless of its sophistication, not just the specific two-head design lever 10 used
— motivating a latent-capacity investigation ahead of any further decoder work.

**D. Two-timescale/dual-operator NNdyn, informed by NDKoop's trend+frequency-specific
branch split and mrDMD's slow/fast separation** (§1, items 1, 3). Cost: **MAJOR**, and
explicitly **not cheaper than, and not meaningfully different in kind from,
`STALL_LITERATURE_NOTES.md`'s own already-identified item 4 escalation** (a full
regime-switching/MoE latent-ODE with 2 expert vector fields + a learned gate,
deliberately held back as a last-resort escalation past the sep-state lever). Listed
here mainly to be explicit that NDKoop/mrDMD do not offer a genuinely cheaper or
structurally novel path to this same idea — they corroborate that "two operators
instead of one, split by timescale" is a sound general pattern (item 3's ECL
benchmark numbers), but do not reduce its cost or specifically de-risk its
application to a forced, moving-geometry, sign-changing aerodynamic event. Should
only be pursued after recommendation A's diagnostic gives positive evidence of a
genuinely separable fast mode — running this blind would repeat the sep-state
lever's own lesson (a hand-assumed structural form for the "fast" component did not
help; there is no reason to expect a hand-assumed two-Koopman-operator split would
fare differently without A's evidence first).

**E. CorrDiff-style generative residual correction** (§4, item 17). Cost: **MAJOR**
(a full conditional diffusion model, new training/sampling infrastructure, likely the
most implementation-alien item in this document relative to this project's existing
TensorFlow/Keras optimization-problem-object codebase) **and ranked lowest despite
being the most novel paradigm**, because of the honest paradigm mismatch documented
in §4 item 17: this project's target is deterministic given its inputs, and
CorrDiff's actual mechanism corrects for genuine *aleatoric* ensemble variability that
this problem does not have an equivalent of. Not recommended as a near-term next
step; recorded for completeness per the task's explicit request to note generative
approaches even where transfer is doubtful.

Items A→B/C→D are meant as a priority order, not a mandatory sequence — A is cheap
enough to run regardless of what else is decided; B and C are alternatives to each
other more than a strict sequence (both are decoder-locality mechanisms genuinely
different from lever 10, evidenced by different papers, and could be prioritized by
whichever the user judges more tractable to implement first rather than run serially);
D should wait on A's result; E is not recommended without a specific, separate
motivation for why a generative framing is needed here.
