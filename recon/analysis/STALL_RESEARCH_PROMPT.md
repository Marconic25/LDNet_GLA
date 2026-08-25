# Handoff prompt — flap-separation ("D-RES") literature research

Paste everything below the line into a research-capable agent (web search + fetch) to
continue this thread. Full findings from the first pass are in
`recon/analysis/STALL_LITERATURE_NOTES.md` — read that before starting so you don't
re-fetch what's already there.

---

You are doing literature research for an aeroelastic ROM project. Produce a cited,
fact-checked report. Verify every quantitative claim against the actual source (fetch
the paper page, not just the abstract snippet) before including it.

## Context (target the search to this)

We train an LDNet (latent ODE `NNdyn` + coordinate-decoder `NNrec`) on 2-D URANS fields
around a two-element NACA airfoil-with-flap (main element + a physically separate flap
element, ≈0.03-chord gap between them) under gust and flap-deflection excitation.
Champion model: mean-split (store time-mean field, decoder learns only the
fluctuation) + CORAL decoder (shift-modulated SIREN, ω₀=10) + 1-D latent + depth L6.

A residual dubbed "D-RES" survived five independent fix attempts that all touched
either the whole-field decoder or the whole-trajectory dynamics *globally* (decoder
depth, near-wall sampling reweight, optimizer budget, multiple-shooting trajectory
segmentation, rate-only/Neural-CDE-style dynamics conditioning) — all five lost or were
net-negative on the actual target metric.

A local diagnostic (`recon/analysis/decomp_stall.py`) then showed D-RES is not a
diffuse smooth-fit problem at all: at the test case's gust peak (t≈0.584s), a band of
domain points near the flap shows a real, brief (~0.3s), spatially localized burst of
REVERSED streamwise flow relative to each point's own quiescent baseline (FOM peak
23.3% of near-flap nodes reversed vs 0% at rest). The champion ROM almost entirely
misses it (ROM's own reversed-flow fraction at the same instant is only 2.1%) — not a
magnitude error but a SIGN error: at the worst points FOM shows strongly negative
(reversed) velocity while ROM predicts strongly positive (attached-like) velocity, a
sign flip at 21.7% of near-flap points at the event's peak.

## Already established — do NOT re-derive

- **Geometry says flap-gap/cove, not classical leading-edge dynamic stall.** The
  airfoil is a discrete two-element geometry with a real ≈0.03-chord gap between main
  element and flap (`recon/analysis/MEANSPLIT_NOTES.md` line 148); the reversal
  localizes at/downstream of the flap's suction side and trailing edge, i.e. at the
  gap/flap location, not a leading-edge suction peak. Flap-gap/cove separation
  literature is the better-matched family (`STALL_LITERATURE_NOTES.md` §0-1) — the
  most directly relevant hit found so far (arc.aiaa.org/doi/10.2514/1.C032026, "Study
  of Gap Physics of Airfoils with Unsteady Flaps") reports response lag scaling
  linearly with gap size specifically for *unsteady flap deflection*, but its full
  text was never obtained (403) — **the highest-value single fetch to chase next is
  this paper's actual PDF/full text**, plus its companion "The physics of modeling
  unsteady flaps with gaps" (ScienceDirect S0889974612002332), both blocked by
  paywalls in the first pass. Try institutional access, Google Scholar cached
  versions, or author personal pages.
- **Semi-empirical dynamic-stall models share one structural idea**: a scalar
  "attachment state" X(t) with its own first-order lag ODE, driven by the forcing
  *and its rate* (Goman-Khrabrov: `τ₁Ẋ + X = X₀(α − τ₂α̇)`; Leishman-Beddoes and ONERA
  are structurally similar, more elaborate versions). This is the best-matched
  structural analogue to "a separation-lag state" found (`STALL_LITERATURE_NOTES.md`
  §2) — but the Goman-Khrabrov equations quoted there came from search-engine-
  aggregated secondary sources, not a verified primary-source fetch (the PDF at
  arXiv:2110.08516 failed to parse as text). **Do not re-derive the equation form —
  it's very likely right (it's near-universally quoted the same way) — but if this
  lever is actually implemented, get a clean primary-source read of the original
  Goman & Khrabrov 1994 paper or a textbook (e.g. Leishman's *Principles of
  Helicopter Aerodynamics*) first.**
- **A working example of the (z, exogenous-signal)→field structure applied to stall
  specifically exists**: AIAA SciTech 2026-0293 uses a conditional autoencoder
  compressing surface-pressure data to a latent space conditioned on angle of attack,
  across pre-stall/stall/post-stall regimes. Full text not obtained (found via search
  only) — chase this one down if pursuing the separation-lag-state lever, it's the
  closest published analogue to this project's own architecture applied to exactly
  this transition.
- **The loss-weighting hook is confirmed open and unused**: `train_fields.py`
  ~line 432-433 is a plain unweighted `tf.reduce_mean(tf.square(pred-target))` — no
  per-point/per-region weight exists. This is distinct from the already-failed
  *sampling* reweight (which changes which points are seen under a fixed budget);
  this changes how much each already-sampled point's residual counts. Curriculum-PINN
  literature (region-adaptive reweighting by local residual magnitude) is a direct,
  cheap match (`STALL_LITERATURE_NOTES.md` §5) — ranked #1 recommendation, cheapest to
  test, should be tried before anything below.
- **No aero-specific regime-switching/mixture-of-experts latent-ODE paper was found.**
  The two closest matches are in different domains entirely: arXiv:2508.21249
  (spatially-gated mixture of 3 different surrogate *architectures* — DoMINO,
  X-MeshGraphNet, FigConvNet — for steady external-car aerodynamics, real quantitative
  win over best-single-expert and ensemble-average, but never fetched past abstract
  level) and arXiv:2510.09594 (MODE: soft-gated mixture of ODE experts for regime-
  switching dynamics, but tested only on a biological oscillator/cell-cycle data, no
  aerodynamics, no exogenous forcing). If pursuing this lever, get full text of both.
- **A calibration number exists for "how much worse is separated flow, normally"**:
  arXiv:2109.02183 (full-text verified) reports separated-flow L1 error ≈2.7× worse
  than attached-flow, and shock-induced-separation ≈4.3× worse than shock-free, for a
  DL compressible-airfoil-flow surrogate. This project's own miss (ROM reversed-flow
  fraction 2.1% vs FOM 23.3%, i.e. missing the event almost entirely) reads as
  *qualitatively* worse than this reference's *quantitative* 2.5-4× inflation — a
  sign-flip problem, not a magnitude-inflation problem. Useful for framing expectations
  but do not re-search for this number, it's confirmed.
- **A static-field precedent for "discontinuity without knowing its location in
  advance" was fetched in full**: arXiv:2408.00771 ("2D Neural Fields with Learned
  Discontinuities") learns a per-mesh-edge jump-magnitude variable jointly with the
  field by ordinary gradient descent, no pre-specified discontinuity curve needed;
  quantitative gains vs. a continuous baseline (InstantNGP) were confirmed
  (+5.5-11dB PSNR, 3.5× better discontinuity localization). Confirmed limitation: only
  validated on *static* images, no time-varying/transient discontinuity experiment —
  porting this idea here means making the jump-magnitude a function of (z, exogenous
  signals) rather than a free per-edge scalar. Do not re-fetch this paper; if pursuing
  a local/gated decoder lever, the next useful search is for any *follow-up* to this
  specific paper that adds a temporal/conditional jump variable (none found in the
  first pass — may not exist yet).

## Questions (the gaps — in priority order)

1. Fetch the full text of the two unsteady-flap-gap-physics papers named above
   (arc.aiaa.org/doi/10.2514/1.C032026 and ScienceDirect S0889974612002332) — both
   403'd in the first pass. These are the single most load-bearing unverified claims
   in the whole review (item 3's family verdict rests partly on them).
2. Fetch AIAA SciTech 2026-0293 full text (conditional-autoencoder stall model) and
   the AIP Physics of Fluids 2025 Leishman-Beddoes/neural hybrid paper
   (pubs.aip.org/aip/pof/article/37/4/045106) — both 403'd. If either gives concrete
   architecture/loss details (not just "we integrate LB structure"), that would
   directly inform how to build the separation-lag-state input (ranked
   recommendation #2 in the literature notes).
3. Has anyone applied mixture-of-experts / regime-switching dynamics specifically to
   an aerodynamic or fluid-flow latent-ODE (not steady-state field gating, not
   biological/synthetic ODEs)? First pass found neither half combined for this
   domain — confirm this gap is real (a thorough negative search still has value) or
   find the paper that closes it.
4. Is there a temporal/conditional extension of the "learned discontinuity" neural-
   field idea (arXiv:2408.00771's family) — i.e. one where the jump/discontinuity
   strength is itself a function of time or of an external conditioning signal,
   rather than a fixed per-element learned scalar? This is the missing piece to port
   that mechanism to a transient event.
5. Quantitative calibration specifically from the AirfRANS benchmark (or an
   equivalent standard aero-surrogate benchmark) broken out by attached vs. separated/
   near-stall subset — the first pass could not find a leaderboard or paper that
   reports this split explicitly (only the compressible-flow paper in item 25 of the
   notes gave a clean attached-vs-separated number, and that's a different benchmark).
6. Does the flap-gap-vorticity mechanism (item 1 above) have anything to say about
   why the event is specifically tied to the *combination* of gust and flap command
   (not either alone) — this is a previously-unexplained symptom from an earlier
   research pass (`LATENTODE_LITERATURE_NOTES.md` §8) that a gap-vorticity-lag
   mechanism might explain (fast flap deflection during a gust exciting the gap
   response) but this connection was speculative in the first pass, not verified
   against a source that actually studies combined gust+flap forcing.

## Deliverable

A synthesis ranked by (expected impact on the D-RES sign-flip/localization problem,
specifically — not generic accuracy) × (implementation cost in `recon/train_fields.py`:
a loss-weighting change or a new scalar signal are CHEAP; a new state variable with its
own small sub-network is MODERATE; a new decoder family, gating mechanism, or anything
requiring new graph/mesh structure is MAJOR). For each recommendation: the citing
source(s) with the specific quantitative evidence, a falsifiable predicted outcome, and
what to measure in an ablation using `recon/analysis/decomp_stall.py`'s existing
reversed-flow-fraction and sign-flip-rate metrics (already-built instrumentation — do
not propose new metrics without checking this script first). Flag every claim you
could not verify against its primary source.
