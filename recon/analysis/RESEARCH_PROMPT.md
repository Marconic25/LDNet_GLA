# Handoff prompt — near-wall reconstruction literature research (v2)

Paste everything below the line into a research-capable agent (web search + fetch).

---

You are doing literature research for an aeroelastic ROM project. Produce a cited,
fact-checked report. Verify every quantitative claim against the actual source (fetch
the paper page, not just the abstract snippet) before including it.

## Context (target the search to this)
We train an LDNet (Regazzoni et al. 2024, Nature Comm. — latent ODE + coordinate-MLP
decoder mapping (latent state z(t), input signals, x, y) → (vx, vy, p)) on 2-D URANS
fields around a NACA airfoil under gust and trailing-edge-flap excitation. Decoder:
tanh MLP, 4–16 hidden layers × 24 wide; inputs are raw (x,y) — no positional
encoding; uniform random subsampling of 1024/11075 mesh nodes per step; MSE loss on
min-max normalized fields; Adam then full-batch BFGS (scipy, dense Hessian).
Documented symptoms: (a) vx error concentrated in boundary layer + near wake
(~15–20% relative), (b) ~90% of that error is a STATIC mean-flow bias present at all
times (sharp BL gradient smeared), (c) global NRMSE dominated by easy far-field/
pressure, (d) deep-decoder training (24 layers) is seed-fragile: identical runs
collapse or thrive depending on RNG seed, BFGS terminates on budget with large
gradient norms.

## Already established (do NOT re-derive; build on it)
- Spectral bias of plain coordinate MLPs explains BL smearing; Fourier-feature input
  encoding is the standard remedy; multiscale σ=[1,5] variant best on sharp aero
  gradients [arXiv:2006.10739; arXiv:2407.19916].
- Wall-distance input + "boundary layer mask" feature (=1 at wall, quadratic decay,
  τ=0.02) gives ~6× drag-error improvement; INFINITY (FF-INR, wall distance input)
  is AirfRANS SOTA with order-of-magnitude better surface-pressure MSE than
  graph/point baselines [arXiv:2505.14704; arXiv:2307.13538].

## Questions (the gaps — in priority order)
1. VERIFY these specific claims by fetching the sources (they were extracted but
   never checked): (i) Berg & Nyström arXiv:1711.06464 hard-Dirichlet ansatz
   û = G + D·NN with learned distance/extension nets; (ii) Sukumar & Srivastava
   arXiv:2104.08426 exact BC via approximate distance functions + transfinite
   interpolation; (iii) RAD/RAR-D residual-based adaptive sampling gains
   arXiv:2207.10289; (iv) MARIO ablation numbers (C_d error 0.794%→4.780%, near-wall
   velocity oscillations) arXiv:2505.14704.
2. Mean+fluctuation decomposition: find works where a neural field/ROM predicts
   FLUCTUATIONS around a stored (or separately-fit) mean field instead of total
   fields, for wall-bounded flows. Does it demonstrably remove static near-wall
   bias? (Any POD-hybrid where mean+first modes are classical and the net does the
   rest also counts.)
3. Quantitative calibration: what per-field, NEAR-WALL errors (NRMSE/relative L2 on
   velocity in the BL region, surface pressure, C_d/C_l) do published airfoil-flow
   surrogates achieve — AirfRANS leaderboard entries (INFINITY, MARIO, Transolver,
   GINO...), unsteady/URANS cases, gust-response surrogates? Is 15–20% relative vx
   in the BL typical or poor for coordinate-decoder ROMs?
4. Depth/width/latent trade-offs for coordinate decoders and modulation-based INRs:
   evidence on deep-tanh trainability (seed sensitivity, init scaling like
   1/sqrt(depth), residual/skip connections in INRs, curriculum from shallow to
   deep), and anything on (quasi-)second-order optimizers (BFGS/L-BFGS) applied to
   these networks — known pathologies and cures.
5. Unsteady/temporal coordinate ROMs around moving surfaces (flap!): how do works
   handle a DEFORMING geometry in the decoder inputs (time-dependent SDF, reference
   frame remapping)? Our flap moves; wall-distance features would be time-dependent.

## Deliverable
A synthesis ranked by (expected impact on near-wall error) × (implementation cost in
our pipeline: adding decoder input features, changing sampling weights, adding an
input embedding, or a mean-field split are CHEAP; new architectures or losses are
MODERATE; anything touching the latent ODE is EXPENSIVE). For each recommendation:
the citing source(s) with the specific quantitative evidence, and what to measure in
an ablation on our side. Flag every claim you could not verify against its source.
