# Near-wall reconstruction — literature findings (deep-research run 2026-07-16, partial)

Run died mid-verification on the account spend limit: 11 claims adversarially verified
(2-3 independent votes each, zero refuted), 14 more claims extracted+sourced but never
voted on. Synthesis step never ran. Salvaged here verbatim-in-substance.

## VERIFIED (vote = confirmations-refutations)

**Spectral bias is our smearing mechanism, Fourier features are the standard cure**
1. (3-0) Fourier feature mapping of input coords lets a standard MLP learn
   high-frequency functions in low-dim domains — directly addresses smeared sharp
   gradients (BL profiles). [Tancik et al., arXiv:2006.10739]
2. (2-1) Without such encoding a coordinate MLP cannot fit high-frequency content
   ("spectral bias") in theory and practice — explains a static, smeared BL from a
   plain (x,y) tanh decoder. [arXiv:2006.10739]
3. (3-0) FF transforms the effective NTK into a stationary kernel with TUNABLE
   bandwidth → the decoder's passable spatial frequency is a hyperparameter (σ) to
   match BL length scales. [arXiv:2006.10739]
4. (2-1) Plain coordinate MLP oversmooths sharp aero gradients (RAE2822 transonic
   shock) via spectral bias; FF input encoding is the documented remedy.
   [arXiv:2407.19916]
5. (2-1) MULTISCALE FF (two Gaussian encodings σ=1 and σ=5, per-scale outputs
   concatenated before a final linear layer) resolves the single-σ trade-off; best
   reconstructions of sharp-gradient aero fields. [arXiv:2407.19916]
6. (3-0) σ trade-off is documented: too low = low-pass filter (smeared BL), too high
   = noisy reconstruction; tune to target frequency content. [arXiv:2407.19916]

**Wall-distance features work at the wall (AirfRANS SOTA)**
7. (3-0) INFINITY: FF-based INR backbone with shift modulation = state-of-the-art
   AirfRANS coordinate network (not a plain tanh MLP). [arXiv:2307.13538]
8. (1-1) INFINITY feeds the distance function to the airfoil surface as an explicit
   input next to (x,y). [arXiv:2307.13538]
9. (2-0) INFINITY surface-pressure MSE 0.07±0.01 vs 0.39–1.13 for Graph U-Net /
   GraphSAGE / MLP / PointNet — order of magnitude better EXACTLY at the wall.
   [arXiv:2307.13538]
10. (3-0) MARIO decoder inputs: FF-encoded coords + signed distance function +
    surface normals + a "boundary layer mask" feature = 1 at the surface, decaying
    quadratically to 0 within τ=0.02 normalized wall distance. [arXiv:2505.14704]
11. (2-0) Ablating that BL mask ⇒ ~6× increase in drag-coefficient error on
    AirfRANS. [arXiv:2505.14704]

## EXTRACTED, NOT YET VERIFIED (verifiers hit spend limit; none refuted)
- MARIO ablation detail: without BL mask, near-wall velocity oscillates; mask lets a
  single global MLP modulate local sensitivity (C_d error 0.794% → 4.780%).
  [arXiv:2505.14704]
- Hard no-slip: multiply network output by approximate distance function φ ⇒ exact
  homogeneous Dirichlet BC (Sukumar & Srivastava). [arXiv:2104.08426]; transfinite
  interpolation generalizes to inhomogeneous/Neumann/Robin on complex geometries.
- Berg & Nyström ansatz û(x) = G(x) + D(x)·NN(x) with smoothed distance D and
  boundary extension G — exact Dirichlet by construction; D and G fit by tiny
  auxiliary nets (works without analytic distance). [arXiv:1711.06464]
- Residual-based adaptive sampling (RAD / RAR-D): resample training points ∝ current
  residual — significant accuracy gains vs uniform sampling with fewer points; point
  placement is first-order, not a minor hyperparameter. [arXiv:2207.10289]
- FF (64 features, σ=1) + BL mask captures the steep BL velocity profile; σ=1 best
  across the BL. [arXiv:2505.14704]

## Immediate implications for recon/train_fields.py (my read, not a source)
Ranked impact × cost given our symptoms (static smeared BL bias in vx, uniform
sampling, plain tanh decoder):
1. **Fourier features on (x,y)** (multiscale σ≈[1,5] à la 2407.19916) — direct hit on
   the documented smearing mechanism; ~30 lines (input embedding + config persist).
2. **Wall-distance input (+ BL mask feature à la MARIO)** — 6×-class evidence at the
   wall; cheap: precompute d(x) from mesh once, add 1-2 input channels.
3. **Near-wall/residual-weighted point sampling** (RAD-lite: oversample nodes with
   d(x) small or current-error large) — replaces uniform 1024/11075 sampling; our A3
   area-weighting failure is the OPPOSITE direction (it upweighted far-field).
4. **Mean+fluctuation split** (predict deviations from stored mean field) — kills the
   static bias by construction; zero literature found YET (searches died early) but
   trivially testable.
5. **Hard no-slip masking** (Berg & Nyström / Sukumar) — elegant, moderate cost,
   pending verification of the sources.
Open (research died before answering): quantitative near-wall NRMSE calibration
across AirfRANS-class works (is our 15-20% vx typical?); depth/width/latent
trade-offs and second-order-optimizer pathologies in this literature.
