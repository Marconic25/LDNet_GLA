# Design: LaTeX Chapter — LQR Control for Gust Load Alleviation

**Date:** 2026-04-27  
**Project:** LDNet_OF  
**Context:** Standalone report chapter (not part of a larger thesis with pre-existing sections). Must be self-contained but concise.

---

## Purpose

Write a LaTeX chapter presenting the LQR controller for gust load alleviation (GLA) on the LDNet aeroelastic model. The chapter must:
- Present the full mathematics (Jacobian linearization via autodiff, DARE, gain K)
- Show time histories of plunge, pitch, accelerations, gust, C_L, C_M, flap deflection under LQR
- Compare with open-loop (no control) baseline
- Discuss trade-off between oscillation reduction and control effort (Q/R sensitivity — qualitative)

---

## System Description

### Augmented State
ξ = [h, ḣ, α, α̇, z]ᵀ  — 4 structural DOFs + 1 scalar latent aero state from LDNet

### Input
u = δ (flap deflection, ±20° saturation)

### Outputs (penalized)
y = [C_L, C_M]ᵀ

### Structural Parameters (from `src/structural/smd.py`)
- M_hh = 24.09 kg, M_aa = 2.063 kg·m², M_ha = 0.625 kg·m (inertial coupling)
- K_H = 4000 N/m, K_α = 700 N·m/rad
- D_H = 12 N·s/m, D_α = 1.6 N·m·s/rad
- Ref area × span = 0.05 m² (used in q_dyn = 0.5·ρ·U²·S)

### LDNet Aero Model (from `src/aerodynamics/model.py`)
- NNdyn (7-7-1 MLP): z update
- NNrec (24-24-24-24-2): reconstructs [C_L, C_M]
- Key nonlinear property: sign flip of dC_L/dδ under large gust → motivates linear approximation at trim only

### Simulation Conditions
- U∞ = 75 m/s, DT = 0.01 s
- Gust: 1-cosine, W_peak = 60 m/s, duration = 1 s, start at t = 0

---

## Chapter Structure (Approach B — Problem-driven)

### §1 Introduction (~0.5 page)
- Motivation: offline optimal regulator as computationally cheap alternative to MPC
- Problem statement: GLA with single flap input δ, sensor = structural state observer
- Key challenge: model contains neural network → no closed-form Jacobian → autodiff

### §2 Aeroelastic Model & Trim (~1 page)
- 2-DOF equations of motion (heave + pitch) with inertial coupling
- LDNet latent state update equation
- Trim computation at (U∞, W=0): ξ_trim, u_trim = 0
- Output equations y = g(ξ, u)

### §3 Jacobian Linearization via Automatic Differentiation (~1.5 pages)
**This is the mathematical core of the chapter.**
- Discrete-time system: ξₖ₊₁ = F(ξₖ, uₖ) via RK4 + LDNet
- Four Jacobian matrices:
  - A_d = ∂F/∂ξ|_trim  (5×5)
  - B_d = ∂F/∂u|_trim  (5×1)
  - C_y = ∂y/∂ξ|_trim  (2×5)
  - D_y = ∂y/∂u|_trim  (2×1)
- Algorithm: tf.GradientTape differentiates through RK4 AND LDNet layers simultaneously
- RK4 tensoriale: equations of rhs() shown explicitly with mass matrix inversion
- Open-loop stability check: |λ(A_d)| < 1 (lightly damped oscillatory modes)

### §4 LQR Formulation & DARE (~1 page)
- Infinite-horizon discrete LQR cost:
  J = Σₖ [ξₖᵀ Q_aug ξₖ + uₖᵀ R_aug uₖ]
- Output penalty augmentation:
  Q_aug = Q + C_yᵀ Q_y C_y  
  R_aug = R + D_yᵀ Q_y D_y
  (physical meaning: penalizes C_L, C_M deviations without state dimension increase)
- Discrete Algebraic Riccati Equation (DARE):
  P = Q_aug + A_dᵀ P A_d − A_dᵀ P B_d (R_aug + B_dᵀ P B_d)⁻¹ B_dᵀ P A_d
- Optimal gain: K = (R_aug + B_dᵀ P B_d)⁻¹ B_dᵀ P A_d  [1×5]
- Control law: δₜ = −K(ξ̂ₜ − ξ_trim), clipped to [−20°, +20°]
- Closed-loop stability: |λ(A_d − B_d K)| < 1 (verified numerically)

### §5 Results (~1.5 pages + 4 figures)
**Figures (PNG files, already generated or to be generated from test_mpc.py):**
- Fig. A: Time histories [h(t), α(t)] — LQR vs baseline
- Fig. B: Accelerations [ḧ(t), α̈(t)] — LQR vs baseline  
- Fig. C: Aerodynamic outputs [C_L(t), C_M(t)] — LQR vs baseline
- Fig. D: Control input δ(t) — control effort

**Text:**
- Quantitative metrics: % reduction in max |h|, max |α|, max |C_M|
- Discussion of residual first peak (LQR has no gust preview → irreducible C_L spike at t≈0.4s)
- Brief comparison with MPC if LQR metrics are available

### §6 Q/R Sensitivity & Trade-off (~0.5 page)
- Qualitative analysis only (no additional simulations)
- Increasing Q_y (output weight) → stronger gust rejection, higher δ effort
- Increasing R → conservative control, reduced structural damping improvement
- K(Q,R) is implicit via P(Q,R) from DARE — no closed-form sensitivity
- Note on DARE conditioning for extreme Q/R ratios

### §7 Conclusions (~0.25 page)
- Summary of LQR performance vs baseline
- Key advantage: single offline solve, zero runtime cost
- Limitation: linear approximation valid only near trim; large gust excursions degrade performance
- Comparison note: MPC has explicit gust preview, LQR is reactive

---

## Output Files

- `src/report/chapter_lqr.tex` — main LaTeX chapter file
- `src/report/figures/` — figures referenced in the chapter (PNG)

---

## Key Source Files

| File | Role |
|------|------|
| `src/control/lqr.py` | Full LQR implementation to transcribe into math |
| `src/structural/smd.py` | Structural parameters (M, K, D matrices) |
| `src/aerodynamics/model.py` | LDNet architecture and step_tf |
| `src/test_mpc.py` | Simulation loop + plotting (LQR case already included) |

---

## LaTeX Style Notes

- Use `\bm{}` (bm package) for bold vectors/matrices
- Equation environments: `equation`, `align` for multi-line derivations
- Use `\text{DARE}` label in equation, cite `\cite{anderson1990optimal}` for DARE theory
- Figures: `\includegraphics[width=\textwidth]{figures/...}`, `subfigure` for multi-panel layouts
- Keep derivations tight: skip intermediate algebra, show key steps only

---

## Verification

After writing the chapter:
1. Compile with `pdflatex chapter_lqr.tex` — zero errors, zero undefined references
2. Check all figure paths resolve
3. Verify DARE formula in §4 matches `scipy.linalg.solve_discrete_are` convention
4. Cross-check gain K formula with `src/control/lqr.py:113`
