# Design: One-step Q_Δα̇ controller vs MPC

**Date:** 2026-06-29  
**Status:** implemented and validated

## Problem

The MPC (H=6) is the only prior controller stable on both design and strong gusts.
Goal: determine whether MPC is theoretically necessary, or whether a simpler one-step
controller with a better-targeted pitch-rate penalty suffices.

## Root cause of Phase 3 failure

Phase 3 penalised `Q · α̇(t+1)²` — the absolute pitch rate at the next step. During a
gust α̇ rises naturally, so the cost fights the physical response. Suppressing it requires
Q ≥ 1e4, at which point the flap saturates to kill pitch rate at the expense of C_L
alleviation. No setting yields both stability and GLA.

## Design A: Q_Δα̇ = Q · (α̇(t+1) − α̇(t))²

**Theoretical basis:** from the structural equations,
```
α̈ ∝ Mz(δ, W) − D_α·α̇ − K_α·α
```
Mz depends linearly on δ, so (α̇_next − α̇_curr) = α̈·dt is directly proportional to the
flap-induced moment. Penalising Δα̇ penalises the controller's own contribution to pitch
acceleration — not the natural gust response. This directly targets the first link of the
resonance cascade (δ → Mz → α̈ → α̇ → ΔC_L → δ) at one order lower weight.

**Implementation:** one line in `clean/controller.py:178`
```python
# before
self.Q_alpha_dot * x_next[3]**2
# after
self.Q_alpha_dot * (x_next[3] - state[3])**2
```
No new parameters, no new infrastructure.

## Validated results

Sweep: `clean/adot_sweep.py`, NH=1, NGRID=7, NSTEPS=800, rollout model.

| Controller | A_025 (design, exc=0.241) | A_027 (strong, exc=0.584) |
|---|---|---|
| Proportional G=20 | −59% stable | modest |
| 1-step Q_CL+R only (Phase 1) | −73% stable | EXPLODE |
| 1-step Q_alpha_dot (Phase 3) | −29% unstable | worsens |
| **Design A (Qad=5, R=1e-3)** | **−70% stable** | −12% stable |
| MPC H=6 | −46% stable | −20% stable |

## Conclusions

1. **MPC not needed for design gust.** Design A gives −70% vs MPC −46%, with a
   single-line cost change and no receding-horizon rollout.

2. **Strong gust gap (8%) is flap authority, not cost formulation.** At CLexc=0.584,
   the authority limit (dC_L/dδ ≈ 0.014/deg, δ_max=14°) saturates the flap under any
   aggressive setting. MPC recovers ~8% by sequencing commands across H steps to delay
   saturation — a multi-step effect the one-step cost cannot reproduce.

3. **Theoretical boundary:**
   - Q_Δα̇ is sufficient where pitch excitation mechanism dominates (moderate gusts).
   - MPC horizon is needed only where flap saturation makes command sequencing decisive.

4. **Best single cross-gust setting:** Qad=5–10, R=5e-3 (stable everywhere, +12%
   on strong gusts). For design-gust peak: Qad=5, R=5e-4 (−75%, ad=2.5x open).
