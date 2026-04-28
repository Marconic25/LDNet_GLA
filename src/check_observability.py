#!/usr/bin/env python3
"""
Observability analysis for the LQG design.

Computes the discrete observability matrix O for the augmented pair
(A_aug, C_obs) where:
    A_aug  in R^{6x6}  — augmented LQR system matrix (ξ, W)
    C_obs  in R^{2x6}  — measurement matrix for outputs (h_ddot, alpha)

Output equation derivation:
  - alpha is a direct state:  y_alpha = [0 0 1 0 0 0] @ xi_aug
  - h_ddot comes from the structural EOM linearized at trim:
        h_ddot = (M_aa * RHS_h - M_ha * RHS_a) / det
    with RHS_h = -Fy - D_H*hd - K_H*h,  RHS_a = Mz - D_alpha*ad - K_alpha*a
    At trim (delta=0, W=0), Fy = q_dyn * C_L  and  Mz = q_dyn * C_M.
    Linearizing:
        delta_Fy  = q_dyn * C_y[0,:] @ delta_xi        (row 0 of C_y)
        delta_Mz  = q_dyn * C_y[1,:] @ delta_xi        (row 1 of C_y)
    So:
        delta_h_ddot = [(M_aa*(-q_dyn*C_y[0,:]) - M_ha*(q_dyn*C_y[1,:])) / det
                        - (M_aa*D_H*e2 + M_ha*D_alpha*e4)/det
                        - (M_aa*K_H*e1 + M_ha*K_alpha*e3)/det]
    where e_i is the i-th unit vector in R^5 (1-indexed: h=1, hd=2, a=3, ad=4, z=5).

    This collapses to a single row of C_obs of length 6 (append zero for W column,
    since W affects h_ddot only through C_L which is already captured by C_y).
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import (M_WING, M_FLAP, I_WING, I_FLAP_EA,
                             D_H, D_ALPHA, K_H, K_ALPHA, _D_X, _D_Y, _D2)
from control.lqr import compute_jacobians
from aerodynamics.model import LDNetModel as AeroModel

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS  (must match test_mpc.py)
# ──────────────────────────────────────────────────────────────────────────────
U_INF     = 75.0
DT        = 0.01
LAMBDA_W  = 0.98

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL AND COMPUTE TRIM
# ──────────────────────────────────────────────────────────────────────────────
models_dir = Path(__file__).parent.parent / 'models'
if not models_dir.exists():
    models_dir = Path(__file__).parent.parent.parent / 'models'

print("Loading aerodynamic model...")
aero_model = AeroModel(str(models_dir))

print("Computing trim latent state...")
z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    z_trim, CL_trim, CM_trim = aero_model.step(z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)
CL_trim = float(CL_trim)
CM_trim = float(CM_trim)
print(f"  C_L_trim = {CL_trim:.5f},  C_M_trim = {CM_trim:.5f}")

# ──────────────────────────────────────────────────────────────────────────────
# LINEARIZATION
# ──────────────────────────────────────────────────────────────────────────────
print("Computing Jacobians via AD...")
x_trim = np.zeros(4)
A_d, B_d, C_y, D_y, B_w = compute_jacobians(
    aero_model, x_trim, z_trim, U_INF, DT, CL_trim, CM_trim)

# Build augmented system  [xi(5), W(1)]
A_aug = np.block([[A_d,              B_w.reshape(5, 1)     ],
                  [np.zeros((1, 5)), np.array([[LAMBDA_W]])]])   # (6, 6)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCT C_obs  (2 x 6)
# ──────────────────────────────────────────────────────────────────────────────
q_dyn = 0.5 * 1.225 * U_INF**2 * 0.05

M_hh = M_WING + M_FLAP
M_aa = I_WING + I_FLAP_EA
M_ha = M_FLAP * _D_X
det  = M_hh * M_aa - M_ha**2

# row_h_ddot in R^5: linearized h_ddot as function of delta_xi (5D)
#   from structural EOM:
#       h_ddot = (M_aa * RHS_h - M_ha * RHS_a) / det
#       RHS_h = -delta_Fy - D_H*delta_hd - K_H*delta_h
#       RHS_a =  delta_Mz - D_ALPHA*delta_ad - K_ALPHA*delta_a
#   with:
#       delta_Fy = q_dyn * C_y[0,:] @ delta_xi   (C_y row 0 = dCL/dxi)
#       delta_Mz = q_dyn * C_y[1,:] @ delta_xi   (C_y row 1 = dCM/dxi)

row_h_ddot_xi = (
    (M_aa * (-q_dyn * C_y[0, :]) - M_ha * (q_dyn * C_y[1, :])) / det
)
# structural terms: contributions from h (idx 0), hd (idx 1), a (idx 2), ad (idx 3)
row_h_ddot_xi[0] += (-M_aa * K_H) / det          # -K_H * h term
row_h_ddot_xi[1] += (-M_aa * D_H) / det          # -D_H * hd term
row_h_ddot_xi[2] += (M_ha * K_ALPHA) / det       # +M_ha * K_alpha * a / det  (sign: -M_ha*RHS_a/det, RHS_a has -K_alpha*a)
row_h_ddot_xi[3] += (M_ha * D_ALPHA) / det       # +M_ha * D_alpha * ad / det

# Append zero for W column (W affects h_ddot through C_L, already in C_y rows,
# but C_y was computed at W=0 so B_w accounts for the W->CL path separately;
# for the output equation at the current step, the W contribution is through
# the aero state, which is captured by the z column of C_y)
row_h_ddot = np.append(row_h_ddot_xi, 0.0)   # (6,)

# row_alpha: alpha is state index 2 (0-based), W index 5
row_alpha = np.zeros(6)
row_alpha[2] = 1.0   # direct measurement of alpha

C_obs = np.vstack([row_h_ddot, row_alpha])   # (2, 6)

print("\nC_obs (measurement matrix, 2x6):")
print("  rows: [h_ddot, alpha]")
print("  cols: [h, hd, alpha, alpha_dot, z, W]")
for i, name in enumerate(['h_ddot', 'alpha ']):
    print(f"  {name}: {C_obs[i]}")

# ──────────────────────────────────────────────────────────────────────────────
# OBSERVABILITY MATRIX  O = [C; C*A; C*A²; ...; C*A^{n-1}]  (2n x 6)
# ──────────────────────────────────────────────────────────────────────────────
n = 6   # state dimension
O = np.vstack([C_obs @ np.linalg.matrix_power(A_aug, k) for k in range(n)])

# Singular value decomposition
U_sv, sv, Vt = np.linalg.svd(O)
sv_max = sv[0]
tol_rel = 1e-6   # relative tolerance for numerical rank

rank = int(np.sum(sv / sv_max > tol_rel))

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  DISCRETE OBSERVABILITY ANALYSIS  —  (A_aug, C_obs)")
print("=" * 62)
print(f"  State dimension n = {n}")
print(f"  O shape          = {O.shape}")
print(f"  Relative tol     = {tol_rel:.0e}  (σ_i / σ_max)")
print()

print(f"  {'Index':>5}  {'Singular value':>16}  {'σ/σ_max':>10}  {'Above tol':>9}")
print("  " + "-" * 48)
for i, s in enumerate(sv):
    ratio = s / sv_max
    flag  = "YES" if ratio > tol_rel else "NO "
    print(f"  {i+1:>5}  {s:>16.6e}  {ratio:>10.3e}  {flag:>9}")

print()
print(f"  Numerical rank = {rank} / {n}  (system is "
      f"{'OBSERVABLE' if rank == n else 'NOT FULLY OBSERVABLE'})")

# ──────────────────────────────────────────────────────────────────────────────
# INTERPRETATION: which directions are weakly observable?
# ──────────────────────────────────────────────────────────────────────────────
print()
print("  Right singular vectors (V columns) for the 3 smallest σ:")
print("  (most-unobservable directions in state space)")
print()
state_names = ['h', 'hd', 'alpha', 'alpha_dot', 'z', 'W']
for i in range(min(3, n)):
    v = Vt[n - 1 - i]   # smallest σ first
    print(f"  σ_{n-i} = {sv[n-1-i]:.3e}:")
    for j, name in enumerate(state_names):
        bar = '█' * int(abs(v[j]) * 40)
        print(f"    {name:>9}: {v[j]:+.4f}  {bar}")
    print()

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT  A_aug and C_obs for re-use in Kalman filter
# ──────────────────────────────────────────────────────────────────────────────
np.save(Path(__file__).parent / 'obs_A_aug.npy', A_aug)
np.save(Path(__file__).parent / 'obs_C_obs.npy', C_obs)
print("  Saved A_aug → src/obs_A_aug.npy")
print("  Saved C_obs → src/obs_C_obs.npy")
print()
