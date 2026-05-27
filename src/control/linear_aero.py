"""
Quasi-steady Theodorsen linear aerodynamic model for Phase 1 Greedy controller.

C_L = 2π*(α + W/U + ḣ/U) + C_Lδ * δ_rad
C_M = C_Mα * α + C_Mδ * δ_rad

δ is in degrees (same convention as rest of codebase); converted internally.
Convention: δ > 0 = TE down = C_L increases.
"""
import numpy as np

# Flap effectiveness coefficients (30% chord trailing-edge flap, thin airfoil)
C_La     = 2.0 * np.pi   # lift-curve slope [rad⁻¹]
C_Ldelta = +0.7           # flap lift coefficient [rad⁻¹]  (positive: δ>0 = TE down → C_L up)
C_Ma     = -0.1           # pitch moment vs alpha [rad⁻¹]
C_Mdelta = +0.35          # pitch moment vs flap [rad⁻¹]   (positive: δ>0 = TE down → C_M up)


def predict(x, delta_deg, W, U):
    """
    Quasi-steady linear aerodynamic prediction.

    Parameters
    ----------
    x : array-like, shape (4,)  — [h, ḣ, α, α̇]  (SI units: m, m/s, rad, rad/s)
    delta_deg : float            — flap deflection [degrees]
    W : float                   — gust vertical velocity [m/s]
    U : float                   — freestream velocity [m/s]

    Returns
    -------
    C_L : float
    C_M : float
    """
    h, hd, a, ad = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    delta_rad = np.deg2rad(float(delta_deg))
    W = float(W)
    U = max(float(U), 1.0)  # avoid division by zero

    C_L = C_La * (a + W / U + hd / U) + C_Ldelta * delta_rad
    C_M = C_Ma * a + C_Mdelta * delta_rad
    return float(C_L), float(C_M)
