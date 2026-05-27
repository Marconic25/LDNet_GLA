"""
Quasi-steady linear aerodynamic model (Theodorsen, incompressible thin-airfoil).

Lift and moment coefficients as a function of the structural state, gust velocity,
and trailing-edge flap deflection:

    C_L = 2π (α + W/U + ḣ/U) + C_Lδ · δ
    C_M = C_Mα · α             + C_Mδ · δ

The effective angle of attack combines pitch (α), gust (W/U), and heave rate
(ḣ/U). Upward heave velocity reduces the effective incidence because the wing
moves into the oncoming flow.

"""
import numpy as np

CL_ALPHA = 2.0 * np.pi   # thin-airfoil lift-curve slope [1/rad]
CL_DELTA = 0.7            # flap lift effectiveness [1/rad]
CM_ALPHA = -0.1           # pitch moment slope w.r.t. α [1/rad]
CM_DELTA = 0.35           # pitch moment slope w.r.t. flap [1/rad]


def predict(state, delta_deg, W, U):
    """
    Compute aerodynamic coefficients at the current instant.

    Parameters
    ----------
    state     : array-like, shape (4,)  [h (m), ḣ (m/s), α (rad), α̇ (rad/s)]
    delta_deg : float   flap deflection [degrees]
    W         : float   vertical gust velocity [m/s]
    U         : float   freestream velocity [m/s]

    Returns
    -------
    C_L : float
    C_M : float
    """
    _, hd, alpha, _ = state
    delta_rad = np.deg2rad(delta_deg)
    U = max(float(U), 1.0)

    C_L = CL_ALPHA * (alpha + W / U + hd / U) + CL_DELTA * delta_rad
    C_M = CM_ALPHA * alpha + CM_DELTA * delta_rad

    return float(C_L), float(C_M)
