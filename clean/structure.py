"""
Two-degree-of-freedom aeroelastic structural model.

Degrees of freedom: heave h [m] and pitch α [rad] of the wing section.
A trailing-edge flap is kinematically attached to the wing; its mass and
inertia appear as off-diagonal entries in the augmented mass matrix and
as generalised forcing terms.

Equations of motion (vector form):
    M_aug · [h_ddot; α_ddot] = f_aero + f_struct + f_flap_kinematics

where M_aug is the 2×2 augmented mass matrix coupling wing and flap inertia.
"""
import numpy as np

# --- Wing structural parameters (Hodges-Pierce benchmark, from cosim_driver.py) ---
M_WING  = 19.24          # mass [kg]
I_WING  = 1.155          # moment of inertia about elastic axis [kg·m²]
K_H     = 25460.0        # heave spring stiffness [N/m]   (ω_h ≈ 36.4 rad/s)
D_H     = 88.0           # heave damping coefficient [N·s/m]      (ζ ≈ 2 %)
K_ALPHA = 9530.0         # pitch spring stiffness [N·m/rad]       (ω_α ≈ 90.9 rad/s)
D_ALPHA = 6.6            # pitch damping coefficient [N·m·s/rad]  (ζ ≈ 2 %)

# --- Flap parameters ------------------------------------------------------
M_FLAP    = 1.19   # flap mass [kg]
I_FLAP_CG = 0.006  # flap moment of inertia about its own centre of gravity [kg·m²]

# Elastic axis and hinge positions (measured from leading edge)
EA_X     = 0.40    # elastic axis x [m]  (40%c — Hodges-Pierce)
HINGE_X  = 0.779   # flap hinge x [m]
DX       = HINGE_X - EA_X   # 0.379 m — horizontal offset EA → hinge
DY       = 0.0               # vertical offset (hinge on chord line)

# Flap moment of inertia about the elastic axis (parallel-axis theorem)
I_FLAP_EA    = I_FLAP_CG + M_FLAP * (DX**2 + DY**2)
I_FLAP_HINGE = I_FLAP_EA  # same reference distance

# --- Augmented mass matrix (constant, computed once) ----------------------
M_HH = M_WING + M_FLAP        # heave–heave
M_AA = I_WING + I_FLAP_EA     # pitch–pitch
M_HA = M_FLAP * DX            # coupling (symmetric)
DET  = M_HH * M_AA - M_HA**2  # determinant for 2×2 inversion


def rhs(state, Fy, Mz, delta_dot=0.0, delta_ddot=0.0):
    """
    Right-hand side of the structural equations of motion.

    Parameters
    ----------
    state       : array-like (h, ḣ, α, α̇)
    Fy          : float   aerodynamic heave force [N]  (positive up)
    Mz          : float   aerodynamic pitch moment [N·m]
    delta_dot   : float   flap angular rate [rad/s]   (used for Coriolis term)
    delta_ddot  : float   flap angular acceleration [rad/s²]

    Returns
    -------
    [ḣ, h_ddot, α̇, α_ddot]
    """
    h, hd, a, ad = state

    # Generalised forces from the prescribed flap motion (small-angle approximation)
    Q_h_flap = -M_FLAP * DY * delta_ddot - M_FLAP * DX * (2.0 * ad * delta_dot)
    Q_a_flap = -I_FLAP_HINGE * delta_ddot

    # Assemble right-hand sides for the two DOFs
    rhs_h = -Fy - D_H * hd - K_H * h + Q_h_flap
    rhs_a =  Mz - D_ALPHA * ad - K_ALPHA * a + Q_a_flap

    # Solve the 2×2 augmented mass system
    h_ddot = (M_AA * rhs_h - M_HA * rhs_a) / DET
    a_ddot = (M_HH * rhs_a - M_HA * rhs_h) / DET

    return [hd, h_ddot, ad, a_ddot]


def step_dp45(state, Fy, Mz, dt):
    """
    Advance the structural state by one time step using Dormand-Prince RK45.

    Uses scipy.integrate.solve_ivp (method='RK45', rtol=1e-8, atol=1e-10,
    max_step=2*dt), matching the parameters of cosim_driver.integrate_structural.
    Aerodynamic forces Fy and Mz are held constant over the step.

    Parameters
    ----------
    state : array-like (h, ḣ, α, α̇)
    Fy    : float   aerodynamic heave force [N]
    Mz    : float   aerodynamic pitch moment [N·m]
    dt    : float   time step [s]

    Returns
    -------
    state_next : np.ndarray (h, ḣ, α, α̇)
    """
    from scipy.integrate import solve_ivp
    x = np.asarray(state, dtype=float)

    sol = solve_ivp(lambda t, s: rhs(s, Fy, Mz), [0.0, dt], x,
                    method='RK45', rtol=1e-8, atol=1e-10, max_step=2.0 * dt)
    return sol.y[:, -1]
