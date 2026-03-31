#!/usr/bin/env python3
"""
Wing structural model (2-DOF, heave + pitch):
    m   * h_ddot     + d_h * h_dot     + k_h * h     = -Fy(t)
    I_z * alpha_ddot + d_a * alpha_dot + k_a * alpha  =  Mz(t)

Flap: input δ(t) from controller, kinematically constrained to the wing structure.
"""
import numpy as np
from scipy.integrate import solve_ivp

# ─────────────────────────── configuration ──────────────────────────────────

# Structural parameters — wing only
M_WING   = 22.9          # wing mass [kg]
I_WING   = 2.057121362   # wing MoI about z-axis [kg·m²]
K_H      = 4000.0        # heave spring stiffness [N/m]
D_H      = 2.0           # heave damping [N·s/m]
K_ALPHA  = 700.0         # pitch spring stiffness [N·m/rad]
D_ALPHA  = 0.5           # pitch damping [N·m·s/rad]

# Geometry (initial mesh coordinates)
EA_X, EA_Y = 0.25, 0.0          # elastic axis (CoR) initial position
HINGE_X    = 0.779               # flap hinge initial x
HINGE_Y    = 0.0                 # flap hinge initial y

# Flap physical parameters — mass and inertia contribute to wing EOM via constraint forces.
# The flap motion remains prescribed (kinematic); its inertia appears as forcing in the EOM.
# Geometry: chord≈0.243m (x: 0.750→0.993), span=0.05m, aluminium-equivalent density.
# Calibrate M_FLAP and I_FLAP_CG to match the real flap structure.
M_FLAP     = 1.19    # flap mass [kg]  (estimated: ρ_Al * A_section * span)
I_FLAP_CG  = 0.006   # flap MoI about its own CG [kg·m²]  (rectangular approx)
# Derived inertia quantities (computed once at module load)
_D_X       = HINGE_X - EA_X                        # 0.525 m  (EA → hinge, x)
_D_Y       = HINGE_Y - EA_Y                        # -0.045 m (EA → hinge, y)
_D2        = _D_X**2 + _D_Y**2                     # |d|² [m²]
I_FLAP_EA  = I_FLAP_CG + M_FLAP * _D2             # MoI about elastic axis [kg·m²]
I_FLAP_HINGE = I_FLAP_CG + M_FLAP * _D2           # MoI about hinge (same d) [kg·m²]


# ─────────────────────── structural integrator ──────────────────────────────

def structural_rhs(t, state, Fy_interp, Mz_interp, delta_dot_interp, delta_ddot_interp):
    """
    RHS of augmented 2-DOF EOM: state = [h, h_dot, alpha, alpha_dot].

    System: wing_main + flap (prescribed δ(t)).
    The flap is kinematically constrained → its mass/inertia appear as
    augmented mass matrix entries and generalised forcing terms.

    Augmented mass matrix [2×2]:
        M_hh = M_WING + M_FLAP
        M_αα = I_WING + I_FLAP_EA
        M_hα = M_αh = M_FLAP * d_x   (inertial coupling)

    Generalised forcing from flap kinematics (small-angle, leading-order):
        Q_h  = -M_FLAP * d_y * δ_ddot        (inertial — heave)
               -M_FLAP * d_x * 2*αd*δ_dot   (Coriolis — heave)
        Q_α  = -I_FLAP_HINGE * δ_ddot        (reaction torque — pitch)

    Forces aero on wing_main only (flap aero included in postProcessing/forces
    via patch list, but for EOM correctness only wing_main forces should drive
    the wing DOFs; flap aero goes into the actuator, not the wing structure).
    NOTE: forces.dat currently integrates both patches — acceptable approximation
    for small δ where flap lift ≪ wing lift.
    """
    h, hd, a, ad = state
    Fy = float(Fy_interp(t))
    Mz = float(Mz_interp(t))

    dlt_ddot = float(delta_ddot_interp(t))   # [rad/s²]
    dlt_dot  = float(delta_dot_interp(t))    # [rad/s]

    # Augmented mass matrix entries
    M_hh = M_WING + M_FLAP
    M_aa = I_WING + I_FLAP_EA
    M_ha = M_FLAP * _D_X        # off-diagonal (symmetric)

    # Generalised inertial forces from prescribed flap acceleration (small-angle)
    Q_h_flap = -M_FLAP * _D_Y * dlt_ddot - M_FLAP * _D_X * (2.0 * ad * dlt_dot)
    Q_a_flap = -I_FLAP_HINGE * dlt_ddot

    # Right-hand sides before solving the 2×2 mass system
    RHS_h = -Fy - D_H * hd - K_H * h + Q_h_flap
    RHS_a =  Mz - D_ALPHA * ad - K_ALPHA * a + Q_a_flap

    # Solve [M_hh M_ha; M_ha M_aa] * [h_ddot; a_ddot] = [RHS_h; RHS_a]
    det    = M_hh * M_aa - M_ha * M_ha
    h_ddot = (M_aa * RHS_h - M_ha * RHS_a) / det
    a_ddot = (M_hh * RHS_a - M_ha * RHS_h) / det

    return [hd, h_ddot, ad, a_ddot]


def integrate_structural(h0, hd0, a0, ad0, t_win, Fy_arr, Mz_arr, delta_dot_arr, delta_ddot_arr):
    """
    Integrate augmented 2-DOF structural model over time window t_win.
    Forces Fy_arr, Mz_arr are sampled at t_win points.
    Returns final state (h, hd, alpha, ad) and full trajectory arrays.
    """
    from scipy.interpolate import interp1d
    Fy_interp = interp1d(t_win, Fy_arr, kind="linear", fill_value="extrapolate")
    Mz_interp = interp1d(t_win, Mz_arr, kind="linear", fill_value="extrapolate")
    delta_dot_interp = interp1d(t_win, delta_dot_arr, kind="linear", fill_value="extrapolate")
    delta_ddot_interp = interp1d(t_win, delta_ddot_arr, kind="linear", fill_value="extrapolate")
    
    sol = solve_ivp(
        structural_rhs,
        [t_win[0], t_win[-1]],
        [h0, hd0, a0, ad0],
        args=(Fy_interp, Mz_interp, delta_dot_interp, delta_ddot_interp),
        t_eval=t_win,
        max_step=(t_win[1] - t_win[0]) * 2,
        rtol=1e-8, atol=1e-10,
    )
    h_arr     = sol.y[0]
    hd_arr    = sol.y[1]
    alpha_arr = sol.y[2]
    ad_arr    = sol.y[3]
    return h_arr[-1], hd_arr[-1], alpha_arr[-1], ad_arr[-1], h_arr, alpha_arr


