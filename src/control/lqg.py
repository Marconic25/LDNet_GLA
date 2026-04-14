import numpy as np
from scipy.linalg import solve_continuous_are
from pathlib import Path


def design_lqr(A_s, B_s, Q, R):

    P = solve_continuous_are(A_s, B_s, Q, R)
    K = np.linalg.inv(R) @ B_s.T @ P
    return K

def design_Kalman(A_s, Q_noise, R_noise):
    """
    Design Kalman filter observer gain.

    Assumes measurement of accelerations [h_ddot, a_ddot] which are rows 1,3 of A_s @ x.

    Args:
        A_s: structural state matrix (4×4)
        Q_noise: process noise covariance (4×4)
        R_noise: measurement noise covariance (2×2)

    Returns:
        L: Kalman observer gain (4×2)
    """
    # C_obs selects accelerations from state: h_ddot = A_s[1,:] @ x, a_ddot = A_s[3,:] @ x
    C_obs = np.array([A_s[1, :], A_s[3, :]])  # shape (2, 4)

    # Solve Riccati for observer: A.T, C.T required
    P = solve_continuous_are(A_s.T, C_obs.T, Q_noise, R_noise)
    L = P @ C_obs.T @ np.linalg.inv(R_noise)
    return L


def run_lqg_simulation(U_INF, T_END, DT, aero_model, K, L, A_s, B_s, delta_max=20.0):
    """
    Closed-loop LQG simulation with LDNet surrogate.

    Args:
        U_INF: freestream velocity [m/s]
        T_END: simulation end time [s]
        DT: time step [s]
        aero_model: LDNetModel instance
        K: LQR gain (1×4)
        L: Kalman gain (4×2)
        A_s: structural state matrix (4×4)
        B_s: structural input matrix (4×1)
        delta_max: deflection limit [°]

    Returns:
        dict with time history of [h, hd, a, ad, delta, C_L, C_M, h_ddot, a_ddot]
    """
    from aerodynamics.model import LDNetModel
    from structural.smd import structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA, D_H, D_ALPHA, K_H, K_ALPHA, _D_X

    # Aero/structural parameters
    rho_inf = 1.225  # air density [kg/m³]
    S_ref = 0.05     # reference area [m²]
    c = 1.0          # reference chord [m]

    # Time grid
    t_win = np.linspace(0.0, T_END, int(T_END / DT) + 1)
    N = len(t_win)

    # Gust function
    def gust_velocity(t):
        GUST_W0 = 60.0
        GUST_T_START = 0.0
        GUST_T_END = 0.8
        t_rel = t - GUST_T_START
        T_g = GUST_T_END - GUST_T_START
        if 0.0 <= t_rel <= T_g:
            return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t_rel / T_g))
        return 0.0

    # Initialize state x = [h, hd, a, ad]
    x = np.array([0.0, 0.0, 0.0, 0.0])
    x_hat = np.array([0.0, 0.0, 0.0, 0.0])  # estimated state
    z = np.zeros((aero_model.num_latent_states,))  # LDNet latent state

    # Precompute gust
    W_gust_arr = np.array([gust_velocity(t) for t in t_win])

    # History arrays
    h_hist = np.zeros(N)
    hd_hist = np.zeros(N)
    a_hist = np.zeros(N)
    ad_hist = np.zeros(N)
    delta_hist = np.zeros(N)
    C_L_hist = np.zeros(N)
    C_M_hist = np.zeros(N)
    h_ddot_hist = np.zeros(N)
    a_ddot_hist = np.zeros(N)

    # Loop
    for i, t in enumerate(t_win):
        # Save initial state
        h_hist[i] = x[0]
        hd_hist[i] = x[1]
        a_hist[i] = x[2]
        ad_hist[i] = x[3]

        # Step 1: LQR control law
        delta = -K @ x_hat
        delta = np.clip(delta[0], -delta_max, delta_max)
        delta_hist[i] = delta

        # Step 2: LDNet aerodynamic step
        z, C_L, C_M = aero_model.step(z, x[0], x[1], x[2], x[3], delta, W_gust_arr[i], U_INF, DT)
        C_L_hist[i] = C_L
        C_M_hist[i] = C_M

        # Step 3: Convert to forces
        Fy = 0.5 * rho_inf * U_INF**2 * S_ref * C_L
        Mz = 0.5 * rho_inf * U_INF**2 * S_ref * c * C_M

        # Step 4: Integrate structure with RK4
        def struct_rhs_wrapper(state, Fy_val, Mz_val):
            # Use structural_rhs with delta_dot=0, delta_ddot=0 (no feedback through flap dynamics)
            return np.array(structural_rhs(t, state, Fy_val, Mz_val, 0.0, 0.0))

        # RK4 step
        k1 = struct_rhs_wrapper(x, Fy, Mz)
        k2 = struct_rhs_wrapper(x + 0.5*DT*k1, Fy, Mz)
        k3 = struct_rhs_wrapper(x + 0.5*DT*k2, Fy, Mz)
        k4 = struct_rhs_wrapper(x + DT*k3, Fy, Mz)
        x_new = x + (DT/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Extract accelerations (last evaluation of RHS at final state)
        _, h_ddot, _, a_ddot = struct_rhs_wrapper(x_new, Fy, Mz)
        h_ddot_hist[i] = h_ddot
        a_ddot_hist[i] = a_ddot

        # Step 5: Measure accelerations
        y = np.array([h_ddot, a_ddot])

        # Step 6: Predict accelerations from state estimate
        y_hat = A_s[1, :] @ x_hat, A_s[3, :] @ x_hat
        y_hat = np.array([A_s[1, :] @ x_hat, A_s[3, :] @ x_hat])

        # Step 7: Kalman update
        x_hat_dot = A_s @ x_hat + B_s[:, 0] * delta + L @ (y - y_hat)
        x_hat = x_hat + DT * x_hat_dot

        # Update state for next iteration
        x = x_new

    return {
        't': t_win,
        'h': h_hist,
        'hd': hd_hist,
        'a': a_hist,
        'ad': ad_hist,
        'delta': delta_hist,
        'C_L': C_L_hist,
        'C_M': C_M_hist,
        'h_ddot': h_ddot_hist,
        'a_ddot': a_ddot_hist,
    }