import numpy as np
from aerodynamics.model import LDNetModel as AeroModel
from structural.smd import integrate_structural
from pathlib import Path

"""
Aeroelastic system definition and parameters.
For each step:
1. Calculate W_gust using the gust_velocity function.
2. Update the aerodynamic model using the step function, which computes the new latent state and the aerodynamic coefficients C_L and C_M based on the current state, input signals, and input parameters.
3.Convert C_L and C_M to aerodynamic forces and moments, and use these to update the structural model (not shown here) to get the new state of the system.
4. Structural model integration and update h, hd, a, ad based on the aerodynamic forces and moments.
5. Save the history of states, aerodynamic coefficients
"""

# Aero parameters to compute F_L and M from C_L and C_M

rho_inf = 1.225  # air density [kg/m^3]
S_ref = 0.05      # reference area [m^2]
c = 1.0          # reference chord length [m]

# Gust parameters (cosine gust, EASA CS-25 profile)
U_INF        = 80.0   # freestream velocity [m/s]
GUST_W0      = 60.0   # peak gust velocity [m/s]
GUST_T_START = 0.0    # gust onset [s]
GUST_T_END   = 0.8    # gust end [s]  (~2 flutter periods, T_flutter≈0.4s)

def gust_velocity(t):
    """Cosine gust vertical velocity component [m/s] at time t."""
    t_rel = t - GUST_T_START
    T_g   = GUST_T_END - GUST_T_START
    if 0.0 <= t_rel <= T_g:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t_rel / T_g))
    return 0.0

T_END = 5.0  # total simulation time [s]
DT = 0.01    # time step for integration [s]
def run_aeroelastic_simulation(delta_control, U_INF, T_END, DT, aero_model=None):
    """Run the aeroelastic simulation, integrating the structural model and updating the aerodynamic model at each time step."""
    if aero_model is None:
        aero_model = AeroModel(str(Path(__file__).parent.parent.parent / 'models'))

    # Time window for simulation
    t_win = np.linspace(0.0, T_END, int(T_END/DT))  # simulate for T_END seconds with DT time step

    # Initialize state variables (z, h, hd, a, ad)
    z0 = np.zeros((aero_model.num_latent_states,))  # initial latent state (can be adjusted based on the problem)
    h0 = 0.0   # initial vertical displacement [m]
    hd0 = 0.0  # initial vertical velocity [m/s]
    a0 = 0.0   # initial angle of attack [rad]
    ad0 = 0.0  # initial angular velocity [rad/s]
    z= z0
    # Precompute gust velocities and aerodynamic forces/moments for the time window
    W_gust_arr = np.array([gust_velocity(t) for t in t_win])
    C_L_arr = np.zeros_like(t_win)
    C_M_arr = np.zeros_like(t_win)
    Fy_arr = np.zeros_like(t_win)
    Mz_arr = np.zeros_like(t_win)
    delta_dot_arr = np.zeros_like(t_win)
    delta_ddot_arr = np.zeros_like(t_win)

    def struct_deriv(h, hd, a, ad, Fy, Mz):

        return hd,h_ddot, ad, a_ddot

    for i, t in enumerate(t_win):
         z, C_L, C_M = aero_model.step(z,h0,hd0,a0,ad0, delta_control[i], W_gust_arr[i], U_INF,DT)
         C_L_arr[i] = C_L
         C_M_arr[i] = C_M
              # Convert C_L and C_M to aerodynamic forces and moments
         Fy_arr[i] = 0.5 * rho_inf * U_INF**2 * S_ref * C_L_arr[i]  # lift force array [N]
         Mz_arr[i] = 0.5 * rho_inf * U_INF**2 * S_ref * c * C_M_arr[i]  # pitching moment array [N*m]

    # Integrate structural model over time window
    h_arr, hd_arr, a_arr, ad_arr, h_traj, alpha_traj = integrate_structural(h0, hd0, a0, ad0, t_win, Fy_arr, Mz_arr, delta_dot_arr, delta_ddot_arr)
    

    return h_arr, hd_arr, a_arr,ad_arr, C_L_arr, C_M_arr, h_traj, alpha_traj