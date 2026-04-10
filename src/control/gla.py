import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from aeroelastic.system import run_aeroelastic_simulation
from aerodynamics.model import LDNetModel as AeroModel
from config import *


aero_model = None  # global variable to hold the aerodynamic model instance
def cost_function(delta_control, U_INF, T_END, DT, r):
    """Cost function for optimization: minimize max vertical displacement (heave) during gust response."""
    global aero_model
    if aero_model is None:
        aero_model = AeroModel(str(Path(__file__).parent.parent.parent / 'models'))
    _, _, _, _, C_L_arr, _, _, _ = run_aeroelastic_simulation(delta_control, U_INF, T_END, DT, aero_model)

    J_load =np.sum(C_L_arr**2) * DT  # Gust load cost (integral of squared lift coefficient
    J_control = r * np.sum(delta_control**2) * DT  # Control effort cost (integral of squared control input)
    return J_load + J_control

def optimize_gla(U_INF=U_INF_DEFAULT, T_END=T_END_DEFAULT, DT=DT_DEFAULT, r=R_DEFAULT):
    """Optimize GLA control input to minimize cost function."""
    N = int(T_END / DT)  # number of time steps
    delta0 = np.zeros(N)  # initial guess: zero control input
    bounds = [(-DELTA_MAX, DELTA_MAX)] * N  # control input bounds
    # Vincoli di velocità del flap
    constraints = []
    max_step = DELTA_DOT_MAX * DT
    for i in range(N - 1):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: max_step - (x[i+1] - x[i])})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: max_step + (x[i+1] - x[i])})
    
    result = minimize(cost_function, delta0, args=(U_INF, T_END, DT, r), bounds=bounds, constraints = constraints, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
    return result.x  # optimized control input array    