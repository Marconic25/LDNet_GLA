p='clean/run.py'
s=open(p).read()
# 1) add --settle arg
arg_anchor="_parser.add_argument('--mpc-horizon', type=int, default=6, dest='mpc_horizon',"
assert s.count(arg_anchor)==1,('arg',s.count(arg_anchor))
# insert a new add_argument line right before the mpc-horizon one
new_arg="_parser.add_argument('--settle', type=int, default=0,\n                     help='Relax joint latent+structure to equilibrium for N steps before sim (damped model physical init).')\n"
s=s.replace(arg_anchor, new_arg+arg_anchor)
# 2) replace the trim block with a settle-aware version
old='''    # Compute trim aero loads (W=0, delta=0) to subtract as static offset.
    # The structural integrator only sees perturbations from trim equilibrium.
    if hasattr(_aero_module, 'reset'):
        from scipy.optimize import fsolve

        def _trim_residual(x_eq):
            C_L_eq, C_M_eq = _aero_module.predict(
                np.array([x_eq[0], 0., x_eq[1], 0.]), 0., 0., U_INF)
            dFy = q_dyn * C_L_eq
            dMz = q_dyn * C_M_eq * C_REF
            return [-dFy - structure.K_H * x_eq[0],
                     dMz - structure.K_ALPHA * x_eq[1]]

        x_eq = fsolve(_trim_residual, [0., 0.])
        x0   = np.array([x_eq[0], 0., x_eq[1], 0.])
        C_L_trim, C_M_trim = _aero_module.predict(x0, 0., 0., U_INF)
        Fy_trim = q_dyn * C_L_trim
        Mz_trim = q_dyn * C_M_trim * C_REF
    else:
        x0 = np.zeros(4)
        C_L_trim = 0.0
        Fy_trim, Mz_trim = 0.0, 0.0'''
assert s.count(old)==1,('trim block',s.count(old))
new='''    # Compute trim aero loads (W=0, delta=0) to subtract as static offset.
    # The structural integrator only sees perturbations from trim equilibrium.
    _settle = getattr(_args, 'settle', 0)
    if hasattr(_aero_module, 'reset') and _settle > 0 and hasattr(_aero_module, 'advance'):
        # Damped model: relax the joint latent+structure system to its physical
        # equilibrium so the sim starts at a true trim (no startup transient).
        xs = np.zeros(4)
        for _ in range(_settle):
            cl_s, cm_s = _aero_module.predict(xs, 0., 0., U_INF)
            _aero_module.advance(xs, 0., 0., U_INF, DT)
            xs = structure.step_rk4(xs, q_dyn * cl_s, q_dyn * cm_s * C_REF, DT)
        x0 = xs.copy()
        C_L_trim, C_M_trim = _aero_module.predict(x0, 0., 0., U_INF)
        Fy_trim = q_dyn * C_L_trim
        Mz_trim = q_dyn * C_M_trim * C_REF
    elif hasattr(_aero_module, 'reset'):
        from scipy.optimize import fsolve

        def _trim_residual(x_eq):
            C_L_eq, C_M_eq = _aero_module.predict(
                np.array([x_eq[0], 0., x_eq[1], 0.]), 0., 0., U_INF)
            dFy = q_dyn * C_L_eq
            dMz = q_dyn * C_M_eq * C_REF
            return [-dFy - structure.K_H * x_eq[0],
                     dMz - structure.K_ALPHA * x_eq[1]]

        x_eq = fsolve(_trim_residual, [0., 0.])
        x0   = np.array([x_eq[0], 0., x_eq[1], 0.])
        C_L_trim, C_M_trim = _aero_module.predict(x0, 0., 0., U_INF)
        Fy_trim = q_dyn * C_L_trim
        Mz_trim = q_dyn * C_M_trim * C_REF
    else:
        x0 = np.zeros(4)
        C_L_trim = 0.0
        Fy_trim, Mz_trim = 0.0, 0.0'''
s=s.replace(old,new)
open(p,'w').write(s)
import ast; ast.parse(s)
print('RUN PATCH OK')
