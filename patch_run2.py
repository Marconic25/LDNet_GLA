p='clean/run.py'
s=open(p).read()
old='''        x0 = xs.copy()
        C_L_trim, C_M_trim = _aero_module.predict(x0, 0., 0., U_INF)
        Fy_trim = q_dyn * C_L_trim
        Mz_trim = q_dyn * C_M_trim * C_REF
    elif hasattr(_aero_module, 'reset'):'''
assert s.count(old)==1,('settle trim',s.count(old))
new='''        x0 = xs.copy()
        C_L_trim, C_M_trim = _aero_module.predict(x0, 0., 0., U_INF)
        # x0=xs is already a joint equilibrium at rest under ABSOLUTE loads
        # (spring balances absolute aero load). Integrate absolute loads -> do NOT
        # subtract trim, else the spring force -K_H*h at x0 has no aero counterpart
        # and the structure rings (spurious startup transient). C_L_trim kept only
        # as the controller reference.
        Fy_trim = 0.0
        Mz_trim = 0.0
    elif hasattr(_aero_module, 'reset'):'''
s=s.replace(old,new)
open(p,'w').write(s)
import ast; ast.parse(s); print('PATCH2 OK')
