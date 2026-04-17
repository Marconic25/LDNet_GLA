import numpy as np, sys
from pathlib import Path
sys.path.insert(0, '/home/marco/LDNet_OF/src')
from structural.smd import get_space_state_matrices
from aerodynamics.model import LDNetModel as AeroModel
from control.mpc import run_mpc_simulation

models_dir = Path('/home/marco/LDNet_OF/models')
aero_model = AeroModel(str(models_dir))
A_s, B_s, _, _ = get_space_state_matrices()
res = run_mpc_simulation(80., 1., 0.01, aero_model, None, A_s, B_s, use_ekf=False)
t = res['t']; DT = 0.01; U=80.
W_t = res['W_gust']
h = res['h']; hd = res['hd']; a = res['a']; ad = res['ad']

_z_trim = np.zeros(aero_model.num_latent_states)
for _ in range(200):
    _z_trim, _, _ = aero_model.step(_z_trim, 0., 0., 0., 0., 0., 0., U, DT)

z_true = _z_trim.copy()
z_W20 = _z_trim.copy()

# Also test: what CL does z_W20 give vs z_true at same x state?
print('t     W_true  z_true   z_W20   dz       CL(z_true,Wtrue)  CL(z_W20,W20)')
for i in range(len(t)-1):
    z_true, CL_true, _ = aero_model.step(z_true, h[i], hd[i], a[i], ad[i], 0., W_t[i], U, DT)
    z_W20, CL_W20, _ = aero_model.step(z_W20, h[i], hd[i], a[i], ad[i], 0., 20., U, DT)
    if i+1 in [20,25,30,35,40,50,60,70,80]:
        zt = float(z_true); z20 = float(z_W20)
        print(f't={t[i+1]:.2f}  W={W_t[i+1]:.1f}  zt={zt:.5f}  z20={z20:.5f}  dz={z20-zt:.5f}  CL_t={CL_true:.4f}  CL_20={CL_W20:.4f}')
