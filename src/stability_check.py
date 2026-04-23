import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from aerodynamics.model import LDNetModel
from structural.smd import structural_rhs
import numpy as np

aero = LDNetModel(str(Path(__file__).parent.parent / 'models'))

for U in [60.0, 70.0, 75.0]:
    z = np.zeros(aero.num_latent_states)
    for _ in range(200):
        z, _, _ = aero.step(z, 0.,0.,0.,0.,0.,0.,U,0.01)
    x = np.zeros(4)
    x[0] = 0.001
    DT = 0.01
    for i in range(1000):
        z, CL, CM = aero.step(z, x[0],x[1],x[2],x[3], 0., 0., U, DT)
        q = 0.5*1.225*U**2*0.05
        Fy = q*float(CL); Mz = q*float(CM)
        k1 = np.array(structural_rhs(0,x,Fy,Mz,0,0))
        k2 = np.array(structural_rhs(0,x+0.5*DT*k1,Fy,Mz,0,0))
        k3 = np.array(structural_rhs(0,x+0.5*DT*k2,Fy,Mz,0,0))
        k4 = np.array(structural_rhs(0,x+DT*k3,Fy,Mz,0,0))
        x = x + (DT/6)*(k1+2*k2+2*k3+k4)
    stable = abs(x[0]) < 0.005 and abs(x[2]) < 0.05
    print(f"U={U:.0f}  t=10s: h={abs(x[0])*1000:.3f}mm  a={abs(np.rad2deg(x[2])):.4f}deg -> {'STABLE' if stable else 'UNSTABLE'}")
