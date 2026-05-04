"""
hinf_step1_linearize.py
=======================
Calcola la linearizzazione AD e salva le matrici su hinf_linearization.npz.
Va eseguito da src/ con il venv attivo, PRIMA di hinf_run_synthesis.py.

    cd /home/marco/LDNet_OF/src
    source venv/bin/activate
    python hinf_step1_linearize.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from aerodynamics.model import LDNetModel
from control.lqr import compute_jacobians

U_INF  = 75.0   # m/s — velocità nominale (sistema instabile, flutter ~3 Hz)
DT     = 0.01
OUTPUT = "hinf_linearization.npz"

print(f"Trim a U_inf={U_INF} m/s...")
aero = LDNetModel(str(Path(__file__).parent.parent / "models"))
z = np.zeros(aero.num_latent_states)
for _ in range(200):
    z, CL_trim, CM_trim = aero.step(z, 0., 0., 0., 0., 0., 0., U_INF, DT)
print(f"  z_trim={z}, CL_trim={CL_trim:.6f}, CM_trim={CM_trim:.6f}")

print("Jacobiani via AD...")
A_d, B_d, C_y, D_y, B_w = compute_jacobians(
    aero, np.zeros(4), z, U_INF, DT, CL_trim, CM_trim)

print(f"  A_d {A_d.shape}, B_d {B_d.shape}, C_y {C_y.shape}, B_w {B_w.shape}")
np.savez(OUTPUT,
         A_d=A_d, B_d=B_d, C_y=C_y, D_y=D_y, B_w=B_w,
         z_trim=z,
         CL_trim=np.array(CL_trim), CM_trim=np.array(CM_trim),
         U_INF=np.array(U_INF), DT=np.array(DT))
print(f"Salvato su {OUTPUT}")
