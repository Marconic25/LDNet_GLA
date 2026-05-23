#!/usr/bin/env python3
"""
Validate LDNet against open-loop simulation data.

Loads sim_A_006_test.csv, drives the model with the recorded inputs
(h, hd, alpha, alpha_dot, delta, W_g) and compares predicted C_L, C_M
against the reference values.

Outputs 4 figures (matching OLvsLQR fig1/fig2/fig4 style):
  fig1_state.png       — h, hd, alpha, alpha_dot
  fig2_accels.png      — (not available in CSV, skipped — replaced by W_g)
  fig3_gust.png        — W_gust reference vs model input
  fig4_aero.png        — C_L, C_M: model vs data
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SRC_DIR    = Path(__file__).parent
MODELS_DIR = SRC_DIR.parent / 'models'
DATA_FILE  = SRC_DIR.parent / 'data' / 'GLA_data' / 'timeseries' / 'sim_A_006_test.csv'
OUT_DIR    = SRC_DIR.parent / 'results' / 'validate_model'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from aerodynamics.model import LDNetModel

U_INF = 80.0   # m/s — matches preprocess_GLA.py
# DT is read from the CSV (t[1]-t[0])

# ── load data ─────────────────────────────────────────────────────────────────
print(f"Loading {DATA_FILE.name} ...")
df = pd.read_csv(DATA_FILE)
t      = df['t'].values
h      = df['h'].values
hd     = df['h_dot'].values
alpha  = df['alpha'].values
ad     = df['alpha_dot'].values
delta  = df['delta'].values
W_g    = df['W_g'].values
CL_ref = df['C_L'].values
CM_ref = df['C_M'].values

DT = float(t[1] - t[0])
print(f"  N={len(t)} steps  DT={DT:.4f}s  T_end={t[-1]:.3f}s  U_INF={U_INF} m/s")

# ── load model ────────────────────────────────────────────────────────────────
print("Loading LDNet model ...")
aero = LDNetModel(str(MODELS_DIR))
print(f"  num_latent_states = {aero.num_latent_states}")

# compute trim latent state
z = np.zeros(aero.num_latent_states)
for _ in range(200):
    z, CL_tr, CM_tr = aero.step(z, 0., 0., 0., 0., 0., 0., U_INF, DT)
print(f"  CL_trim={float(CL_tr):.5f}  CM_trim={float(CM_tr):.5f}")

# ── drive model with recorded inputs ─────────────────────────────────────────
print("Running model with recorded inputs ...")
CL_pred = np.zeros(len(t))
CM_pred = np.zeros(len(t))

for i in range(len(t)):
    z, CL, CM = aero.step(z, h[i], hd[i], alpha[i], ad[i],
                           delta[i], W_g[i], U_INF, DT)
    CL_pred[i] = float(CL)
    CM_pred[i] = float(CM)

# ── metrics ───────────────────────────────────────────────────────────────────
nrmse_CL = np.sqrt(np.mean((CL_pred - CL_ref)**2)) / (CL_ref.max() - CL_ref.min())
nrmse_CM = np.sqrt(np.mean((CM_pred - CM_ref)**2)) / (CM_ref.max() - CM_ref.min())
print(f"\n  NRMSE C_L = {nrmse_CL:.4f}  ({nrmse_CL*100:.2f}%)")
print(f"  NRMSE C_M = {nrmse_CM:.4f}  ({nrmse_CM*100:.2f}%)")

# ── plot style ────────────────────────────────────────────────────────────────
REF   = dict(color='steelblue', lw=1.5, alpha=0.85, label='CFD data')
MODEL = dict(color='crimson',   lw=1.5, alpha=0.85, ls='--', label='LDNet')

def fmt(ax, ylabel, title=None):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

# ── Figure 1 — Structural state (from data) ───────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig1.suptitle('Structural State  —  sim_A_006_test', fontsize=13)
axes[0,0].plot(t, h,                **REF); fmt(axes[0,0], 'h [m]',    'Heave displacement')
axes[0,1].plot(t, hd,               **REF); fmt(axes[0,1], 'ḣ [m/s]',  'Heave velocity')
axes[1,0].plot(t, np.rad2deg(alpha),**REF); fmt(axes[1,0], 'α [°]',    'Pitch angle')
axes[1,1].plot(t, np.rad2deg(ad),   **REF); fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
fig1.tight_layout()
fig1.savefig(OUT_DIR / 'fig1_state.png', dpi=150); plt.close(fig1)
print("[OK] fig1_state.png")

# ── Figure 2 — Gust profile ────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 4))
fig2.suptitle('Gust Profile  —  sim_A_006_test', fontsize=13)
ax.plot(t, W_g, color='steelblue', lw=1.8, label='$W_{gust}$ [m/s]')
fmt(ax, 'W [m/s]', 'Gust velocity')
fig2.tight_layout()
fig2.savefig(OUT_DIR / 'fig2_gust.png', dpi=150); plt.close(fig2)
print("[OK] fig2_gust.png")

# ── Figure 3 — Control input ──────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(10, 4))
fig3.suptitle('Control Input  —  sim_A_006_test', fontsize=13)
ax.plot(t, np.rad2deg(delta), color='darkcyan', lw=1.8, label='δ [°]')
fmt(ax, 'δ [°]', 'Flap deflection')
fig3.tight_layout()
fig3.savefig(OUT_DIR / 'fig3_control.png', dpi=150); plt.close(fig3)
print("[OK] fig3_control.png")

# ── Figure 4 — Aerodynamic coefficients: model vs data ────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig4.suptitle(f'Aerodynamic Coefficients  —  LDNet vs CFD data\n'
              f'NRMSE $C_L$={nrmse_CL*100:.1f}%  '
              f'NRMSE $C_M$={nrmse_CM*100:.1f}%', fontsize=12)
axes[0].plot(t, CL_ref,  **REF);   axes[0].plot(t, CL_pred, **MODEL)
fmt(axes[0], '$C_L$', 'Lift coefficient')
axes[1].plot(t, CM_ref,  **REF);   axes[1].plot(t, CM_pred, **MODEL)
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
fig4.tight_layout()
fig4.savefig(OUT_DIR / 'fig4_aero.png', dpi=150); plt.close(fig4)
print("[OK] fig4_aero.png")

print(f"\nDone. Results in: {OUT_DIR}")
