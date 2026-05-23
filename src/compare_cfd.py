#!/usr/bin/env python3
"""
Compare CFD data: sim_A_000_train (no gust) vs sim_A_006_test (with gust).
4 figures: structural state, gust + delta, aerodynamic coefficients, differences.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'GLA_data' / 'timeseries'
OUT_DIR  = Path(__file__).parent.parent / 'results' / 'compare_cfd'
OUT_DIR.mkdir(parents=True, exist_ok=True)

d0 = pd.read_csv(DATA_DIR / 'sim_A_000_train.csv')
d6 = pd.read_csv(DATA_DIR / 'sim_A_006_test.csv')

t = d0['t'].values

A0 = dict(color='steelblue', lw=1.5, alpha=0.85, label='A_000 (no gust)')
A6 = dict(color='crimson',   lw=1.5, alpha=0.85, label='A_006 (gust)')

def fmt(ax, ylabel, title=None):
    ax.set_xlabel('t [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

# ── Figure 1 — Structural state ───────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig1.suptitle('Structural State  —  A_000 vs A_006', fontsize=13)
for d, kw in [(d0, A0), (d6, A6)]:
    axes[0,0].plot(t, d['h'],                    **kw)
    axes[0,1].plot(t, d['h_dot'],                **kw)
    axes[1,0].plot(t, np.rad2deg(d['alpha']),    **kw)
    axes[1,1].plot(t, np.rad2deg(d['alpha_dot']),**kw)
fmt(axes[0,0], 'h [m]',    'Heave displacement')
fmt(axes[0,1], 'ḣ [m/s]',  'Heave velocity')
fmt(axes[1,0], 'α [°]',    'Pitch angle')
fmt(axes[1,1], 'α̇ [°/s]', 'Pitch rate')
fig1.tight_layout()
fig1.savefig(OUT_DIR / 'fig1_state.png', dpi=150); plt.close(fig1)
print("[OK] fig1_state.png")

# ── Figure 2 — Gust and delta ─────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig2.suptitle('Inputs  —  A_000 vs A_006', fontsize=13)
for d, kw in [(d0, A0), (d6, A6)]:
    axes[0].plot(t, d['W_g'],              **kw)
    axes[1].plot(t, np.rad2deg(d['delta']),**kw)
fmt(axes[0], 'W [m/s]', 'Gust velocity')
fmt(axes[1], 'δ [°]',   'Flap deflection')
fig2.tight_layout()
fig2.savefig(OUT_DIR / 'fig2_inputs.png', dpi=150); plt.close(fig2)
print("[OK] fig2_inputs.png")

# ── Figure 3 — Aerodynamic coefficients ──────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig3.suptitle('Aerodynamic Coefficients  —  A_000 vs A_006', fontsize=13)
for d, kw in [(d0, A0), (d6, A6)]:
    axes[0].plot(t, d['C_L'], **kw)
    axes[1].plot(t, d['C_M'], **kw)
fmt(axes[0], '$C_L$', 'Lift coefficient')
fmt(axes[1], '$C_M$', 'Pitching moment coefficient')
fig3.tight_layout()
fig3.savefig(OUT_DIR / 'fig3_aero.png', dpi=150); plt.close(fig3)
print("[OK] fig3_aero.png")

# ── Figure 4 — Differences (A_006 - A_000) ───────────────────────────────────
fig4, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
fig4.suptitle('Δ (A_006 − A_000)  —  effect of gust', fontsize=13)
DIFF = dict(color='darkorange', lw=1.5, alpha=0.85)
axes[0,0].plot(t, d6['h']     - d0['h'],                     **DIFF); fmt(axes[0,0], 'Δh [m]',    'Heave')
axes[0,1].plot(t, d6['h_dot'] - d0['h_dot'],                 **DIFF); fmt(axes[0,1], 'Δḣ [m/s]',  'Heave velocity')
axes[0,2].plot(t, d6['W_g']   - d0['W_g'],                   **DIFF); fmt(axes[0,2], 'ΔW [m/s]',  'Gust (A_006 gust input)')
axes[1,0].plot(t, np.rad2deg(d6['alpha'] - d0['alpha']),      **DIFF); fmt(axes[1,0], 'Δα [°]',    'Pitch angle')
axes[1,1].plot(t, d6['C_L']   - d0['C_L'],                   **DIFF); fmt(axes[1,1], 'ΔC_L',      'Lift difference')
axes[1,2].plot(t, d6['C_M']   - d0['C_M'],                   **DIFF); fmt(axes[1,2], 'ΔC_M',      'Moment difference')
fig4.tight_layout()
fig4.savefig(OUT_DIR / 'fig4_diff.png', dpi=150); plt.close(fig4)
print("[OK] fig4_diff.png")

# ── print summary stats ───────────────────────────────────────────────────────
print(f"\n── Summary ─────────────────────────────────────────────")
print(f"{'':12s}  {'A_000':>10s}  {'A_006':>10s}  {'Δmax':>10s}")
for col, label in [('C_L','C_L'), ('C_M','C_M'), ('h','h [m]'),
                   ('alpha','α [rad]'), ('W_g','W_g [m/s]')]:
    v0 = d0[col].values; v6 = d6[col].values
    print(f"  {label:<10s}  max={v0.max():7.4f}  max={v6.max():7.4f}  Δmax={v6.max()-v0.max():+7.4f}")

print(f"\nDone. Results in: {OUT_DIR}")
