#!/usr/bin/env python3
"""Replay validation: drive LDNetAero with recorded sim_A_025_test inputs and
compare predicted F_y/M_z against the reference structural_trajectory.csv.

Iteration log lives in clean/REPLAY_LOG.md. Reports NRMSE, max error, first
50N-breach step, and z_norm at t=0,0.5,1,2s.
"""
import numpy as np, csv, sys
from pathlib import Path
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ldnet_aero import LDNetAero

MODEL_DIR = HERE / 'models' / 'latent_10'
CSV = Path('/work/u10677113/NACA2312/dataset_v5/sim_A_025_test/structural_trajectory.csv')
U_INF = 80.0
Q_DYN = 0.5 * 1.225 * U_INF**2 * 0.05   # 196.0 ; F = C * q_dyn

aero = LDNetAero(str(MODEL_DIR))
print(f'num_latent_states = {aero._num_z}')
dt_ref = aero._dt_ref
print(f'dt_ref (training) = {dt_ref}')

with open(CSV) as f:
    rows = list(csv.reader(f))
data = np.array([[float(v) for v in r] for r in rows[1:]])
# cols: t,h,hd,alpha,ad,Fy,Mz,W_gust,delta
t_raw = data[:,0]
raw_dt = float(t_raw[1] - t_raw[0])
stride = max(1, round(dt_ref / raw_dt))
idx = list(range(0, len(data), stride))
print(f'raw_dt={raw_dt:.3e}  stride={stride}  n_steps={len(idx)}  eff_dt={stride*raw_dt:.4e}')

aero.reset(dt=dt_ref)           # z = 0 at t=0 (baseline)
print(f'z_norm after reset = {np.linalg.norm(aero._z):.4f}')

t_a, Fy_p, Mz_p, Fy_r, Mz_r, zn = [], [], [], [], [], []
for i in idx:
    h, hd, a, ad = data[i,1], data[i,2], data[i,3], data[i,4]
    Fy, Mz, W, delta = data[i,5], data[i,6], data[i,7], data[i,8]
    state = np.array([h, hd, a, ad])
    C_L, C_M = aero.predict(state, delta, W, U_INF)   # read-only on z
    Fy_p.append(C_L * Q_DYN); Mz_p.append(C_M * Q_DYN)
    Fy_r.append(Fy); Mz_r.append(Mz)
    t_a.append(t_raw[i]); zn.append(np.linalg.norm(aero._z))
    aero.advance(state, delta, W, U_INF, dt_ref)      # step z forward

t_a = np.array(t_a); Fy_p=np.array(Fy_p); Mz_p=np.array(Mz_p)
Fy_r=np.array(Fy_r); Mz_r=np.array(Mz_r); zn=np.array(zn)

def nrmse(p, r):
    rng = r.max() - r.min()
    return np.sqrt(np.mean((p-r)**2)) / rng if rng > 0 else np.nan

nF, nM = nrmse(Fy_p, Fy_r), nrmse(Mz_p, Mz_r)
err = np.abs(Fy_p - Fy_r)
imax = int(np.argmax(err))
breach = np.where(err > 50.0)[0]
t_breach = (f'{t_a[breach[0]]:.2f}s' if len(breach) else 'never')

def zat(tt):
    j = int(np.argmin(np.abs(t_a - tt)))
    return round(float(zn[j]), 3)

print()
print(f'NRMSE F_y: {nF:.3f}   NRMSE M_z: {nM:.3f}')
print(f'Max |F_y_err|: {err[imax]:.1f} N   at t={t_a[imax]:.2f}s')
print(f'First step where |F_y_err| > 50N: {t_breach}')
print(f'z_norm at t=0, t=0.5s, t=1s, t=2s: [{zat(0)}, {zat(0.5)}, {zat(1)}, {zat(2)}]')
print(f'(z_norm max over run: {zn.max():.3f}  final: {zn[-1]:.3f})')

if HAVE_PLT:
    (HERE/'results').mkdir(exist_ok=True)
    fig, ax = plt.subplots(2,1, figsize=(10,7), sharex=True)
    ax[0].plot(t_a, Fy_r, 'steelblue', lw=1.3, label='ref F_y')
    ax[0].plot(t_a, Fy_p, 'crimson', lw=1.0, ls='--', label='LDNet F_y')
    ax[0].set_ylabel('F_y [N]'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f'replay sim_A_025_test  NRMSE_Fy={nF:.3f} NRMSE_Mz={nM:.3f}')
    ax[1].plot(t_a, err, 'k', lw=1.0); ax[1].axhline(50, color='r', ls=':')
    ax[1].set_ylabel('|F_y err| [N]'); ax[1].set_xlabel('t [s]'); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE/'results'/'replay_Fy.png', dpi=130)
    print('saved results/replay_Fy.png')
else:
    print('(matplotlib unavailable - skipped plot)')
