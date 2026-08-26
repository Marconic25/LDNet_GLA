#!/usr/bin/env python3
"""DMD diagnostic (DYNAMIC_CONTRIBUTION_LITERATURE_NOTES.md recommendation A):
does the D-RES near-flap separation event correspond to a distinct, low-rank
DYNAMICAL MODE (its own growth/oscillation rate, its own localized spatial
support), separable from the smooth background dynamics via linear modal
analysis -- or is its variance smeared across many modes with no scale
separation? Pure offline SVD/eigendecomposition on the FOM dump already on
disk. Zero training, zero cluster compute -- exactly the class of cheap
diagnostic that reframed the whole D-RES investigation once already
(decomp_stall.py).

Method: exact DMD (Tu et al. 2014) on the near-airfoil region's vx
FLUCTUATION (time-mean subtracted, matching this project's established
static/dynamic convention) for sim_Cc_060. For each mode, compute:
  - continuous-time eigenvalue omega = log(mu)/dt -> growth rate + frequency
  - flap-concentration: fraction of the mode's spatial L2 energy inside the
    near-flap band (decomp_stall.py's is_flap_near mask) vs the whole near
    region
  - event-concentration: fraction of the mode's reconstructed TEMPORAL
    envelope energy inside the confirmed event window (t=0.5-0.8s,
    MEANSPLIT_NOTES.md's STALL/SEPARATION HYPOTHESIS) vs the full trajectory
A mode that is high on BOTH concentration metrics, with a non-trivial
oscillation/growth rate, and explains a meaningful share of near-flap
variance is the falsifiable signature this diagnostic is looking for.
"""
import numpy as np
from pathlib import Path

AN = Path(__file__).resolve().parent
RES = AN.parent / "results"
D = RES / "ms_coral_o10_s0_rom_cc060"

region_label = np.load(AN / "region_labels.npy")
air = np.load(AN / "airfoil_nodes.npy")
N = len(region_label)
near = region_label == 0

fom = np.load(D / "fom_sim_Cc_060.npy").astype(np.float64)   # [T,N,3]
pts = np.load(D / "rom_points.npy").astype(np.float64)
times = np.load(D / "rom_times.npy").astype(np.float64)
t_rel = times - times[0]
T = fom.shape[0]
dt = float(np.mean(np.diff(t_rel)))

# near-flap sub-mask within the near region (same convention as decomp_stall.py)
d2 = ((pts[:, None, :] - pts[air][None, :, :]) ** 2).sum(-1)
nn = d2.argmin(1)
is_flap_near_full = near & (nn >= 292)

near_idx = np.where(near)[0]
flap_within_near = is_flap_near_full[near_idx]   # boolean mask INTO near_idx
print(f"near region: {len(near_idx)} nodes, of which {flap_within_near.sum()} "
      f"are near-flap ({100*flap_within_near.mean():.1f}%)")

vx_near = fom[:, near_idx, 0]                      # (T, n_near)
vx_fluct = vx_near - vx_near.mean(axis=0, keepdims=True)   # time-mean subtracted
X = vx_fluct.T                                     # (n_near, T) snapshot matrix

# --- exact DMD ---
X1, X2 = X[:, :-1], X[:, 1:]
r = 20   # truncation rank -- generous enough to capture the event, small enough to interpret
U, S, Vh = np.linalg.svd(X1, full_matrices=False)
U_r, S_r, V_r = U[:, :r], S[:r], Vh[:r, :].conj().T
energy_captured = (S[:r] ** 2).sum() / (S ** 2).sum()
print(f"DMD rank r={r} captures {100*energy_captured:.2f}% of near-region fluctuation energy")

A_tilde = U_r.conj().T @ X2 @ V_r @ np.diag(1.0 / S_r)
mu, W = np.linalg.eig(A_tilde)
Phi = X2 @ V_r @ np.diag(1.0 / S_r) @ W             # (n_near, r) DMD modes
omega = np.log(mu) / dt                             # continuous-time eigenvalues

# DMD amplitudes via the initial snapshot
b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

# --- per-mode diagnostics ---
event_mask = (t_rel >= 0.45) & (t_rel <= 0.85)   # confirmed event window +/- margin
rows = []
for j in range(r):
    mode_energy_flap = np.sum(np.abs(Phi[flap_within_near, j]) ** 2)
    mode_energy_total = np.sum(np.abs(Phi[:, j]) ** 2)
    flap_conc = mode_energy_flap / (mode_energy_total + 1e-30)

    time_series = b[j] * np.exp(omega[j] * t_rel)     # (T,) complex temporal coefficient
    env = np.abs(time_series)
    event_energy = np.sum(env[event_mask] ** 2)
    total_energy = np.sum(env ** 2)
    event_conc = event_energy / (total_energy + 1e-30)

    growth = float(np.real(omega[j]))
    freq_hz = float(np.imag(omega[j]) / (2 * np.pi))
    mode_variance_share = float(np.abs(b[j]) ** 2 * np.sum(np.abs(Phi[:, j]) ** 2))
    rows.append((j, growth, freq_hz, flap_conc, event_conc, mode_variance_share))

rows.sort(key=lambda r_: -r_[5])   # rank by variance contribution
total_var = sum(r_[5] for r_ in rows)

print(f"\n{'mode':>4s} {'growth[1/s]':>12s} {'freq[Hz]':>9s} {'flap-conc':>10s} "
      f"{'event-conc':>11s} {'var-share':>10s}")
for j, growth, freq_hz, flap_conc, event_conc, var in rows[:12]:
    print(f"{j:4d} {growth:12.3f} {freq_hz:9.3f} {flap_conc:10.3f} "
          f"{event_conc:11.3f} {100*var/total_var:9.2f}%")

# --- summary verdict ---
print("\n=== VERDICT ===")
candidate = [r_ for r_ in rows if r_[3] > 0.5 and r_[4] > 0.5 and r_[5] / total_var > 0.02]
if candidate:
    print(f"{len(candidate)} mode(s) found with BOTH flap-concentration>0.5 AND "
          f"event-concentration>0.5 AND >2% variance share -- a candidate separable "
          f"fast/localized mode exists:")
    for j, growth, freq_hz, flap_conc, event_conc, var in candidate:
        print(f"  mode {j}: growth={growth:.3f}/s freq={freq_hz:.3f}Hz "
              f"flap-conc={flap_conc:.2f} event-conc={event_conc:.2f} "
              f"var-share={100*var/total_var:.2f}%")
else:
    print("NO mode found with both flap-concentration>0.5 and event-concentration>0.5 "
          "carrying >2% variance -- the event's variance does NOT cleanly separate "
          "into a single low-rank localized/transient DMD mode at this rank/threshold.")
