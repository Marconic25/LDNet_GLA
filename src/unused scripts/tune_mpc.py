#!/usr/bin/env python3
"""
MPC tuning script: test diverse combinazioni di Q e R.
Analizza trade-off tra riduzione dello stato e sforzo di controllo.
"""
import numpy as np
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from structural.smd import get_space_state_matrices
from control.mpc import MPCController, run_mpc_simulation
from aerodynamics.model import LDNetModel as AeroModel

# Test parameters (fixed)
U_INF = 80.0
T_END = 5.0
DT = 0.01
MPC_HORIZON = 10

# Load structural model
print("Loading structural state-space matrices...")
A_s, B_s, C_s, D_s = get_space_state_matrices()

# Load aerodynamic model
print("Loading aerodynamic model...")
models_dir = Path(__file__).parent.parent / 'models'
if not models_dir.exists():
    models_dir = Path(__file__).parent.parent.parent / 'models'
aero_model = AeroModel(str(models_dir))

# EKF noise covariances
Q_noise = np.eye(4) * 0.01
R_noise = np.eye(2) * 0.1

# Baseline (run once)
print(f"\n{'='*70}")
print(f"Running BASELINE simulation (no control)...")
print(f"{'='*70}")
result_baseline = run_mpc_simulation(U_INF, T_END, DT, aero_model, None, A_s, B_s,
                                      use_ekf=False)

h_amp_baseline = (result_baseline['h'].max() - result_baseline['h'].min()) / 2
a_amp_baseline = (result_baseline['a'].max() - result_baseline['a'].min()) / 2
print(f"Baseline h amplitude: {h_amp_baseline:.6f} m")
print(f"Baseline a amplitude: {a_amp_baseline:.6f} rad")

# Tuning grid
# Q_weight: penalize state (heave + pitch)
# R_weight: penalize control
# Q_acc_weight: penalize accelerations (structural loads)
Q_weights = [10.0, 50.0, 100.0, 200.0, 500.0]
R_weights = [0.001, 0.005, 0.01, 0.05, 0.1]
Q_acc_weights = [0.0, 5.0, 10.0, 20.0]  # 0.0 = no acceleration penalty (baseline MPC)

results = []

print(f"\n{'='*70}")
print(f"TUNING GRID: {len(Q_weights)} × {len(R_weights)} × {len(Q_acc_weights)} = {len(Q_weights)*len(R_weights)*len(Q_acc_weights)} configurations")
print(f"{'='*70}\n")

for q_val in Q_weights:
    for r_val in R_weights:
        for q_acc_val in Q_acc_weights:
            Q_mpc = np.diag([q_val, 1.0, q_val, 1.0])
            R_mpc = r_val
            Q_acc_mpc = np.array([q_acc_val, q_acc_val])

        print(f"Testing Q={q_val:.1f}, R={r_val:.4f}, Q_acc={q_acc_val:.1f}...", end=" ", flush=True)

        # Initialize MPC
        mpc = MPCController(aero_model, U_INF, DT, Q_mpc, R_mpc, P=Q_mpc, N=MPC_HORIZON, delta_max=20.0, Q_acc=Q_acc_mpc)

        # Run simulation
        result = run_mpc_simulation(U_INF, T_END, DT, aero_model, mpc, A_s, B_s,
                                    use_ekf=True, Q_noise=Q_noise, R_noise=R_noise)

        # Compute metrics
        h_amp = (result['h'].max() - result['h'].min()) / 2
        a_amp = (result['a'].max() - result['a'].min()) / 2
        h_reduction = (h_amp_baseline - h_amp) / h_amp_baseline * 100
        a_reduction = (a_amp_baseline - a_amp) / a_amp_baseline * 100

        delta_min = result['delta'].min()
        delta_max = result['delta'].max()
        delta_range = delta_max - delta_min

        # Control effort (RMS of delta)
        delta_rms = np.sqrt(np.mean(result['delta']**2))

        # Average absolute delta (active control)
        delta_mean_abs = np.mean(np.abs(result['delta']))

        config = {
            'Q': q_val,
            'R': r_val,
            'Q_acc': q_acc_val,
            'h_amplitude': h_amp,
            'a_amplitude': a_amp,
            'h_reduction_pct': h_reduction,
            'a_reduction_pct': a_reduction,
            'delta_min': delta_min,
            'delta_max': delta_max,
            'delta_range': delta_range,
            'delta_rms': delta_rms,
            'delta_mean_abs': delta_mean_abs,
        }
        results.append(config)

        print(f"h_red={h_reduction:+6.1f}%, a_red={a_reduction:+6.1f}%, h_ddot+a_ddot_pct_reduction={((h_reduction+a_reduction)/2):+6.1f}%, Δ_rms={delta_rms:.4f}°")

# Analyze results
print(f"\n{'='*70}")
print(f"TUNING RESULTS SUMMARY")
print(f"{'='*70}\n")

# Sort by h_reduction (prioritize heave control)
results_by_h = sorted(results, key=lambda x: x['h_reduction_pct'], reverse=True)

print("Top 10 configurations (by heave reduction):")
print(f"{'Q':<8} {'R':<10} {'Q_acc':<8} {'h_red%':<10} {'a_red%':<10} {'Δ_rms°':<10} {'Δ_range°':<10}")
print("-" * 75)
for i, r in enumerate(results_by_h[:10]):
    print(f"{r['Q']:<8.1f} {r['R']:<10.4f} {r['Q_acc']:<8.1f} {r['h_reduction_pct']:<10.1f} {r['a_reduction_pct']:<10.1f} {r['delta_rms']:<10.4f} {r['delta_range']:<10.2f}")

# Find Pareto-optimal (no NaN in any metric)
print(f"\n{'='*70}")
print(f"PARETO ANALYSIS: Best trade-offs")
print(f"{'='*70}\n")

# Best overall control (highest average reduction)
avg_reduction = [(r['h_reduction_pct'] + r['a_reduction_pct'])/2 for r in results]
best_overall_idx = np.argmax(avg_reduction)
best_overall = results[best_overall_idx]

print("Best overall control (max avg reduction):")
print(f"  Q={best_overall['Q']}, R={best_overall['R']}, Q_acc={best_overall['Q_acc']}")
print(f"  h reduction: {best_overall['h_reduction_pct']:+.1f}%")
print(f"  a reduction: {best_overall['a_reduction_pct']:+.1f}%")
print(f"  δ RMS: {best_overall['delta_rms']:.4f}°, range: {best_overall['delta_range']:.2f}°")

# Most conservative (minimal control effort)
best_conservative_idx = np.argmin([r['delta_rms'] for r in results])
best_conservative = results[best_conservative_idx]

print(f"\nMost conservative (min control effort):")
print(f"  Q={best_conservative['Q']}, R={best_conservative['R']}, Q_acc={best_conservative['Q_acc']}")
print(f"  h reduction: {best_conservative['h_reduction_pct']:+.1f}%")
print(f"  a reduction: {best_conservative['a_reduction_pct']:+.1f}%")
print(f"  δ RMS: {best_conservative['delta_rms']:.4f}°, range: {best_conservative['delta_range']:.2f}°")

# Best control without saturation
non_saturated = [r for r in results if abs(r['delta_max']) < 20.0 and abs(r['delta_min']) < 20.0]
if non_saturated:
    best_ns_idx = np.argmax([r['h_reduction_pct'] for r in non_saturated])
    best_ns = non_saturated[best_ns_idx]
    print(f"\nBest control (no saturation):")
    print(f"  Q={best_ns['Q']}, R={best_ns['R']}, Q_acc={best_ns['Q_acc']}")
    print(f"  h reduction: {best_ns['h_reduction_pct']:+.1f}%")
    print(f"  a reduction: {best_ns['a_reduction_pct']:+.1f}%")
    print(f"  δ RMS: {best_ns['delta_rms']:.4f}°, range: {best_ns['delta_range']:.2f}°")

# Save detailed results to JSON
results_file = Path(__file__).parent.parent / 'mpc_tuning_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[OK] Detailed results saved to: {results_file}")

# Save summary as CSV for Excel/plotting
csv_file = Path(__file__).parent.parent / 'mpc_tuning_summary.csv'
with open(csv_file, 'w') as f:
    f.write("Q,R,Q_acc,h_amplitude_m,a_amplitude_rad,h_reduction_pct,a_reduction_pct,delta_min_deg,delta_max_deg,delta_rms_deg,delta_mean_abs_deg\n")
    for r in results:
        f.write(f"{r['Q']},{r['R']},{r['Q_acc']},{r['h_amplitude']:.6f},{r['a_amplitude']:.6f},{r['h_reduction_pct']:.1f},{r['a_reduction_pct']:.1f},{r['delta_min']:.2f},{r['delta_max']:.2f},{r['delta_rms']:.4f},{r['delta_mean_abs']:.4f}\n")
print(f"[OK] CSV summary saved to: {csv_file}")
