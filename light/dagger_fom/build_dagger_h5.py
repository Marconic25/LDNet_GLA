#!/usr/bin/env python3
"""
Build a standalone, closed-loop-only HDF5 fine-tune dataset from collected
real-FOM structural_trajectory.csv files, in the exact format
src/sensitivity_latent_rollout.py (via utils.load_gla_h5) expects:

    points            (1, 2)          fixed eval point [[0, 0]] (scalar CL/CM)
    times             (T,)            uniform grid, dt = model's dt_ref
    input_parameters  (N, 1)          U_inf = 80.0 for every trajectory
    input_signals     (N, T, 6)       [h, hd, alpha, ad, delta, W_gust]
    output_signals    (N, T, 1, 2)    [Fy, Mz]
    output_fields     (N, T, 1, 2)    zeros (unused, matches existing data)
    sim_families      (N,)            string tags (informational only)

Does NOT touch data/preprocess_GLA.py or data/GLA_*.h5 — fully self-contained
under light/dagger_fom/. All trajectories are resampled onto a COMMON time
grid (write_h5's np.stack requires identical row count across trajectories;
our collected cells have different natural durations T_show=Tg+0.5) via
linear interpolation of the raw (~2.8e-4s) CSV samples.

Usage:
    python3 build_dagger_h5.py --iter 1 \
        --cell 30 0.40 /path/to/W30_Tg0.40/structural_trajectory.csv train \
        --cell 30 0.30 /path/to/W30_Tg0.30/structural_trajectory.csv train \
        --cell 30 0.50 /path/to/W30_Tg0.50/structural_trajectory.csv valid
"""
import argparse
import csv as _csv
import os

import h5py
import numpy as np

DT_REF = 0.002   # must match config.json's normalization/time_constant
U_INF = 80.0


def load_csv(path):
    with open(path) as f:
        rows = list(_csv.reader(f))
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    # columns: t,h,hd,alpha,ad,Fy,Mz,W_gust,delta
    return dict(t=data[:, 0], h=data[:, 1], hd=data[:, 2], alpha=data[:, 3],
                ad=data[:, 4], Fy=data[:, 5], Mz=data[:, 6],
                W_gust=data[:, 7], delta=data[:, 8])


def resample(traj, t_grid):
    t0 = traj['t'][0]
    t_rel = traj['t'] - t0
    out = {}
    for k in ['h', 'hd', 'alpha', 'ad', 'delta', 'W_gust', 'Fy', 'Mz']:
        out[k] = np.interp(t_grid, t_rel, traj[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iter', type=int, required=True, help='DAgger iteration number')
    ap.add_argument('--cell', action='append', nargs=4,
                     metavar=('W0', 'TG', 'CSV', 'SPLIT'),
                     required=True,
                     help='repeatable: W0 Tg csv_path {train,valid}')
    ap.add_argument('--t-common', type=float, default=None,
                     help='common trajectory length [s] (default: min over '
                          'cells of Tg+0.5, i.e. the shortest collected cell)')
    ap.add_argument('--out-dir', type=str, default=None,
                     help='default: light/dagger_fom/data/iterN')
    args = ap.parse_args()

    _THIS = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(_THIS, 'data', f'iter{args.iter}')
    os.makedirs(out_dir, exist_ok=True)

    cells = [(float(w0), float(tg), csv, split) for w0, tg, csv, split in args.cell]
    t_common = args.t_common or min(tg + 0.5 for _, tg, _, _ in cells)
    t_grid = np.arange(0.0, t_common + DT_REF * 0.5, DT_REF)
    T = len(t_grid)
    print(f"iter{args.iter}: {len(cells)} cells, t_common={t_common:.3f}s, T={T} samples @ dt={DT_REF}s")

    by_split = {'train': [], 'valid': []}
    for w0, tg, csv_path, split in cells:
        traj = load_csv(csv_path)
        r = resample(traj, t_grid)
        input_signals = np.stack([r['h'], r['hd'], r['alpha'], r['ad'],
                                   r['delta'], r['W_gust']], axis=-1)   # (T,6)
        output_signals = np.stack([r['Fy'], r['Mz']], axis=-1)[:, None, :]  # (T,1,2)
        by_split[split].append(dict(W0=w0, Tg=tg, input_signals=input_signals,
                                     output_signals=output_signals))
        print(f"  W0={w0:g} Tg={tg:.2f} -> {split}  ({csv_path})")

    for split, entries in by_split.items():
        if not entries:
            continue
        N = len(entries)
        input_signals = np.stack([e['input_signals'] for e in entries])      # (N,T,6)
        output_signals = np.stack([e['output_signals'] for e in entries])    # (N,T,1,2)
        output_fields = np.zeros_like(output_signals)                        # (N,T,1,2)
        input_parameters = np.full((N, 1), U_INF)
        points = np.array([[0.0, 0.0]])
        sim_families = np.array([f"D_W{e['W0']:g}Tg{e['Tg']:.2f}".encode()
                                  for e in entries])

        out_path = os.path.join(out_dir, f'GLA_{split}.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('points', data=points)
            f.create_dataset('times', data=t_grid)
            f.create_dataset('input_parameters', data=input_parameters)
            f.create_dataset('input_signals', data=input_signals)
            f.create_dataset('output_signals', data=output_signals)
            f.create_dataset('output_fields', data=output_fields)
            f.create_dataset('sim_families', data=sim_families)
        print(f"wrote {out_path}  (N={N}, T={T})")


if __name__ == '__main__':
    main()
