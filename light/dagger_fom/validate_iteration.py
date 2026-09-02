#!/usr/bin/env python3
"""
End-of-iteration validation: given a real-FOM structural_trajectory.csv
(baseline or candidate-controller/candidate-model run), report:
  - oscillation count (sign changes in the flap command's per-window derivative)
  - CLred using the ROM's own open-loop exo as baseline (matches this
    session's earlier verify_clred.py methodology, reused here)
  - optionally, teacher-forcing prediction error against a given LDNet model
    (reused from this session's teacher_force_check.py probe)

Usage:
    python3 validate_iteration.py --csv PATH --w0 30 --tg 0.40 --r 0.0003 \
        [--model MODEL_DIR] [--damult 3] [--label "qadot=0.1"]
"""
import argparse
import csv as _csv
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..'))  # light/
import run as Rn  # noqa: E402


def load_csv(path):
    with open(path) as f:
        rows = list(_csv.reader(f))
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    return dict(t=data[:, 0], h=data[:, 1], hd=data[:, 2], alpha=data[:, 3],
                ad=data[:, 4], Fy=data[:, 5], Mz=data[:, 6],
                W_gust=data[:, 7], delta=data[:, 8])


def count_oscillation(t, delta, tg):
    mask = t - t[0] <= tg + 0.5
    d = delta[mask]
    diffs = np.diff(d)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def teacher_force(csv_path, model_dir):
    """Reused from this session's teacher_force_check.py."""
    from ldnet_aero import LDNetAero
    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    t_raw = data[:, 0]
    raw_dt = float(t_raw[1] - t_raw[0])
    q = Rn.RHO * 0.5 * Rn.U ** 2 * Rn.S

    aero = LDNetAero(model_dir)
    dt_ref = aero._dt_ref
    stride = max(1, round(dt_ref / raw_dt))
    idx = list(range(0, len(data) - stride, stride))

    aero.reset(dt=dt_ref)
    errs = []
    for i in idx:
        h, hd, a, ad = data[i, 1], data[i, 2], data[i, 3], data[i, 4]
        Fy = data[i, 5]
        W, delta = data[i, 7], data[i, 8]
        cl_pred, _ = aero.predict((h, hd, a, ad), delta, W, Rn.U)
        errs.append(cl_pred - Fy / q)
        aero.advance((h, hd, a, ad), delta, W, Rn.U, dt_ref)
    errs = np.array(errs)
    return dict(rmse=float(np.sqrt(np.mean(errs ** 2))),
                max_abs=float(np.max(np.abs(errs))),
                mean=float(np.mean(errs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--w0', type=float, required=True)
    ap.add_argument('--tg', type=float, required=True)
    ap.add_argument('--r', type=float, required=True)
    ap.add_argument('--model', type=str, default=None,
                     help='if given, also run the teacher-forcing accuracy probe')
    ap.add_argument('--label', type=str, default='')
    args = ap.parse_args()

    traj = load_csv(args.csv)
    q = Rn.RHO * 0.5 * Rn.U ** 2 * Rn.S
    CL_fom = traj['Fy'] / q

    OL = Rn.simulate('open', args.w0, args.tg, TEND=3.0)
    t_rom = OL['_t']
    mw_rom = Rn._gust_window(t_rom, args.tg)
    exo = float(np.max(np.abs(OL['CL'][mw_rom] - Rn.CLTRIM)))

    t_fom = traj['t'] - traj['t'][0]
    mw_fom = t_fom <= args.tg + 0.5
    exc_fom = float(np.max(np.abs(CL_fom[mw_fom] - Rn.CLTRIM)))
    clred_fom = (exo - exc_fom) / exo * 100.0 if exo > 1e-12 else 0.0
    flap_max = float(np.max(np.abs(traj['delta'][mw_fom])))
    osc = count_oscillation(traj['t'], traj['delta'], args.tg)

    print(f"=== {args.label or args.csv} ===")
    print(f"  CLred = {clred_fom:+.1f}%   flap_max = {flap_max:.2f} deg   osc_count = {osc}")

    if args.model:
        tf_res = teacher_force(args.csv, args.model)
        print(f"  teacher-force vs {args.model}:")
        print(f"    rmse={tf_res['rmse']:.4f}  max_abs={tf_res['max_abs']:.4f}  mean={tf_res['mean']:.4f}")


if __name__ == '__main__':
    main()
