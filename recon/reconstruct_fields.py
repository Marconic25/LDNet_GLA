#!/usr/bin/env python3
"""Reconstruct LDNet fields for a sim and dump ROM npy (cluster-side, TF container).

Loads a trained field-LDNet (NNdyn/NNrec weights + config.json), evolves the latent state
from the sim's input signals, queries NNrec over the FULL mesh grid, denormalizes, and saves
the reconstruction so the (TF-free) viz tool can compare it to the FOM fields.

  rom_<name>.npy   [T, Npts, 3]  (vx, vy, p)   LDNet reconstruction
  fom_<name>.npy   [T, Npts, 3]                ground truth from the h5 (resampled grid)
  rom_points.npy   [Npts, 2]                   grid
  rom_times.npy    [T]

Usage (inside the TF container):
  python reconstruct_fields.py --model models/latent_10 --data data/FIELDS_test.h5 \
      --index 0 --name sim_A_025 --out recon_fields/rom
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import utils                                   # noqa: E402
from train_fields import build_networks, make_ldnet, FIELD_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir with config.json + weights")
    ap.add_argument("--data", required=True, help="FIELDS h5 containing the sim")
    ap.add_argument("--index", type=int, default=0, help="sim index within the h5")
    ap.add_argument("--name", default="sim")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    model = Path(args.model)
    with open(model / "config.json") as f:
        cfg = json.load(f)
    problem, norm, nls = cfg["problem"], cfg["normalization"], cfg["num_latent_states"]
    output_nl = cfg.get("output_nl", "cubic")
    dt_base = norm["time"]["time_constant"]; dt = dt_base

    NNdyn, NNrec = build_networks(nls, problem, dt, dt_base)
    # build by calling once, then load weights
    _ = NNdyn(tf.zeros((1, nls + 1 + 6), tf.float64))
    _ = NNrec(tf.zeros((1, 1, 1, nls + 6 + 2), tf.float64))
    NNdyn.load_weights(str(model / "NNdyn_weights.weights.h5"))
    NNrec.load_weights(str(model / "NNrec_weights.weights.h5"))
    ldnet, _ = make_ldnet(NNdyn, NNrec, nls, problem, dt, dt_base, output_nl=output_nl)

    ds = utils.load_gla_h5(args.data)
    # keep only the requested sim
    for k in ["input_parameters", "input_signals", "output_signals", "output_fields"]:
        ds[k] = ds[k][args.index:args.index + 1]
    if ds.get("sim_families") is not None:
        ds["sim_families"] = ds["sim_families"][args.index:args.index + 1]
    points = ds["points"].copy()
    times = ds["times"].copy()
    utils.process_dataset(ds, problem, norm, dt=None)   # full grid

    out_n = ldnet(ds)
    fmin = np.array([norm["output_fields"][n]["min"] for n in FIELD_NAMES])
    fmax = np.array([norm["output_fields"][n]["max"] for n in FIELD_NAMES])
    rom = utils.normalize_back(out_n, fmin, fmax, axis=3).numpy()[0]   # [T,Npts,3]
    fom = utils.normalize_back(ds["output_fields"], fmin, fmax, axis=3).numpy()[0]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"rom_{args.name}.npy", rom.astype(np.float32))
    np.save(out / f"fom_{args.name}.npy", fom.astype(np.float32))
    np.save(out / "rom_points.npy", points.astype(np.float32))
    np.save(out / "rom_times.npy", times.astype(np.float32))
    err = np.sqrt(np.mean((rom - fom) ** 2)) / (fom.max() - fom.min())
    print(f"saved ROM {rom.shape} -> {out}  combined NRMSE {err:.3e}")


if __name__ == "__main__":
    main()
