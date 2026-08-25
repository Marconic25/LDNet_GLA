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
from train_fields import (build_networks, make_ldnet, wall_features,        # noqa: E402
                          signal_rate_channels, flap_loss_weights,
                          FIELD_NAMES, SIGNAL_NAMES, SepStateDynamics)


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
    arch = cfg.get("architecture", {})   # absent in pre-depth-study models -> defaults
    dt_base = norm["time"]["time_constant"]; dt = dt_base

    # Fourier-feature encoding widens the decoder spatial input beyond space dim.
    ff = cfg.get("fourier")
    fourier_B = np.load(model / "fourier_B.npy") if ff else None
    n_extra = problem["space"]["dimension"] - 2
    rec_space_dim = (2 * ff["m"] * len(ff["scales"]) + n_extra) if ff else None

    # decoder architecture: 'mlp' (tanh, default) or 'coral' (shift-modulated SIREN)
    decoder = cfg.get("decoder", "mlp")
    siren = cfg.get("siren") or {}
    dyn_cond = cfg.get("dyn_cond", "concat")
    dyn_sep_state = bool(cfg.get("dyn_sep_state", False))
    sep_cfg = cfg.get("sep_state") or {}

    NNdyn, NNrec = build_networks(nls, problem, dt, dt_base,
                                  dyn_layers=arch.get("dyn_layers", 2),
                                  dyn_width=arch.get("dyn_width", 7),
                                  rec_layers=arch.get("rec_layers", 4),
                                  rec_width=arch.get("rec_width", 24),
                                  rec_space_dim=rec_space_dim, decoder=decoder,
                                  siren_omega0=siren.get("omega0", 30.0),
                                  siren_mod_layers=siren.get("mod_layers", 2),
                                  siren_mod_width=siren.get("mod_width"),
                                  siren_mod_type=siren.get("mod_type", "shift"),
                                  dyn_cond=dyn_cond, dyn_sep_state=dyn_sep_state)
    # build by calling once, then load weights (space dim may include wall features / FF;
    # signal count may include --add-signal-rates' +2 channels -- read from problem
    # (saved in config.json), never hardcoded, to avoid a silent shape mismatch)
    sdim = rec_space_dim if rec_space_dim is not None else problem["space"]["dimension"]
    n_sig = len(problem["input_signals"])
    n_sep = 1 if dyn_sep_state else 0
    if dyn_cond == "cde":
        _ = NNdyn(tf.zeros((1, nls), tf.float64))
    else:
        _ = NNdyn(tf.zeros((1, nls + n_sep + 1 + n_sig), tf.float64))
    _ = NNrec(tf.zeros((1, 1, 1, nls + n_sep + n_sig + sdim), tf.float64))
    NNdyn.load_weights(str(model / "NNdyn_weights.weights.h5"))
    NNrec.load_weights(str(model / "NNrec_weights.weights.h5"))

    sepnet = None
    if dyn_sep_state:
        sepnet = SepStateDynamics(tau1_init=sep_cfg.get("tau1_init", 5.0))
        sepnet.x0(tf.zeros((1, 2), tf.float64))   # force-build sublayers before loading
        sepnet.load_weights(str(model / "sepstate_weights.weights.h5"))
        print(f"sep-state ON: learned tau1={float(sepnet.tau1.numpy()):.3g} steps")

    lw = cfg.get("loss_weight")
    n_weight_cols = lw["n_weight_cols"] if lw else 0
    ldnet, _ = make_ldnet(NNdyn, NNrec, nls, problem, dt, dt_base, output_nl=output_nl,
                          fourier_B=fourier_B, dyn_cond=dyn_cond, n_weight_cols=n_weight_cols,
                          dyn_sep_state=dyn_sep_state, sepnet=sepnet)

    ds = utils.load_gla_h5(args.data)
    # keep only the requested sim
    for k in ["input_parameters", "input_signals", "output_signals", "output_fields"]:
        ds[k] = ds[k][args.index:args.index + 1]
    if ds.get("sim_families") is not None:
        ds["sim_families"] = ds["sim_families"][args.index:args.index + 1]
    points = ds["points"].copy()
    times = ds["times"].copy()
    wf = cfg.get("wall_feats")
    if wf:
        axy = np.load(model / "airfoil_xy.npy")
        ds["points"] = np.concatenate(
            [ds["points"], wall_features(ds["points"], axy, wf["tau"])], axis=1)
    if cfg.get("add_signal_rates"):
        w_idx, d_idx = SIGNAL_NAMES.index("W_gust"), SIGNAL_NAMES.index("delta")
        r = signal_rate_channels(ds["input_signals"], dt_base, w_idx, d_idx)
        ds["input_signals"] = np.concatenate([ds["input_signals"], r], axis=2)
    if dyn_sep_state:
        # Same rate computation/normalization convention as training (main()'s
        # sep_state_norm block) -- indices into the BASE 6-channel SIGNAL_NAMES
        # layout, valid regardless of whether --add-signal-rates already
        # appended extra columns to ds['input_signals'] above.
        sw_idx, sd_idx = SIGNAL_NAMES.index("W_gust"), SIGNAL_NAMES.index("delta")
        raw_r = signal_rate_channels(ds["input_signals"], dt_base, sw_idx, sd_idx)
        rn = sep_cfg["rate_norm"]
        sep_lo = np.array([rn["Wd"]["min"], rn["deltad"]["min"]])
        sep_hi = np.array([rn["Wd"]["max"], rn["deltad"]["max"]])
        ds["sep_rates"] = tf.convert_to_tensor(
            utils.normalize_forw(raw_r, sep_lo, sep_hi, axis=2), tf.float64)
    if lw and lw["mode"] == "flap":
        # loss weighting is training-only (no target to weight against at inference) --
        # this column is appended ONLY so ds['points']' width matches what the saved
        # normalization['space'] ranges expect (computed from the training points array,
        # which had this column appended); the actual values are discarded before NNrec
        # (n_weight_cols strips them in reconstruct_states), so recomputing the real
        # sigma vs. a placeholder is just for clarity/consistency, not correctness.
        fxy = np.load(model / "flap_xy.npy")
        ds["points"] = np.concatenate(
            [ds["points"], flap_loss_weights(ds["points"], fxy, lw["tau"])], axis=1)
    utils.process_dataset(ds, problem, norm, dt=None)   # full grid

    if dyn_sep_state:
        out_n, xtraj = ldnet(ds, return_aux=True)
    else:
        out_n = ldnet(ds)
    fmin = np.array([norm["output_fields"][n]["min"] for n in FIELD_NAMES])
    fmax = np.array([norm["output_fields"][n]["max"] for n in FIELD_NAMES])
    rom = utils.normalize_back(out_n, fmin, fmax, axis=3).numpy()[0]   # [T,Npts,3]
    fom = utils.normalize_back(ds["output_fields"], fmin, fmax, axis=3).numpy()[0]
    if cfg.get("mean_split"):
        # mean-split model: the net predicts fluctuations around the stored train mean.
        # fom is unaffected (raw totals normalize->denormalize round-trip exactly).
        rom = rom + np.load(model / "mean_fields.npy")[None]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"rom_{args.name}.npy", rom.astype(np.float32))
    np.save(out / f"fom_{args.name}.npy", fom.astype(np.float32))
    np.save(out / "rom_points.npy", points.astype(np.float32))
    np.save(out / "rom_times.npy", times.astype(np.float32))
    if dyn_sep_state:
        # X(t) trajectory for the requested sim -- (T,) attachment-state readout
        # (--dyn-sep-state analysis: does X actually dip during the gust+flap event?)
        np.save(out / f"xsep_{args.name}.npy", xtraj.numpy()[0, :, 0].astype(np.float32))
    err = np.sqrt(np.mean((rom - fom) ** 2)) / (fom.max() - fom.min())
    print(f"saved ROM {rom.shape} -> {out}  combined NRMSE {err:.3e}")


if __name__ == "__main__":
    main()
