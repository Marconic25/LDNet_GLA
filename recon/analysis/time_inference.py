#!/usr/bin/env python3
"""Wall-clock timing of the champion field-LDNet's forward pass (cost side of the
cost-vs-accuracy comparison). Reuses reconstruct_fields.py's exact model-loading and
evolve+reconstruct call -- no reimplementation of the inference path -- and adds
perf_counter timing around it. Local (WSL) CPU run, single core, no cluster job.

Usage:
  python3 recon/analysis/time_inference.py \
      --model recon/models/coral_o10_s0_synced/latent_1 \
      --data recon/data/FIELDS_A025.h5 --index 0 --name sim_A_025 --reps 5
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

HERE = Path(__file__).resolve().parent.parent  # recon/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import utils                                   # noqa: E402
from train_fields import (build_networks, make_ldnet, wall_features,        # noqa: E402
                          signal_rate_channels, flap_loss_weights,
                          FIELD_NAMES, SIGNAL_NAMES, SepStateDynamics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--name", default="sim")
    ap.add_argument("--reps", type=int, default=5,
                     help="repeated forward passes after warmup, for a stable timing")
    args = ap.parse_args()

    model = Path(args.model)
    with open(model / "config.json") as f:
        cfg = json.load(f)
    problem, norm, nls = cfg["problem"], cfg["normalization"], cfg["num_latent_states"]
    output_nl = cfg.get("output_nl", "cubic")
    arch = cfg.get("architecture", {})
    dt_base = norm["time"]["time_constant"]; dt = dt_base

    # Fourier-feature encoding widens the decoder spatial input beyond space dim.
    ff = cfg.get("fourier")
    fourier_B = np.load(model / "fourier_B.npy") if ff else None
    n_extra = problem["space"]["dimension"] - 2
    rec_space_dim = (2 * ff["m"] * len(ff["scales"]) + n_extra) if ff else None

    decoder = cfg.get("decoder", "mlp")
    siren = cfg.get("siren") or {}
    dyn_cond = cfg.get("dyn_cond", "concat")
    dyn_sep_state = bool(cfg.get("dyn_sep_state", False))
    sep_cfg = cfg.get("sep_state") or {}
    local_decoder = bool(cfg.get("local_decoder", False))
    local_cfg = cfg.get("local") or {}
    local_ref_xy = local_xy_min = local_xy_max = None
    if local_decoder:
        local_ref_xy = np.load(model / "flap_xy.npy")
        local_xy_min = np.array(norm["space"]["min"][:2])
        local_xy_max = np.array(norm["space"]["max"][:2])

    graph_decoder = bool(cfg.get("graph_decoder", False))
    graph_cfg = cfg.get("graph") or {}
    graph_positions_arr = graph_adj_norm_arr = None
    if graph_decoder:
        graph_positions_arr = np.load(model / "graph_nodes.npy")
        graph_adj_norm_arr = np.load(model / "graph_adj_norm.npy")

    t_build0 = time.perf_counter()
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
                                  dyn_cond=dyn_cond, dyn_sep_state=dyn_sep_state,
                                  local_decoder=local_decoder, local_ref_xy=local_ref_xy,
                                  local_tau=local_cfg.get("tau", 0.3),
                                  local_width=local_cfg.get("width", 16),
                                  local_depth=local_cfg.get("depth", 3),
                                  local_omega0=local_cfg.get("omega0", 30.0),
                                  local_gate_hidden=local_cfg.get("gate_hidden", 8),
                                  local_xy_min=local_xy_min, local_xy_max=local_xy_max,
                                  graph_decoder=graph_decoder,
                                  graph_positions=graph_positions_arr,
                                  graph_adj_norm=graph_adj_norm_arr,
                                  graph_hidden=graph_cfg.get("hidden", 16),
                                  graph_relax_steps=graph_cfg.get("relax_steps", 2))
    sdim = rec_space_dim if rec_space_dim is not None else problem["space"]["dimension"]
    n_sig = len(problem["input_signals"])
    n_sep = 1 if dyn_sep_state else 0
    if dyn_cond == "cde":
        _ = NNdyn(tf.zeros((1, nls), tf.float64))
    else:
        _ = NNdyn(tf.zeros((1, nls + n_sep + 1 + n_sig), tf.float64))
    if not graph_decoder:
        _ = NNrec(tf.zeros((1, 1, 1, nls + n_sep + n_sig + sdim), tf.float64))
    NNdyn.load_weights(str(model / "NNdyn_weights.weights.h5"))
    NNrec.load_weights(str(model / "NNrec_weights.weights.h5"))

    sepnet = None
    if dyn_sep_state:
        sepnet = SepStateDynamics(tau1_init=sep_cfg.get("tau1_init", 5.0))
        sepnet.x0(tf.zeros((1, 2), tf.float64))
        sepnet.load_weights(str(model / "sepstate_weights.weights.h5"))

    lw = cfg.get("loss_weight")
    n_weight_cols = lw["n_weight_cols"] if lw else 0
    ldnet, _ = make_ldnet(NNdyn, NNrec, nls, problem, dt, dt_base, output_nl=output_nl,
                          fourier_B=fourier_B, dyn_cond=dyn_cond, n_weight_cols=n_weight_cols,
                          dyn_sep_state=dyn_sep_state, sepnet=sepnet)
    t_build1 = time.perf_counter()

    ds = utils.load_gla_h5(args.data)
    for k in ["input_parameters", "input_signals", "output_signals", "output_fields"]:
        ds[k] = ds[k][args.index:args.index + 1]
    if ds.get("sim_families") is not None:
        ds["sim_families"] = ds["sim_families"][args.index:args.index + 1]
    n_points = ds["points"].shape[0]
    n_times = ds["times"].shape[1] if ds["times"].ndim > 1 else ds["times"].shape[0]
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
        sw_idx, sd_idx = SIGNAL_NAMES.index("W_gust"), SIGNAL_NAMES.index("delta")
        raw_r = signal_rate_channels(ds["input_signals"], dt_base, sw_idx, sd_idx)
        rn = sep_cfg["rate_norm"]
        sep_lo = np.array([rn["Wd"]["min"], rn["deltad"]["min"]])
        sep_hi = np.array([rn["Wd"]["max"], rn["deltad"]["max"]])
        ds["sep_rates"] = tf.convert_to_tensor(
            utils.normalize_forw(raw_r, sep_lo, sep_hi, axis=2), tf.float64)
    if lw and lw["mode"] == "flap":
        fxy = np.load(model / "flap_xy.npy")
        ds["points"] = np.concatenate(
            [ds["points"], flap_loss_weights(ds["points"], fxy, lw["tau"])], axis=1)
    utils.process_dataset(ds, problem, norm, dt=None)   # full grid, no subsampling

    n_params = sum(int(np.prod(v.shape)) for v in NNdyn.trainable_variables) + \
               sum(int(np.prod(v.shape)) for v in NNrec.trainable_variables)

    # warmup (first call pays TF tracing/XLA cost; excluded from the timed reps)
    t_warm0 = time.perf_counter()
    _ = ldnet(ds)
    t_warm1 = time.perf_counter()

    wall_reps = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out_n = ldnet(ds)
        _ = out_n.numpy()   # force materialization (TF ops can be lazily scheduled)
        t1 = time.perf_counter()
        wall_reps.append(t1 - t0)

    wall_reps = np.array(wall_reps)
    result = {
        "model": str(model),
        "data": args.data,
        "sim": args.name,
        "n_points_full_grid": int(n_points),
        "n_times": int(n_times),
        "n_params_total": int(n_params),
        "build_and_load_s": t_build1 - t_build0,
        "first_call_s_incl_tracing": t_warm1 - t_warm0,
        "steady_state_reps_s": wall_reps.tolist(),
        "steady_state_mean_s": float(wall_reps.mean()),
        "steady_state_std_s": float(wall_reps.std()),
        "per_point_per_time_us": float(wall_reps.mean() / (n_points * n_times) * 1e6),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
