#!/usr/bin/env python3
"""Latent-state / input-selection sensitivity sweep for LDNet on the GLA dataset.

Trains an LDNet reconstructing the sectional loads (F_y, M_z) in teacher-forced
(open-loop) mode for each value in LATENT_SWEEP and each input configuration, evaluates
on the test set (NRMSE + Pearson dissimilarity, per output and combined), and saves per
run under <RESULTS_ROOT>/in{INPUT_SET}/:
  - latent_N/{NN*_weights.h5, config.json, metrics.json}
  - latent_N/{loss.png, loss_history.npz}      training loss history
  - latent_N/{timeseries.png, traces.npz}      F_y/M_z FOM-vs-ROM for families A/B/Cc
  - summary/{nrmse_rho_vs_latent.png, all_metrics.json}

Configuration via environment variables:
  INPUT_SET    = 2 | 4 | 6   (default 6; see INPUT_PRESETS)
  LATENT_SWEEP = comma list  (default "1,3,5,10")
  DATA_OVERRIDE, RESULTS_OVERRIDE = data / results roots
  EXPORT_ONLY  = 1           reload weights and re-dump metrics/traces without training
Thesis-quality figures are produced separately by light/plots_ldnet_accuracy.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

# Add src/ to path so utils/optimization are importable when run from any dir
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
import utils
import optimization

# ---------------------------------------------------------------
# Paths / run configuration
import os
DATA_DIR = Path(os.environ.get("DATA_OVERRIDE", "/work/u10677113/LDNet_GLA/data"))

# Input-signal configuration for the number-of-inputs study.
# CANON_SIGNALS is the fixed column order of `input_signals` in the GLA_*.h5 files;
# each preset selects a subset of those columns (see select_input_columns()).
CANON_SIGNALS = ["h", "hd", "a", "ad", "delta", "W_gust"]
INPUT_PRESETS = {
    "2": ["delta", "W_gust"],                         # gust + flap only
    "4": ["h", "a", "delta", "W_gust"],               # + positions, no rates (no hd, ad)
    "6": ["h", "hd", "a", "ad", "delta", "W_gust"],   # full structural state
}
INPUT_SET   = os.environ.get("INPUT_SET", "6")
SEL_SIGNALS = INPUT_PRESETS[INPUT_SET]
SEL_COLS    = [CANON_SIGNALS.index(n) for n in SEL_SIGNALS]

# Outputs are organised per input set: <root>/in{2,4,6}/latent_{d}/ and .../summary/
RESULTS_ROOT = Path(os.environ.get("RESULTS_OVERRIDE", "results/sensitivity"))
RESULTS_DIR  = RESULTS_ROOT / f"in{INPUT_SET}"

# EXPORT_ONLY=1 reloads saved weights and re-dumps traces/metrics without retraining
# (used to add thesis artifacts to an already-trained sweep).
EXPORT_ONLY = os.environ.get("EXPORT_ONLY", "0") == "1"

# ---------------------------------------------------------------
# Hyperparameters
LATENT_SWEEP    = [int(x) for x in os.environ.get("LATENT_SWEEP", "1,3,5,10").split(",")]
dt              = 0.002
dt_base         = 0.002
num_epochs_Adam = 200
num_epochs_BFGS = 20000
# Latent leak (1-lambda damped update). 0 = original undamped LDNet used for the
# teacher-forced selection sweep. Set LAMBDA_DAMP>0 (e.g. via EXPORT_ONLY on the
# deployed rollout model) to evaluate a damped model faithfully:
#   state <- state + dt/dt_base * (NNdyn(inp) - LEAK*state)
LEAK            = float(os.environ.get("LAMBDA_DAMP", "0.0"))
# Tikhonov (L2) weight regularization, as in the reference LDNet implementation
# (Regazzoni et al.: per-layer mean of squared kernels, averaged over layers,
# biases excluded; validation loss is monitored WITHOUT this term). 0 disables.
ALPHA_REG       = float(os.environ.get("ALPHA_REG", "0.0"))
# Per-trajectory inverse-variance weighting of the train MSE (counteracts the
# global-normalization squeeze that starves low-amplitude families, e.g. the
# flap-only family B). Weights are normalized to mean 1 and capped. 0 disables.
FAMILY_W        = os.environ.get("FAMILY_W", "0") == "1"
FAMILY_W_CAP    = float(os.environ.get("FAMILY_W_CAP", "20.0"))
# Early stopping on the validation loss: keep the weights at the validation
# minimum and stop once it has not improved for PATIENCE recorded samples
# (validation is logged every 10 iterations, so PATIENCE=50 -> 500 iterations).
PATIENCE        = int(os.environ.get("PATIENCE", "50"))

# ---------------------------------------------------------------
# Problem definition (input_signals selected by INPUT_SET; see INPUT_PRESETS)
problem = {
    "space": {"dimension": 2},
    "input_parameters": [{"name": "U_inf"}],
    "input_signals": [{"name": n} for n in SEL_SIGNALS],
    "output_signals": [
        {"name": "F_y"},
        {"name": "M_z"},
    ],
    "output_fields": [
        {"name": "ux"},
        {"name": "uy"},
    ],
}


def select_input_columns(dataset):
    """Subset the loaded input_signals columns to the SEL_SIGNALS chosen for this run.

    The GLA_*.h5 files always store all CANON_SIGNALS columns; the {2,4}-input
    studies keep only the selected ones, in the same order as problem['input_signals'].
    """
    if dataset.get('input_signals') is not None:
        dataset['input_signals'] = dataset['input_signals'][:, :, SEL_COLS]
    return dataset

normalization = {
    'space': {'min': [0, 0], 'max': [1, 1]},
    'time':  {'time_constant': dt_base},
    'input_parameters': {
        'U_inf': {'min': 0, 'max': 120},
    },
    'input_signals': {
        'h':      {'min': -0.029, 'max':  0.009},
        'hd':     {'min': -0.54,  'max':  0.62 },
        'a':      {'min': -0.014, 'max':  0.011},
        'ad':     {'min': -0.93,  'max':  0.93 },
        'delta':  {'min': -14.09, 'max': 14.55 },
        'W_gust': {'min':   0.0,  'max': 47.81 },
    },
    'output_signals': {
        'F_y': {'min': -18.65, 'max': 547.29},
        'M_z': {'min': -44.99, 'max':  42.24},
    },
    'output_fields': {
        'ux': {'min': -50,  'max': 150},
        'uy': {'min': -100, 'max': 100},
    },
}

# ---------------------------------------------------------------
# Model builder

def build_networks(num_latent_states):
    n_inp = num_latent_states + len(problem['input_parameters']) + len(problem['input_signals'])
    NNdyn = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation=tf.nn.tanh, input_shape=(n_inp,)),
        tf.keras.layers.Dense(7, activation=tf.nn.tanh),
        tf.keras.layers.Dense(num_latent_states),
    ])
    n_rec = num_latent_states + len(problem['input_signals']) + problem['space']['dimension']
    NNrec = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation=tf.nn.tanh, input_shape=(None, None, n_rec)),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(len(problem['output_signals'])),
    ])
    return NNdyn, NNrec

# ---------------------------------------------------------------
# LDNet forward pass

def make_ldnet(NNdyn, NNrec, num_latent_states):
    weight_direction = 0
    epsilon = 1e-4
    alpha = 0.05

    def evolve_dynamics(dataset):
        num_samples = dataset['input_signals'].shape[0]
        num_times   = dataset['input_signals'].shape[1]
        state = tf.zeros((num_samples, num_latent_states), dtype=tf.float64)
        history = tf.TensorArray(tf.float64, size=num_times)
        history = history.write(0, state)
        for i in tf.range(num_times - 1):
            inp = tf.concat([
                state,
                tf.expand_dims(dataset['input_parameters'][:, 0], axis=-1),
                dataset['input_signals'][:, i, :],
            ], axis=-1)
            state = state + dt / dt_base * (NNdyn(inp) - LEAK * state)
            history = history.write(i + 1, state)
        return tf.transpose(history.stack(), perm=(1, 0, 2))

    def reconstruct_output(dataset, states):
        ns, nt, np_ = dataset['num_samples'], dataset['num_times'], dataset['num_points']
        s_exp = tf.broadcast_to(tf.expand_dims(states, 2), [ns, nt, np_, num_latent_states])
        u_exp = tf.broadcast_to(tf.expand_dims(dataset['input_signals'], 2),
                                [ns, nt, np_, len(problem['input_signals'])])
        out = NNrec(tf.concat([s_exp, u_exp, dataset['points_full']], axis=3))
        out = (out**3 + alpha * out) / (1 + alpha)
        return out

    def ldnet(dataset):
        return reconstruct_output(dataset, evolve_dynamics(dataset))

    def get_direction(v):
        return tf.math.divide(v, epsilon + tf.expand_dims(tf.norm(v, axis=3), axis=-1))

    def loss_fn(dataset, target, target_dir, sample_w=None):
        pred = ldnet(dataset)
        sq = tf.square(pred - tf.convert_to_tensor(target, tf.float64))
        if sample_w is not None:
            sq = sample_w[:, None, None, None] * sq
        mse_v = tf.reduce_mean(sq)
        dir_  = get_direction(pred)
        mse_d = tf.reduce_mean(tf.square(dir_ - tf.convert_to_tensor(target_dir, tf.float64)))
        return mse_v + weight_direction * mse_d

    def reg_term():
        # Reference-implementation Tikhonov term: mean of squared kernels per
        # layer, averaged over layers, kernels only (no biases).
        def wreg(NN):
            ks = [lay.kernel for lay in NN.layers if hasattr(lay, 'kernel')]
            return tf.add_n([tf.reduce_mean(tf.square(k)) for k in ks]) / len(ks)
        return ALPHA_REG * (wreg(NNdyn) + wreg(NNrec))

    return ldnet, loss_fn, get_direction, reg_term

# ---------------------------------------------------------------
# Train one configuration

def train_one(num_latent_states, dataset_train, dataset_valid):
    print(f"\n{'='*60}")
    print(f"  Training num_latent_states = {num_latent_states}")
    print(f"{'='*60}")

    np.random.seed(0)
    tf.random.set_seed(0)

    NNdyn, NNrec = build_networks(num_latent_states)
    ldnet, loss_fn, get_direction, reg_term = make_ldnet(NNdyn, NNrec, num_latent_states)

    tgt_dir_train = get_direction(dataset_train['output_signals'])
    tgt_dir_valid = get_direction(dataset_valid['output_signals'])

    # Optional per-trajectory inverse-variance weights (mean 1, capped).
    sample_w = None
    if FAMILY_W:
        tgt = np.asarray(dataset_train['output_signals'], dtype=np.float64)
        var = tgt.reshape(tgt.shape[0], -1).var(axis=1)
        w = 1.0 / (var + 1e-8)
        w = np.minimum(w / w.mean(), FAMILY_W_CAP)
        w = w / w.mean()
        sample_w = tf.constant(w, dtype=tf.float64)
        print(f'  FAMILY_W on: weights min {w.min():.2f} max {w.max():.2f}')

    # Regularization enters the TRAIN objective only; validation stays plain
    # MSE so that checkpoint selection compares reconstruction quality.
    loss_train = lambda: loss_fn(dataset_train, dataset_train['output_signals'], tgt_dir_train,
                                 sample_w) + reg_term()
    loss_valid = lambda: loss_fn(dataset_valid, dataset_valid['output_signals'], tgt_dir_valid)

    variables = NNdyn.variables + NNrec.variables
    opt = optimization.OptimizationProblem(variables, loss_train, loss_valid)
    opt.track_best_valid = True
    opt.patience = PATIENCE

    print('  Adam...')
    opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
    print('  BFGS...')
    opt.optimize_BFGS(num_epochs_BFGS)

    best_valid = opt.restore_best()
    if best_valid is not None:
        print(f'  Restored best-validation weights (valid loss {best_valid:.3e})')

    return NNdyn, NNrec, ldnet, opt

# ---------------------------------------------------------------
# Evaluate on test set

def evaluate(ldnet, dataset_tests):
    out_norm = ldnet(dataset_tests)
    fom = utils.denormalize_output(dataset_tests['output_signals'], problem, normalization).numpy()
    rom = utils.denormalize_output(out_norm, problem, normalization).numpy()

    metrics = {}
    for i, name in enumerate(['F_y', 'M_z']):
        f = fom[:, :, 0, i]
        r = rom[:, :, 0, i]
        nrmse = np.sqrt(np.mean((r - f)**2)) / (np.max(f) - np.min(f))
        rho, _ = scipy.stats.pearsonr(r.ravel(), f.ravel())
        metrics[f'NRMSE_{name}'] = float(nrmse)
        metrics[f'rho_{name}']   = float(rho)
        print(f'  NRMSE {name}: {nrmse:.3e}   1-rho: {1-rho:.3e}')

    nrmse_all = np.sqrt(np.mean((rom - fom)**2)) / (np.max(fom) - np.min(fom))
    rho_all, _ = scipy.stats.pearsonr(rom.ravel(), fom.ravel())
    metrics['NRMSE'] = float(nrmse_all)
    metrics['rho']   = float(rho_all)
    print(f'  NRMSE combined: {nrmse_all:.3e}   1-rho: {1-rho_all:.3e}')

    return metrics, fom, rom

# ---------------------------------------------------------------
# Timeseries plot (3 rows × 3 cols: F_y, M_z, |err|  ×  family A, B, Cc)

def plot_timeseries(fom, rom, time_axis, family_indices, num_latent_states, out_dir):
    families = ['A', 'B', 'Cc']
    fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharey='row')
    fig.suptitle(f'LDNet  $d_n = {num_latent_states}$  — test set', fontsize=13)

    for col, fam in enumerate(families):
        idx = family_indices.get(fam)
        if idx is None:
            continue

        t = time_axis

        # F_y
        axs[0, col].plot(t, fom[idx, :, 0, 0], 'b-',  lw=1.2, label='FOM')
        axs[0, col].plot(t, rom[idx, :, 0, 0], 'r--', lw=1.2, label='ROM')
        axs[0, col].set_title(f'Family {fam}')
        if col == 0:
            axs[0, col].set_ylabel('$F_y$ [N]')
        axs[0, col].legend(fontsize=8)

        # M_z
        axs[1, col].plot(t, fom[idx, :, 0, 1], 'b-',  lw=1.2, label='FOM')
        axs[1, col].plot(t, rom[idx, :, 0, 1], 'r--', lw=1.2, label='ROM')
        if col == 0:
            axs[1, col].set_ylabel('$M_z$ [N·m]')

        # Absolute error
        axs[2, col].plot(t, np.abs(fom[idx, :, 0, 0] - rom[idx, :, 0, 0]), 'b-',  lw=1, label='$|err\\ F_y|$')
        axs[2, col].plot(t, np.abs(fom[idx, :, 0, 1] - rom[idx, :, 0, 1]), 'r--', lw=1, label='$|err\\ M_z|$')
        axs[2, col].set_xlabel('Time [s]')
        if col == 0:
            axs[2, col].set_ylabel('Abs. error')
        axs[2, col].legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / 'timeseries.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')

# ---------------------------------------------------------------
# Summary NRMSE / 1-rho vs latent states

def plot_summary(all_metrics, latent_values, summary_dir):
    nrmse_vals = [m['NRMSE']       for m in all_metrics]
    rho_vals   = [1 - m['rho']     for m in all_metrics]
    nrmse_Fy   = [m['NRMSE_F_y']   for m in all_metrics]
    nrmse_Mz   = [m['NRMSE_M_z']   for m in all_metrics]
    rho_Fy     = [1 - m['rho_F_y'] for m in all_metrics]
    rho_Mz     = [1 - m['rho_M_z'] for m in all_metrics]

    fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    axs[0].semilogy(latent_values, nrmse_vals, 'ko-', lw=1.5, label='combined')
    axs[0].semilogy(latent_values, nrmse_Fy,  'b^--', lw=1,   label='$F_y$')
    axs[0].semilogy(latent_values, nrmse_Mz,  'rs--', lw=1,   label='$M_z$')
    axs[0].set_ylabel('NRMSE')
    axs[0].legend()
    axs[0].grid(True, which='both', ls=':')

    axs[1].semilogy(latent_values, rho_vals, 'ko-', lw=1.5, label='combined')
    axs[1].semilogy(latent_values, rho_Fy,  'b^--', lw=1,   label='$F_y$')
    axs[1].semilogy(latent_values, rho_Mz,  'rs--', lw=1,   label='$M_z$')
    axs[1].set_ylabel('$1 - \\rho$')
    axs[1].set_xlabel('num. latent states')
    axs[1].legend()
    axs[1].grid(True, which='both', ls=':')
    axs[1].set_xticks(latent_values)

    fig.suptitle('LDNet sensitivity — GLA dataset', fontsize=12)
    fig.tight_layout()
    out_path = summary_dir / 'nrmse_rho_vs_latent.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved summary plot: {out_path}')

# ---------------------------------------------------------------
# Main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'summary').mkdir(exist_ok=True)

    # Load datasets once (process_dataset modifies in-place, reload each time)
    import time
    print('Loading datasets...', flush=True)
    t0 = time.time(); raw_train = utils.load_gla_h5(DATA_DIR / 'GLA_train.h5'); print(f'  train loaded {time.time()-t0:.1f}s', flush=True)
    t0 = time.time(); raw_valid = utils.load_gla_h5(DATA_DIR / 'GLA_valid.h5'); print(f'  valid loaded {time.time()-t0:.1f}s', flush=True)
    t0 = time.time(); raw_tests = utils.load_gla_h5(DATA_DIR / 'GLA_test.h5');  print(f'  test  loaded {time.time()-t0:.1f}s', flush=True)

    # Resolve one test sim index per family
    families = raw_tests['sim_families']
    if families is None:
        raise RuntimeError("sim_families not found in GLA_test.h5 — regenerate with updated preprocess_GLA.py")
    family_indices = {}
    for fam in ['A', 'B', 'Cc']:
        for i, f in enumerate(families):
            if f == fam:
                family_indices[fam] = i
                break
    print(f'Family indices in test set: {family_indices}')

    all_metrics = []

    for num_latent_states in LATENT_SWEEP:
        out_dir = RESULTS_DIR / f'latent_{num_latent_states}'
        out_dir.mkdir(exist_ok=True)

        # Reload raw data each iteration (process_dataset mutates the dict)
        import time
        print(f'  loading HDF5...', flush=True)
        t0 = time.time(); dataset_train = utils.load_gla_h5(DATA_DIR / 'GLA_train.h5'); print(f'  train loaded {time.time()-t0:.1f}s', flush=True)
        t0 = time.time(); dataset_valid = utils.load_gla_h5(DATA_DIR / 'GLA_valid.h5'); print(f'  valid loaded {time.time()-t0:.1f}s', flush=True)
        t0 = time.time(); dataset_tests = utils.load_gla_h5(DATA_DIR / 'GLA_test.h5');  print(f'  test  loaded {time.time()-t0:.1f}s', flush=True)

        for _ds in (dataset_train, dataset_valid, dataset_tests):
            select_input_columns(_ds)

        print(f'  processing train...', flush=True); t0 = time.time()
        utils.process_dataset(dataset_train, problem, normalization, dt=dt, num_points_subsample=1)
        print(f'  done {time.time()-t0:.1f}s', flush=True)
        print(f'  processing valid...', flush=True); t0 = time.time()
        utils.process_dataset(dataset_valid, problem, normalization, dt=dt, num_points_subsample=1)
        print(f'  done {time.time()-t0:.1f}s', flush=True)
        print(f'  processing test...', flush=True); t0 = time.time()
        utils.process_dataset(dataset_tests, problem, normalization, dt=dt)
        print(f'  done {time.time()-t0:.1f}s', flush=True)

        if EXPORT_ONLY:
            # Reload a previously trained model and only regenerate thesis artifacts.
            NNdyn, NNrec = build_networks(num_latent_states)
            NNdyn.load_weights(str(out_dir / 'NNdyn_weights.weights.h5'))
            NNrec.load_weights(str(out_dir / 'NNrec_weights.weights.h5'))
            ldnet, _, _, _ = make_ldnet(NNdyn, NNrec, num_latent_states)
            print(f'  [export-only] loaded weights from {out_dir}')
        else:
            NNdyn, NNrec, ldnet, opt = train_one(num_latent_states, dataset_train, dataset_valid)

            # Save weights
            NNdyn.save_weights(str(out_dir / 'NNdyn_weights.weights.h5'))
            NNrec.save_weights(str(out_dir / 'NNrec_weights.weights.h5'))

            # Save config.json for clean/ inference
            import json as _json
            config = {
                'problem': problem,
                'normalization': normalization,
                'num_latent_states': num_latent_states,
                'input_set': INPUT_SET,
                'input_signals': SEL_SIGNALS,
            }
            with open(out_dir / 'config.json', 'w') as _fp:
                _json.dump(config, _fp, indent=2)
            print(f'  Saved config.json to {out_dir}')

            # Save loss history (data for the thesis loss figure) + a dev PNG
            np.savez(out_dir / 'loss_history.npz',
                     iterations=np.asarray(opt.iterations_history),
                     train=np.asarray(opt.loss_train_history),
                     valid=np.asarray(opt.loss_valid_history),
                     adam_epochs=num_epochs_Adam)
            fig, ax = plt.subplots()
            ax.loglog(opt.iterations_history, opt.loss_train_history, 'o-', label='train')
            ax.loglog(opt.iterations_history, opt.loss_valid_history, 'o-', label='valid')
            ax.axvline(num_epochs_Adam, color='k', ls='--', lw=0.8)
            ax.set_xlabel('epochs')
            ax.set_ylabel('loss')
            ax.set_title(f'$d_n = {num_latent_states}$')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / 'loss.png', dpi=120)
            plt.close(fig)

        # Evaluate
        metrics, fom, rom = evaluate(ldnet, dataset_tests)
        metrics['num_latent_states'] = num_latent_states
        all_metrics.append(metrics)

        with open(out_dir / 'metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=2)

        # Dev timeseries plot (titled) + thesis-ready trace dump (F_y, M_z per family)
        time_axis = np.asarray(dataset_tests['times']) * dt_base
        plot_timeseries(fom, rom, time_axis, family_indices, num_latent_states, out_dir)
        traces = {'time': time_axis}
        for fam, idx in family_indices.items():
            traces[f'{fam}_fom_Fy'] = fom[idx, :, 0, 0]
            traces[f'{fam}_rom_Fy'] = rom[idx, :, 0, 0]
            traces[f'{fam}_fom_Mz'] = fom[idx, :, 0, 1]
            traces[f'{fam}_rom_Mz'] = rom[idx, :, 0, 1]
        np.savez(out_dir / 'traces.npz', **traces)

    # Summary plots
    plot_summary(all_metrics, LATENT_SWEEP, RESULTS_DIR / 'summary')

    # Save all metrics together
    with open(RESULTS_DIR / 'summary' / 'all_metrics.json', 'w') as fp:
        json.dump(all_metrics, fp, indent=2)

    print('\nSweep complete.')


if __name__ == '__main__':
    main()
