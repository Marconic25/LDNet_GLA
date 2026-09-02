#!/usr/bin/env python3
"""Closed-loop ROLLOUT training sweep over INPUT_SET x NUM_LATENT for LDNet GLA.

Parametrized version of sensitivity_latent_rollout.py (which stays untouched and
handles the single 6-input / d_s=10 config). For each (INPUT_SET, NUM_LATENT) this
script:
  1. builds NNdyn/NNrec with input dims matching the INPUT_SET preset,
  2. warm-starts from the teacher-forced sweep weights under
     WARMSTART_ROOT/in{INPUT_SET}/latent_{NUM_LATENT},
  3. trains in closed-loop rollout mode with the proven recipe: the 2-DOF
     structural ODE (clean/structure.py, fixed-step DP45 in TF) is propagated
     from the model's OWN predicted loads inside the loss; delta and W_gust stay
     exogenous; loss = MSE(rolled structural states vs data) + W_LOAD*MSE(loads);
     PURE BFGS (NADAM=0), best-VALIDATION early stopping via optimization.py
     (track_best_valid/patience/restore_best),
  4. evaluates on the test set BOTH (a) closed-loop replay (model rolls its own
     structural state, exogenous delta/W only) and (b) teacher-forced open-loop
     (structural states fed from data each step) NRMSE / Pearson rho,
  5. writes metrics.json + traces.npz + loss_history.npz + weights + config.json
     under RESULTS_OVERRIDE/in{INPUT_SET}/latent_{NUM_LATENT}/.

CONCEPTUAL NOTE (in2 degeneracy): for INPUT_SET=2 the networks see only
(delta, W_gust), so the model's own structural state NEVER enters the nets and
there is no closed-loop feedback path through the networks. Rollout training for
in2 therefore degenerates to (nearly) open-loop training: the only difference vs
teacher-forced is that the state-matching loss term backpropagates through the
structural ODE into the predicted loads, which carries the same information as
the load-matching term itself. Expect rollout training to genuinely matter only
for the 4- and 6-input configurations; in2 is run anyway for grid completeness.

Env: INPUT_SET (2|4|6), NUM_LATENT, DATA_OVERRIDE, RESULTS_OVERRIDE,
     WARMSTART_ROOT, LAMBDA_DAMP (0.003), ROLLOUT_LEN (800), NADAM (0),
     NBFGS (500), W_LOAD (1.0), PATIENCE (20, in validation samples = 10 iters
     each), SELFTEST (1 = only report warm-start rollout accuracy and exit).
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
# structure.py lives in <repo>/clean, not in src/ (on the cluster too); make it
# importable regardless of PYTHONPATH.
_clean = (SRC_DIR / '..' / 'clean').resolve()
if (_clean / 'structure.py').exists():
    sys.path.insert(0, str(_clean))
import utils, optimization, structure

# ---------------------------------------------------------------
# Run configuration
DATA_DIR       = Path(os.environ.get("DATA_OVERRIDE", "/work/u10677113/LDNet_GLA/data"))
RESULTS_ROOT   = Path(os.environ.get("RESULTS_OVERRIDE", "/work/u10677113/LDNet_GLA/results/sensitivity_rollout"))
WARMSTART_ROOT = Path(os.environ.get("WARMSTART_ROOT", "/work/u10677113/LDNet_GLA/results/sensitivity"))
INPUT_SET      = os.environ.get("INPUT_SET", "6")
NUM_LATENT     = int(os.environ.get("NUM_LATENT", "10"))
LAMBDA_DAMP    = float(os.environ.get("LAMBDA_DAMP", "0.003"))
ROLLOUT_LEN    = int(os.environ.get("ROLLOUT_LEN", "800"))
num_epochs_Adam = int(os.environ.get("NADAM", "0"))      # proven recipe: pure BFGS
num_epochs_BFGS = int(os.environ.get("NBFGS", "500"))
W_LOAD         = float(os.environ.get("W_LOAD", "1.0"))
PATIENCE       = int(os.environ.get("PATIENCE", "20"))   # validation samples (x10 iters)
SELFTEST       = os.environ.get("SELFTEST", "0") == "1"

# Input presets (identical to sensitivity_latent.py). CANON_SIGNALS is the fixed
# column order in the GLA_*.h5 files. IMPORTANT: unlike the teacher-forced sweep
# we NEVER subset the data columns — the full structural state (h,hd,a,ad) is
# always needed from data for the rollout initial condition and the state-match
# targets. Only the NETWORK inputs are subset, via SEL_COLS gathers.
CANON_SIGNALS = ["h", "hd", "a", "ad", "delta", "W_gust"]
INPUT_PRESETS = {
    "2": ["delta", "W_gust"],
    "4": ["h", "a", "delta", "W_gust"],
    "6": ["h", "hd", "a", "ad", "delta", "W_gust"],
}
SEL_SIGNALS = INPUT_PRESETS[INPUT_SET]
SEL_COLS    = tf.constant([CANON_SIGNALS.index(n) for n in SEL_SIGNALS], tf.int32)
N_SIG       = len(SEL_SIGNALS)

WARMSTART = WARMSTART_ROOT / f"in{INPUT_SET}" / f"latent_{NUM_LATENT}"
OUT_DIR   = RESULTS_ROOT / f"in{INPUT_SET}" / f"latent_{NUM_LATENT}"

dt = 0.002; dt_base = 0.002; DT_PHYS = 0.002
alpha_cub = 0.05

# Problem dict used for DATA processing keeps all 6 canonical signals; the
# network-facing signal list is SEL_SIGNALS (recorded in config.json).
problem = {
    "space": {"dimension": 2},
    "input_parameters": [{"name": "U_inf"}],
    "input_signals": [{"name": n} for n in CANON_SIGNALS],
    "output_signals": [{"name": "F_y"}, {"name": "M_z"}],
    "output_fields": [{"name": "ux"}, {"name": "uy"}],
}
normalization = {
    'space': {'min': [0,0], 'max': [1,1]},
    'time':  {'time_constant': dt_base},
    'input_parameters': {'U_inf': {'min': 0, 'max': 120}},
    'input_signals': {
        'h':{'min':-0.029,'max':0.009},'hd':{'min':-0.54,'max':0.62},
        'a':{'min':-0.014,'max':0.011},'ad':{'min':-0.93,'max':0.93},
        'delta':{'min':-14.09,'max':14.55},'W_gust':{'min':0.0,'max':47.81}},
    'output_signals': {'F_y':{'min':-18.65,'max':547.29},'M_z':{'min':-44.99,'max':42.24}},
    'output_fields': {'ux':{'min':-50,'max':150},'uy':{'min':-100,'max':100}},
}

# ---- normalization tensors (identical to sensitivity_latent_rollout.py) ----
SIG_MIN = tf.constant([-0.029,-0.54,-0.014,-0.93], tf.float64)   # h,hd,a,ad
SIG_MAX = tf.constant([ 0.009, 0.62, 0.011, 0.93], tf.float64)
OUT_MIN = tf.constant([-18.65,-44.99], tf.float64)               # F_y, M_z
OUT_MAX = tf.constant([547.29, 42.24], tf.float64)
def norm_states(xp):   return (2.0*xp - SIG_MIN - SIG_MAX) / (SIG_MAX - SIG_MIN)
def denorm_states(xn): return 0.5*(SIG_MIN + SIG_MAX + (SIG_MAX - SIG_MIN)*xn)
def denorm_loads(on):  return 0.5*(OUT_MIN + OUT_MAX + (OUT_MAX - OUT_MIN)*on)

# ---- structural model in TF (matches clean/structure.py, no flap inertial terms) ----
M_HH = tf.constant(structure.M_HH, tf.float64); M_AA = tf.constant(structure.M_AA, tf.float64)
M_HA = tf.constant(structure.M_HA, tf.float64); DET = tf.constant(structure.DET, tf.float64)
K_H = tf.constant(structure.K_H, tf.float64); D_H = tf.constant(structure.D_H, tf.float64)
K_ALPHA = tf.constant(structure.K_ALPHA, tf.float64); D_ALPHA = tf.constant(structure.D_ALPHA, tf.float64)
def struct_rhs(x, Fy, Mz):
    h=x[:,0]; hd=x[:,1]; a=x[:,2]; ad=x[:,3]
    rhs_h = -Fy - D_H*hd - K_H*h
    rhs_a =  Mz - D_ALPHA*ad - K_ALPHA*a
    hdd = (M_AA*rhs_h - M_HA*rhs_a)/DET
    add = (M_HH*rhs_a - M_HA*rhs_h)/DET
    return tf.stack([hd, hdd, ad, add], axis=-1)
def struct_step(x, Fy, Mz, h):
    """Fixed-step Dormand-Prince RK45 (5th-order, 6 stages)."""
    k1 = struct_rhs(x, Fy, Mz)
    k2 = struct_rhs(x + h*(1.0/5)*k1, Fy, Mz)
    k3 = struct_rhs(x + h*(3.0/40*k1 + 9.0/40*k2), Fy, Mz)
    k4 = struct_rhs(x + h*(44.0/45*k1 - 56.0/15*k2 + 32.0/9*k3), Fy, Mz)
    k5 = struct_rhs(x + h*(19372.0/6561*k1 - 25360.0/2187*k2 + 64448.0/6561*k3 - 212.0/729*k4), Fy, Mz)
    k6 = struct_rhs(x + h*(9017.0/3168*k1 - 355.0/33*k2 + 46732.0/5247*k3 + 49.0/176*k4 - 5103.0/18656*k5), Fy, Mz)
    return x + h*(35.0/384*k1 + 500.0/1113*k3 + 125.0/192*k4 - 2187.0/6784*k5 + 11.0/84*k6)

def build_networks(nz):
    n_inp = nz + 1 + N_SIG
    NNdyn = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation=tf.nn.tanh, input_shape=(n_inp,)),
        tf.keras.layers.Dense(7, activation=tf.nn.tanh),
        tf.keras.layers.Dense(nz)])
    n_rec = nz + N_SIG + 2
    NNrec = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation=tf.nn.tanh, input_shape=(n_rec,)),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(24, activation=tf.nn.tanh),
        tf.keras.layers.Dense(2)])
    return NNdyn, NNrec

def make_rollout(NNdyn, NNrec, nz, L):
    """Closed-loop rollout: structural state propagated from the model's own loads.

    Only SEL_COLS of the full normalized 6-signal vector are fed to the nets;
    for INPUT_SET=2 that means the fed-back state never reaches the networks
    (see module docstring: in2 degenerates to open-loop)."""
    def rollout(ds):
        sig = ds['input_signals']          # (B,T,6) normalized, full canon order
        Upar = tf.expand_dims(ds['input_parameters'][:,0], -1)   # (B,1) normalized
        pt = ds['point_n']                 # (B,2) normalized spatial point
        B = tf.shape(sig)[0]
        z = tf.zeros((B, nz), tf.float64)
        x = denorm_states(sig[:,0,0:4])    # physical IC from data
        xs = tf.TensorArray(tf.float64, size=L)
        lo = tf.TensorArray(tf.float64, size=L)
        for i in tf.range(L):
            sig_struct = norm_states(x)                    # (B,4)
            sig_exo = sig[:, i, 4:6]                       # (B,2) delta,W (exogenous)
            s_full = tf.concat([sig_struct, sig_exo], axis=-1)   # (B,6) canon order
            s_net = tf.gather(s_full, SEL_COLS, axis=-1)   # (B,N_SIG) preset subset
            rec_in = tf.concat([z, s_net, pt], axis=-1)
            out = NNrec(rec_in)
            out = (out**3 + alpha_cub*out)/(1+alpha_cub)   # (B,2) normalized loads
            loads = denorm_loads(out)
            Fy = loads[:,0]; Mz = loads[:,1]
            inp = tf.concat([z, Upar, s_net], axis=-1)
            z = z + dt/dt_base*(NNdyn(inp) - LAMBDA_DAMP*z)
            x = struct_step(x, Fy, Mz, DT_PHYS)
            x = tf.clip_by_value(x, -10.0, 10.0)           # NaN guard
            xs = xs.write(i, norm_states(x))               # normalized predicted state at i+1
            lo = lo.write(i, out)                          # normalized predicted load at i
        xs = tf.transpose(xs.stack(), (1,0,2))   # (B,L,4)
        lo = tf.transpose(lo.stack(), (1,0,2))   # (B,L,2)
        return xs, lo
    def loss_fn(ds):
        xs, lo = rollout(ds)
        sig = ds['input_signals']
        tgt_x = sig[:, 1:L+1, 0:4]                    # true states at steps 1..L
        tgt_l = ds['output_signals'][:, 0:L, 0, :]    # true loads at steps 0..L-1
        mse_x = tf.reduce_mean(tf.square(xs - tgt_x))
        mse_l = tf.reduce_mean(tf.square(lo - tgt_l))
        return mse_x + W_LOAD*mse_l
    return rollout, loss_fn

def make_teacher_forced(NNdyn, NNrec, nz):
    """Teacher-forced open-loop forward pass (states fed from data each step),
    evaluated at the sectional load point, with the SAME damped latent update
    used in rollout training (leak = LAMBDA_DAMP) so the trained model is
    evaluated faithfully. Returns normalized loads (B, T-1, 2) for steps 0..T-2."""
    def forward(ds, T):
        sig = ds['input_signals']
        Upar = tf.expand_dims(ds['input_parameters'][:,0], -1)
        pt = ds['point_n']
        B = tf.shape(sig)[0]
        z = tf.zeros((B, nz), tf.float64)
        lo = tf.TensorArray(tf.float64, size=T-1)
        for i in tf.range(T-1):
            s_net = tf.gather(sig[:, i, :], SEL_COLS, axis=-1)
            rec_in = tf.concat([z, s_net, pt], axis=-1)
            out = NNrec(rec_in)
            out = (out**3 + alpha_cub*out)/(1+alpha_cub)
            lo = lo.write(i, out)
            inp = tf.concat([z, Upar, s_net], axis=-1)
            z = z + dt/dt_base*(NNdyn(inp) - LAMBDA_DAMP*z)
        return tf.transpose(lo.stack(), (1,0,2))   # (B,T-1,2)
    return forward

def load_data(name):
    ds = utils.load_gla_h5(DATA_DIR / name)
    fams = ds.get('sim_families')
    utils.process_dataset(ds, problem, normalization, dt=dt)
    ds['point_n'] = tf.convert_to_tensor(
        np.broadcast_to(ds['points_full'][0,0,0,:], (ds['num_samples'], 2)).copy(), tf.float64)
    for k in ['input_signals','input_parameters','output_signals']:
        ds[k] = tf.convert_to_tensor(ds[k], tf.float64)
    ds['sim_families'] = fams
    return ds

def loads_metrics(fom, rom, prefix=''):
    """NRMSE + Pearson rho per output and combined; fom/rom physical (B,L,2)."""
    import scipy.stats
    m = {}
    for i, name in enumerate(['F_y','M_z']):
        f = fom[:,:,i]; r = rom[:,:,i]
        m[f'{prefix}NRMSE_{name}'] = float(np.sqrt(np.mean((r-f)**2)) / (np.max(f)-np.min(f)))
        m[f'{prefix}rho_{name}']   = float(scipy.stats.pearsonr(r.ravel(), f.ravel())[0])
    m[f'{prefix}NRMSE'] = float(np.sqrt(np.mean((rom-fom)**2)) / (np.max(fom)-np.min(fom)))
    m[f'{prefix}rho']   = float(scipy.stats.pearsonr(rom.ravel(), fom.ravel())[0])
    return m

def main():
    print(f"ROLLOUT SWEEP  INPUT_SET={INPUT_SET} ({SEL_SIGNALS})  NUM_LATENT={NUM_LATENT}", flush=True)
    print(f"  WARMSTART={WARMSTART}", flush=True)
    print(f"  LAMBDA={LAMBDA_DAMP} ROLLOUT_LEN={ROLLOUT_LEN} W_LOAD={W_LOAD} "
          f"NADAM={num_epochs_Adam} NBFGS={num_epochs_BFGS} PATIENCE={PATIENCE}", flush=True)

    print("Loading data...", flush=True)
    ds_tr = load_data('GLA_train.h5'); ds_va = load_data('GLA_valid.h5'); ds_te = load_data('GLA_test.h5')
    print(f"  train {ds_tr['input_signals'].shape}  valid {ds_va['input_signals'].shape}"
          f"  test {ds_te['input_signals'].shape}", flush=True)

    np.random.seed(0); tf.random.set_seed(0)
    NNdyn, NNrec = build_networks(NUM_LATENT)
    _ = NNdyn(tf.zeros((1, NUM_LATENT+1+N_SIG), tf.float64))
    _ = NNrec(tf.zeros((1, NUM_LATENT+N_SIG+2), tf.float64))
    NNdyn.load_weights(str(WARMSTART/'NNdyn_weights.weights.h5'))
    NNrec.load_weights(str(WARMSTART/'NNrec_weights.weights.h5'))
    print("  warm-start weights loaded", flush=True)

    # --- verify TF structural step matches numpy reference ---
    xr = np.random.randn(3,4)*np.array([0.01,0.3,0.005,0.3])
    Fy = np.array([100.,-50.,200.]); Mz = np.array([5.,-3.,10.])
    tf_step = struct_step(tf.constant(xr), tf.constant(Fy), tf.constant(Mz), DT_PHYS).numpy()
    np_step = np.array([structure.step_dp45(xr[i], Fy[i], Mz[i], DT_PHYS) for i in range(3)])
    err = np.max(np.abs(tf_step - np_step))
    print(f"  TF-vs-numpy structure step max err = {err:.2e}  {'OK' if err<1e-8 else 'MISMATCH!'}", flush=True)
    assert err < 1e-8, "TF structure step does not match numpy"

    rollout, loss_fn = make_rollout(NNdyn, NNrec, NUM_LATENT, ROLLOUT_LEN)

    # --- warm-start closed-loop sanity on validation ---
    t0 = time.time()
    xs, _ = tf.function(rollout)(ds_va)
    tgt_x = ds_va['input_signals'][:, 1:ROLLOUT_LEN+1, 0:4].numpy(); xsn = xs.numpy()
    print(f"  [warmstart rollout] ({time.time()-t0:.1f}s) state NRMSE per dof:", flush=True)
    for j,nm in enumerate(['h','hd','a','ad']):
        f=tgt_x[:,:,j]; r=xsn[:,:,j]
        print(f"      {nm}: {np.sqrt(np.mean((r-f)**2))/(f.max()-f.min()+1e-9):.3f}", flush=True)
    print(f"  any NaN in warm-start rollout: {np.any(np.isnan(xsn))}", flush=True)
    if SELFTEST:
        print("SELFTEST done.", flush=True); return

    # --- train: pure BFGS, best-valid early stop (proven recipe) ---
    loss_train = lambda: loss_fn(ds_tr)
    loss_valid = lambda: loss_fn(ds_va)
    # Keras 3 compat shim: model.variables yields KerasVariable wrappers that
    # optimization.py's GradientTape/stitcher cannot handle; unwrap to the
    # backing tf.Variable via .value. No-op on Keras 2 (cluster image).
    variables = [v if isinstance(v, tf.Variable) else v.value
                 for v in NNdyn.variables + NNrec.variables]
    opt = optimization.OptimizationProblem(variables, loss_train, loss_valid)
    opt.track_best_valid = True
    opt.patience = PATIENCE

    if num_epochs_Adam > 0:
        print("  Adam...", flush=True)
        opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-3))
    print("  BFGS...", flush=True)
    opt.optimize_BFGS(num_epochs_BFGS)
    best_valid = opt.restore_best()
    if best_valid is not None:
        print(f"  Restored best-validation weights (valid rollout loss {best_valid:.4e})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NNdyn.save_weights(str(OUT_DIR/'NNdyn_weights.weights.h5'))
    NNrec.save_weights(str(OUT_DIR/'NNrec_weights.weights.h5'))
    cfg = {'problem': problem, 'normalization': normalization,
           'num_latent_states': NUM_LATENT, 'lambda_damp': LAMBDA_DAMP,
           'input_set': INPUT_SET, 'input_signals': SEL_SIGNALS,
           'rollout_len': ROLLOUT_LEN, 'w_load': W_LOAD,
           'warmstart': str(WARMSTART), 'training': 'closed-loop rollout, pure BFGS, best-valid'}
    json.dump(cfg, open(OUT_DIR/'config.json','w'), indent=2)
    np.savez(OUT_DIR/'loss_history.npz',
             iterations=np.asarray(opt.iterations_history),
             train=np.asarray([float(v) for v in opt.loss_train_history]),
             valid=np.asarray([float(v) for v in opt.loss_valid_history]),
             adam_epochs=num_epochs_Adam)

    # ============ EVALUATION on test set ============
    T = int(ds_te['input_signals'].shape[1]); L = T - 1
    time_axis = np.asarray(ds_te['times']) * dt_base

    # (a) closed-loop replay over the FULL test trajectories
    print(f"  [eval] closed-loop replay over {L} steps...", flush=True)
    ro_full, _ = make_rollout(NNdyn, NNrec, NUM_LATENT, L)
    t0 = time.time()
    xs, lo = tf.function(ro_full)(ds_te)
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)
    fom_l = denorm_loads(ds_te['output_signals'][:, 0:L, 0, :]).numpy()
    rom_l_cl = denorm_loads(lo).numpy()
    metrics = {'num_latent_states': NUM_LATENT, 'input_set': INPUT_SET,
               'input_signals': SEL_SIGNALS}
    metrics.update(loads_metrics(fom_l, rom_l_cl, prefix='closed_'))
    fom_x = denorm_states(ds_te['input_signals'][:, 1:L+1, 0:4]).numpy()
    rom_x = denorm_states(xs).numpy()
    for j, nm in enumerate(['h','hd','a','ad']):
        f = fom_x[:,:,j]; r = rom_x[:,:,j]
        metrics[f'closed_NRMSE_{nm}'] = float(np.sqrt(np.mean((r-f)**2))/(np.max(f)-np.min(f)))
    metrics['closed_any_nan'] = bool(np.any(np.isnan(rom_x)) or np.any(np.isnan(rom_l_cl)))
    print(f"    closed-loop: NRMSE={metrics['closed_NRMSE']:.3e} rho={metrics['closed_rho']:.4f} "
          f"NaN={metrics['closed_any_nan']}", flush=True)

    # (b) teacher-forced open-loop on the same trajectories/steps
    print("  [eval] teacher-forced open-loop...", flush=True)
    tf_fwd = make_teacher_forced(NNdyn, NNrec, NUM_LATENT)
    t0 = time.time()
    lo_tf = tf.function(lambda d: tf_fwd(d, T))(ds_te)
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)
    rom_l_tf = denorm_loads(lo_tf).numpy()
    metrics.update(loads_metrics(fom_l, rom_l_tf, prefix='teacher_'))
    print(f"    teacher-forced: NRMSE={metrics['teacher_NRMSE']:.3e} rho={metrics['teacher_rho']:.4f}", flush=True)

    with open(OUT_DIR/'metrics.json','w') as fp:
        json.dump(metrics, fp, indent=2)

    # traces per family (A, B, Cc): loads closed-loop + teacher-forced + states
    fams = ds_te['sim_families']
    family_indices = {}
    if fams is not None:
        for fam in ['A','B','Cc']:
            idx = [i for i,f in enumerate(fams) if f == fam]
            if idx: family_indices[fam] = idx[0]
    traces = {'time': time_axis[0:L]}
    for fam, idx in family_indices.items():
        traces[f'{fam}_fom_Fy']    = fom_l[idx,:,0];    traces[f'{fam}_fom_Mz']    = fom_l[idx,:,1]
        traces[f'{fam}_closed_Fy'] = rom_l_cl[idx,:,0]; traces[f'{fam}_closed_Mz'] = rom_l_cl[idx,:,1]
        traces[f'{fam}_teacher_Fy']= rom_l_tf[idx,:,0]; traces[f'{fam}_teacher_Mz']= rom_l_tf[idx,:,1]
        for j, nm in enumerate(['h','hd','a','ad']):
            traces[f'{fam}_fom_{nm}']    = fom_x[idx,:,j]
            traces[f'{fam}_closed_{nm}'] = rom_x[idx,:,j]
    np.savez(OUT_DIR/'traces.npz', **traces)

    print(f"Done. Artifacts in {OUT_DIR}", flush=True)

if __name__ == '__main__':
    main()
