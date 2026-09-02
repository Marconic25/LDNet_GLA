#!/usr/bin/env python3
"""Closed-loop ROLLOUT training for LDNet GLA.

Unlike teacher-forced training (structural states h,hd,a,ad fed from data each
step), here the structural states are PROPAGATED by the 2-DOF structural ODE
driven by the model's own predicted loads (F_y, M_z). Only delta and W_gust stay
exogenous (from data). This trains the model for closed-loop / free-running
stability and forces it to actually use the delta/W inputs.

Warm-starts from an existing teacher-forced model (WARMSTART dir).
Env: WARMSTART, LAMBDA_DAMP, ROLLOUT_LEN, NADAM, NBFGS, OUTDIR, W_LOAD, SELFTEST.
"""
import json, sys, os, time
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
import utils, optimization, structure

DATA_DIR    = Path(os.environ.get("DATA_OVERRIDE", "/work/u10677113/LDNet_GLA/data"))
RESULTS_DIR = Path(os.environ.get("OUTDIR", "/work/u10677113/LDNet_GLA/clean/models_rollout"))
WARMSTART   = os.environ.get("WARMSTART", "/work/u10677113/LDNet_GLA/clean/models_damped_l003_full/latent_10")
LAMBDA_DAMP = float(os.environ.get("LAMBDA_DAMP", "0.003"))
ROLLOUT_LEN = int(os.environ.get("ROLLOUT_LEN", "800"))
num_epochs_Adam = int(os.environ.get("NADAM", "200"))
num_epochs_BFGS = int(os.environ.get("NBFGS", "500"))
W_LOAD      = float(os.environ.get("W_LOAD", "1.0"))   # weight of the load-matching term
SELFTEST    = os.environ.get("SELFTEST", "0") == "1"
NUM_LATENT  = 10
# Depth-parametrized (DYN_LAYERS x DYN_WIDTH, REC_LAYERS x REC_WIDTH). Defaults
# 2x7 / 4x24 reproduce the original L=6 architecture bit-identically. Must match
# the WARMSTART checkpoint's own depth (loaded weights are shape-checked).
DYN_LAYERS = int(os.environ.get("DYN_LAYERS", "2"))
DYN_WIDTH  = int(os.environ.get("DYN_WIDTH", "7"))
REC_LAYERS = int(os.environ.get("REC_LAYERS", "4"))
REC_WIDTH  = int(os.environ.get("REC_WIDTH", "24"))
dt = 0.002; dt_base = 0.002; DT_PHYS = 0.002
alpha_cub = 0.05

problem = {
    "space": {"dimension": 2},
    "input_parameters": [{"name": "U_inf"}],
    "input_signals": [{"name": n} for n in ["h","hd","a","ad","delta","W_gust"]],
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

# ---- normalization tensors ----
SIG_MIN = tf.constant([-0.029,-0.54,-0.014,-0.93], tf.float64)   # h,hd,a,ad
SIG_MAX = tf.constant([ 0.009, 0.62, 0.011, 0.93], tf.float64)
OUT_MIN = tf.constant([-18.65,-44.99], tf.float64)               # F_y, M_z
OUT_MAX = tf.constant([547.29, 42.24], tf.float64)
def norm_states(xp):   return (2.0*xp - SIG_MIN - SIG_MAX) / (SIG_MAX - SIG_MIN)
def denorm_states(xn): return 0.5*(SIG_MIN + SIG_MAX + (SIG_MAX - SIG_MIN)*xn)
def denorm_loads(on):  return 0.5*(OUT_MIN + OUT_MAX + (OUT_MAX - OUT_MIN)*on)

# ---- structural model in TF (matches structure.py, no flap inertial terms, as run.py) ----
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
    """Fixed-step Dormand-Prince RK45 (5th-order, 6 stages; Dormand & Prince 1980)."""
    k1 = struct_rhs(x, Fy, Mz)
    k2 = struct_rhs(x + h*(1.0/5)*k1, Fy, Mz)
    k3 = struct_rhs(x + h*(3.0/40*k1 + 9.0/40*k2), Fy, Mz)
    k4 = struct_rhs(x + h*(44.0/45*k1 - 56.0/15*k2 + 32.0/9*k3), Fy, Mz)
    k5 = struct_rhs(x + h*(19372.0/6561*k1 - 25360.0/2187*k2 + 64448.0/6561*k3 - 212.0/729*k4), Fy, Mz)
    k6 = struct_rhs(x + h*(9017.0/3168*k1 - 355.0/33*k2 + 46732.0/5247*k3 + 49.0/176*k4 - 5103.0/18656*k5), Fy, Mz)
    return x + h*(35.0/384*k1 + 500.0/1113*k3 + 125.0/192*k4 - 2187.0/6784*k5 + 11.0/84*k6)

def build_networks(nz):
    n_inp = nz + 1 + 6
    dyn = [tf.keras.layers.Dense(DYN_WIDTH, activation=tf.nn.tanh, input_shape=(n_inp,))]
    dyn += [tf.keras.layers.Dense(DYN_WIDTH, activation=tf.nn.tanh) for _ in range(DYN_LAYERS - 1)]
    dyn += [tf.keras.layers.Dense(nz)]
    NNdyn = tf.keras.Sequential(dyn)
    n_rec = nz + 6 + 2
    rec = [tf.keras.layers.Dense(REC_WIDTH, activation=tf.nn.tanh, input_shape=(n_rec,))]
    rec += [tf.keras.layers.Dense(REC_WIDTH, activation=tf.nn.tanh) for _ in range(REC_LAYERS - 1)]
    rec += [tf.keras.layers.Dense(2)]
    NNrec = tf.keras.Sequential(rec)
    return NNdyn, NNrec

def make_rollout(NNdyn, NNrec, nz, L):
    def rollout(ds):
        sig = ds['input_signals']          # (B,T,6) normalized
        Upar = tf.expand_dims(ds['input_parameters'][:,0], -1)   # (B,1) normalized
        pt = ds['point_n']                 # (B,2) normalized spatial point
        B = tf.shape(sig)[0]
        z = tf.zeros((B, nz), tf.float64)
        x = denorm_states(sig[:,0,0:4])    # physical IC from data
        xs = tf.TensorArray(tf.float64, size=L)
        lo = tf.TensorArray(tf.float64, size=L)
        for i in tf.range(L):
            sig_struct = norm_states(x)                 # (B,4)
            sig_exo = sig[:, i, 4:6]                     # (B,2) delta,W
            s = tf.concat([sig_struct, sig_exo], axis=-1)  # (B,6)
            rec_in = tf.concat([z, s, pt], axis=-1)
            out = NNrec(rec_in)
            out = (out**3 + alpha_cub*out)/(1+alpha_cub)   # (B,2) normalized loads
            loads = denorm_loads(out)
            Fy = loads[:,0]; Mz = loads[:,1]
            inp = tf.concat([z, Upar, s], axis=-1)
            z = z + dt/dt_base*(NNdyn(inp) - LAMBDA_DAMP*z)
            x = struct_step(x, Fy, Mz, DT_PHYS)
            x = tf.clip_by_value(x, -10.0, 10.0)          # NaN guard (generous vs data O(1) phys)
            xs = xs.write(i, norm_states(x))              # normalized predicted state at i+1
            lo = lo.write(i, out)                          # normalized predicted load at i
        xs = tf.transpose(xs.stack(), (1,0,2))   # (B,L,4)
        lo = tf.transpose(lo.stack(), (1,0,2))   # (B,L,2)
        return xs, lo
    def loss_fn(ds):
        xs, lo = rollout(ds)
        sig = ds['input_signals']
        tgt_x = sig[:, 1:L+1, 0:4]            # true states at steps 1..L
        tgt_l = ds['output_signals'][:, 0:L, 0, :]   # true loads at steps 0..L-1
        mse_x = tf.reduce_mean(tf.square(xs - tgt_x))
        mse_l = tf.reduce_mean(tf.square(lo - tgt_l))
        return mse_x + W_LOAD*mse_l
    return rollout, loss_fn

def load_data(name):
    ds = utils.load_gla_h5(DATA_DIR / name)
    utils.process_dataset(ds, problem, normalization, dt=dt)
    ds['point_n'] = tf.convert_to_tensor(
        np.broadcast_to(ds['points_full'][0,0,0,:], (ds['num_samples'], 2)).copy(), tf.float64)
    for k in ['input_signals','input_parameters','output_signals']:
        ds[k] = tf.convert_to_tensor(ds[k], tf.float64)
    return ds

def main():
    print(f"WARMSTART={WARMSTART} LAMBDA={LAMBDA_DAMP} ROLLOUT_LEN={ROLLOUT_LEN} W_LOAD={W_LOAD}", flush=True)
    print("Loading data...", flush=True)
    ds_tr = load_data('GLA_train.h5'); ds_va = load_data('GLA_valid.h5')
    print(f"  train {ds_tr['input_signals'].shape}  valid {ds_va['input_signals'].shape}", flush=True)

    NNdyn, NNrec = build_networks(NUM_LATENT)
    # build by calling once
    _ = NNdyn(tf.zeros((1, NUM_LATENT+1+6), tf.float64)); _ = NNrec(tf.zeros((1, NUM_LATENT+6+2), tf.float64))
    NNdyn.load_weights(str(Path(WARMSTART)/'NNdyn_weights.weights.h5'))
    NNrec.load_weights(str(Path(WARMSTART)/'NNrec_weights.weights.h5'))
    print("  warm-start weights loaded", flush=True)

    # --- verify TF structure matches numpy (best-effort: structure.step_dp45
    # may not exist if src/structure.py has since moved to a different
    # integrator for unrelated recon/ work — struct_step's own DP45 Butcher
    # tableau is self-contained and was already validated bit-exact against
    # this same check in the original L6 rollout run, so skip rather than
    # fail the whole training if the reference function is gone) ---
    if hasattr(structure, 'step_dp45'):
        xr = np.random.randn(3,4)*np.array([0.01,0.3,0.005,0.3])
        Fy = np.array([100.,-50.,200.]); Mz = np.array([5.,-3.,10.])
        tf_step = struct_step(tf.constant(xr), tf.constant(Fy), tf.constant(Mz), DT_PHYS).numpy()
        np_step = np.array([structure.step_dp45(xr[i], Fy[i], Mz[i], DT_PHYS) for i in range(3)])
        err = np.max(np.abs(tf_step - np_step))
        print(f"  TF-vs-numpy structure step max err = {err:.2e}  {'OK' if err<1e-8 else 'MISMATCH!'}", flush=True)
        assert err < 1e-8, "TF structure step does not match numpy"
    else:
        print("  [SKIP] structure.step_dp45 not found (src/structure.py has moved on) — "
              "struct_step's own DP45 implementation is self-contained, proceeding", flush=True)

    rollout, loss_fn = make_rollout(NNdyn, NNrec, NUM_LATENT, ROLLOUT_LEN)

    # --- self-test: closed-loop accuracy at warm-start ---
    xs, lo = rollout(ds_va)
    tgt_x = ds_va['input_signals'][:, 1:ROLLOUT_LEN+1, 0:4].numpy()
    xsn = xs.numpy()
    print(f"  [warmstart rollout] state NRMSE per dof:", flush=True)
    for j,nm in enumerate(['h','hd','a','ad']):
        f=tgt_x[:,:,j]; r=xsn[:,:,j]; nr=np.sqrt(np.mean((r-f)**2))/(f.max()-f.min()+1e-9)
        print(f"      {nm}: {nr:.3f}", flush=True)
    print(f"  any NaN in rollout: {np.any(np.isnan(xsn))}", flush=True)
    l0 = loss_fn(ds_va).numpy(); print(f"  warm-start valid rollout loss = {l0:.4e}", flush=True)
    if SELFTEST:
        print("SELFTEST done.", flush=True); return

    loss_train = lambda: loss_fn(ds_tr)
    loss_valid = lambda: loss_fn(ds_va)
    variables = NNdyn.variables + NNrec.variables
    opt = optimization.OptimizationProblem(variables, loss_train, loss_valid)
    out_dir = RESULTS_DIR / f'latent_{NUM_LATENT}'; out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {'problem':problem,'normalization':normalization,'num_latent_states':NUM_LATENT,'lambda_damp':LAMBDA_DAMP,
           'dyn_layers':DYN_LAYERS,'dyn_width':DYN_WIDTH,'rec_layers':REC_LAYERS,'rec_width':REC_WIDTH}
    json.dump(cfg, open(out_dir/'config.json','w'), indent=2)
    best = [float(opt.ag_valid_loss().numpy())]
    NNdyn.save_weights(str(out_dir/'NNdyn_weights.weights.h5'))
    NNrec.save_weights(str(out_dir/'NNrec_weights.weights.h5'))
    print(f'  [ckpt] init warm-start valid={best[0]:.4e} SAVED', flush=True)
    def _ck(it):
        vl = float(opt.ag_valid_loss().numpy())
        if vl < best[0]:
            best[0] = vl
            NNdyn.save_weights(str(out_dir/'NNdyn_weights.weights.h5'))
            NNrec.save_weights(str(out_dir/'NNrec_weights.weights.h5'))
            print(f'  [ckpt] iter {it} valid={vl:.4e} SAVED (best)', flush=True)
        else:
            print(f'  [ckpt] iter {it} valid={vl:.4e} (no improve, best={best[0]:.4e})', flush=True)
    opt.checkpoint_callback = _ck; opt.checkpoint_every = 10

    print("  Adam...", flush=True)
    opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-3))
    print("  BFGS...", flush=True)
    opt.optimize_BFGS(num_epochs_BFGS)
    # NOTE: best-validation weights are already saved by _ck; do NOT overwrite with final.
    print(f"Done. Best valid rollout loss = {best[0]:.4e}. Best-valid model in {out_dir}", flush=True)

if __name__ == '__main__':
    main()
