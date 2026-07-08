#!/usr/bin/env python3
"""Field-reconstruction LDNet training (vx, vy, p) with a latent-state sweep.

Adapted from src/sensitivity_latent.py, but trains the reconstruction network against
the FULL spatial fields (output_fields = [vx, vy, p]) instead of the scalar forces:
  * NNrec output dim = len(output_fields) = 3
  * loss matches NNrec(z, signals, x, y) to the CFD fields over a (subsampled) point cloud
  * latent sweep reproduces the NRMSE-vs-num-latent-states curve (paper panel b)

Reuses the validated LDNet field machinery in src/: utils.process_dataset / dataset_normalize
(point subsampling + normalization) and optimization.OptimizationProblem (Adam + BFGS).
Normalization ranges are computed from the data (esp. pressure), not hardcoded.

Runs on the cluster GPU (TF / Keras 2.14 container). Example:
  DATA=recon/data python3 recon/train_fields.py \
     --train recon/data/FIELDS_train.h5 --valid recon/data/FIELDS_valid.h5 \
     --test recon/data/FIELDS_test.h5 --out recon/models --latents 1,5,10
"""
import argparse, csv, json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf

tf.keras.backend.set_floatx("float64")

# import the project's field pipeline (utils + optimization)
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import utils          # noqa: E402
import optimization   # noqa: E402

FIELD_NAMES = ["vx", "vy", "p"]
SIGNAL_NAMES = ["h", "hd", "a", "ad", "delta", "W_gust"]


def problem_def():
    return {
        "space": {"dimension": 2},
        "input_parameters": [{"name": "U_inf"}],
        "input_signals": [{"name": n} for n in SIGNAL_NAMES],
        "output_signals": [{"name": "F_y"}, {"name": "M_z"}],
        "output_fields": [{"name": n} for n in FIELD_NAMES],
    }


def _rng(lo, hi):
    """Min/max guarded against a zero range (e.g. delta==0 for gust-only family A),
    which would divide-by-zero in normalization. A constant channel maps to 0."""
    lo, hi = float(lo), float(hi)
    if hi - lo < 1e-9:
        c = 0.5 * (lo + hi); lo, hi = c - 0.5, c + 0.5
    return {"min": lo, "max": hi}


def compute_normalization(train, dt_base):
    """Min/max normalization ranges from the training arrays."""
    inp = train["input_signals"]      # (N,T,6)
    of  = train["output_fields"]      # (N,T,P,3)
    os_ = train["output_signals"]     # (N,T,1,2)
    pts = train["points"]             # (P,2)
    norm = {
        "space": {"min": [float(pts[:, 0].min()), float(pts[:, 1].min())],
                  "max": [float(pts[:, 0].max()), float(pts[:, 1].max())]},
        "time": {"time_constant": dt_base},
        "input_parameters": {"U_inf": {"min": 0.0, "max": 120.0}},
        "input_signals": {SIGNAL_NAMES[i]: _rng(inp[:, :, i].min(), inp[:, :, i].max())
                          for i in range(6)},
        "output_signals": {"F_y": _rng(os_[..., 0].min(), os_[..., 0].max()),
                           "M_z": _rng(os_[..., 1].min(), os_[..., 1].max())},
        "output_fields": {FIELD_NAMES[i]: _rng(of[..., i].min(), of[..., i].max())
                          for i in range(3)},
    }
    return norm


def build_networks(num_latent_states, problem, dt, dt_base,
                   dyn_layers=2, dyn_width=7, rec_layers=4, rec_width=24):
    """Defaults (2x7 dyn, 4x24 rec) reproduce the original hardcoded architecture
    exactly (same layer order -> same weight-init RNG draws for a given seed)."""
    n_inp = num_latent_states + len(problem["input_parameters"]) + len(problem["input_signals"])
    dyn = [tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh, input_shape=(n_inp,))]
    dyn += [tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh)
            for _ in range(dyn_layers - 1)]
    dyn += [tf.keras.layers.Dense(num_latent_states)]
    NNdyn = tf.keras.Sequential(dyn)
    n_rec = num_latent_states + len(problem["input_signals"]) + problem["space"]["dimension"]
    rec = [tf.keras.layers.Dense(rec_width, activation=tf.nn.tanh, input_shape=(None, None, n_rec))]
    rec += [tf.keras.layers.Dense(rec_width, activation=tf.nn.tanh)
            for _ in range(rec_layers - 1)]
    rec += [tf.keras.layers.Dense(len(problem["output_fields"]))]   # 3 = vx,vy,p
    NNrec = tf.keras.Sequential(rec)
    return NNdyn, NNrec


class RecordingOptimizationProblem(optimization.OptimizationProblem):
    """OptimizationProblem that records loss histories for later plotting without
    changing the optimization trajectory.

    - `history` collects rows (phase, kind, step, train_loss, valid_loss, wall_s):
        kind='iter' : per-iteration (Adam epoch / BFGS outer iteration), val loss included
        kind='fev'  : per BFGS function evaluation (train loss only; free, cached)
    - record_every=10 reproduces the base class cadence/printout exactly
      (train loss at BFGS iterations is taken from the cached last function
      evaluation = same weights, same value, one forward pass cheaper).
    - optimize_BFGS additionally stores the scipy result (`bfgs_result`) so the
      termination reason (maxiter vs line-search precision loss) is preserved.
    """

    def __init__(self, variables, loss_train, loss_valid,
                 record_every=10, history=None, phase="opt", t0=None):
        self.record_every = max(1, int(record_every))
        self.history = history if history is not None else []
        self.phase = phase
        self.t0 = time.time() if t0 is None else t0
        self._last_eval_loss = None
        self._n_evals = 0
        super().__init__(variables, loss_train, loss_valid)

    def ag_train_loss_grad_numpy(self, params_1d):
        loss, grad = self.ag_train_loss_grad(params_1d)
        l = float(loss.numpy())
        self._n_evals += 1
        self._last_eval_loss = l
        self.history.append((self.phase, "fev", self._n_evals, l, "", time.time() - self.t0))
        return loss.numpy(), grad.numpy()

    def iteration_callback(self):
        if self.iteration % self.record_every == 0:
            tl = self._last_eval_loss if self._last_eval_loss is not None \
                else float(self.ag_train_loss().numpy())
            vl = float(self.ag_valid_loss().numpy())
            self.history.append((self.phase, "iter", self.iteration, tl, vl,
                                 time.time() - self.t0))
            self.iterations_history.append(self.iteration)
            self.loss_train_history.append(tl)
            self.loss_valid_history.append(vl)
            if self.iteration % 10 == 0:
                print('epoch% 5d   -   training loss: %1.3e   -   validation loss %1.3e' %
                      (self.iteration, tl, vl))
        if self.checkpoint_callback is not None and self.iteration > 0 \
                and self.iteration % self.checkpoint_every == 0:
            self.checkpoint_callback(self.iteration)
        self.iteration += 1

    def optimize_BFGS(self, num_epochs):
        import scipy.optimize as sopt
        options = {'maxiter': num_epochs, 'gtol': 1e-100}
        init_params = self.stitcher.stitch(self.variables).numpy()

        def callback(_):
            self.iteration_callback()
            return False

        res = sopt.minimize(fun=self.ag_train_loss_grad_numpy, x0=init_params,
                            method='BFGS', jac=True, tol=1e-100,
                            options=options, callback=callback)
        self.bfgs_result = {
            "nit": int(res.nit), "nfev": int(res.nfev), "status": int(res.status),
            "success": bool(res.success), "message": str(res.message),
            "final_fun": float(res.fun),
            "grad_inf_norm": float(np.max(np.abs(res.jac))),
        }
        # NOTE: like the base class, variables are deliberately left at the last
        # evaluated point (not res.x) to keep behavior identical to src/optimization.py.
        return self.bfgs_result


def make_ldnet(NNdyn, NNrec, num_latent_states, problem, dt, dt_base, output_nl="cubic"):
    """output_nl: 'cubic' = (out^3 + alpha*out)/(1+alpha) tail-compression (baseline,
    ~21x gradient attenuation near 0); 'linear' = identity (healthy unit gradient)."""
    alpha = 0.05

    def evolve_dynamics(dataset):
        ns = dataset["input_signals"].shape[0]
        nt = dataset["input_signals"].shape[1]
        state = tf.zeros((ns, num_latent_states), dtype=tf.float64)
        history = tf.TensorArray(tf.float64, size=nt).write(0, state)
        for i in tf.range(nt - 1):
            inp = tf.concat([state,
                             tf.expand_dims(dataset["input_parameters"][:, 0], -1),
                             dataset["input_signals"][:, i, :]], axis=-1)
            state = state + dt / dt_base * NNdyn(inp)
            history = history.write(i + 1, state)
        return tf.transpose(history.stack(), perm=(1, 0, 2))

    def reconstruct(dataset, states):
        ns, nt, npx = dataset["num_samples"], dataset["num_times"], dataset["num_points"]
        s = tf.broadcast_to(tf.expand_dims(states, 2), [ns, nt, npx, num_latent_states])
        u = tf.broadcast_to(tf.expand_dims(dataset["input_signals"], 2),
                            [ns, nt, npx, len(problem["input_signals"])])
        out = NNrec(tf.concat([s, u, dataset["points_full"]], axis=3))
        if output_nl == "linear":
            return out
        return (out ** 3 + alpha * out) / (1 + alpha)

    def ldnet(dataset):
        return reconstruct(dataset, evolve_dynamics(dataset))

    def loss_fn(dataset, target):
        return tf.reduce_mean(tf.square(ldnet(dataset) - tf.convert_to_tensor(target, tf.float64)))

    return ldnet, loss_fn


def evaluate(ldnet, dataset, problem, norm):
    out_n = ldnet(dataset)
    # denormalize using field ranges (output_fields)
    fmin = np.array([norm["output_fields"][n]["min"] for n in FIELD_NAMES])
    fmax = np.array([norm["output_fields"][n]["max"] for n in FIELD_NAMES])
    rom = utils.normalize_back(out_n, fmin, fmax, axis=3).numpy()
    fom = utils.normalize_back(dataset["output_fields"], fmin, fmax, axis=3).numpy()
    def safe_rho(a, b):
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))) or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(scipy.stats.pearsonr(a, b)[0])

    metrics = {}
    for i, n in enumerate(FIELD_NAMES):
        f, r = fom[..., i], rom[..., i]
        nrmse = float(np.sqrt(np.mean((r - f) ** 2)) / (f.max() - f.min()))
        rho = safe_rho(r.ravel(), f.ravel())
        metrics[f"NRMSE_{n}"] = nrmse
        metrics[f"rho_{n}"] = rho
        print(f"  {n}: NRMSE {nrmse:.3e}  1-rho {1-rho:.3e}")
    nrmse_all = float(np.sqrt(np.mean((rom - fom) ** 2)) / (fom.max() - fom.min()))
    metrics["NRMSE"] = nrmse_all
    metrics["rho"] = safe_rho(rom.ravel(), fom.ravel())
    print(f"  combined NRMSE {nrmse_all:.3e}")
    return metrics


def area_weighted_subset(points, k, tri=None, seed=0):
    """Pick k node indices with probability proportional to node area, to de-bias the
    loss away from the mesh-refined centerline (uniform-over-nodes over-weights it ~7:1).

    Node area = 1/3 * sum of incident triangle areas if a triangulation is given;
    otherwise inverse local node density via a 2D histogram (inverse-density ~ cell area).
    """
    n = points.shape[0]
    if k >= n:
        return np.arange(n)
    if tri is not None:
        v = points[tri]                      # (Ntri, 3, 2)
        a = 0.5 * np.abs((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
                         - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
        w = np.zeros(n)
        for c in range(3):
            np.add.at(w, tri[:, c], a / 3.0)
    else:
        nb = 64
        H, xe, ye = np.histogram2d(points[:, 0], points[:, 1], bins=nb)
        ix = np.clip(np.searchsorted(xe, points[:, 0]) - 1, 0, nb - 1)
        iy = np.clip(np.searchsorted(ye, points[:, 1]) - 1, 0, nb - 1)
        dens = H[ix, iy]
        w = 1.0 / np.maximum(dens, 1.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False, p=w))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latents", default="1,5,10")
    ap.add_argument("--subsample", type=int, default=1024, help="train/valid point subsample")
    ap.add_argument("--adam", type=int, default=200)
    ap.add_argument("--bfgs", type=int, default=2000)
    ap.add_argument("--output-nl", choices=["cubic", "linear"], default="cubic",
                    help="NNrec output transform (cubic=tail-compress baseline, linear=identity)")
    ap.add_argument("--restarts", type=int, default=1,
                    help="Adam restarts from different seeds; BFGS runs on best-val winner")
    ap.add_argument("--sampling", choices=["uniform", "area"], default="uniform",
                    help="point subsampling: uniform-over-nodes (baseline) or area-weighted")
    ap.add_argument("--tri", default=None, help="mesh_triangles.npy for exact area weighting")
    ap.add_argument("--dyn-layers", type=int, default=2,
                    help="number of hidden layers in NNdyn (default 2 = original)")
    ap.add_argument("--dyn-width", type=int, default=7,
                    help="hidden width of NNdyn (default 7 = original)")
    ap.add_argument("--rec-layers", type=int, default=4,
                    help="number of hidden layers in NNrec (default 4 = original)")
    ap.add_argument("--rec-width", type=int, default=24,
                    help="hidden width of NNrec (default 24 = original)")
    ap.add_argument("--log-every", type=int, default=10,
                    help="record train/val loss every N iterations (10 = original cadence; "
                         "1 = full per-epoch/per-iteration history for convergence studies)")
    args = ap.parse_args()

    latents = [int(x) for x in args.latents.split(",")]
    out_dir = Path(args.out); (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    problem = problem_def()

    raw_train0 = utils.load_gla_h5(args.train)
    times = raw_train0["times"]
    dt_base = float(times[1] - times[0]); dt = dt_base
    norm = compute_normalization(raw_train0, dt_base)
    with open(out_dir / "normalization.json", "w") as f:
        json.dump(norm, f, indent=2)

    tri_arr = np.load(args.tri) if args.tri else None

    all_metrics = []
    for nls in latents:
        print(f"\n{'='*60}\n  num_latent_states = {nls}  "
              f"[output_nl={args.output_nl} sampling={args.sampling} "
              f"restarts={args.restarts} adam={args.adam} bfgs={args.bfgs}]\n{'='*60}")
        np.random.seed(0); tf.random.set_seed(0)

        d_tr = utils.load_gla_h5(args.train)
        d_va = utils.load_gla_h5(args.valid)
        d_te = utils.load_gla_h5(args.test)

        if args.sampling == "area":
            idx = area_weighted_subset(d_tr["points"], args.subsample, tri=tri_arr)
            for d in (d_tr, d_va):
                d["points"] = d["points"][idx]
                d["output_fields"] = d["output_fields"][:, :, idx, :]
            utils.process_dataset(d_tr, problem, norm, dt=None)
            utils.process_dataset(d_va, problem, norm, dt=None)
        else:
            utils.process_dataset(d_tr, problem, norm, dt=None, num_points_subsample=args.subsample)
            utils.process_dataset(d_va, problem, norm, dt=None, num_points_subsample=args.subsample)
        utils.process_dataset(d_te, problem, norm, dt=None)

        def fresh():
            NNdyn, NNrec = build_networks(nls, problem, dt, dt_base,
                                          dyn_layers=args.dyn_layers, dyn_width=args.dyn_width,
                                          rec_layers=args.rec_layers, rec_width=args.rec_width)
            ldnet, loss_fn = make_ldnet(NNdyn, NNrec, nls, problem, dt, dt_base,
                                        output_nl=args.output_nl)
            return NNdyn, NNrec, ldnet, loss_fn

        history = []          # (phase, kind, step, train_loss, valid_loss, wall_s)
        phase_wall = {}
        t_run0 = time.time()
        n_params = {}

        # --- Adam phase with restarts; keep the best init by validation loss ---
        best = None  # (val_loss, NNdyn_weights, NNrec_weights)
        for r in range(args.restarts):
            np.random.seed(r); tf.random.set_seed(r)
            NNdyn, NNrec, ldnet, loss_fn = fresh()
            if not n_params:
                n_params = {"NNdyn": int(NNdyn.count_params()),
                            "NNrec": int(NNrec.count_params())}
                n_params["total"] = n_params["NNdyn"] + n_params["NNrec"]
                print(f"  arch: NNdyn {args.dyn_layers}x{args.dyn_width} "
                      f"({n_params['NNdyn']} params), NNrec {args.rec_layers}x{args.rec_width} "
                      f"({n_params['NNrec']} params), total {n_params['total']} params")
            loss_tr = lambda: loss_fn(d_tr, d_tr["output_fields"])
            loss_va = lambda: loss_fn(d_va, d_va["output_fields"])
            t_ph = time.time()
            opt = RecordingOptimizationProblem(NNdyn.variables + NNrec.variables, loss_tr, loss_va,
                                               record_every=args.log_every, history=history,
                                               phase=f"adam_r{r}", t0=t_run0)
            print(f"  Adam (restart {r+1}/{args.restarts})..." if args.restarts > 1 else "  Adam...")
            opt.optimize_keras(args.adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
            phase_wall[f"adam_r{r}"] = time.time() - t_ph
            vl = float(opt.ag_valid_loss().numpy())
            if args.restarts > 1:
                print(f"    restart {r}: val loss after Adam = {vl:.3e}")
            if best is None or vl < best[0]:
                best = (vl, NNdyn.get_weights(), NNrec.get_weights())

        # --- BFGS polish on the best Adam init ---
        NNdyn, NNrec, ldnet, loss_fn = fresh()
        NNdyn.set_weights(best[1]); NNrec.set_weights(best[2])
        loss_tr = lambda: loss_fn(d_tr, d_tr["output_fields"])
        loss_va = lambda: loss_fn(d_va, d_va["output_fields"])
        t_ph = time.time()
        opt = RecordingOptimizationProblem(NNdyn.variables + NNrec.variables, loss_tr, loss_va,
                                           record_every=args.log_every, history=history,
                                           phase="bfgs", t0=t_run0)
        print(f"  BFGS (best Adam val={best[0]:.3e})...")
        bfgs_result = opt.optimize_BFGS(args.bfgs)
        phase_wall["bfgs"] = time.time() - t_ph
        print(f"  BFGS done: nit={bfgs_result['nit']} nfev={bfgs_result['nfev']} "
              f"message='{bfgs_result['message']}' final={bfgs_result['final_fun']:.3e}")

        md = out_dir / f"latent_{nls}"; md.mkdir(exist_ok=True)
        NNdyn.save_weights(str(md / "NNdyn_weights.weights.h5"))
        NNrec.save_weights(str(md / "NNrec_weights.weights.h5"))
        with open(md / "config.json", "w") as f:
            json.dump({"problem": problem, "normalization": norm,
                       "num_latent_states": nls, "output_nl": args.output_nl,
                       "architecture": {"dyn_layers": args.dyn_layers, "dyn_width": args.dyn_width,
                                        "rec_layers": args.rec_layers, "rec_width": args.rec_width,
                                        "n_params_dyn": n_params.get("NNdyn"),
                                        "n_params_rec": n_params.get("NNrec")}}, f, indent=2)

        # --- persist loss history + run info (plot-ready, no re-run needed) ---
        with open(md / "loss_history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "kind", "step", "train_loss", "valid_loss", "wall_s"])
            for row in history:
                w.writerow(row)
        final_tr = next((h[3] for h in reversed(history) if h[1] == "iter"), None)
        final_va = next((h[4] for h in reversed(history) if h[1] == "iter"), None)
        with open(md / "run_info.json", "w") as f:
            json.dump({"argv": sys.argv[1:],
                       "num_latent_states": nls,
                       "dyn_layers": args.dyn_layers, "dyn_width": args.dyn_width,
                       "rec_layers": args.rec_layers, "rec_width": args.rec_width,
                       "n_params": n_params,
                       "restarts": args.restarts, "adam": args.adam, "bfgs": args.bfgs,
                       "output_nl": args.output_nl, "log_every": args.log_every,
                       "best_adam_val": best[0],
                       "bfgs_result": bfgs_result,
                       "final_train_loss": final_tr, "final_valid_loss": final_va,
                       "phase_wall_s": phase_wall,
                       "total_wall_s": time.time() - t_run0}, f, indent=2)

        print("  eval (test, full grid):")
        m = evaluate(ldnet, d_te, problem, norm)
        m["num_latent_states"] = nls
        all_metrics.append(m)
        with open(md / "metrics.json", "w") as f:
            json.dump(m, f, indent=2)

    # NRMSE-vs-latent summary
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(latents, [m["NRMSE"] for m in all_metrics], "ko-", label="combined")
    for n, mk in zip(FIELD_NAMES, ["b^--", "rs--", "gd--"]):
        ax.semilogy(latents, [m[f"NRMSE_{n}"] for m in all_metrics], mk, label=n)
    ax.set_xlabel("num. latent states"); ax.set_ylabel("NRMSE")
    ax.set_xticks(latents); ax.grid(True, which="both", ls=":"); ax.legend()
    ax.set_title("Field-LDNet sensitivity")
    fig.tight_layout(); fig.savefig(out_dir / "summary" / "nrmse_vs_latent.png", dpi=140)
    with open(out_dir / "summary" / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nsaved summary -> {out_dir/'summary'/'nrmse_vs_latent.png'}")


if __name__ == "__main__":
    main()
