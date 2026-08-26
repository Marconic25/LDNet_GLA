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


def compute_normalization(train, dt_base, signal_names=None):
    """Min/max normalization ranges from the training arrays. Space bounds are
    per-column, so extra point-feature columns (wall distance, BL mask) are
    normalized alongside x,y with zero changes in src/. signal_names overrides
    the default 6-channel SIGNAL_NAMES list (e.g. +2 for --add-signal-rates);
    length must match train['input_signals'].shape[-1]."""
    names = signal_names if signal_names is not None else SIGNAL_NAMES
    inp = train["input_signals"]      # (N,T,len(names))
    of  = train["output_fields"]      # (N,T,P,3)
    os_ = train["output_signals"]     # (N,T,1,2)
    pts = train["points"]             # (P, 2+F)
    norm = {
        "space": {"min": [float(pts[:, j].min()) for j in range(pts.shape[1])],
                  "max": [float(pts[:, j].max()) for j in range(pts.shape[1])]},
        "time": {"time_constant": dt_base},
        "input_parameters": {"U_inf": {"min": 0.0, "max": 120.0}},
        "input_signals": {names[i]: _rng(inp[:, :, i].min(), inp[:, :, i].max())
                          for i in range(len(names))},
        "output_signals": {"F_y": _rng(os_[..., 0].min(), os_[..., 0].max()),
                           "M_z": _rng(os_[..., 1].min(), os_[..., 1].max())},
        "output_fields": {FIELD_NAMES[i]: _rng(of[..., i].min(), of[..., i].max())
                          for i in range(3)},
    }
    return norm


def signal_rate_channels(input_signals, dt_base, w_idx, delta_idx):
    """Finite-difference rate channels for W_gust and delta (--add-signal-rates lever,
    stall investigation): h,alpha already have hd,ad provided as inputs, but W_gust
    and delta have NO rate channel anywhere in the baseline -- exactly the gap the
    Neural-CDE experiment's post-mortem identified (it REPLACED value-conditioning
    with rate-only conditioning and lost; this ADDS the missing rates on top of the
    standard concatenation instead, the untested middle ground flagged there).
    np.gradient uses one-sided differences at the edges -- same length as input,
    no shape change. Returns (N,T,2) [Wdot, deltadot] to concatenate onto
    input_signals; dt_base is the shared uniform time spacing (as used elsewhere
    for the dt/dt_base rollout ratio)."""
    Wd = np.gradient(input_signals[:, :, w_idx], axis=1) / dt_base
    dd = np.gradient(input_signals[:, :, delta_idx], axis=1) / dt_base
    return np.stack([Wd, dd], axis=2)


def build_fourier_B(scales, m, seed=12345):
    """Gaussian random Fourier-feature frequency matrix B: (2, m*len(scales)).
    Columns for scale s are drawn ~ N(0, s^2). Seed is FIXED (not the run seed) so
    every seed/arm shares the same feature basis -> a fair 'does FF help' test; only
    the network init varies with the run seed."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, float(s), size=(2, m)) for s in scales], axis=1)


def fourier_encode(coords, B):
    """coords [...,2] (normalized x,y) -> [cos, sin](2*pi * coords @ B), dim 2*M."""
    proj = 2.0 * np.pi * tf.matmul(coords, tf.constant(B, tf.float64))
    return tf.concat([tf.cos(proj), tf.sin(proj)], axis=-1)


class ModulatedSiren(tf.keras.Model):
    """Shift-modulated SIREN decoder (CORAL / Functa style) as a drop-in NNrec
    (D-RES lever, [[dynamic-residual-levers]]).

    Receives the SAME concatenated tensor the tanh decoder gets,
    [latent_state (nls) | input_signals (nsig) | coords (din_coord)], and splits
    it back into a conditioning code c=[z,u] and the spatial coords. A small
    modulation MLP maps c -> per-sine-layer modulation; the SIREN base maps
    coords -> field, then a linear head. Sine activations supply the
    high-frequency capacity directly (no Fourier features needed), and the
    modulation conditions the field on (z,u) without touching the latent ODE.

    mod_type: 'shift' (CORAL/Functa, default) -> sin(omega0*(W h + b) + beta_l);
    'film' (FiLM, scale+shift) -> sin(omega0*(gamma_l * (W h + b)) + beta_l) with
    gamma_l = 1 + modnet output (so it starts at ~1 and film reduces to a plain
    SIREN at init -> stable). film gives the decoder a multiplicative per-channel
    conditioning knob on top of the additive shift.

    Deviation from CORAL: the modulation is produced amortized by an MLP from
    (z,u), trained end-to-end, not meta-learned per-sample latent codes. SIREN
    init per Sitzmann 2020: first layer U(-1/fan_in, 1/fan_in), hidden layers
    U(-sqrt(6/fan_in)/omega0, +...), linear output head."""

    def __init__(self, din_mod, din_coord, width, depth, out_dim,
                 omega0=30.0, mod_layers=2, mod_width=None, mod_type="shift", **kw):
        super().__init__(**kw)
        self.din_mod = int(din_mod)
        self.width = int(width)
        self.depth = int(depth)
        self.omega0 = float(omega0)
        self.mod_type = str(mod_type)
        self.n_mod = 2 if self.mod_type == "film" else 1   # (scale, shift) vs shift-only
        mod_width = self.width if mod_width is None else int(mod_width)
        self.siren = []
        for l in range(self.depth):
            fan_in = int(din_coord) if l == 0 else self.width
            lim = (1.0 / fan_in) if l == 0 else (np.sqrt(6.0 / fan_in) / self.omega0)
            self.siren.append(tf.keras.layers.Dense(
                self.width, activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(-lim, lim),
                bias_initializer="zeros"))
        self.head = tf.keras.layers.Dense(int(out_dim), activation=None)
        mod = [tf.keras.layers.Dense(mod_width, activation=tf.nn.tanh)
               for _ in range(int(mod_layers))]
        mod += [tf.keras.layers.Dense(self.depth * self.width * self.n_mod, activation=None)]
        self.modnet = tf.keras.Sequential(mod)

    def call(self, x):
        code = x[..., :self.din_mod]          # [z, u], broadcast across points
        h = x[..., self.din_mod:]             # coords (raw, or +wall features)
        m = self.modnet(code)                 # (..., depth*width*n_mod)
        W = self.width
        for l in range(self.depth):
            pre = self.siren[l](h)
            if self.mod_type == "film":
                b = l * W * 2
                scale = 1.0 + m[..., b:b + W]                 # ~1 at init -> stable
                shift = m[..., b + W:b + 2 * W]
                h = tf.sin(self.omega0 * (scale * pre) + shift)
            else:
                shift = m[..., l * W:(l + 1) * W]
                h = tf.sin(self.omega0 * pre + shift)
        return self.head(h)


class SepStateDynamics(tf.keras.Model):
    """Dedicated PARALLEL scalar lag-ODE update for a flow-attachment indicator
    X(t) in (0,1) (1=fully attached, 0=fully separated), Goman-Khrabrov form
    (`--dyn-sep-state` lever, stall/separation investigation,
    STALL_LITERATURE_NOTES.md section 2 item 4/9):
        tau1 * dX/dt + X = X0(gust_rate, flap_rate)
    discretized exactly like NNdyn's own forward-Euler rollout:
        X_next = X + dt/dt_base * (X0(rates) - X) / tau1

    Deliberately a SEPARATE small update, not folded into NNdyn's generic
    concat-MLP output (i.e. NOT part of num_latent_states): the lag-ODE is a
    SPECIFIC, constrained recurrence (bounded exponential relaxation toward a
    target), structurally different from NNdyn's free vector field
    `state = state + dt/dt_base * NNdyn(...)`. Folding X into NNdyn's own
    output vector would let the optimizer learn ANY dynamics for that
    channel, discarding exactly the structural prior -- shared by every
    semi-empirical dynamic-stall model reviewed (Goman-Khrabrov,
    Leishman-Beddoes, ONERA) -- that motivates this lever. X is still made
    VISIBLE to both NNdyn's regular update and NNrec's decoder input (via
    `extra_cond` in reconstruct_states / the dyn_sep_state branch in
    make_ldnet's evolve_dynamics), matching the literature recommendation's
    "concatenated as an extra input alongside z" -- only X's OWN evolution
    uses this dedicated recurrence, not NNdyn's generic one.

    X0(.) MLP: 2 inputs (normalized Wdot, deltadot -- see signal_rate_channels;
    computed independently of --add-signal-rates, which this lever does not
    require -- see main()'s sep_state_norm block), one small tanh hidden layer
    (width=4, fixed -- not a CLI knob, keeps this a CHEAP-MODERATE lever), then
    a sigmoid output. Output-layer bias is initialized to +2.0 (sigmoid(2)
    ~=0.88) so X0(0,0) starts near "mostly attached at rest" -- a
    physically-motivated prior matching this project's quiescent-start
    initial conditions (every sim starts attached), not a free 0.5 coin flip.

    tau1: single trainable scalar (softplus-parameterized, always >0 by
    construction) shared across all samples/times -- the literature calls
    this "a relaxation time constant", not a per-condition quantity. Units:
    number of dt_base steps (dt/dt_base is always 1 in every run in this
    project, but kept general to match NNdyn's own dt/dt_base convention).
    Default init 5 steps -- same order as the confirmed ~0.3s / ~15-dt_base-
    step separation burst width at the real cluster dt_base~=0.02s
    (STALL_LITERATURE_NOTES.md section 2, MEANSPLIT_NOTES.md's STALL/
    SEPARATION HYPOTHESIS section): fast enough to plausibly track the
    event, slow enough to be a genuine LAG (tau1->0 would degenerate to
    X=X0 exactly, discarding the whole point of a lag state)."""

    def __init__(self, hidden=4, tau1_init=5.0, **kw):
        super().__init__(**kw)
        self.x0_hidden = tf.keras.layers.Dense(int(hidden), activation=tf.nn.tanh)
        self.x0_out = tf.keras.layers.Dense(
            1, activation=None,
            bias_initializer=tf.keras.initializers.Constant(2.0))
        raw0 = float(np.log(np.expm1(float(tau1_init))))   # softplus^-1(tau1_init)
        self.tau1_raw = tf.Variable(raw0, trainable=True, dtype=tf.float64,
                                    name="sepstate_tau1_raw")

    def x0(self, rates_n):
        """rates_n: (..., 2) normalized [Wdot, deltadot] -> (..., 1) in (0,1)."""
        return tf.sigmoid(self.x0_out(self.x0_hidden(rates_n)))

    @property
    def tau1(self):
        return tf.nn.softplus(self.tau1_raw) + 1e-3


class GatedLocalDecoder(tf.keras.Model):
    """Additive local-correction decoder, gated by spatial proximity to a
    reference point set (the flap surface) AND a learned function of the
    conditioning code (z,u) -- `--local-decoder` lever, stall/separation
    investigation, STALL_LITERATURE_NOTES.md section 9 item 3 (the one
    remaining, genuinely-new-architecture candidate after the 4 cheaper
    stall levers -- residual-curriculum weighting, signal-rates, flap
    loss-weight, Goman-Khrabrov sep-state -- all lost, see
    MEANSPLIT_NOTES.md's CUMULATIVE SUMMARY).

        output = global_net(x) + spatial_gate(coords) * dynamic_gate(code) * local_net(x)

    Drop-in NNrec replacement (same call signature as ModulatedSiren: takes
    the [code|coords] concatenated tensor, returns (...,3) fields) -- zero
    changes needed to reconstruct_states/make_ldnet, exactly like CORAL
    itself was a drop-in replacement for the plain tanh MLP NNrec.

    global_net / local_net: two INDEPENDENT ModulatedSiren instances with
    their own weights (not a shared/modified single network) -- the "small
    MoE-style... 2-3 decoder heads" recommendation. local_net is typically
    smaller (--local-width/--local-depth) but higher-frequency
    (--local-omega0, default 30 vs the champion's 10): the D-RES investigation
    established the OVERALL smooth residual wants LOW omega0 (high-omega hurt,
    STALL_LITERATURE_NOTES.md/MEANSPLIT_NOTES.md), but this local head's
    target -- a genuinely sharp, sign-changing reversal event -- is a
    structurally different, plausibly higher-frequency signal; kept
    independently configurable rather than assumed.

    spatial_gate: computed fresh each call from the REAL (denormalized,
    physical chord-unit) x,y coordinates against `ref_xy` (the flap surface
    points, also physical units) -- reuses the exact tau=0.3c distance-decay
    convention already validated by the flap loss-weight lever
    (flap_loss_weights), just evaluated inside the model instead of riding
    through the data pipeline as an extra points column (simpler: no
    n_weight_cols-style column bookkeeping, no risk of the child nets
    accidentally seeing an extra input column). xy_min/xy_max invert the
    project's standard normalize_forw ([min,max]->[-1,1]) affine map -- same
    formula as utils.normalize_back, just inlined as a tf op so it can run
    inside the decoder's forward pass. Denormalizing INSIDE the model (not
    once at data-prep time) means the gate stays correct even under multiple
    restarts/fresh() rebuilds with no re-plumbing.

    dynamic_gate: small MLP mapping the conditioning code -> scalar in [0,1]
    via sigmoid, bias-initialized so the gate starts near 0 (sigmoid(-4)~0.018)
    -- ADDITIVE correction, so gate~0 at init means the composite reproduces
    the plain global_net (i.e. the champion) almost exactly at the start of
    training, a conservative, safe initialization (unlike a blend/mixture,
    which would need careful mixing-fraction init to avoid instability)."""

    def __init__(self, global_net, local_net, din_mod, ref_xy, tau,
                 xy_min, xy_max, gate_hidden=8, **kw):
        super().__init__(**kw)
        self.global_net = global_net
        self.local_net = local_net
        self.din_mod = int(din_mod)
        self.ref_xy = tf.constant(np.asarray(ref_xy, dtype=np.float64))   # (R,2) physical
        self.tau = float(tau)
        self.xy_min = tf.constant(np.asarray(xy_min, dtype=np.float64))   # (2,)
        self.xy_max = tf.constant(np.asarray(xy_max, dtype=np.float64))   # (2,)
        self.gate_hidden = tf.keras.layers.Dense(int(gate_hidden), activation=tf.nn.tanh)
        self.gate_out = tf.keras.layers.Dense(
            1, activation=None,
            bias_initializer=tf.keras.initializers.Constant(-4.0))

    def call(self, x):
        code = x[..., :self.din_mod]
        coords_xy_n = x[..., self.din_mod:self.din_mod + 2]     # normalized x,y (first 2 cols)
        coords_xy = 0.5 * (self.xy_min + self.xy_max) + \
            0.5 * (self.xy_max - self.xy_min) * coords_xy_n     # -> physical chord units
        d = tf.reduce_min(
            tf.norm(coords_xy[..., None, :] - self.ref_xy[None, None, None, :, :], axis=-1),
            axis=-1)                                             # (...,) min dist to flap
        # tf.maximum converts its args independently via tf.convert_to_tensor
        # BEFORE comparing dtypes (unlike tensor operators +,-,*, which infer
        # from the tensor side) -- a bare Python 0.0 there resolves to TF's
        # GLOBAL default float32, clashing with d's float64 (this project runs
        # entirely in float64 via tf.keras.backend.set_floatx). Caught by the
        # real cluster training run (TF 2.14), NOT by a local eager check
        # (TF 2.21) -- same class of environment-dependent gap as the
        # sep-state autograph bug; explicit zeros_like sidesteps it.
        spatial = tf.square(tf.maximum(tf.zeros_like(d), 1.0 - d / self.tau))   # (...,) in [0,1]
        dyn = tf.sigmoid(self.gate_out(self.gate_hidden(code)))    # (...,1)
        gate = spatial[..., None] * dyn                             # (...,1)
        return self.global_net(x) + gate * self.local_net(x)


class GraphRelaxDecoder(tf.keras.Model):
    """Additive correction via GENUINE local mesh-graph message-passing on a
    small, fixed near-flap subgraph -- `--graph-decoder` lever, stall/
    separation investigation, DYNAMIC_CONTRIBUTION_LITERATURE_NOTES.md
    recommendation B ("Read, Write, Relax," arXiv:2608.21677): interleave the
    existing GLOBAL CORAL decoder (the "read/write" step) with a small number
    of LOCAL relaxation sweeps restricted to real mesh neighbors (the
    "relax" step), multigrid-cycle style.

    Unlike lever 10 (GatedLocalDecoder): that was a second, INDEPENDENT,
    per-point SIREN head -- every point decoded from (z,u,x,y) alone, no
    information ever crossed between neighboring points. This class performs
    REAL local mixing: each graph node's feature is updated from its actual
    mesh neighbors' features via `adj_norm` (built from mesh_triangles.npy,
    graph_adjacency_norm), K rounds, before producing a correction. This is
    the "genuine message-passing, not a fixed gate" distinction the
    literature review flagged as the reason lever 10's null result does not
    close off decoder-locality as a family.

    Structural constraint this project's fully-batched (all samples x all
    times x all points as ONE dense tensor) training loop imposes: a full
    all-pairs local-mixing operation over every point is computationally
    infeasible (O(n_points^2) per (sample,time), and this project processes
    every (sample,time) simultaneously in one call -- an 11075-point full
    grid would need ~10^11-entry attention/adjacency tensors). The graph is
    therefore a SMALL, FIXED subset of ~200 near-flap nodes (`graph_nodes`,
    `--sampling graph` guarantees their presence at known positions in every
    training batch), not the whole domain.

    graph_positions: integer array of positions WITHIN THE CURRENT points
    tensor's point axis where the graph nodes live -- `tf.range(len(graph_
    nodes))` for training (graph_sampling_idx always places them first) or
    the graph nodes' actual (scattered) indices into the full reference grid
    for inference (reconstruct_fields.py, no subsampling, natural order).
    Different arrays, same class -- the scatter-back step (a one-hot matmul,
    robust to either contiguous-prefix or scattered positions, avoiding
    tf.scatter_nd's batched-index edge cases) does not care which case it is.

    adj_norm: constant (Ng,Ng) row-normalized adjacency + self-loops
    (graph_adjacency_norm)."""

    def __init__(self, global_net, din_mod, graph_positions, adj_norm,
                 hidden=16, n_relax=2, **kw):
        super().__init__(**kw)
        self.global_net = global_net
        self.din_mod = int(din_mod)
        self.graph_positions = tf.constant(np.asarray(graph_positions), dtype=tf.int32)
        self.adj_norm = tf.constant(np.asarray(adj_norm, dtype=np.float64))
        self.n_relax = int(n_relax)
        self.in_proj = tf.keras.layers.Dense(int(hidden), activation=tf.nn.tanh)
        self.self_layers = [tf.keras.layers.Dense(int(hidden), activation=None)
                            for _ in range(self.n_relax)]
        self.neigh_layers = [tf.keras.layers.Dense(int(hidden), activation=None)
                             for _ in range(self.n_relax)]
        # zero-initialized final layer -> composite ~= global_net alone at
        # init (same conservative-init rationale as GatedLocalDecoder's gate)
        self.out_proj = tf.keras.layers.Dense(
            3, activation=None,
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros())

    def call(self, x):
        npx = tf.shape(x)[2]
        x_graph = tf.gather(x, self.graph_positions, axis=2)   # (ns,nt,Ng,feat)
        h = self.in_proj(x_graph)                              # (ns,nt,Ng,hidden)
        for k in range(self.n_relax):
            agg = tf.einsum("ij,...jf->...if", self.adj_norm, h)   # local mean-aggregation
            h = tf.nn.tanh(self.self_layers[k](h) + self.neigh_layers[k](agg))
        corr_graph = self.out_proj(h)                          # (ns,nt,Ng,3)
        # scatter correction back to full point-axis width via a one-hot
        # matmul (robust for both contiguous-prefix and scattered positions,
        # avoids tf.scatter_nd's batched multi-dim index bookkeeping)
        onehot = tf.one_hot(self.graph_positions, depth=npx, dtype=tf.float64)   # (Ng,npx)
        corr_full = tf.einsum("...gd,gp->...pd", corr_graph, onehot)   # (ns,nt,npx,3)
        return self.global_net(x) + corr_full


def build_networks(num_latent_states, problem, dt, dt_base,
                   dyn_layers=2, dyn_width=7, rec_layers=4, rec_width=24,
                   rec_space_dim=None, decoder="mlp",
                   siren_omega0=30.0, siren_mod_layers=2, siren_mod_width=None,
                   siren_mod_type="shift", dyn_cond="concat", dyn_sep_state=False,
                   local_decoder=False, local_ref_xy=None, local_tau=0.3,
                   local_width=16, local_depth=3, local_omega0=30.0,
                   local_gate_hidden=8, local_xy_min=None, local_xy_max=None,
                   graph_decoder=False, graph_positions=None, graph_adj_norm=None,
                   graph_hidden=16, graph_relax_steps=2):
    """Defaults (2x7 dyn, 4x24 rec) reproduce the original hardcoded architecture
    exactly (same layer order -> same weight-init RNG draws for a given seed).
    rec_space_dim overrides the decoder's spatial-input width (Fourier-feature
    encoding widens it beyond problem['space']['dimension']).
    decoder='coral' swaps the tanh NNrec for a shift-modulated SIREN (D-RES arm);
    NNdyn (the latent ODE) uses the plain concatenation update unless dyn_cond
    is 'cde' (see evolve_dynamics_cde) -- then it outputs a
    (num_latent_states x n_channels) matrix and takes z ALONE as input.
    dyn_sep_state=True (--dyn-sep-state lever) widens BOTH NNdyn's input and
    NNrec's input by one extra scalar conditioning channel (the sep-state X,
    see SepStateDynamics) -- NOT added to num_latent_states itself. Only
    implemented for the standard 'concat' dyn_cond path (asserted mutually
    exclusive with dyn_cond='cde' by the caller, main()).
    local_decoder=True (--local-decoder lever, only valid with decoder='coral')
    wraps the plain CORAL NNrec into a GatedLocalDecoder: the CORAL net becomes
    `global_net`, a second smaller/higher-omega0 ModulatedSiren `local_net` is
    built alongside it, gated by local_ref_xy/local_tau (flap proximity,
    physical chord units) and a learned function of (z,u). local_xy_min/
    local_xy_max are the space normalization range's x,y columns (from `norm`,
    already computed by the caller) needed to denormalize coords inside the
    gate -- see GatedLocalDecoder.
    graph_decoder=True (--graph-decoder lever, only valid with decoder='coral',
    mutually exclusive with local_decoder -- asserted by the caller, main())
    wraps the plain CORAL NNrec into a GraphRelaxDecoder: genuine local
    mesh-graph message-passing on a small fixed near-flap node subset
    (graph_positions: where those nodes live in the CURRENT points tensor;
    graph_adj_norm: their precomputed row-normalized adjacency) -- see
    GraphRelaxDecoder."""
    n_sep = 1 if dyn_sep_state else 0
    if dyn_cond == "cde":
        # NB bias init: f_theta depends on z ALONE and z starts at the physical
        # zero initial condition. Keras' default zero-bias init makes every
        # all-tanh layer collapse to an EXACT-zero cascade at input=0
        # (tanh(kernel^T@0 + 0) = 0, feeding 0 into the next layer, etc.), so
        # f_theta(0) = 0 identically regardless of the kernels -> dz = f(0)@dX
        # = 0 forever -> the state can never leave the origin and every kernel
        # gets an exact-zero gradient (verified: caught via a local smoke test
        # before this was ever launched on the cluster). A small random bias
        # breaks the degenerate fixed point so the first step is genuinely
        # kernel-dependent and the state can move.
        bias_init = tf.keras.initializers.RandomNormal(stddev=0.05)
        n_channels = len(problem["input_parameters"]) + len(problem["input_signals"])
        dyn = [tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh,
                                     bias_initializer=bias_init,
                                     input_shape=(num_latent_states,))]
        dyn += [tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh,
                                      bias_initializer=bias_init)
                for _ in range(dyn_layers - 1)]
        dyn += [tf.keras.layers.Dense(num_latent_states * n_channels,
                                      bias_initializer=bias_init)]
        NNdyn = tf.keras.Sequential(dyn)
    else:
        n_inp = num_latent_states + n_sep + len(problem["input_parameters"]) + len(problem["input_signals"])
        dyn = [tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh, input_shape=(n_inp,))]
        dyn += [tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh)
                for _ in range(dyn_layers - 1)]
        dyn += [tf.keras.layers.Dense(num_latent_states)]
        NNdyn = tf.keras.Sequential(dyn)
    sdim = problem["space"]["dimension"] if rec_space_dim is None else rec_space_dim
    n_rec = num_latent_states + n_sep + len(problem["input_signals"]) + sdim
    din_mod = num_latent_states + n_sep + len(problem["input_signals"])
    if decoder == "coral":
        global_net = ModulatedSiren(
            din_mod=din_mod, din_coord=sdim, width=rec_width, depth=rec_layers,
            out_dim=len(problem["output_fields"]), omega0=siren_omega0,
            mod_layers=siren_mod_layers, mod_width=siren_mod_width,
            mod_type=siren_mod_type)
        if local_decoder:
            local_net = ModulatedSiren(
                din_mod=din_mod, din_coord=sdim, width=local_width, depth=local_depth,
                out_dim=len(problem["output_fields"]), omega0=local_omega0,
                mod_layers=siren_mod_layers, mod_width=None, mod_type=siren_mod_type)
            NNrec = GatedLocalDecoder(
                global_net, local_net, din_mod=din_mod, ref_xy=local_ref_xy,
                tau=local_tau, xy_min=local_xy_min, xy_max=local_xy_max,
                gate_hidden=local_gate_hidden)
        elif graph_decoder:
            NNrec = GraphRelaxDecoder(
                global_net, din_mod=din_mod, graph_positions=graph_positions,
                adj_norm=graph_adj_norm, hidden=graph_hidden, n_relax=graph_relax_steps)
        else:
            NNrec = global_net
        # force-build variables; graph_decoder's tf.gather(x, graph_positions,
        # axis=2) needs a points axis at least as large as the largest fixed
        # graph-node index, unlike every other decoder which is pointwise and
        # tolerates the generic npx=1 dummy.
        dummy_npx = int(tf.reduce_max(graph_positions).numpy()) + 1 if graph_decoder else 1
        NNrec(tf.zeros((1, 1, dummy_npx, n_rec), dtype=tf.float64))
    else:
        rec = [tf.keras.layers.Dense(rec_width, activation=tf.nn.tanh,
                                     input_shape=(None, None, n_rec))]
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


def reconstruct_states(NNrec, dataset, states, num_latent_states, problem,
                       output_nl="cubic", fourier_B=None, n_weight_cols=0,
                       extra_cond=None):
    """Decode a (ns, nt, num_latent_states) state trajectory into fields at every
    query point. Shared by the standard single-shot rollout (make_ldnet) and the
    multiple-shooting training rollout (evolve_dynamics_shooting) so both paths
    decode identically -- only how `states` was produced differs.

    n_weight_cols: trailing columns of dataset['points_full'] that are loss-weight
    signals (--loss-weight-mode), NOT real coordinates -- stripped before NNrec sees
    them (NNrec's input width, from problem['space']['dimension'], never counts
    them). 0 (default) = no such columns, byte-identical to pre-lever behavior.

    extra_cond: optional (ns, nt, k) extra per-sample-per-time conditioning
    tensor, concatenated onto the broadcast latent-state code BEFORE u/coords
    (--dyn-sep-state's X trajectory, k=1 -- see make_ldnet). None (default) is
    a pure no-op, byte-identical to the pre-lever behavior."""
    alpha = 0.05
    ns, nt, npx = dataset["num_samples"], dataset["num_times"], dataset["num_points"]
    s = tf.broadcast_to(tf.expand_dims(states, 2), [ns, nt, npx, num_latent_states])
    if extra_cond is not None:
        k = extra_cond.shape[-1]
        ec = tf.broadcast_to(tf.expand_dims(extra_cond, 2), [ns, nt, npx, k])
        s = tf.concat([s, ec], axis=3)
    u = tf.broadcast_to(tf.expand_dims(dataset["input_signals"], 2),
                        [ns, nt, npx, len(problem["input_signals"])])
    coords = dataset["points_full"]
    if n_weight_cols:
        coords = coords[..., :-n_weight_cols]
    if fourier_B is not None:
        coords = tf.concat([fourier_encode(coords[..., :2], fourier_B),
                            coords[..., 2:]], axis=-1)
    out = NNrec(tf.concat([s, u, coords], axis=3))
    if output_nl == "linear":
        return out
    return (out ** 3 + alpha * out) / (1 + alpha)


def evolve_dynamics_shooting(NNdyn, dataset, num_latent_states, dt, dt_base,
                             seg_bounds, seg_init):
    """Multiple-shooting rollout (training only -- see --shooting-segments).

    Splits [0, nt) into len(seg_bounds)-1 segments at the (Python, static)
    indices in seg_bounds (seg_bounds[0]=0, seg_bounds[-1]=nt-1). Segment 0
    starts from the physical z=0 initial condition, exactly like the standard
    evolve_dynamics(); segments 1..K-1 start from their own FREE trainable
    state seg_init[:, k-1, :] (shape (ns, K-1, num_latent_states)) instead of
    inheriting whatever the previous segment's own rollout landed on -- this
    is what decouples long-horizon error accumulation from local model fit
    during training (Turan & Jaeschke 2109.06786).

    Returns:
      states: (ns, nt, num_latent_states), same shape/semantics as
        evolve_dynamics()'s output, so reconstruct_states() is unchanged. At
        each internal boundary the FREE variable (not the previous segment's
        prediction) is what gets decoded/fit against the true field.
      residuals: list of (ns, num_latent_states) tensors, one per internal
        boundary: (segment k's own predicted end state) - (segment k+1's
        free start state). Squared and averaged into the continuity penalty
        by the caller; NOT part of `states` (this is exactly the quantity
        multiple shooting drives to zero as training progresses).
    """
    ns = dataset["input_signals"].shape[0]
    K = len(seg_bounds) - 1
    seg_states, residuals = [], []
    for k in range(K):
        t0, t1 = seg_bounds[k], seg_bounds[k + 1]
        state = tf.zeros((ns, num_latent_states), dtype=tf.float64) if k == 0 \
            else seg_init[:, k - 1, :]
        steps = [state]
        for i in range(t0, t1):
            inp = tf.concat([state,
                             tf.expand_dims(dataset["input_parameters"][:, 0], -1),
                             dataset["input_signals"][:, i, :]], axis=-1)
            state = state + dt / dt_base * NNdyn(inp)
            steps.append(state)
        if k < K - 1:
            residuals.append(state - seg_init[:, k, :])
            seg_states.append(tf.stack(steps[:-1], axis=1))   # drop the overlapping endpoint
        else:
            seg_states.append(tf.stack(steps, axis=1))        # last segment: keep through nt-1
    return tf.concat(seg_states, axis=1), residuals


def evolve_dynamics_cde(NNdyn, dataset, num_latent_states, n_channels):
    """Neural-CDE-style latent update: dz = f_theta(z) @ dX, where X_t is the
    exogenous path [input_parameter, input_signals] and dX is its per-step
    increment (finite-difference derivative, X[i+1]-X[i]). f_theta (NNdyn, built
    with dyn_cond='cde' in build_networks) depends on z ALONE -- no concatenated
    exogenous input -- and outputs a (num_latent_states x n_channels) matrix per
    sample; the control signal enters structurally through the matrix-vector
    contraction with dX, not by nonlinear mixing into the MLP's input like the
    standard concat NNdyn (Kidger, Morrill, Foster, Lyons, Neural Controlled
    Differential Equations for Irregular Time Series, NeurIPS 2020,
    arXiv:2005.08926).

    Deliberately skips the paper's own cubic-spline interpolation of X: its
    stated purpose there is numerical stability of the ADJOINT backward pass
    (their Appendix A.2) -- this project differentiates through the unrolled
    loop directly (no adjoint sensitivity), so that justification does not
    transfer; a per-step finite difference captures the same structural idea
    (rate-of-change-driven, multiplicative coupling) at far lower
    implementation cost/risk. See LATENTODE_LITERATURE_NOTES.md section 6."""
    ns = dataset["input_signals"].shape[0]
    nt = dataset["input_signals"].shape[1]
    param = tf.expand_dims(dataset["input_parameters"][:, 0], -1)   # (ns,1), constant in t
    state = tf.zeros((ns, num_latent_states), dtype=tf.float64)
    history = tf.TensorArray(tf.float64, size=nt).write(0, state)
    for i in tf.range(nt - 1):
        X_i = tf.concat([param, dataset["input_signals"][:, i, :]], axis=-1)
        X_ip1 = tf.concat([param, dataset["input_signals"][:, i + 1, :]], axis=-1)
        dX = X_ip1 - X_i                                            # (ns, n_channels)
        f_mat = tf.reshape(NNdyn(state), [ns, num_latent_states, n_channels])
        state = state + tf.einsum("ndc,nc->nd", f_mat, dX)
        history = history.write(i + 1, state)
    return tf.transpose(history.stack(), perm=(1, 0, 2))


def make_ldnet(NNdyn, NNrec, num_latent_states, problem, dt, dt_base, output_nl="cubic",
               fourier_B=None, dyn_cond="concat", n_weight_cols=0, loss_weight_boost=0.0,
               loss_weight_mode="none", loss_weight_residual_power=1.0,
               dyn_sep_state=False, sepnet=None):
    """output_nl: 'cubic' = (out^3 + alpha*out)/(1+alpha) tail-compression (baseline,
    ~21x gradient attenuation near 0); 'linear' = identity (healthy unit gradient).
    fourier_B: if given, the decoder's (x,y) columns are Fourier-feature encoded
    (spectral-bias remedy); any extra spatial columns (wall features) pass through.
    dyn_cond='cde' switches the latent update to the Neural-CDE-style rollout
    (evolve_dynamics_cde) -- NNdyn must have been built with build_networks(...,
    dyn_cond='cde') to match, or shapes will not line up.

    loss_weight_mode: --loss-weight-mode lever (stall investigation), three-way:
      'none'     -- exact original unweighted MSE (byte-identical, default).
      'flap'     -- STATIC, precomputed geometric weight: the trailing
                    n_weight_cols columns of points_full are a [0,1] flap-proximity
                    signal (flap_loss_weights), NOT coordinates -- stripped before
                    NNrec (reconstruct_states) and used ONLY here to upweight the
                    squared error as (1 + loss_weight_boost*sigma).
      'residual' -- DYNAMIC, curriculum weight built from the model's OWN current
                    prediction: w = stop_gradient(|pred-target|_2 across output
                    fields)^loss_weight_residual_power, mean-normalized. No extra
                    points columns (n_weight_cols stays 0) since nothing is
                    precomputed -- the weight is a pure function of this step's
                    forward pass. Recomputed fresh every loss_fn call (Adam epoch
                    AND every BFGS function evaluation, including rejected
                    line-search trials) rather than an EMA across steps: an EMA
                    would need persistent state that a rejected BFGS trial
                    evaluation would corrupt (scipy's line search calls the loss
                    multiple times per outer iteration, not all of which are
                    accepted), whereas a pure function of the current step's
                    detached residual has no such hazard and needs no extra
                    trainable/state variables. power=0 -> weight==1 everywhere,
                    identical to 'none' up to floating-point (see main()'s
                    byte-identical-when-off check, which tests mode='none' itself,
                    not power=0, as the off-switch). See MEANSPLIT_NOTES.md's
                    dated RESIDUAL-CURRICULUM section for the full design writeup.
    n_weight_cols: trailing weight-signal columns of points_full to strip before
    NNrec sees them (flap mode only; 0 for 'none'/'residual').

    dyn_sep_state/sepnet: --dyn-sep-state lever (stall/separation
    investigation). When True, sepnet (a SepStateDynamics instance) must be
    given; its own dedicated lag-ODE update produces a scalar attachment-
    state trajectory X(t), initialized X(0)=1 (fully attached, matching every
    sim's quiescent start), integrated from dataset['sep_rates'] (normalized
    [Wdot, deltadot], precomputed by the caller via signal_rate_channels --
    independent of --add-signal-rates). X is concatenated as an extra scalar
    channel into BOTH NNdyn's regular update input and NNrec's decoder input
    (via reconstruct_states' extra_cond) -- NOT into num_latent_states
    itself. Mutually exclusive with dyn_cond='cde' and shooting (asserted by
    the caller, main()). False/None (default) is a pure no-op, byte-identical
    to the pre-lever behavior."""
    n_channels = len(problem["input_parameters"]) + len(problem["input_signals"])

    def evolve_dynamics(dataset):
        if dyn_cond == "cde":
            return evolve_dynamics_cde(NNdyn, dataset, num_latent_states, n_channels), None
        ns = dataset["input_signals"].shape[0]
        nt = dataset["input_signals"].shape[1]
        state = tf.zeros((ns, num_latent_states), dtype=tf.float64)
        history = tf.TensorArray(tf.float64, size=nt).write(0, state)
        # NB: dyn_sep_state branches the WHOLE loop, not individual statements
        # inside one shared tf.range loop -- autograph's for_stmt conversion
        # does its own static pass over a loop body's assigned names to set up
        # tf.while_loop's loop-carried variables; a variable (x_history) that's
        # only conditionally assigned INSIDE the loop body (even under a
        # Python-constant `if dyn_sep_state:`) trips "must be defined before
        # the loop" the moment dyn_sep_state=False, since that trace path never
        # executes the branch that would have defined it. Two fully separate
        # loops (one per branch) sidesteps this entirely -- verified against
        # the ValueError this exact pattern threw in the real (autograph-
        # traced, cluster) training path; a local eager forward-pass call does
        # NOT catch this class of bug (no tracing happens), only real training.
        if dyn_sep_state:
            xsep = tf.ones((ns, 1), dtype=tf.float64)   # X(0)=1: fully attached IC
            x_history = tf.TensorArray(tf.float64, size=nt).write(0, xsep)
            rates = dataset["sep_rates"]                # (ns, nt, 2), normalized
            for i in tf.range(nt - 1):
                inp = tf.concat([state, xsep,
                                 tf.expand_dims(dataset["input_parameters"][:, 0], -1),
                                 dataset["input_signals"][:, i, :]], axis=-1)
                state = state + dt / dt_base * NNdyn(inp)
                history = history.write(i + 1, state)
                x0v = sepnet.x0(rates[:, i, :])                      # (ns,1)
                xsep = xsep + dt / dt_base * (x0v - xsep) / sepnet.tau1
                x_history = x_history.write(i + 1, xsep)
            xtraj = tf.transpose(x_history.stack(), perm=(1, 0, 2))
        else:
            for i in tf.range(nt - 1):
                inp = tf.concat([state,
                                 tf.expand_dims(dataset["input_parameters"][:, 0], -1),
                                 dataset["input_signals"][:, i, :]], axis=-1)
                state = state + dt / dt_base * NNdyn(inp)
                history = history.write(i + 1, state)
            xtraj = None
        states = tf.transpose(history.stack(), perm=(1, 0, 2))
        return states, xtraj

    def reconstruct(dataset, states_and_x):
        states, xtraj = states_and_x
        return reconstruct_states(NNrec, dataset, states, num_latent_states, problem,
                                  output_nl=output_nl, fourier_B=fourier_B,
                                  n_weight_cols=n_weight_cols, extra_cond=xtraj)

    def ldnet(dataset, return_aux=False):
        states_and_x = evolve_dynamics(dataset)
        fields = reconstruct(dataset, states_and_x)
        if return_aux:
            return fields, states_and_x[1]
        return fields

    def loss_fn(dataset, target):
        err2 = tf.square(ldnet(dataset) - tf.convert_to_tensor(target, tf.float64))
        if loss_weight_mode == "none":
            return tf.reduce_mean(err2)
        if loss_weight_mode == "flap":
            # undo the [0,1]->[-1,1] min-max normalization applied to the trailing
            # weight column (exact: constructed to touch both 0 and 1, see
            # flap_loss_weights/main()'s assert), then form a properly normalized
            # weighted mean (w=1 everywhere reduces exactly to the plain MSE above).
            sigma = 0.5 * (dataset["points_full"][..., -1] + 1.0)
            w = tf.expand_dims(1.0 + loss_weight_boost * sigma, -1)
        elif loss_weight_mode == "residual":
            # CURRICULUM weight: stop-gradient'd per-point residual magnitude (L2
            # across the 3 output fields, matching how the project's combined-NRMSE
            # already treats fields together), raised to a power and mean-
            # normalized. tf.stop_gradient means this branch contributes NO
            # gradient of its own -- only the `err2` factor below (computed from
            # `ldnet(dataset)` with a live gradient) does, so the network cannot
            # cheat by shrinking the weight instead of the error. The final
            # reduce_sum(w*err2)/reduce_sum(w) ratio is already invariant to any
            # overall constant scale of w (it cancels top and bottom, value AND
            # gradient), so the explicit /mean(w_raw) below is not load-bearing
            # for that invariance -- it is kept so the logged/inspectable weight
            # values are centered at 1 (consistent with 'flap's convention and
            # the task's "mean weight ~=1" framing), and so power=0 (w_raw==1
            # everywhere) is visibly a true no-op.
            resid = tf.sqrt(tf.reduce_sum(tf.stop_gradient(err2), axis=-1)) + 1e-12
            w_raw = tf.pow(resid, loss_weight_residual_power)
            w = tf.expand_dims(w_raw / (tf.reduce_mean(w_raw) + 1e-30), -1)
        else:
            raise ValueError(f"unknown loss_weight_mode {loss_weight_mode!r}")
        return tf.reduce_sum(w * err2) / (tf.reduce_sum(w) * tf.cast(tf.shape(err2)[-1], tf.float64))

    return ldnet, loss_fn


def evaluate(ldnet, dataset, problem, norm, mean_fields=None):
    out_n = ldnet(dataset)
    # denormalize using field ranges (output_fields)
    fmin = np.array([norm["output_fields"][n]["min"] for n in FIELD_NAMES])
    fmax = np.array([norm["output_fields"][n]["max"] for n in FIELD_NAMES])
    rom = utils.normalize_back(out_n, fmin, fmax, axis=3).numpy()
    fom = utils.normalize_back(dataset["output_fields"], fmin, fmax, axis=3).numpy()
    if mean_fields is not None:
        # mean-split: rom/fom above are fluctuations; add the stored mean back so all
        # metrics stay in TOTAL-field units with the same full-range denominators as
        # every previous run (directly comparable NRMSE).
        rom = rom + mean_fields[None, None]
        fom = fom + mean_fields[None, None]
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


def wall_features(points, airfoil_xy, tau):
    """Per-node wall features from the airfoil surface polyline (R2 lever):
      d      = distance to the nearest airfoil-surface node (chord units, c=1)
      sigma  = (max(0, 1 - d/tau))^2   BL mask: 1 at the wall, quadratic decay,
               0 beyond tau (MARIO-style; their C_d ablation 0.794%->4.780%
               without it). Returns (P, 2) [d, sigma]."""
    from scipy.spatial import cKDTree
    d = cKDTree(airfoil_xy).query(points[:, :2])[0]
    sig = np.maximum(0.0, 1.0 - d / tau) ** 2
    return np.stack([d, sig], axis=1)


def flap_loss_weights(points, flap_xy, tau):
    """Per-node flap-proximity weight signal for the loss-REweighting lever (stall
    investigation, [[dynamic-residual-levers]]): sigma = (max(0,1-d/tau))^2 in [0,1],
    d = distance to the nearest FLAP-surface node only (not the whole airfoil like
    wall_features) -- tau is sized to cover the localized transient flow-reversal band
    identified in decomp_stall.py (~0.25c behind the flap TE), not just the boundary
    layer (wall_tau=0.02c). Rides through dataset_normalize exactly like
    wall_features' sigma column and is recovered at loss time via the closed-form
    inverse of the [0,1]->[-1,1] min-max map -- exact, since sigma is constructed to
    touch both 0 (far field) and 1 (points AT a flap surface node), so the empirical
    column min/max are exactly 0/1 (asserted at call time in main())."""
    from scipy.spatial import cKDTree
    d = cKDTree(flap_xy).query(points[:, :2])[0]
    sigma = np.maximum(0.0, 1.0 - d / tau) ** 2
    return sigma[:, None]


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


def graph_sampling_idx(points, graph_nodes, k, seed=0):
    """--sampling graph: FIXED idx = graph_nodes (always present, always at the
    FRONT, same order every call) + uniform-random fill for the remaining
    budget. Same fixed-idx-then-process_dataset-without-subsample pattern as
    area_weighted_subset/nearwall_weighted_subset -- this is what guarantees
    the local graph's nodes are present at KNOWN positions (indices
    0..len(graph_nodes)-1) in every training batch, which GraphRelaxDecoder's
    message-passing step depends on (--graph-decoder lever, stall/separation
    investigation, DYNAMIC_CONTRIBUTION_LITERATURE_NOTES.md recommendation B:
    genuine local message-passing, unlike lever 10's fixed-gated additive
    head)."""
    n = points.shape[0]
    graph_nodes = np.asarray(graph_nodes)
    if k >= n:
        return np.arange(n)
    assert k > len(graph_nodes), \
        f"--subsample ({k}) must exceed len(graph_nodes) ({len(graph_nodes)}) " \
        "to leave room for far-field fill points"
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(n), graph_nodes, assume_unique=False)
    fill = rng.choice(pool, size=k - len(graph_nodes), replace=False)
    return np.concatenate([graph_nodes, fill])   # graph nodes FIRST, fixed order


def graph_adjacency_norm(graph_nodes, tri):
    """Row-normalized (mean-aggregation) adjacency + self-loops for the FIXED
    local graph (--graph-decoder), built from real mesh connectivity
    (mesh_triangles.npy) restricted to edges where BOTH endpoints are in
    graph_nodes. Returns a dense (len(graph_nodes), len(graph_nodes)) float64
    matrix -- dense, not sparse, because the graph is small by construction
    (a few hundred nodes) and every use site (GraphRelaxDecoder.call) needs a
    matmul that broadcasts over the (sample,time) batch dims, which a dense
    small matrix does trivially via tf.einsum."""
    graph_nodes = np.asarray(graph_nodes)
    Ng = len(graph_nodes)
    pos = {int(n): i for i, n in enumerate(graph_nodes)}
    A = np.eye(Ng, dtype=np.float64)   # self-loops
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    for a, b in edges:
        ia, ib = pos.get(int(a)), pos.get(int(b))
        if ia is not None and ib is not None and ia != ib:
            A[ia, ib] = 1.0
            A[ib, ia] = 1.0
    A = A / A.sum(axis=1, keepdims=True)   # row-normalize (mean aggregation)
    return A


def nearwall_weighted_subset(points, airfoil_xy, k, tau, boost=4.0, seed=0):
    """RAD-lite: pick k node indices oversampling the near-wall band, using the SAME
    static reference-geometry distance as wall_features (near-wall lit review #3).
    Deliberately time-invariant (no flap-deflection tracking): unlike a hard no-slip
    OUTPUT mask, a wrong/stale near-wall label here only mis-targets the sampling
    density -- it can never force a physically wrong value, so it is safe even where
    the static reference band does not track the moving flap surface exactly.

    weight = 1 + boost * sigma_bl(d), sigma_bl = (max(0, 1-d/tau))^2 in [0,1], so
    far-field points keep baseline weight 1 and wall-adjacent points get up to
    (1+boost)x the baseline draw probability. boost=0 reduces to uniform."""
    from scipy.spatial import cKDTree
    n = points.shape[0]
    if k >= n:
        return np.arange(n)
    d = cKDTree(airfoil_xy).query(points[:, :2])[0]
    sigma = np.maximum(0.0, 1.0 - d / tau) ** 2
    w = 1.0 + boost * sigma
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
    ap.add_argument("--mean-split", action="store_true",
                    help="store the train-set ensemble+time mean field per point and train "
                         "the decoder on FLUCTUATIONS around it (normalization recomputed on "
                         "fluctuations; mean added back at eval so metrics stay total-field). "
                         "Targets the static near-wall bias (M-SPLIT study).")
    ap.add_argument("--mean-ref", choices=["ensemble", "t0"], default="ensemble",
                    help="mean-split reference: 'ensemble' = mean over sims AND times "
                         "(POD practice, default), 't0' = mean over sims of the first "
                         "snapshot (= trim state; all sims start from the same checkpoint)")
    ap.add_argument("--alpha-reg", type=float, default=0.0,
                    help="Tikhonov weight regularization, reference-LDNet form (per-layer "
                         "mean of squared kernels averaged over layers, biases excluded, "
                         "NNdyn+NNrec; validation monitored WITHOUT the term). 0 disables. "
                         "Loads-sweep finding: 3e-4 flips the d_s verdict there.")
    ap.add_argument("--wall-feats", action="store_true",
                    help="append wall distance d + BL mask (1-d/tau)^2_+ as extra decoder "
                         "point-feature columns (R2 lever); requires --airfoil-nodes")
    ap.add_argument("--airfoil-nodes", default=None,
                    help="npy of airfoil-surface node INDICES into the reference grid "
                         "(analysis_hmetric/airfoil_nodes.npy)")
    ap.add_argument("--wall-tau", type=float, default=0.02,
                    help="BL-mask decay length in chord units (default 0.02, MARIO tau)")
    ap.add_argument("--fourier-scales", default=None,
                    help="comma list of Gaussian RFF scales sigma to encode the decoder "
                         "(x,y) inputs, e.g. '1,5' (multiscale, Aero-Nef best) or '10'. "
                         "Off by default = raw coordinates. Spectral-bias remedy (D-RES arm A).")
    ap.add_argument("--fourier-m", type=int, default=16,
                    help="number of random Fourier frequencies PER scale (feature dim "
                         "= 2*m*len(scales))")
    ap.add_argument("--decoder", choices=["mlp", "coral"], default="mlp",
                    help="NNrec architecture: 'mlp' = tanh MLP (baseline) or 'coral' = "
                         "shift-modulated SIREN decoder (D-RES lever; sine base over "
                         "coords, (z,u) injected as per-layer shifts). Latent ODE "
                         "unchanged either way. coral forces output-nl=linear and is "
                         "mutually exclusive with --fourier-scales (SIREN IS the "
                         "spectral-bias remedy).")
    ap.add_argument("--siren-omega0", type=float, default=30.0,
                    help="SIREN first-/hidden-layer frequency omega0 (Sitzmann default 30)")
    ap.add_argument("--siren-mod-layers", type=int, default=2,
                    help="hidden layers in the (z,u)->shifts modulation MLP (coral only)")
    ap.add_argument("--siren-mod-width", type=int, default=None,
                    help="width of the modulation MLP (coral only; default = --rec-width)")
    ap.add_argument("--siren-mod-type", choices=["shift", "film"], default="shift",
                    help="coral modulation: 'shift' (CORAL, additive beta, default) or "
                         "'film' (scale+shift: gamma*(Wh+b)+beta, more decoder capacity)")
    ap.add_argument("--dyn-cond", choices=["concat", "cde"], default="concat",
                    help="NNdyn conditioning on exogenous signals: 'concat' (baseline, "
                         "[z,params,signals] nonlinearly mixed into one MLP) or 'cde' "
                         "(Neural-CDE style: dz = f_theta(z) @ dX, control enters "
                         "structurally via the signal path's per-step increment, not "
                         "raw concatenation; f_theta depends on z alone). Mutually "
                         "exclusive with --shooting-segments >1 (not implemented together).")
    ap.add_argument("--shooting-segments", type=int, default=1,
                    help="multiple-shooting training rollout: split each trajectory into "
                         "this many segments, each with its own free trainable initial "
                         "latent state (seg 0 still starts from the physical z=0 IC), "
                         "plus a continuity penalty pulling segments back into one "
                         "consistent trajectory (Turan & Jaeschke 2109.06786). 1 = "
                         "disabled, bit-identical to the standard single-shot rollout. "
                         "Training-only: validation/test/inference always use the "
                         "standard full free-running rollout (evolve_dynamics), matching "
                         "actual deployment (no oracle segment states at inference).")
    ap.add_argument("--shooting-lambda", type=float, default=1.0,
                    help="--shooting-segments >1: weight of the continuity penalty "
                         "(mean squared segment-boundary mismatch) added to the data-fit "
                         "MSE. Fixed (not annealed) for this first pass.")
    ap.add_argument("--restarts", type=int, default=1,
                    help="Adam restarts from different seeds; BFGS runs on best-val winner")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="base RNG seed; restart r uses seed-base+r (default 0 = "
                         "historical behavior, bit-identical to all previous runs)")
    ap.add_argument("--sampling", choices=["uniform", "area", "near-wall", "graph"],
                    default="uniform",
                    help="point subsampling: uniform-over-nodes (baseline), area-weighted, "
                         "near-wall-weighted (RAD-lite, requires --airfoil-nodes), or "
                         "'graph' (--graph-decoder: FIXED idx = --graph-nodes first, "
                         "uniform-random fill for the rest -- guarantees the local graph's "
                         "nodes are present at known positions in every training batch)")
    ap.add_argument("--tri", default=None, help="mesh_triangles.npy for exact area weighting "
                    "(also required by --graph-decoder for its adjacency)")
    ap.add_argument("--nearwall-boost", type=float, default=4.0,
                    help="--sampling near-wall: draw-probability multiplier at the wall "
                         "relative to the far field (uses --wall-tau/--airfoil-nodes; "
                         "0 = uniform, higher = more concentrated near the surface)")
    ap.add_argument("--add-signal-rates", action="store_true",
                    help="append finite-difference rate channels for W_gust and "
                         "delta (Wdot, deltadot) to input_signals, fed to BOTH "
                         "NNdyn and NNrec via the standard concatenation -- h,alpha "
                         "already have hd,ad given, W_gust/delta do not. Additive: "
                         "keeps the standard value-conditioning, unlike --dyn-cond "
                         "cde (which REPLACED it with rate-only conditioning and "
                         "lost cleanly, see MEANSPLIT_NOTES.md); this is the "
                         "explicitly-flagged untested middle ground from that "
                         "post-mortem (stall/separation investigation).")
    ap.add_argument("--loss-weight-mode", choices=["none", "flap", "residual"], default="none",
                    help="per-point LOSS reweighting (not sampling): 'flap' upweights "
                         "squared error near the flap surface/wake by "
                         "(1 + loss-weight-boost*sigma), sigma a smooth distance-based "
                         "decay (chord units, --loss-weight-tau) from --flap-nodes -- "
                         "a STATIC, precomputed-before-training geometric weight. "
                         "'residual' instead upweights by the model's OWN current "
                         "detached prediction residual magnitude (curriculum "
                         "weighting; see --loss-weight-residual-power) -- a DYNAMIC "
                         "weight recomputed every training step from that step's "
                         "forward pass, no --flap-nodes required. Distinct from "
                         "--sampling near-wall: does not change WHICH points are "
                         "seen (safe under a fixed subsample budget -- near-wall "
                         "sampling lost by starving far-field coverage), only how "
                         "much each residual counts once selected. Both submodes "
                         "target the transient flap-region flow-reversal event "
                         "found in decomp_stall.py (stall/separation investigation).")
    ap.add_argument("--flap-nodes", default=None,
                    help="npy of FLAP-surface (not whole-airfoil) node INDICES into "
                         "the reference grid, e.g. recon/analysis/flap_nodes.npy. "
                         "Required if --loss-weight-mode flap.")
    ap.add_argument("--loss-weight-tau", type=float, default=0.3,
                    help="--loss-weight-mode flap: distance decay length in chord "
                         "units (default 0.3, sized to cover the ~0.25c-behind-"
                         "flap-TE reversal band found in decomp_stall.py -- much "
                         "larger than --wall-tau's boundary-layer scale)")
    ap.add_argument("--loss-weight-boost", type=float, default=5.0,
                    help="--loss-weight-mode flap: extra weight multiplier at the "
                         "flap surface itself (weight = 1 + boost*sigma, sigma in "
                         "[0,1], matches --nearwall-boost's default for comparability)")
    ap.add_argument("--loss-weight-residual-power", type=float, default=1.0,
                    help="--loss-weight-mode residual: per-point/time weight = "
                         "(stop_gradient(|pred-target|) L2-across-fields)^power, "
                         "mean-normalized (~=1). power=0 -> uniform weight, a true "
                         "no-op (same value as --loss-weight-mode none up to fp). "
                         "Since err2 in the loss is already squared, the effective "
                         "per-point loss contribution scales as resid^(power+2): "
                         "power=1.0 (default, 'moderate') -> resid^3; power=2.0 "
                         "('strong') -> resid^4, concentrating much more sharply "
                         "on the worst-fit points/times (curriculum-PINN framing, "
                         "STALL_LITERATURE_NOTES.md section 5).")
    ap.add_argument("--dyn-sep-state", action="store_true",
                    help="add one extra scalar 'attachment state' X in (0,1) to "
                         "NNdyn's rollout, governed by its OWN dedicated "
                         "Goman-Khrabrov-form lag-ODE (tau1*Xdot + X = "
                         "X0(gust_rate, flap_rate), forward-Euler discretized "
                         "like NNdyn's own update) rather than folded into "
                         "NNdyn's generic MLP output -- see SepStateDynamics. "
                         "X(0)=1 (fully attached IC); X is concatenated as an "
                         "extra conditioning channel into BOTH NNdyn's regular "
                         "update and NNrec's decoder input (NOT into "
                         "num_latent_states). Computes its own Wdot/deltadot "
                         "rate inputs via signal_rate_channels independent of "
                         "--add-signal-rates. Mutually exclusive with "
                         "--dyn-cond cde and --shooting-segments >1. "
                         "STALL_LITERATURE_NOTES.md section 2 item 4/9 "
                         "(#2-ranked recommendation, section 9).")
    ap.add_argument("--sep-state-tau1-init", type=float, default=5.0,
                    help="--dyn-sep-state: initial tau1 guess, in units of "
                         "dt_base steps (softplus-parameterized during "
                         "training, always >0). Default 5 steps -- same "
                         "order as the confirmed ~15-dt_base-step separation "
                         "burst width at the real cluster dt_base~=0.02s.")
    ap.add_argument("--local-decoder", action="store_true",
                    help="wrap NNrec (--decoder coral only) into a GatedLocalDecoder: "
                         "output = global_net(x) + gate(coords,z,u)*local_net(x), an "
                         "ADDITIVE correction from a second, independent (smaller, "
                         "higher-omega0) ModulatedSiren head, gated by flap-proximity "
                         "(--local-tau, physical chord units, --flap-nodes required) "
                         "times a learned function of (z,u) -- gate starts near 0 "
                         "(safe init: composite ~= plain global_net at start of "
                         "training). Genuinely new architecture (STALL_LITERATURE_"
                         "NOTES.md section 9 item 3), the one remaining candidate "
                         "after 4 cheaper stall levers (residual-curriculum weight, "
                         "signal-rates, flap loss-weight, Goman-Khrabrov sep-state) "
                         "all lost -- see GatedLocalDecoder.")
    ap.add_argument("--local-width", type=int, default=16,
                    help="--local-decoder: local_net SIREN width (default 16, "
                         "smaller than the champion global_net's 24)")
    ap.add_argument("--local-depth", type=int, default=3,
                    help="--local-decoder: local_net SIREN depth (default 3, "
                         "smaller than the champion global_net's 4)")
    ap.add_argument("--local-omega0", type=float, default=30.0,
                    help="--local-decoder: local_net SIREN omega0 (default 30, "
                         "the original Sitzmann default -- HIGHER than the "
                         "champion global_net's 10, deliberately: the OVERALL "
                         "smooth residual wanted low omega0, but this head's "
                         "target -- a sharp, sign-changing reversal event -- is "
                         "a structurally different, plausibly higher-frequency "
                         "signal; independently configurable, not assumed)")
    ap.add_argument("--local-tau", type=float, default=0.3,
                    help="--local-decoder: flap-proximity spatial gate decay "
                         "length in physical chord units (default 0.3, same "
                         "convention as --loss-weight-tau)")
    ap.add_argument("--local-gate-hidden", type=int, default=8,
                    help="--local-decoder: hidden width of the (z,u)->dynamic-gate "
                         "MLP (default 8)")
    ap.add_argument("--graph-decoder", action="store_true",
                    help="wrap NNrec (--decoder coral only, mutually exclusive with "
                         "--local-decoder) into a GraphRelaxDecoder: output = "
                         "global_net(x) + scatter(relax(gather(x))), genuine local "
                         "mesh-graph message-passing (real neighbor mixing via real "
                         "mesh connectivity, K rounds) on a small FIXED near-flap "
                         "node subset (--graph-nodes), scattered back as an additive "
                         "correction. Requires --sampling graph and --tri (mesh "
                         "triangulation, for the adjacency). Distinct from "
                         "--local-decoder (lever 10, which never mixed information "
                         "between points -- every point decoded independently from "
                         "(z,u,x,y) alone): this is REAL local coupling, the "
                         "'Read, Write, Relax' mechanism (arXiv:2608.21677), "
                         "DYNAMIC_CONTRIBUTION_LITERATURE_NOTES.md recommendation B, "
                         "the ranked-highest candidate after the DMD diagnostic "
                         "(recommendation A) cheaply ruled out a two-timescale split.")
    ap.add_argument("--graph-nodes", default=None,
                    help="npy of FIXED near-flap node indices for --graph-decoder / "
                         "--sampling graph, e.g. recon/analysis/graph_nodes.npy "
                         "(~200 nodes, a subset of the near-flap band)")
    ap.add_argument("--graph-hidden", type=int, default=16,
                    help="--graph-decoder: hidden width of the message-passing MLPs "
                         "(default 16)")
    ap.add_argument("--graph-relax-steps", type=int, default=2,
                    help="--graph-decoder: number of local relaxation (message-"
                         "passing) rounds (default 2, matching a small multigrid "
                         "V-cycle's relax count, not a deep GNN stack)")
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

    if args.decoder == "coral":
        assert not args.fourier_scales, \
            "--decoder coral is mutually exclusive with --fourier-scales (SIREN sine " \
            "activations are the spectral-bias remedy; do not double up)"
        if args.output_nl != "linear":
            print("coral decoder: forcing --output-nl linear (cubic tail-compression is "
                  "meaningless with sine activations)")
            args.output_nl = "linear"
    assert not (args.dyn_cond == "cde" and args.shooting_segments > 1), \
        "--dyn-cond cde + --shooting-segments >1 not implemented together " \
        "(evolve_dynamics_shooting is concat-only); test them in isolation"
    assert not (args.loss_weight_mode != "none" and args.shooting_segments > 1), \
        "--loss-weight-mode + --shooting-segments >1 not implemented together " \
        "(the shooting training-loss path builds its own MSE, not loss_fn); " \
        "test them in isolation"
    assert not (args.dyn_sep_state and args.dyn_cond == "cde"), \
        "--dyn-sep-state + --dyn-cond cde not implemented together " \
        "(evolve_dynamics_cde has a structurally different update, no " \
        "z-concat loop to attach X to); test them in isolation"
    assert not (args.dyn_sep_state and args.shooting_segments > 1), \
        "--dyn-sep-state + --shooting-segments >1 not implemented together " \
        "(evolve_dynamics_shooting doesn't build/thread a sep-state " \
        "trajectory); test them in isolation"
    if args.local_decoder:
        assert args.decoder == "coral", "--local-decoder requires --decoder coral"
        assert args.flap_nodes, "--local-decoder requires --flap-nodes"
    if args.graph_decoder:
        assert args.decoder == "coral", "--graph-decoder requires --decoder coral"
        assert not args.local_decoder, \
            "--graph-decoder + --local-decoder not implemented together " \
            "(both wrap the same global_net); test in isolation"
        assert args.graph_nodes, "--graph-decoder requires --graph-nodes"
        assert args.tri, "--graph-decoder requires --tri (mesh_triangles.npy, for the adjacency)"
        assert args.sampling == "graph", \
            "--graph-decoder requires --sampling graph (otherwise the graph nodes " \
            "are not guaranteed present at known positions in the training batch)"
    if args.sampling == "graph":
        assert args.graph_nodes, "--sampling graph requires --graph-nodes"

    latents = [int(x) for x in args.latents.split(",")]
    out_dir = Path(args.out); (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    problem = problem_def()
    signal_names = list(SIGNAL_NAMES)
    w_idx = d_idx = None
    if args.add_signal_rates:
        signal_names += ["Wd_gust", "deltad"]
        problem["input_signals"] = [{"name": n} for n in signal_names]
        w_idx, d_idx = SIGNAL_NAMES.index("W_gust"), SIGNAL_NAMES.index("delta")
    # --dyn-sep-state's own rate indices into the base 6-channel SIGNAL_NAMES layout
    # -- fixed/valid regardless of --add-signal-rates (which only ever APPENDS
    # columns after index 5, never reorders), so this lever's X0(.) always has its
    # rate inputs independent of that unrelated flag.
    sep_w_idx, sep_d_idx = SIGNAL_NAMES.index("W_gust"), SIGNAL_NAMES.index("delta")

    raw_train0 = utils.load_gla_h5(args.train)
    times = raw_train0["times"]
    dt_base = float(times[1] - times[0]); dt = dt_base
    if args.add_signal_rates:
        rates0 = signal_rate_channels(raw_train0["input_signals"], dt_base, w_idx, d_idx)
        raw_train0["input_signals"] = np.concatenate([raw_train0["input_signals"], rates0], axis=2)
        print(f"signal-rates ON: +2 input channels (Wd_gust, deltad), "
              f"range Wd=[{rates0[...,0].min():.3g},{rates0[...,0].max():.3g}] "
              f"deltad=[{rates0[...,1].min():.3g},{rates0[...,1].max():.3g}]")
    sep_state_norm = None
    if args.dyn_sep_state:
        # Rate-input normalization range for the X0(.) MLP, computed from the raw
        # (pre-normalization) train signals, same _rng-guarded min/max convention as
        # compute_normalization(). Independent of --add-signal-rates: reads straight
        # off raw_train0["input_signals"][..., sep_w_idx/sep_d_idx] which is always
        # valid regardless of whether that flag has already appended extra columns.
        sep_rates0 = signal_rate_channels(raw_train0["input_signals"], dt_base,
                                          sep_w_idx, sep_d_idx)
        sep_state_norm = {"Wd": _rng(sep_rates0[..., 0].min(), sep_rates0[..., 0].max()),
                          "deltad": _rng(sep_rates0[..., 1].min(), sep_rates0[..., 1].max())}
        print(f"sep-state ON (--dyn-sep-state): X0 rate-input norm ranges "
              f"Wd=[{sep_state_norm['Wd']['min']:.3g},{sep_state_norm['Wd']['max']:.3g}] "
              f"deltad=[{sep_state_norm['deltad']['min']:.3g},{sep_state_norm['deltad']['max']:.3g}], "
              f"tau1_init={args.sep_state_tau1_init:g} steps, X(0)=1 (fully attached)")
    mean_fields = None
    if args.mean_split:
        # mean over the TRAIN set only, per node, per field -> (P,3). 'ensemble' =
        # mean over sims and times (POD practice); 't0' = mean over sims of the first
        # snapshot (trim state; all sims start from the same checkpoint). All FIELDS
        # h5 share the reference extraction grid, so the same mean applies
        # point-aligned to valid/test. Computed BEFORE normalization so the min-max
        # ranges are re-derived on the fluctuation fields.
        if args.mean_ref == "t0":
            mean_fields = raw_train0["output_fields"][:, 0].mean(axis=0)
        else:
            mean_fields = raw_train0["output_fields"].mean(axis=(0, 1))
        raw_train0["output_fields"] = raw_train0["output_fields"] - mean_fields[None, None]
        np.save(out_dir / "mean_fields.npy", mean_fields)
        print(f"mean-split ON (ref={args.mean_ref}): stored train mean "
              f"(P={mean_fields.shape[0]}), fluct ranges: " + ", ".join(
                  f"{FIELD_NAMES[i]} [{raw_train0['output_fields'][..., i].min():.3g}, "
                  f"{raw_train0['output_fields'][..., i].max():.3g}]" for i in range(3)))

    airfoil_xy = None
    if args.wall_feats or args.sampling == "near-wall":
        assert args.airfoil_nodes, \
            "--wall-feats/--sampling near-wall requires --airfoil-nodes"
        air_idx = np.load(args.airfoil_nodes)
        airfoil_xy = raw_train0["points"][air_idx, :2].copy()
    if args.wall_feats:
        raw_train0["points"] = np.concatenate(
            [raw_train0["points"], wall_features(raw_train0["points"], airfoil_xy,
                                                 args.wall_tau)], axis=1)
        problem["space"]["dimension"] = raw_train0["points"].shape[1]
        print(f"wall-feats ON: +2 decoder inputs (d, sigma_bl tau={args.wall_tau}), "
              f"{len(air_idx)} surface nodes, space dim -> "
              f"{problem['space']['dimension']}")
    if args.sampling == "near-wall":
        print(f"near-wall sampling ON: boost={args.nearwall_boost} tau={args.wall_tau} "
              f"(RAD-lite, static reference geometry, training-subsample only)")

    flap_xy_w = None
    n_weight_cols = 0
    if args.loss_weight_mode == "flap":
        assert args.flap_nodes, "--loss-weight-mode flap requires --flap-nodes"
        flap_idx_w = np.load(args.flap_nodes)
        flap_xy_w = raw_train0["points"][flap_idx_w, :2].copy()
        sigma0 = flap_loss_weights(raw_train0["points"], flap_xy_w, args.loss_weight_tau)
        assert abs(float(sigma0.min())) < 1e-9 and abs(float(sigma0.max()) - 1.0) < 1e-9, \
            f"flap loss-weight sigma expected exact range [0,1], got " \
            f"[{sigma0.min()},{sigma0.max()}] -- check --flap-nodes/--loss-weight-tau"
        raw_train0["points"] = np.concatenate([raw_train0["points"], sigma0], axis=1)
        n_weight_cols = 1
        print(f"loss-weight ON (flap): boost={args.loss_weight_boost} "
              f"tau={args.loss_weight_tau}c, {len(flap_idx_w)} flap surface nodes, "
              f"{int((sigma0[:, 0] > 0).sum())}/{len(sigma0)} points touched (sigma>0)")
    elif args.loss_weight_mode == "residual":
        # no precomputed points column -- the weight is a pure function of each
        # training step's own forward-pass residual (see make_ldnet's loss_fn).
        print(f"loss-weight ON (residual curriculum): power="
              f"{args.loss_weight_residual_power} (per-point/time detached "
              f"|pred-target| L2-across-fields, mean-normalized, recomputed every "
              f"Adam epoch / BFGS function evaluation -- no persistent state)")

    local_ref_xy = None
    if args.local_decoder:
        # reuse flap_xy_w if --loss-weight-mode flap already loaded the SAME
        # --flap-nodes reference (avoids loading/asserting twice); otherwise
        # load fresh. Physical (x,y), NOT normalized -- GatedLocalDecoder
        # denormalizes its own coords internally to match.
        if flap_xy_w is not None:
            local_ref_xy = flap_xy_w
        else:
            local_ref_xy = raw_train0["points"][np.load(args.flap_nodes), :2].copy()
        print(f"local-decoder ON: local_net {args.local_depth}x{args.local_width} "
              f"omega0={args.local_omega0:g}, gate tau={args.local_tau}c, "
              f"{len(local_ref_xy)} flap reference nodes")

    graph_nodes_arr = None
    graph_adj_norm_arr = None
    if args.graph_decoder or args.sampling == "graph":
        graph_nodes_arr = np.load(args.graph_nodes)
    if args.graph_decoder:
        tri_for_graph = np.load(args.tri)
        graph_adj_norm_arr = graph_adjacency_norm(graph_nodes_arr, tri_for_graph)
        avg_degree = float((graph_adj_norm_arr > 0).sum(axis=1).mean() - 1)   # -1: exclude self-loop
        print(f"graph-decoder ON: {len(graph_nodes_arr)} fixed near-flap nodes, "
              f"{args.graph_relax_steps} relax steps, hidden={args.graph_hidden}, "
              f"avg mesh-neighbor degree={avg_degree:.1f}")
    # Fourier-feature encoding of the decoder (x,y) inputs (D-RES arm A).
    fourier_B = None
    rec_space_dim = None
    if args.fourier_scales:
        scales = [float(s) for s in args.fourier_scales.split(",")]
        fourier_B = build_fourier_B(scales, args.fourier_m)
        n_extra = problem["space"]["dimension"] - 2   # wall-feat columns, if any
        rec_space_dim = 2 * args.fourier_m * len(scales) + n_extra
        np.save(out_dir / "fourier_B.npy", fourier_B)
        print(f"fourier-feats ON: scales={scales} m={args.fourier_m} -> "
              f"decoder spatial dim {rec_space_dim} (2*{args.fourier_m}*{len(scales)}"
              f"{f' + {n_extra} wall' if n_extra else ''})")

    norm = compute_normalization(raw_train0, dt_base, signal_names=signal_names)
    norm["sep_state_rates"] = sep_state_norm   # None unless --dyn-sep-state (traceability only)
    with open(out_dir / "normalization.json", "w") as f:
        json.dump(norm, f, indent=2)
    local_xy_min = np.array(norm["space"]["min"][:2]) if args.local_decoder else None
    local_xy_max = np.array(norm["space"]["max"][:2]) if args.local_decoder else None

    tri_arr = np.load(args.tri) if args.tri else None

    all_metrics = []
    for nls in latents:
        print(f"\n{'='*60}\n  num_latent_states = {nls}  "
              f"[output_nl={args.output_nl} sampling={args.sampling} "
              f"restarts={args.restarts} adam={args.adam} bfgs={args.bfgs}]\n{'='*60}")
        np.random.seed(args.seed_base); tf.random.set_seed(args.seed_base)

        d_tr = utils.load_gla_h5(args.train)
        d_va = utils.load_gla_h5(args.valid)
        d_te = utils.load_gla_h5(args.test)

        if args.add_signal_rates:
            for d in (d_tr, d_va, d_te):
                r = signal_rate_channels(d["input_signals"], dt_base, w_idx, d_idx)
                d["input_signals"] = np.concatenate([d["input_signals"], r], axis=2)

        if args.dyn_sep_state:
            # Raw (pre-normalization) rates for the X0(.) MLP, normalized via the
            # fixed range computed once from raw_train0 above (sep_state_norm) --
            # same convention as the standard input_signals normalization, but kept
            # as a SEPARATE dataset key ('sep_rates', not folded into
            # input_signals) since this lever must not require --add-signal-rates.
            sep_lo = np.array([sep_state_norm["Wd"]["min"], sep_state_norm["deltad"]["min"]])
            sep_hi = np.array([sep_state_norm["Wd"]["max"], sep_state_norm["deltad"]["max"]])
            for d in (d_tr, d_va, d_te):
                raw_r = signal_rate_channels(d["input_signals"], dt_base, sep_w_idx, sep_d_idx)
                d["sep_rates"] = tf.convert_to_tensor(
                    utils.normalize_forw(raw_r, sep_lo, sep_hi, axis=2), tf.float64)

        if mean_fields is not None:
            for d in (d_tr, d_va, d_te):
                assert d["output_fields"].shape[2] == mean_fields.shape[0], \
                    "mean-split requires all datasets on the shared reference grid"
                d["output_fields"] = d["output_fields"] - mean_fields[None, None]

        if args.wall_feats:
            for d in (d_tr, d_va, d_te):
                d["points"] = np.concatenate(
                    [d["points"], wall_features(d["points"], airfoil_xy,
                                                args.wall_tau)], axis=1)

        if args.loss_weight_mode == "flap":
            for d in (d_tr, d_va, d_te):
                d["points"] = np.concatenate(
                    [d["points"], flap_loss_weights(d["points"], flap_xy_w,
                                                    args.loss_weight_tau)], axis=1)

        if args.sampling == "area":
            idx = area_weighted_subset(d_tr["points"], args.subsample, tri=tri_arr)
            for d in (d_tr, d_va):
                d["points"] = d["points"][idx]
                d["output_fields"] = d["output_fields"][:, :, idx, :]
            utils.process_dataset(d_tr, problem, norm, dt=None)
            utils.process_dataset(d_va, problem, norm, dt=None)
        elif args.sampling == "near-wall":
            idx = nearwall_weighted_subset(d_tr["points"], airfoil_xy, args.subsample,
                                           args.wall_tau, args.nearwall_boost,
                                           seed=args.seed_base)
            for d in (d_tr, d_va):
                d["points"] = d["points"][idx]
                d["output_fields"] = d["output_fields"][:, :, idx, :]
            utils.process_dataset(d_tr, problem, norm, dt=None)
            utils.process_dataset(d_va, problem, norm, dt=None)
        elif args.sampling == "graph":
            # graph nodes FIRST (positions 0..Ng-1, fixed every call) + random
            # fill -- GraphRelaxDecoder's graph_positions=arange(Ng) at train
            # time relies on this exact ordering (see fresh() below).
            idx = graph_sampling_idx(d_tr["points"], graph_nodes_arr, args.subsample,
                                     seed=args.seed_base)
            for d in (d_tr, d_va):
                d["points"] = d["points"][idx]
                d["output_fields"] = d["output_fields"][:, :, idx, :]
            utils.process_dataset(d_tr, problem, norm, dt=None)
            utils.process_dataset(d_va, problem, norm, dt=None)
        else:
            utils.process_dataset(d_tr, problem, norm, dt=None, num_points_subsample=args.subsample)
            utils.process_dataset(d_va, problem, norm, dt=None, num_points_subsample=args.subsample)
        utils.process_dataset(d_te, problem, norm, dt=None)

        K = args.shooting_segments
        ns_tr = d_tr["input_signals"].shape[0]
        nt_tr = d_tr["input_signals"].shape[1]
        if K > 1:
            seg_bounds = [round(i * (nt_tr - 1) / K) for i in range(K + 1)]
            print(f"  multiple-shooting ON: {K} segments (bounds {seg_bounds}), "
                  f"continuity lambda={args.shooting_lambda:g}")

        def fresh():
            NNdyn, NNrec = build_networks(nls, problem, dt, dt_base,
                                          dyn_layers=args.dyn_layers, dyn_width=args.dyn_width,
                                          rec_layers=args.rec_layers, rec_width=args.rec_width,
                                          rec_space_dim=rec_space_dim, decoder=args.decoder,
                                          siren_omega0=args.siren_omega0,
                                          siren_mod_layers=args.siren_mod_layers,
                                          siren_mod_width=args.siren_mod_width,
                                          siren_mod_type=args.siren_mod_type,
                                          dyn_cond=args.dyn_cond,
                                          dyn_sep_state=args.dyn_sep_state,
                                          local_decoder=args.local_decoder,
                                          local_ref_xy=local_ref_xy, local_tau=args.local_tau,
                                          local_width=args.local_width, local_depth=args.local_depth,
                                          local_omega0=args.local_omega0,
                                          local_gate_hidden=args.local_gate_hidden,
                                          local_xy_min=local_xy_min, local_xy_max=local_xy_max,
                                          graph_decoder=args.graph_decoder,
                                          graph_positions=(tf.range(len(graph_nodes_arr))
                                                          if args.graph_decoder else None),
                                          graph_adj_norm=graph_adj_norm_arr,
                                          graph_hidden=args.graph_hidden,
                                          graph_relax_steps=args.graph_relax_steps)
            sepnet = None
            if args.dyn_sep_state:
                sepnet = SepStateDynamics(tau1_init=args.sep_state_tau1_init)
                sepnet.x0(tf.zeros((1, 2), dtype=tf.float64))   # force-build sublayers
            ldnet, loss_fn = make_ldnet(NNdyn, NNrec, nls, problem, dt, dt_base,
                                        output_nl=args.output_nl, fourier_B=fourier_B,
                                        dyn_cond=args.dyn_cond, n_weight_cols=n_weight_cols,
                                        loss_weight_boost=args.loss_weight_boost,
                                        loss_weight_mode=args.loss_weight_mode,
                                        loss_weight_residual_power=args.loss_weight_residual_power,
                                        dyn_sep_state=args.dyn_sep_state, sepnet=sepnet)
            seg_init = tf.Variable(tf.zeros((ns_tr, K - 1, nls), dtype=tf.float64),
                                   trainable=True) if K > 1 else None
            return NNdyn, NNrec, sepnet, ldnet, loss_fn, seg_init

        def make_losses(NNdyn, NNrec, loss_fn, seg_init):
            """Train loss (+ optional reference-LDNet Tikhonov: per-layer mean of
            squared kernels averaged over layers, biases excluded, NNdyn+NNrec);
            validation loss monitored WITHOUT the term (as in sensitivity_latent).
            Validation ALWAYS uses the standard single-shot rollout (loss_fn/ldnet),
            matching real inference -- shooting only ever changes the TRAIN loss."""
            loss_va = lambda: loss_fn(d_va, d_va["output_fields"])

            def reg():
                def wreg(NN):
                    ks = [l.kernel for l in NN.layers if hasattr(l, "kernel")]
                    return tf.add_n([tf.reduce_mean(tf.square(k)) for k in ks]) / len(ks)
                return args.alpha_reg * (wreg(NNdyn) + wreg(NNrec))

            if K > 1:
                def loss_tr():
                    states, residuals = evolve_dynamics_shooting(
                        NNdyn, d_tr, nls, dt, dt_base, seg_bounds, seg_init)
                    pred = reconstruct_states(NNrec, d_tr, states, nls, problem,
                                              output_nl=args.output_nl, fourier_B=fourier_B)
                    data_mse = tf.reduce_mean(tf.square(
                        pred - tf.convert_to_tensor(d_tr["output_fields"], tf.float64)))
                    continuity = tf.add_n([tf.reduce_mean(tf.square(r)) for r in residuals]) \
                        / len(residuals)
                    loss = data_mse + args.shooting_lambda * continuity
                    return loss + reg() if args.alpha_reg > 0 else loss
            elif args.alpha_reg > 0:
                loss_tr = lambda: loss_fn(d_tr, d_tr["output_fields"]) + reg()
            else:
                loss_tr = lambda: loss_fn(d_tr, d_tr["output_fields"])
            return loss_tr, loss_va

        history = []          # (phase, kind, step, train_loss, valid_loss, wall_s)
        phase_wall = {}
        t_run0 = time.time()
        n_params = {}

        # --- Adam phase with restarts; keep the best init by validation loss ---
        best = None  # (val_loss, NNdyn_weights, NNrec_weights, sepnet_weights_or_None, seg_init_value_or_None)
        for r in range(args.restarts):
            np.random.seed(args.seed_base + r); tf.random.set_seed(args.seed_base + r)
            NNdyn, NNrec, sepnet, ldnet, loss_fn, seg_init = fresh()
            if not n_params:
                n_params = {"NNdyn": int(NNdyn.count_params()),
                            "NNrec": int(NNrec.count_params())}
                if sepnet is not None:
                    # sepnet.count_params() would raise: SepStateDynamics has no
                    # call(), only .x0()/.tau1, so Keras never flips its own
                    # `built` flag even though its sublayers built individually
                    # on first use -- sum trainable variables directly instead.
                    n_params["sepnet"] = int(sum(int(np.prod(v.shape))
                                                 for v in sepnet.trainable_variables))
                n_params["total"] = n_params["NNdyn"] + n_params["NNrec"] + n_params.get("sepnet", 0)
                rec_desc = (f"coral-SIREN {args.rec_layers}x{args.rec_width} "
                            f"(omega0={args.siren_omega0:g}, {args.siren_mod_type} mod "
                            f"{args.siren_mod_layers}L)"
                            if args.decoder == "coral"
                            else f"NNrec {args.rec_layers}x{args.rec_width}")
                sep_desc = (f", sepnet ({n_params['sepnet']} params, "
                           f"tau1_init={args.sep_state_tau1_init:g})"
                           if sepnet is not None else "")
                print(f"  arch: NNdyn {args.dyn_layers}x{args.dyn_width} "
                      f"({n_params['NNdyn']} params), {rec_desc} "
                      f"({n_params['NNrec']} params){sep_desc}, total {n_params['total']} params")
            loss_tr, loss_va = make_losses(NNdyn, NNrec, loss_fn, seg_init)
            train_vars = NNdyn.variables + NNrec.variables + \
                        (list(sepnet.variables) if sepnet is not None else []) + \
                        ([seg_init] if K > 1 else [])
            t_ph = time.time()
            opt = RecordingOptimizationProblem(train_vars, loss_tr, loss_va,
                                               record_every=args.log_every, history=history,
                                               phase=f"adam_r{r}", t0=t_run0)
            print(f"  Adam (restart {r+1}/{args.restarts})..." if args.restarts > 1 else "  Adam...")
            opt.optimize_keras(args.adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
            phase_wall[f"adam_r{r}"] = time.time() - t_ph
            vl = float(opt.ag_valid_loss().numpy())
            if args.restarts > 1:
                print(f"    restart {r}: val loss after Adam = {vl:.3e}")
            if best is None or vl < best[0]:
                best = (vl, NNdyn.get_weights(), NNrec.get_weights(),
                        sepnet.get_weights() if sepnet is not None else None,
                        seg_init.numpy() if K > 1 else None)

        # --- BFGS polish on the best Adam init ---
        NNdyn, NNrec, sepnet, ldnet, loss_fn, seg_init = fresh()
        NNdyn.set_weights(best[1]); NNrec.set_weights(best[2])
        if sepnet is not None:
            sepnet.set_weights(best[3])
        if K > 1:
            seg_init.assign(best[4])
        loss_tr, loss_va = make_losses(NNdyn, NNrec, loss_fn, seg_init)
        train_vars = NNdyn.variables + NNrec.variables + \
                    (list(sepnet.variables) if sepnet is not None else []) + \
                    ([seg_init] if K > 1 else [])
        t_ph = time.time()
        opt = RecordingOptimizationProblem(train_vars, loss_tr, loss_va,
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
        if sepnet is not None:
            sepnet.save_weights(str(md / "sepstate_weights.weights.h5"))
        if mean_fields is not None:
            np.save(md / "mean_fields.npy", mean_fields)   # self-contained model dir
        if airfoil_xy is not None:
            np.save(md / "airfoil_xy.npy", airfoil_xy)     # for recon-time features
        if fourier_B is not None:
            np.save(md / "fourier_B.npy", fourier_B)       # for recon-time FF encoding
        if flap_xy_w is not None or local_ref_xy is not None:
            # saved once, reused by whichever of the two levers is active (flap
            # loss-weight and local-decoder use the identical raw flap points)
            np.save(md / "flap_xy.npy",
                   flap_xy_w if flap_xy_w is not None else local_ref_xy)
        if args.graph_decoder:
            # persisted so reconstruct_fields.py never needs --tri again --
            # graph_nodes_arr are natural (full-grid) indices, reused as-is
            # for reconstruction (which always runs on the full grid).
            np.save(md / "graph_nodes.npy", graph_nodes_arr)
            np.save(md / "graph_adj_norm.npy", graph_adj_norm_arr)
        with open(md / "config.json", "w") as f:
            json.dump({"problem": problem, "normalization": norm,
                       "num_latent_states": nls, "output_nl": args.output_nl,
                       "mean_split": bool(args.mean_split),
                       "mean_ref": args.mean_ref if args.mean_split else None,
                       "alpha_reg": args.alpha_reg,
                       "wall_feats": ({"tau": args.wall_tau, "n_extra": 2}
                                      if args.wall_feats else None),
                       "fourier": ({"scales": [float(s) for s in args.fourier_scales.split(",")],
                                    "m": args.fourier_m}
                                   if args.fourier_scales else None),
                       "decoder": args.decoder,
                       "siren": ({"omega0": args.siren_omega0,
                                  "mod_layers": args.siren_mod_layers,
                                  "mod_width": args.siren_mod_width,
                                  "mod_type": args.siren_mod_type}
                                 if args.decoder == "coral" else None),
                       "architecture": {"dyn_layers": args.dyn_layers, "dyn_width": args.dyn_width,
                                        "rec_layers": args.rec_layers, "rec_width": args.rec_width,
                                        "n_params_dyn": n_params.get("NNdyn"),
                                        "n_params_rec": n_params.get("NNrec")},
                       "shooting": ({"segments": K, "lambda": args.shooting_lambda,
                                     "bounds": seg_bounds} if K > 1 else None),
                       "dyn_cond": args.dyn_cond,
                       "add_signal_rates": bool(args.add_signal_rates),
                       "loss_weight": ({"mode": args.loss_weight_mode,
                                        "tau": args.loss_weight_tau,
                                        "boost": args.loss_weight_boost,
                                        "residual_power": (args.loss_weight_residual_power
                                                            if args.loss_weight_mode == "residual"
                                                            else None),
                                        "n_weight_cols": n_weight_cols}
                                       if args.loss_weight_mode != "none" else None),
                       "dyn_sep_state": bool(args.dyn_sep_state),
                       "sep_state": ({"tau1_init": args.sep_state_tau1_init, "hidden": 4,
                                      "rate_norm": sep_state_norm,
                                      "n_params": n_params.get("sepnet"),
                                      "tau1_learned": float(sepnet.tau1.numpy())}
                                     if args.dyn_sep_state else None),
                       "local_decoder": bool(args.local_decoder),
                       "local": ({"width": args.local_width, "depth": args.local_depth,
                                  "omega0": args.local_omega0, "tau": args.local_tau,
                                  "gate_hidden": args.local_gate_hidden}
                                if args.local_decoder else None),
                       "graph_decoder": bool(args.graph_decoder),
                       "graph": ({"n_nodes": int(len(graph_nodes_arr)),
                                  "hidden": args.graph_hidden,
                                  "relax_steps": args.graph_relax_steps,
                                  "graph_nodes_path": str(args.graph_nodes)}
                                if args.graph_decoder else None)}, f, indent=2)

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
                       "mean_split": bool(args.mean_split),
                       "mean_ref": args.mean_ref if args.mean_split else None,
                       "alpha_reg": args.alpha_reg,
                       "wall_feats": bool(args.wall_feats),
                       "shooting_segments": K, "shooting_lambda": args.shooting_lambda if K > 1 else None,
                       "add_signal_rates": bool(args.add_signal_rates),
                       "loss_weight_mode": args.loss_weight_mode,
                       "loss_weight_boost": args.loss_weight_boost if n_weight_cols else None,
                       "loss_weight_residual_power": (args.loss_weight_residual_power
                                                       if args.loss_weight_mode == "residual"
                                                       else None),
                       "dyn_sep_state": bool(args.dyn_sep_state),
                       "sep_state_tau1_init": (args.sep_state_tau1_init
                                               if args.dyn_sep_state else None),
                       "sep_state_tau1_learned": (float(sepnet.tau1.numpy())
                                                  if sepnet is not None else None),
                       "local_decoder": bool(args.local_decoder),
                       "graph_decoder": bool(args.graph_decoder),
                       "seed_base": args.seed_base,
                       "best_adam_val": best[0],
                       "bfgs_result": bfgs_result,
                       "final_train_loss": final_tr, "final_valid_loss": final_va,
                       "phase_wall_s": phase_wall,
                       "total_wall_s": time.time() - t_run0}, f, indent=2)

        ldnet_eval = ldnet
        if args.graph_decoder:
            # d_te is the FULL, naturally-ordered grid (never subsampled) --
            # NNrec's train-time graph_positions=arange(Ng) assumed the
            # "--sampling graph" reordering (graph nodes forced to the front
            # of a size-args.subsample array) and is meaningless here. Build a
            # second thin wrapper around the SAME trained sub-layers (Dense
            # objects are shared by reference, not copied) with
            # graph_positions pointing at the nodes' real indices in the full
            # grid instead -- see graph_nodes.npy / graph_adjacency_norm().
            NNrec_eval = GraphRelaxDecoder(
                NNrec.global_net, din_mod=NNrec.din_mod,
                graph_positions=graph_nodes_arr, adj_norm=graph_adj_norm_arr,
                hidden=args.graph_hidden, n_relax=args.graph_relax_steps)
            NNrec_eval.in_proj = NNrec.in_proj
            NNrec_eval.self_layers = NNrec.self_layers
            NNrec_eval.neigh_layers = NNrec.neigh_layers
            NNrec_eval.out_proj = NNrec.out_proj
            ldnet_eval, _ = make_ldnet(NNdyn, NNrec_eval, nls, problem, dt, dt_base,
                                       output_nl=args.output_nl, fourier_B=fourier_B,
                                       dyn_cond=args.dyn_cond, n_weight_cols=n_weight_cols,
                                       loss_weight_boost=args.loss_weight_boost,
                                       loss_weight_mode=args.loss_weight_mode,
                                       loss_weight_residual_power=args.loss_weight_residual_power,
                                       dyn_sep_state=args.dyn_sep_state, sepnet=sepnet)
        print("  eval (test, full grid):")
        m = evaluate(ldnet_eval, d_te, problem, norm, mean_fields=mean_fields)
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
