#!/usr/bin/env python3
"""Local eager sanity check for GraphRelaxDecoder (--graph-decoder), before
spending a real cluster smoke test. Per this project's own established
lesson, this CANNOT catch autograph/dtype-class bugs (a cluster smoke test
remains mandatory regardless) -- it only catches shape/logic errors:

  1. train-mode forward pass (small subsampled tensor, graph_positions =
     arange(Ng), graph nodes forced to the front) produces finite output of
     the right shape.
  2. eval-mode forward pass (full-grid-sized tensor, graph_positions =
     scattered natural indices) produces finite output of the right shape.
  3. REBIND correctness: build an eval-mode wrapper and a train-mode wrapper
     that share the SAME (randomly-initialized-but-fixed) sub-layer weights,
     feed both the IDENTICAL feature vectors at the graph nodes (just
     arranged at different positions in each tensor), and check the
     corrections they produce at those nodes match exactly. This is exactly
     the mechanism train_fields.py's eval-time NNrec_eval rebind and
     reconstruct_fields.py both rely on.
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_fields import GraphRelaxDecoder, graph_sampling_idx, graph_adjacency_norm

rng = np.random.default_rng(0)

# --- tiny fake mesh: 50 points, 6 of them are "graph nodes" ---
n_full = 50
n_rec = 7          # din_mod(=4) + sdim(=2) + 1 filler, doesn't matter here
Ng = 6
ns, nt = 2, 3
graph_nodes = np.array([5, 12, 20, 21, 33, 40])   # scattered natural indices

# fake triangulation: connect each graph node to its two neighbors in the list
# (toy connectivity, just needs SOME real edges for adjacency to be non-trivial)
tri = np.array([[graph_nodes[i], graph_nodes[(i + 1) % Ng], graph_nodes[(i + 2) % Ng]]
               for i in range(Ng)])
adj_norm = graph_adjacency_norm(graph_nodes, tri)
assert adj_norm.shape == (Ng, Ng)
assert np.allclose(adj_norm.sum(axis=1), 1.0), "adjacency rows must be normalized"
print(f"adjacency OK: shape={adj_norm.shape}, avg row-sum={adj_norm.sum(axis=1).mean():.3f}")

global_net = tf.keras.layers.Dense(3, activation=None)
global_net(tf.zeros((1, 1, 1, n_rec), dtype=tf.float64))  # force-build

# ---- (1) train-mode wrapper: small subsampled tensor, positions=arange(Ng) ----
subsample = 15
idx = graph_sampling_idx(np.zeros((n_full, 2)), graph_nodes, subsample, seed=0)
assert list(idx[:Ng]) == list(graph_nodes), "graph nodes must be first"
assert len(idx) == subsample and len(set(idx)) == subsample

train_positions = tf.range(Ng)
dec_train = GraphRelaxDecoder(global_net, din_mod=4, graph_positions=train_positions,
                              adj_norm=adj_norm, hidden=8, n_relax=2)

x_full = tf.constant(rng.normal(size=(ns, nt, n_full, n_rec)))
x_train = tf.gather(x_full, idx, axis=2)   # (ns, nt, subsample, n_rec) -- graph nodes first
out_train = dec_train(x_train)
assert out_train.shape == (ns, nt, subsample, 3)
assert np.all(np.isfinite(out_train.numpy())), "train-mode output has non-finite values"
print(f"train-mode forward OK: shape={out_train.shape}")

# ---- (2) eval-mode wrapper: full-grid tensor, scattered natural positions ----
dec_eval = GraphRelaxDecoder(global_net, din_mod=4, graph_positions=graph_nodes,
                             adj_norm=adj_norm, hidden=8, n_relax=2)
dec_eval.in_proj = dec_train.in_proj
dec_eval.self_layers = dec_train.self_layers
dec_eval.neigh_layers = dec_train.neigh_layers
dec_eval.out_proj = dec_train.out_proj

out_eval = dec_eval(x_full)
assert out_eval.shape == (ns, nt, n_full, 3)
assert np.all(np.isfinite(out_eval.numpy())), "eval-mode output has non-finite values"
print(f"eval-mode forward OK: shape={out_eval.shape}")

# ---- (3) rebind correctness: same feature values at graph nodes -> same correction ----
# x_train's first Ng columns ARE x_full's graph_nodes columns (both gathered
# from the same x_full via idx[:Ng] == graph_nodes), so the two wrappers see
# IDENTICAL x_graph input despite different points-axis sizes/positions.
corr_train_at_nodes = (out_train - dec_train.global_net(x_train)).numpy()[:, :, :Ng, :]
corr_eval_at_nodes = (out_eval - dec_eval.global_net(x_full)).numpy()[:, :, graph_nodes, :]
max_diff = np.abs(corr_train_at_nodes - corr_eval_at_nodes).max()
assert max_diff < 1e-10, f"train/eval rebind mismatch at graph nodes: max_diff={max_diff:.3e}"
print(f"rebind correctness OK: train-mode vs eval-mode correction at graph nodes "
      f"match to {max_diff:.2e}")

# ---- (4) far-field points get ZERO correction (one-hot scatter must not leak) ----
far_idx = [i for i in range(n_full) if i not in set(graph_nodes.tolist())]
corr_eval_far = (out_eval - dec_eval.global_net(x_full)).numpy()[:, :, far_idx, :]
assert np.abs(corr_eval_far).max() == 0.0, "non-graph points must get exactly zero correction"
print("scatter isolation OK: non-graph points get exactly zero correction")

print("\nALL LOCAL CHECKS PASSED (shape/logic only -- cluster smoke test still required "
      "for autograph/dtype-class bugs, per project history)")
