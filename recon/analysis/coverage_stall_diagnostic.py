#!/usr/bin/env python3
"""COVERAGE DIAGNOSTIC: does the TRAINING SET actually contain flow-separation
events, or is the champion ROM being asked to extrapolate outside its training
envelope at test time?

Motivation: 11 independent levers have now been tried against the D-RES
near-flap dynamic residual and all lost (see MEANSPLIT_NOTES.md). Every one of
them changed HOW the model learns -- decoder architecture, decoder
conditioning, local gating, genuine mesh-graph message-passing, point
sampling, optimizer, training procedure, dynamics conditioning, loss
weighting, dynamics-side regime memory. NONE of them changed WHAT it learns
from. If the training trajectories never contain a separation event of the
magnitude seen in the hard test case (sim_Cc_060, 23.35% of near-flap nodes
reversed at the gust peak), then no architecture can learn that response and
the correct conclusion is a DATA-COVERAGE limit, not an architectural
ceiling -- a different claim, with a concrete remedy (add separated-flow
cases to training) instead of a dead end.

Method: identical separation indicator to decomp_stall.py -- per near-flap
node, project velocity at every timestep onto that node's OWN t=0 velocity
direction; negative projection = local reversal relative to its own attached
baseline. Computed per training sample, then compared against the test cases.

Caveat checked explicitly below: the t=0-as-attached-baseline assumption is
only valid if each trajectory genuinely starts quiescent (gust and flap both
~zero at t=0). The script verifies this on the signals and reports it rather
than assuming it.

Pure numpy + h5py on the existing dumps. Zero training, zero cluster queue.
"""
import sys
import numpy as np
import h5py
from pathlib import Path

AN = Path(__file__).resolve().parent
DATA = AN.parent / "data"
SIGNAL_NAMES = ["h", "hd", "a", "ad", "delta", "W_gust"]

region_label = np.load(AN / "region_labels.npy")
air = np.load(AN / "airfoil_nodes.npy")
near = region_label == 0


def flap_mask(points):
    """near-flap boolean mask over the full grid (decomp_stall.py convention:
    nearest airfoil-surface node index >= 292 => flap element, not main)."""
    d2 = ((points[:, None, :] - points[air][None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(1)
    return near & (nn >= 292)


def reversal_peaks(h5path, is_flap_near, label):
    """Per-sample peak near-flap reversed-flow fraction + signal sanity check."""
    with h5py.File(h5path, "r") as f:
        of = f["output_fields"]
        sig = f["input_signals"][:]          # (ns, nt, n_signals)
        ns = of.shape[0]
        peaks = np.zeros(ns)
        for i in range(ns):
            vel = np.asarray(of[i, :, :, :2], dtype=np.float64)   # (nt, N, 2)
            vref = vel[0]
            vref_n = vref / (np.linalg.norm(vref, axis=1, keepdims=True) + 1e-12)
            proj = (vel * vref_n[None]).sum(-1)                    # (nt, N)
            rev = proj < 0
            peaks[i] = rev[:, is_flap_near].mean(1).max()
    d_idx, w_idx = SIGNAL_NAMES.index("delta"), SIGNAL_NAMES.index("W_gust")
    t0_delta = np.abs(sig[:, 0, d_idx]).max()
    t0_W = np.abs(sig[:, 0, w_idx]).max()
    print(f"\n--- {label}: {ns} samples ---")
    print(f"  t=0 quiescence check: max|delta(0)|={t0_delta:.4g}, "
          f"max|W_gust(0)|={t0_W:.4g}"
          f"{'  [OK, attached baseline valid]' if max(t0_delta, t0_W) < 1e-6 else '  [WARNING: t=0 not quiescent, baseline assumption weakened]'}")
    print(f"  peak near-flap reversed-flow fraction across samples:")
    print(f"    min={100*peaks.min():.2f}%  median={100*np.median(peaks):.2f}%  "
          f"mean={100*peaks.mean():.2f}%  max={100*peaks.max():.2f}%")
    for thr in (0.05, 0.10, 0.15, 0.20):
        n = int((peaks >= thr).sum())
        print(f"    samples with peak >= {100*thr:4.0f}% : {n:3d}/{ns} ({100*n/ns:5.1f}%)")
    return peaks


def main():
    targets = [
        (DATA / "FIELDS_div_train.h5", "TRAINING SET (FIELDS_div_train)"),
        (DATA / "FIELDS_div_valid.h5", "VALIDATION SET (FIELDS_div_valid)"),
        (DATA / "FIELDS_Cc060.h5", "TEST sim_Cc_060 (the hard case)"),
        (DATA / "FIELDS_A025.h5", "TEST sim_A_025 (gust-only)"),
    ]
    # grid/mask from the first available file (all datasets share the reference grid)
    first = next(p for p, _ in targets if p.exists())
    with h5py.File(first, "r") as f:
        points = f["points"][:]
    assert len(points) == len(region_label), \
        f"grid mismatch: {len(points)} points vs {len(region_label)} region labels"
    is_flap_near = flap_mask(points)
    print(f"near-flap band: {int(is_flap_near.sum())} of {len(points)} nodes")

    results = {}
    for path, label in targets:
        if not path.exists():
            print(f"\n--- {label}: MISSING ({path}), skipped ---")
            continue
        results[label] = reversal_peaks(path, is_flap_near, label)

    print("\n=== VERDICT ===")
    tr = next((v for k, v in results.items() if k.startswith("TRAINING")), None)
    te = next((v for k, v in results.items() if "Cc_060" in k), None)
    if tr is None or te is None:
        print("training and/or Cc_060 missing -- cannot compare")
        return
    te_peak = te.max()
    n_ge = int((tr >= te_peak).sum())
    print(f"hard test case peak reversal = {100*te_peak:.2f}%")
    print(f"training samples reaching that level: {n_ge}/{len(tr)}")
    print(f"training max = {100*tr.max():.2f}%, training median = {100*np.median(tr):.2f}%")
    if n_ge == 0:
        print("\nEXTRAPOLATION: no training sample ever reaches the test case's\n"
              "separation level. The model is being asked to reproduce an event\n"
              "outside its training envelope -- this reframes the 11-lever\n"
              "negative result as a DATA-COVERAGE limit, not (only) an\n"
              "architectural ceiling. Concrete remedy: add separated-flow cases.")
    elif n_ge < 0.05 * len(tr):
        print(f"\nSPARSE COVERAGE: only {n_ge} training samples ({100*n_ge/len(tr):.1f}%)\n"
              "reach the test case's separation level -- the event is in-envelope\n"
              "but heavily under-represented. Partially reframes the negative\n"
              "result; class-imbalance/resampling is a concrete untried remedy.")
    else:
        print(f"\nWELL COVERED: {n_ge} training samples ({100*n_ge/len(tr):.1f}%) reach\n"
              "the test case's separation level. The training envelope genuinely\n"
              "contains this regime -- the architectural-ceiling conclusion from\n"
              "the 11-lever campaign STANDS, and is now stronger for having ruled\n"
              "out the data-coverage explanation.")


if __name__ == "__main__":
    main()
