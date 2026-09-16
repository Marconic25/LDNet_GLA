#!/usr/bin/env python3
"""
gen_matrix.py -- source-of-truth generator for the dataset_v6 FSI campaign.

Writes final/design/run_matrix.csv: 146 rows describing every run of the v6
simulation campaign (families A / B / Cc), consumed downstream by
final/cluster/submit_campaign.sh (one PBS job per row), the field/loads
preprocessing in final/data/, and any v6 study in final/studies/.

Column contract (fixed -- other agents code against this, do not reorder or
rename):

    sim_id,family,split,W0,Tg,r_g,controller,K_ff,t_d,delta_pk,rate_max,R_star,t_end

See README.md in this directory for the full design rationale. Key sources
this script mirrors:
  - light/latex/chapter2.tex, \\label{subsubsec:campaign} and
    \\label{tab:dataset_families}   -- family definitions and LHS intervals
  - light/tests/cs25_combo_study.py -- TG_LIST / W0_LIST / R_GRID
  - light/results_cs25_combo/summary.md -- per-cell MPC R* (6 canonical Tg)
  - clean/data/Family A/sim_info.txt, clean/data/Family B/sim_info.txt
    -- v5 per-run metadata format (mirrored by --from-v5)

Usage:
    python3 final/design/gen_matrix.py --dry-run
    python3 final/design/gen_matrix.py --out final/design/run_matrix.csv
    python3 final/design/gen_matrix.py --n-cc 100   # widen Cc, extra -> prop
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import qmc
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only where scipy is absent
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]          # .../final/design/gen_matrix.py -> repo root
DEFAULT_OUT = THIS_FILE.parent / "run_matrix.csv"
SUMMARY_MD = REPO_ROOT / "light" / "results_cs25_combo" / "summary.md"

COLUMNS = [
    "sim_id", "family", "split", "W0", "Tg", "r_g", "controller",
    "K_ff", "t_d", "delta_pk", "rate_max", "R_star", "t_end",
]

# ---------------------------------------------------------------------------
# Constants mirrored from the sources named above. Keep in sync if those
# files change.
# ---------------------------------------------------------------------------
U_INF = 80.0  # m/s, freestream (chapter2.tex subsubsec:campaign)

# light/tests/cs25_combo_study.py
CC_W0_LIST = [10.0, 20.0, 30.0]
CANONICAL_TG = [0.30, 0.40, 0.50, 0.70, 1.00, 1.20]   # == summary.md's 6 Tg rows
R_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

# Family A / Cc gust bounds (chapter2.tex, tab:dataset_families)
RG_LO, RG_HI = 0.10, 0.60
TG_LO, TG_HI = 0.30, 1.20

# W0/r_g dependency resolution
# ------------------------------------------------------------------
# tab:dataset_families quotes BOTH W0 in [8,48] m/s AND r_g in [0.10,0.60] as
# LHS intervals for families A and Cc, but r_g = W0/U_inf (chapter2.tex,
# subsubsec:campaign) with U_inf = 80 m/s fixed -- the two are not
# independent, so they cannot both be free LHS dimensions. Resolution used
# here: sample r_g on its LHS interval and DERIVE W0 = r_g * U_INF. This is
# consistent iff the derived W0 interval reproduces the quoted one exactly:
#     0.10 * 80 = 8   and   0.60 * 80 = 48
# which is exactly the quoted W0 range, so the table's two rows are two
# views of the same one degree of freedom and this resolution loses nothing.
assert abs(RG_LO * U_INF - 8.0) < 1e-9, "r_g/W0 resolution inconsistent at low end"
assert abs(RG_HI * U_INF - 48.0) < 1e-9, "r_g/W0 resolution inconsistent at high end"

# Family B bounds (tab:dataset_families)
DPK_LO, DPK_HI = 2.0, 15.0      # deg, magnitude (sign handled separately)
RATE_LO, RATE_HI = 20.0, 200.0  # deg/s

# Family row counts (tab:dataset_families / task brief)
N_A_TRAIN, N_A_VAL, N_A_TEST = 20, 5, 5
N_B_TRAIN, N_B_VAL, N_B_TEST = 30, 8, 8
N_CC_DEFAULT = 70
# Baseline Cc split fractions (50/10/10 out of 70); generalised to --n-cc.
CC_TRAIN_FRAC, CC_VAL_FRAC = 50 / 70, 10 / 70

ROUND = 4  # decimal places for physical quantities in the CSV


# ---------------------------------------------------------------------------
# Latin hypercube sampling (scipy if present, else a documented numpy
# fallback implementing the same classical construction).
# ---------------------------------------------------------------------------
def _numpy_lhs(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified-permutation Latin hypercube sample in [0,1)^dim.

    Classical LHS construction (McKay, Conover & Beckman 1979), used only
    when scipy is not importable: for each dimension independently, split
    [0,1) into n equal-probability strata, draw one uniform sample inside
    each stratum, then independently permute the n stratum draws across
    dimensions. Marginally this stratifies every dimension into n bins
    exactly like scipy.stats.qmc.LatinHypercube; it lacks scipy's
    optimisation of pairwise correlation between dimensions, which is not
    needed here (each family draws at most 2 free parameters).
    """
    out = np.empty((n, dim))
    for d in range(dim):
        cut = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(cut)
        out[:, d] = cut
    return out


def lhs_unit(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """n x dim samples in [0,1)^dim, scipy LHS if available else fallback."""
    if n == 0:
        return np.empty((0, dim))
    if _HAVE_SCIPY:
        sampler = qmc.LatinHypercube(d=dim, seed=rng)
        return sampler.random(n)
    return _numpy_lhs(n, dim, rng)


def scale(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + u * (hi - lo)


# ---------------------------------------------------------------------------
# R* interpolation: summary.md has R* only at the 6 canonical Tg values, per
# W0 in {10,20,30}. We need it at every Tg on the (possibly finer) Cc grid.
# R is treated as a discrete design choice (R_GRID), never a continuous
# knob, so we interpolate in log10(R) vs Tg and snap the result to the
# nearest grid point in log10 space.
# ---------------------------------------------------------------------------
def load_rstar_table(summary_md: Path) -> dict:
    """Parse summary.md's markdown table into {(W0, Tg): R*}.

    Fails loudly (raises) if the file is missing or the expected columns
    are not found, rather than silently falling back to hardcoded numbers.
    """
    if not summary_md.exists():
        raise FileNotFoundError(
            f"R* source table not found: {summary_md}\n"
            "gen_matrix.py refuses to hardcode the CS-25 R* table -- it must "
            "be parsed from light/results_cs25_combo/summary.md at runtime. "
            "Regenerate/restore that file (see light/tests/cs25_combo_study.py) "
            "before running gen_matrix.py."
        )
    header = None
    table = {}
    for raw in summary_md.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            if "W0" in cells and "Tg" in cells:
                header = cells
            continue
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue  # markdown header separator row
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        try:
            w0 = float(row["W0"]); tg = float(row["Tg"]); rstar = float(row["combo R*"])
        except (KeyError, ValueError):
            continue
        table[(w0, tg)] = rstar

    if header is None or "combo R*" not in header:
        raise ValueError(
            f"Could not find a 'combo R*' column in {summary_md} -- format changed?"
        )
    missing = [(w0, tg) for w0 in CC_W0_LIST for tg in CANONICAL_TG if (w0, tg) not in table]
    if missing:
        raise ValueError(f"summary.md is missing R* for cells {missing}")
    return table


def snap_to_r_grid(r: float) -> float:
    grid = np.asarray(R_GRID, dtype=float)
    j = int(np.argmin(np.abs(np.log10(grid) - np.log10(r))))
    return float(grid[j])


def interpolate_rstar(rstar_table: dict, w0: float, tg: float) -> float:
    """log10(R) linear interpolation over Tg at fixed W0, snapped to R_GRID."""
    xp = np.array(CANONICAL_TG)
    fp = np.log10(np.array([rstar_table[(w0, t)] for t in CANONICAL_TG]))
    log_r = float(np.interp(tg, xp, fp))
    return snap_to_r_grid(10 ** log_r)


# ---------------------------------------------------------------------------
# t_end rules
# ---------------------------------------------------------------------------
def t_end_a(tg: float) -> float:
    return round(min(3.0, tg + 0.9), ROUND)


# Family B run geometry. v5 ran a flat 3.0 s and let the release ramp fill
# "the remainder of the run" (chapter2.tex). v6 shortens the run, so the
# remainder has to be pinned to something -- but it must stay a *slow*
# release, because the contrast between one fast stroke and one slow release
# is the whole point of family B: it is what puts low-frequency flap->loads
# content in the training set. Deriving t_end from the schedule instead
# (e.g. a symmetric ramp at rate_max) collapses 2/3 of the family onto a
# ~0.2 s impulse and throws that content away.
T_B_NOMINAL = 1.8   # [s] run length for every family-B run
T_B_RELEASE = 1.5   # [s] release reaches zero here; 0.3 s of free decay follows


def flap_schedule_b(delta_pk: float, rate_max: float):
    """Knots of the family-B flap schedule: fast stroke, then slow release.

    Returns (times, angles) for the driver's --delta-times/--delta-angles.

        0        -> 0
        t_up     -> delta_pk      fast stroke at the sampled rate limit
        1.5      -> 0             slow release over the remainder
        1.8      -> 0             free decay

    t_up = |delta_pk| / rate_max is at most 15/20 = 0.75 s over the LHS box,
    so the release ramp (|delta_pk| / (1.5 - t_up)) is slower than the stroke
    everywhere except at that single extreme corner, where the two rates are
    equal. Structural context: the heave mode decays with 1/(zeta*omega_h)
    = 1.37 s, so 1.8 s captures the stroke, the release and roughly one
    decay time of the response.
    """
    t_up = abs(delta_pk) / rate_max
    return ([0.0, round(t_up, ROUND), T_B_RELEASE, T_B_NOMINAL],
            [0.0, round(float(delta_pk), ROUND), 0.0, 0.0])


def t_end_b(delta_pk: float, rate_max: float) -> float:
    """Fixed T_B_NOMINAL -- the schedule lives inside the run, not vice versa.

    Arguments are kept for interface symmetry with t_end_a / t_end_cc and so
    that a future schedule change can make the length parameter-dependent
    again without touching the call sites.
    """
    return round(T_B_NOMINAL, ROUND)


def t_end_cc(tg: float) -> float:
    return round(min(3.0, tg + 1.2), ROUND)


# ---------------------------------------------------------------------------
# Family A -- gust only
# ---------------------------------------------------------------------------
def build_family_a(rngs) -> list:
    rows = []
    idx = 1
    for split, n in (("train", N_A_TRAIN), ("val", N_A_VAL), ("test", N_A_TEST)):
        u = lhs_unit(n, 2, next(rngs))
        r_g = scale(u[:, 0], RG_LO, RG_HI)
        tg = scale(u[:, 1], TG_LO, TG_HI)
        for i in range(n):
            w0 = r_g[i] * U_INF
            rows.append({
                "sim_id": f"sim_A_{idx:03d}_{split}",
                "family": "A", "split": split,
                "W0": round(w0, ROUND), "Tg": round(tg[i], ROUND),
                "r_g": round(r_g[i], ROUND),
                "controller": "schedule",
                "K_ff": "", "t_d": "", "delta_pk": "", "rate_max": "",
                "R_star": "",
                "t_end": t_end_a(tg[i]),
            })
            idx += 1
    return rows


# ---------------------------------------------------------------------------
# Family B -- flap only
# ---------------------------------------------------------------------------
def build_family_b(rngs) -> list:
    rows = []
    idx = 1
    for split, n in (("train", N_B_TRAIN), ("val", N_B_VAL), ("test", N_B_TEST)):
        u = lhs_unit(n, 2, next(rngs))
        mag = scale(u[:, 0], DPK_LO, DPK_HI)
        rate = scale(u[:, 1], RATE_LO, RATE_HI)
        # Exactly half negative within every split (n is always even here:
        # 30/8/8), which also gives exactly half negative over the family.
        sign_rng = next(rngs)
        n_neg = n // 2
        sign = np.array([-1.0] * n_neg + [1.0] * (n - n_neg))
        sign_rng.shuffle(sign)
        delta_pk = sign * mag
        for i in range(n):
            rows.append({
                "sim_id": f"sim_B_{idx:03d}_{split}",
                "family": "B", "split": split,
                "W0": 0, "Tg": 0, "r_g": 0,
                "controller": "schedule",
                "K_ff": "", "t_d": "",
                "delta_pk": round(float(delta_pk[i]), ROUND),
                "rate_max": round(float(rate[i]), ROUND),
                "R_star": "",
                "t_end": t_end_b(delta_pk[i], rate[i]),
            })
            idx += 1
    return rows


# ---------------------------------------------------------------------------
# Family Cc -- gust + flap, CS-25 cell coverage (redesigned vs v5)
# ---------------------------------------------------------------------------
def select_prop_cells(n_prop: int) -> list:
    """Pick n_prop (W0, Tg) cells out of the 6x3=18 canonical grid, spread
    for widest coverage.

    Rule: flatten the 18 canonical cells row-major (W0 outer ascending, Tg
    inner ascending). If n_prop <= 18, drop (18 - n_prop) cells at evenly
    spaced positions in that flattened order -- i.e. keep the subsequence
    that is as close as possible to equally spaced, which spreads the kept
    cells across both W0 and Tg rather than clustering them. If n_prop > 18
    (a widened --n-cc), use every canonical cell once and then repeat the
    same flattened order for the extra rows.

    For the default n_prop=13 this keeps at least 4 of 6 Tg values for
    every W0, and every one of the 6 Tg values is kept for at least 2 of
    the 3 W0 rows.
    """
    grid = [(w0, tg) for w0 in CC_W0_LIST for tg in CANONICAL_TG]
    n_grid = len(grid)
    if n_prop <= 0:
        return []
    if n_prop <= n_grid:
        n_drop = n_grid - n_prop
        drop_idx = {int((k + 0.5) * n_grid / n_drop) for k in range(n_drop)} if n_drop else set()
        return [c for i, c in enumerate(grid) if i not in drop_idx]
    cells = list(grid)
    i = n_grid
    while len(cells) < n_prop:
        cells.append(grid[i % n_grid])
        i += 1
    return cells


def build_family_cc(rngs, n_cc: int, tg_step: float, rstar_table: dict) -> list:
    n_steps = round((TG_HI - TG_LO) / tg_step)
    tg_grid = [round(TG_LO + i * tg_step, 6) for i in range(n_steps + 1)]

    mpc_cells = [(w0, tg) for w0 in CC_W0_LIST for tg in tg_grid]
    n_mpc = len(mpc_cells)
    n_prop = n_cc - n_mpc
    if n_prop < 0:
        raise ValueError(
            f"--n-cc={n_cc} is smaller than the {n_mpc}-cell MPC grid "
            f"(3 W0 x {len(tg_grid)} Tg); the MPC grid has priority, raise --n-cc."
        )
    prop_cells = select_prop_cells(n_prop)

    # Split counts: baseline 50/10/10 fractions out of the family total,
    # generalised to any n_cc; test count is the remainder so the three
    # always sum exactly to n_cc.
    n_train = round(n_cc * CC_TRAIN_FRAC)
    n_val = round(n_cc * CC_VAL_FRAC)
    n_test = n_cc - n_train - n_val

    mpc_train = round(n_train * n_mpc / n_cc) if n_cc else 0
    mpc_val = round(n_val * n_mpc / n_cc) if n_cc else 0
    mpc_test = n_mpc - mpc_train - mpc_val
    prop_train = n_train - mpc_train
    prop_val = n_val - mpc_val
    prop_test = n_test - mpc_test
    assert prop_train + prop_val + prop_test == n_prop
    assert min(mpc_train, mpc_val, mpc_test, prop_train, prop_val, prop_test) >= 0, (
        "Cc split allocation went negative -- n_cc too small relative to the "
        "MPC grid for the 50/10/10 fractions to make sense"
    )

    def splits_for(n_group: int, n_val_g: int, n_test_g: int, rng: np.random.Generator) -> list:
        order = rng.permutation(n_group)
        labels = np.empty(n_group, dtype=object)
        labels[order[:n_val_g]] = "val"
        labels[order[n_val_g:n_val_g + n_test_g]] = "test"
        labels[order[n_val_g + n_test_g:]] = "train"
        return list(labels)

    mpc_splits = splits_for(n_mpc, mpc_val, mpc_test, next(rngs))
    prop_splits = splits_for(len(prop_cells), prop_val, prop_test, next(rngs))

    rows = []
    idx = 1
    for (w0, tg), split in zip(mpc_cells, mpc_splits):
        r_star = interpolate_rstar(rstar_table, w0, tg)
        rows.append({
            "sim_id": f"sim_Cc_{idx:03d}_{split}",
            "family": "Cc", "split": split,
            "W0": round(w0, ROUND), "Tg": round(tg, ROUND),
            "r_g": round(w0 / U_INF, ROUND),
            "controller": "mpc",
            "K_ff": "", "t_d": "", "delta_pk": "", "rate_max": "",
            "R_star": r_star,
            "t_end": t_end_cc(tg),
        })
        idx += 1
    for (w0, tg), split in zip(prop_cells, prop_splits):
        rows.append({
            "sim_id": f"sim_Cc_{idx:03d}_{split}",
            "family": "Cc", "split": split,
            "W0": round(w0, ROUND), "Tg": round(tg, ROUND),
            "r_g": round(w0 / U_INF, ROUND),
            "controller": "prop",
            "K_ff": "", "t_d": "", "delta_pk": "", "rate_max": "",
            "R_star": "",
            "t_end": t_end_cc(tg),
        })
        idx += 1
    return rows


# ---------------------------------------------------------------------------
# --from-v5: optional, best-effort reader of an existing v5 campaign tree so
# that families A and B can stay run-for-run identical to v5 instead of
# drawing a fresh LHS. This directory only exists on the cluster; it is
# NOT part of this repo and could not be exercised against real data while
# writing this script -- verify against a real v5 tree before relying on it.
# ---------------------------------------------------------------------------
_INFO_RE_A = re.compile(
    r"#\s*(?P<sim_id>sim_A_\d+_(?P<split>train|val|test)).*?"
    r"R=(?P<r_g>[-0-9.eE]+)\s+T_g=(?P<tg>[-0-9.eE]+)\s+W_g0=(?P<w0>[-0-9.eE]+)",
    re.DOTALL,
)
_INFO_RE_B = re.compile(
    r"#\s*(?P<sim_id>sim_B_\d+_(?P<split>train|val|test)).*?"
    r"delta_max=(?P<delta>-?[0-9.]+)\D*rate_max=(?P<rate>[0-9.]+)",
    re.DOTALL,
)


def _find_sim_info_files(v5_dir: Path, family_letter: str) -> list:
    candidates = [
        v5_dir / f"Family {family_letter}",
        v5_dir / f"Family_{family_letter}",
        v5_dir / family_letter,
        v5_dir,
    ]
    for base in candidates:
        if not base.exists():
            continue
        found = sorted(base.glob(f"**/sim_{family_letter}_*/sim_info.txt"))
        if found:
            return found
        found = sorted(base.glob("sim_info.txt"))
        if found:
            return found
    return []


def load_family_a_from_v5(v5_dir: Path) -> list:
    files = _find_sim_info_files(v5_dir, "A")
    if not files:
        raise FileNotFoundError(f"--from-v5: no Family A sim_info.txt found under {v5_dir}")
    rows = []
    for f in files:
        m = _INFO_RE_A.search(f.read_text(encoding="utf-8"))
        if not m:
            raise ValueError(f"--from-v5: could not parse {f} (Family A format changed?)")
        w0 = float(m["w0"])
        tg = float(m["tg"])
        rows.append({
            "sim_id": m["sim_id"], "family": "A", "split": m["split"],
            "W0": round(w0, ROUND), "Tg": round(tg, ROUND),
            "r_g": round(w0 / U_INF, ROUND),
            "controller": "schedule",
            "K_ff": "", "t_d": "", "delta_pk": "", "rate_max": "",
            "R_star": "", "t_end": t_end_a(tg),
        })
    return rows


def load_family_b_from_v5(v5_dir: Path) -> list:
    files = _find_sim_info_files(v5_dir, "B")
    if not files:
        raise FileNotFoundError(f"--from-v5: no Family B sim_info.txt found under {v5_dir}")
    rows = []
    for f in files:
        m = _INFO_RE_B.search(f.read_text(encoding="utf-8"))
        if not m:
            raise ValueError(f"--from-v5: could not parse {f} (Family B format changed?)")
        delta = float(m["delta"])
        rate = float(m["rate"])
        rows.append({
            "sim_id": m["sim_id"], "family": "B", "split": m["split"],
            "W0": 0, "Tg": 0, "r_g": 0,
            "controller": "schedule",
            "K_ff": "", "t_d": "",
            "delta_pk": round(delta, ROUND), "rate_max": round(rate, ROUND),
            "R_star": "", "t_end": t_end_b(delta, rate),
        })
    return rows


# ---------------------------------------------------------------------------
# Validation + reporting
# ---------------------------------------------------------------------------
def validate(rows: list, n_cc: int) -> None:
    ids = [r["sim_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate sim_id detected"

    by_family = {"A": [], "B": [], "Cc": []}
    for r in rows:
        by_family[r["family"]].append(r)
    assert len(by_family["A"]) == 30, f"Family A count {len(by_family['A'])} != 30"
    assert len(by_family["B"]) == 46, f"Family B count {len(by_family['B'])} != 46"
    assert len(by_family["Cc"]) == n_cc, f"Family Cc count {len(by_family['Cc'])} != {n_cc}"

    def split_counts(fam_rows):
        c = {"train": 0, "val": 0, "test": 0}
        for r in fam_rows:
            c[r["split"]] += 1
        return c

    assert split_counts(by_family["A"]) == {"train": 20, "val": 5, "test": 5}
    assert split_counts(by_family["B"]) == {"train": 30, "val": 8, "test": 8}
    if n_cc == N_CC_DEFAULT:
        assert split_counts(by_family["Cc"]) == {"train": 50, "val": 10, "test": 10}

    n_neg = sum(1 for r in by_family["B"] if r["delta_pk"] < 0)
    assert n_neg == len(by_family["B"]) // 2, f"Family B negative count {n_neg} != half"

    for r in rows:
        assert 1.0 - 1e-9 <= r["t_end"] <= 3.0 + 1e-9, f"{r['sim_id']} t_end {r['t_end']} out of [1,3]"

    cc_mpc = [r for r in by_family["Cc"] if r["controller"] == "mpc"]
    cc_prop = [r for r in by_family["Cc"] if r["controller"] == "prop"]
    if n_cc == N_CC_DEFAULT:
        assert len(cc_mpc) == 57, f"Cc mpc rows {len(cc_mpc)} != 57"
        assert len(cc_prop) == 13, f"Cc prop rows {len(cc_prop)} != 13"
    for r in cc_mpc:
        assert r["R_star"] != "" and float(r["R_star"]) in R_GRID, f"{r['sim_id']} bad R_star"
    for r in cc_prop:
        assert r["R_star"] == "", f"{r['sim_id']} prop row should have empty R_star"


def print_summary(rows: list) -> None:
    by_family = {"A": [], "B": [], "Cc": []}
    for r in rows:
        by_family[r["family"]].append(r)

    print("=== dataset_v6 run matrix -- summary ===")
    print(f"{'family':<8}{'train':>7}{'val':>7}{'test':>7}{'total':>8}")
    tot = {"train": 0, "val": 0, "test": 0}
    for fam in ("A", "B", "Cc"):
        c = {"train": 0, "val": 0, "test": 0}
        for r in by_family[fam]:
            c[r["split"]] += 1
            tot[r["split"]] += 1
        print(f"{fam:<8}{c['train']:>7}{c['val']:>7}{c['test']:>7}{len(by_family[fam]):>8}")
    grand = sum(tot.values())
    print(f"{'TOTAL':<8}{tot['train']:>7}{tot['val']:>7}{tot['test']:>7}{grand:>8}")
    print()

    cc = by_family["Cc"]
    n_mpc = sum(1 for r in cc if r["controller"] == "mpc")
    n_prop = sum(1 for r in cc if r["controller"] == "prop")
    print(f"Cc controller split: mpc={n_mpc}  prop={n_prop}")
    if n_mpc:
        from collections import Counter
        hist = Counter(r["R_star"] for r in cc if r["controller"] == "mpc")
        print("Cc mpc R_star usage:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))

    n_b_neg = sum(1 for r in by_family["B"] if r["delta_pk"] != "" and r["delta_pk"] < 0)
    print(f"Family B sign balance: neg={n_b_neg}  pos={len(by_family['B']) - n_b_neg}"
          f"  (of {len(by_family['B'])})")
    print()

    for fam in ("A", "B", "Cc"):
        t = [r["t_end"] for r in by_family[fam]]
        print(f"t_end[{fam}]: min={min(t):.4f}  max={max(t):.4f}  mean={sum(t) / len(t):.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def make_rngs(seed: int, n: int):
    ss = np.random.SeedSequence(seed)
    for child in ss.spawn(n):
        yield np.random.default_rng(child)


def build_matrix(seed: int, n_cc: int, tg_step: float, summary_md: Path,
                  from_v5: Path | None) -> list:
    rngs = make_rngs(seed, 32)  # generous pool of independent streams

    if from_v5 is not None:
        rows_a = load_family_a_from_v5(from_v5)
        rows_b = load_family_b_from_v5(from_v5)
    else:
        rows_a = build_family_a(rngs)
        rows_b = build_family_b(rngs)

    rstar_table = load_rstar_table(summary_md)
    rows_cc = build_family_cc(rngs, n_cc, tg_step, rstar_table)

    return rows_a + rows_b + rows_cc


def write_csv(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output CSV path (default: {DEFAULT_OUT})")
    p.add_argument("--seed", type=int, default=20260902,
                   help="master RNG seed, fixed default for a reproducible campaign")
    p.add_argument("--n-cc", type=int, default=N_CC_DEFAULT,
                   help="total Family Cc rows; extra rows beyond the 57-cell "
                        "MPC grid go to controller=prop (default: 70)")
    p.add_argument("--tg-step", type=float, default=0.05,
                   help="Tg grid step for the Cc MPC sweep (default: 0.05)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the summary table and write nothing")
    p.add_argument("--from-v5", type=Path, default=None,
                   help="OPTIONAL: read families A/B from an existing v5 "
                        "campaign directory instead of drawing fresh LHS "
                        "(cluster-only path, not used by default)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = build_matrix(args.seed, args.n_cc, args.tg_step, SUMMARY_MD, args.from_v5)
    validate(rows, args.n_cc)
    print_summary(rows)
    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0
    write_csv(rows, args.out)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
