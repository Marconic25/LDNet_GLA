# `final/design/` — run matrix for dataset_v6

`gen_matrix.py` generates `run_matrix.csv`, the 146-row source of truth for the
v6 FSI campaign. Every downstream piece (`final/cluster/submit_campaign.sh`,
one PBS job per row; `final/data/` preprocessing; any v6 study) reads this
file and this file only — no other script should re-derive the campaign
design.

## Regenerating

```
python3 final/design/gen_matrix.py --dry-run          # summary only, writes nothing
python3 final/design/gen_matrix.py                    # writes final/design/run_matrix.csv
python3 final/design/gen_matrix.py --n-cc 100          # widen Cc to 100 (176 total); extra rows go to controller=prop
```

`--seed 20260902` is the fixed default so the campaign is reproducible; pass
a different value only to explore an alternative draw. `--from-v5 <dir>` is
optional and only useful on the cluster (see below); the default path never
touches it.

Requires `scipy` for `scipy.stats.qmc.LatinHypercube`; if scipy is not
importable the script falls back to a documented numpy stratified-permutation
LHS (same classical construction, no pairwise-correlation optimisation, which
is not needed for the 2-parameter draws used here).

## Column contract

```
sim_id,family,split,W0,Tg,r_g,controller,K_ff,t_d,delta_pk,rate_max,R_star,t_end
```

- `sim_id`: `sim_<family>_<idx:03d>_<split>`, e.g. `sim_A_007_train`.
  For A/B, `idx` runs sequentially train-block-then-val-block-then-test-block
  (mirrors the v5 convention seen in `clean/data/Family {A,B}/sim_info.txt`).
  For Cc, `idx` follows *generation* order (the 57-cell MPC grid, then the
  prop cells) — the split label is assigned independently by a seeded
  permutation, so Cc's `idx` and `split` are not contiguous blocks.
- `controller` ∈ `schedule` | `mpc` | `prop`.
- Empty cells are genuinely not-applicable for that row (CSV shows nothing),
  as opposed to Family B's `W0=Tg=r_g=0`, which is a meaningful "gust off".
- `K_ff`/`t_d` are always empty in v6: they were a v5 Family-Cc feed-forward
  artifact; v6's Cc uses `mpc`/`prop` controllers instead (see below), so no
  row ever populates them.

## Family design

**A — 30 runs (20/5/5), gust only, flap fixed at 0, `controller=schedule`.**
LHS over `r_g ∈ [0.10, 0.60]` and `Tg ∈ [0.30, 1.20]` s, drawn independently
per split (so train, val and test are each individually space-filling, not
just the pooled 30). `W0 = 80 · r_g`.

*`W0`/`r_g` dependency.* `tab:dataset_families` quotes both `W0 ∈ [8,48]` m/s
and `r_g ∈ [0.10,0.60]` as LHS intervals, but `r_g = W0/U_inf` with
`U_inf = 80` m/s fixed — they are one degree of freedom, not two. Resolution:
sample `r_g` and derive `W0`, since `0.10·80 = 8` and `0.60·80 = 48` reproduce
the quoted `W0` range exactly (asserted at import time in `gen_matrix.py`).

**B — 46 runs (30/8/8), flap only, gust off (`W0=Tg=r_g=0`), `controller=schedule`.**
LHS over `|delta_pk| ∈ [2,15]` deg and `rate_max ∈ [20,200]` deg/s, drawn per
split. Sign is negative in exactly half of *every* split (15/4/4, all even
counts), which also gives exactly half over the whole family (23/23).

**Cc — 70 runs (50/10/10), gust + flap, redesigned vs v5.**
57 rows systematically cover `W0 ∈ {10,20,30}` × `Tg ∈ [0.30,1.20]` step
`--tg-step` (default 0.05 → 19 values) with `controller=mpc`, each carrying
an `R_star`. The remaining 13 (`n_cc - 57`, generalizes if `--n-cc` is
raised) use `controller=prop`, placed on the 6-canonical-Tg × 3-W0 = 18-cell
grid via `select_prop_cells()`: flatten the 18 cells row-major (`W0` outer,
`Tg` inner) and drop `18 - n_prop` cells at evenly spaced positions in that
order. For the default 13, this keeps ≥4 of 6 `Tg` values per `W0` and every
`Tg` value survives for ≥2 of the 3 `W0` rows — a simple, reproducible
"widest coverage" rule (if `n_prop > 18`, the 18-cell cycle repeats for the
extra rows). Splits inside Cc are allocated proportionally to the 50/10/10
family fractions, separately within the mpc group and the prop group (so
both controllers are represented in every split), then assigned to specific
cells by a seeded permutation.

## `R_star` interpolation

`light/results_cs25_combo/summary.md` only reports the MPC `R*` at the 6
canonical `Tg ∈ {0.30,0.40,0.50,0.70,1.00,1.20}`, per `W0`. `gen_matrix.py`
parses that markdown table at runtime (`load_rstar_table`, keyed off the
`combo R*` header) and **raises if the file is missing** — the table is never
hardcoded. For the other 13 `Tg` values it interpolates linearly in
`log10(R)` vs `Tg` (`R` spans two decades; a discrete design choice, not a
continuous knob) and snaps the result to the nearest point of
`R_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]` in log-space
(`light/tests/cs25_combo_study.py`'s grid).

## `t_end` rule

v5 ran a flat 3.0 s; every downstream study only evaluates `t ≤ Tg + 0.5`, so
the tail is wasted CFD. v6 cuts it:

- A: `min(3.0, Tg + 0.9)`
- B: **fixed `T_B_NOMINAL = 1.8` s** — see below
- Cc: `min(3.0, Tg + 1.2)`

Every `t_end` is rounded to 4 decimals.

### Why Family B has a fixed length

v5's schedule — fast stroke at the sampled rate limit, then a slow release
"over the remainder of the run" — is well defined only because the run was a
flat 3.0 s. Deriving `t_end` from the schedule makes that circular.

Closing the loop by making the release symmetric (same `rate_max` as the
stroke) is self-contained but wrong on the merits: it collapses roughly two
thirds of the family onto a ~0.2 s impulse and discards the low-frequency
flap→loads content that Family B exists to generate. The contrast between one
fast stroke and one slow release IS the design of the family.

So the run length is pinned instead and the schedule lives inside it.

## Family B flap schedule — authoritative knots

`flap_schedule_b(delta_pk, rate_max)` in `gen_matrix.py` is the single source
of truth. `run_sim.pbs` must build `--delta-times` / `--delta-angles` from it
(or reproduce it exactly):

```
t [s]:   0.0    t_up          1.5    1.8
δ [deg]: 0.0    delta_pk      0.0    0.0

t_up = |delta_pk| / rate_max          fast stroke at the sampled rate limit
release rate = |delta_pk| / (1.5 - t_up)   slow release over the remainder
```

`t_up ≤ 15/20 = 0.75 s` over the whole LHS box, so the release is slower than
the stroke everywhere except at that single extreme corner, where the two
rates coincide. `T_B_RELEASE = 1.5` leaves 0.3 s of free decay; for scale, the
heave mode decays with `1/(ζ·ω_h) = 1.37 s`.

## Verification

`gen_matrix.py` self-checks (`validate()`, run on every invocation, dry-run
or not): row/split counts per family, no duplicate `sim_id`, `t_end ∈
[1.0,3.0]`, Family B's exact half-negative balance, and that every Cc `mpc`
row has an `R_star` on `R_GRID` while every `prop` row does not.
