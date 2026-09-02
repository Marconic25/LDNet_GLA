# SDD Progress Ledger — E2-Combo Migration
# Plan: docs/superpowers/plans/2026-07-08-e2combo-migration.md
# Started: 2026-07-08
Task 0: complete (commits 81e830e5..305b829a, review clean)
Task 1: complete (commits 305b829a..beca8c3e, review clean)
Task 2: complete (commits beca8c3e..e751686e, review clean; minor: simulate() docstring not updated)
Task 3: complete (commits e751686e..22e316e4, review clean; minors: TG_SMOKE unfiltered, extra smoke_dp45_baseline.py)
Task 4: complete (commits 22e316e4..e1789628, review clean)
Task 5: complete (commit c5ee7403, smoke pass: frac=0 +80.5% matches dp45 anchor; full run launched)
Task 6+7: complete (commit 9abc780b — CS-25 combo traces + noise results + NOTES.md filled)

# SDD Progress Ledger — MPC robustness axes (A2 calib + B2 timing)
# Plan: docs/superpowers/plans/2026-07-09-mpc-robustness.md
# Started: 2026-07-09
# Base: 9abc780b
# Gates: combo clean +80.51%; K=1 sigma2% seeds 0-1 = 80.3860, 80.5235 (bit-exact)
Task 1: complete (commits 9abc780b..beb29791, review clean; wc-cache assumption verified vs harness by controller)
Task 2: complete (commits beb29791..011c4f38, review clean, K=1 bit-exact verified vs optimal.py; minors: shift_field docstring wording, inert _delta_prev in adapter)
Task 3: complete (smokes PASS: A2 anchor +80.5/bias_del +1.5, B2 anchor +80.5, K=1 bit-exact 80.3860/80.5235 vs W_combo; fulls launched 2026-07-09 ~14:05 pids 3884170/3884256; smoke DATA: bias+5% -> +12.9 flag, shift+2 -> +71.5 flag, K=5 -> +80.4)
Task 4: complete (commit 0ecaf98b — A2/B2 npz + figs + NOTES A2/B2/envelope sections; envelope: bias <+2%W0 / >=-5% free, gain 0.9-1.2, timing [-10,+2] ms, refit free to ~12 Hz)

# SDD Progress Ledger — MPC robustness axes (C2 jitter + D2 mismatch)
# Plan: docs/superpowers/plans/2026-07-10-mpc-robustness-cd.md
# Started: 2026-07-10
# Base: 0ecaf98b
# Gates: anchor +80.51% (C2 k=0 bit-exact by-design; D2 anchor cex0=0.4600)
Task 1 (C2): complete (commits 0ecaf98b..8e9867d9, review clean, k=0 bit-exact verified analytically)
Task 2 (D2): complete (commits 8e9867d9..65982386, review clean, toggle verified vs structure.rhs/dp45_batch call-time reads; minor: DAMULT print fallback, same as all axes)
Task 3 (C2+D2): complete (smokes PASS: C2 k=0 +80.5, D2 anchor +80.5/cex0 0.4600; fulls DONE 2026-07-10)
Task 4 (C2+D2): complete (commit 3ee774cf — npz + figs + NOTES C2/D2/envelope-additions; C2 jitter free to 0.80 m; D2 structural free, ctrl U/CLtrim +-5% catastrophic -57.5/-60.8)
