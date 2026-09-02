# Task 2 Completion Report: D2 Structural Model-Mismatch Robustness

## Summary

Successfully created and committed all four files for the D2 axis (structural model-mismatch robustness) of the LDNet GLA E2-combo robustness study.

## Files Created

1. **light/noise/noise_mismatch_combo.py** (183 lines)
   - Core experiment script implementing deterministic perturbed-plant / nominal-controller mismatch
   - Arms: dalpha, kalpha, uinf, cltrim, anchor
   - Critical structure: `_MismatchCtrl.compute()` with setattr toggle (nominal for MPC, restore perturbed after)
   - Per-point loop: set-perturb → try[OL+closed-loop] → finally-restore ordering
   - Outputs: results/D2_mismatch.npz with axis='D2' records

2. **light/noise/launch_noise_mismatch_combo.sh** (11 lines)
   - Cluster launcher script (cluster-only path /work/u10677113/LDNet_GLA)
   - Supports smoke and full modes

3. **light/noise/status_noise_mismatch_combo.sh** (11 lines)
   - Status monitoring script for D2.smoke and D2 log files
   - Detects DONE/ERROR/RUNNING/DEAD states

4. **light/noise/plots_noise_mismatch_combo.py** (77 lines)
   - Two-panel figure generation (structural left, controller-param right)
   - Reads D2_mismatch.npz, plots CLred vs multiplier/error, flags explosions

## Parse Tests

All four files passed syntax validation:

```
noise_mismatch_combo.py: parse OK
plots_noise_mismatch_combo.py: parse OK
launch_noise_mismatch_combo.sh: launch OK
status_noise_mismatch_combo.sh: status OK
```

## Self-Review Findings

✅ **_MismatchCtrl.compute() method** (lines 95–102):
- Correct setattr toggle: sets nominal, runs MPC, finally restores plant values
- Try/finally indentation and structure exact match to brief

✅ **Per-point main loop** (lines 137–155):
- Correct sequence: set-perturb (line 137–138) → try block (OL+closed-loop) → finally-restore (line 153–155)
- OLp metrics computed against own open-loop rollout (line 152)
- Each arm's pert and mpc_kw build correctly (lines 154)

✅ **Plot structure**:
- Left panel: dalpha + kalpha (both added with anchor for x=1.0)
- Right panel: uinf (% error) + cltrim (% error) (both added with anchor for x=0)
- Flagged explosions shown as red rings (open fill)

✅ **Config values**: W0=30, Tg=0.4, JMAX=50, N=8, R=3e-4, R_du=0, lam=0, anchor=80.51%

## Commit

- **SHA**: 65982386
- **Message**: "feat(noise): D2 structural model-mismatch robustness axis for E2-combo"
- **Co-author**: Claude Fable 5 <noreply@anthropic.com>
- **Files staged and committed**: exactly 4 files (no user changes included)

## Status

**DONE** — All files written with exact content from brief, line endings normalized, parse tests pass (4/4), commit created cleanly.
