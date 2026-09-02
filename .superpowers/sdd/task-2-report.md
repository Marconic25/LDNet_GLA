# Task 2 Report: B2 Timing Robustness Axis

## Status
**DONE**

## Commit Details
- **SHA:** 011c4f38
- **Subject:** feat(noise): B2 timing robustness axis for E2-combo (shift+refit)
- **Files Changed:** 4 (all new files)
  - light/noise/noise_timing_combo.py
  - light/noise/launch_noise_timing_combo.sh
  - light/noise/status_noise_timing_combo.sh
  - light/noise/plots_noise_timing_combo.py

## Parse Test Results
All files passed syntax validation:
- noise_timing_combo.py: parse OK
- plots_noise_timing_combo.py: parse OK
- launch_noise_timing_combo.sh: launch OK
- status_noise_timing_combo.sh: status OK

## Self-Review Findings

### noise_timing_combo.py
- Module docstring correct (shift/refit explanations match brief)
- Config: W0=30.0, Tg=0.4, JMAX=50, N=8, R=3e-4, R_DU=0.0, LAM=0.0
- Smoke gate: SHIFT_CLEAN=[0,2], SHIFT_NOISY=[], REFIT_KS=[1,5]
- Full run: SHIFT_CLEAN=[-10,-5,-2,-1,0,1,2,5,10], SHIFT_NOISY=[-5,5], REFIT_KS=[1,5,10,25,42]
- shift_field() correct with edge padding (lines 53-59)
- _delivered() measures (sigma_del, bias_del) vs next-step gust
- RefitSensor: conditional solve on (i - self._i0) >= self.K, offset calc at lines 126-129
- _ComboCtrl adapter: sensor.last set by wc_fun before compute
- make_mpc() with G=161, correct parameters
- Main loop: open-loop baseline + shift clean + shift noisy + refit
- Output: results/B2_timing.npz

### launch_noise_timing_combo.sh
- Cluster path: /work/u10677113/LDNet_GLA/light/noise
- Smoke: ./run_axis.sh "noise_timing_combo.py --smoke" → B2.smoke.log
- Full: ./run_axis.sh "noise_timing_combo.py" → B2.log
- Proper nohup backgrounding with pid output
- Line endings normalized to LF

### status_noise_timing_combo.sh
- Monitors B2.smoke and B2 logs
- Status detection: DONE (^# DONE), ERROR (Traceback/Error/Killed), RUNNING (pgrep), DEAD
- Extracts last log line
- Line endings normalized to LF

### plots_noise_timing_combo.py
- Loads results/B2_timing.npz with allow_pickle=True
- Two-panel figure (1x2, figsize=(11,4))
- Left panel: shift axis with clean/noisy data + anchor line
- Right panel: refit axis with 100 ms DLR reference + anchor
- Suptitle: "E2-combo timing robustness (W30/Tg0.4, DAMULT=3)"
- Output: fig_noise_timing_combo.png

## Constraints Verified
- Only four new files staged and committed (no user changes)
- Imports from optimal: FusedPreviewSensor, MPCPreviewController only
- Config matches brief exactly
- Seed generation: rng(100+seed)
- Metrics window: t <= Tg+0.5
- No modifications to existing files
- DAMULT=3 already in run_axis.sh

## Implementation Complete
All files created per brief, parse-tested, and committed as 011c4f38.
