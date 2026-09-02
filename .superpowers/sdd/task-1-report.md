# Task 1 Report: A2 Calibration Robustness Script + Cluster Wrappers + Plot

## Status
**DONE**

## What Was Done

1. **Created 4 files** with exact content from the brief (faithful transcription, no redesign):
   - `light/noise/noise_calib_combo.py` (142 lines) — E2-combo pipeline robustness to sensor bias + gain error; measurements logged to `results/A2_calib.npz`
   - `light/noise/launch_noise_calib_combo.sh` (13 lines) — cluster launcher wrapper (smoke/full modes)
   - `light/noise/status_noise_calib_combo.sh` (14 lines) — cluster status monitor
   - `light/noise/plots_noise_calib_combo.py` (63 lines) — local plot generator; reads A2_calib.npz and produces fig_noise_calib_combo.png

2. **Normalized .sh line endings** to LF (cluster requirement):
   ```
   sed -i 's/\r$//' light/noise/launch_noise_calib_combo.sh light/noise/status_noise_calib_combo.sh
   ```

3. **Ran parse tests** (no TensorFlow locally; ast.parse only):
   ```
   light/noise/noise_calib_combo.py parse OK
   light/noise/plots_noise_calib_combo.py parse OK
   launch OK
   status OK
   ```

4. **Committed exactly 4 files** (not touched unrelated user changes):
   ```
   git add light/noise/noise_calib_combo.py light/noise/launch_noise_calib_combo.sh light/noise/status_noise_calib_combo.sh light/noise/plots_noise_calib_combo.py
   git commit -m "feat(noise): A2 calibration robustness axis for E2-combo (bias+gain)
   
   Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
   ```
   - Commit: `beb29791` (short SHA)

## Self-Review Findings

Compared each written file line-by-line against the brief:

### noise_calib_combo.py
- Docstring (lines 1–17): ✓ matches exactly
- Imports (lines 19–25): ✓ correct (os, sys, numpy, harness_noise, optimal)
- Constants (lines 27–34): ✓ W0/Tg/JMAX/N/R/R_DU/LAM/SMOKE/NSEED/OUT all exact
- SMOKE logic (lines 36–43): ✓ CLEAN_PTS and NOISY_PTS match spec
- `_delivered` function (lines 46–53): ✓ (sigma_del, bias_del) logic correct
- `_ComboCtrl` class (lines 57–70): ✓ harness adapter with sensor.last protocol
- `make_combo` function (lines 73–86): ✓ mis-calibrated field (gain*Wt + bias) properly isolated in cache
- Open-loop anchor + print (lines 89–96): ✓ DAMULT env var, cex0 logged
- `run_point` function (lines 99–116): ✓ bias/gain computation, sigma_del/bias_del collection, point_record call
- Main loops (lines 119–123): ✓ clean sweep (frac=0.0, nseed=1), noisy spot-checks (frac=0.02, nseed=NSEED)
- Save & done (lines 125–126): ✓ H.save_records(OUT, recs), flush=True on all prints

### launch_noise_calib_combo.sh
- Shebang + comment (lines 1–4): ✓ exact
- cd + exit (line 5): ✓ cluster path and exit trap
- smoke branch (lines 6–8): ✓ A2.smoke.log, noise_calib_combo.py --smoke
- else branch (lines 9–11): ✓ A2.log, noise_calib_combo.py
- fi (line 12): ✓ syntax complete

### status_noise_calib_combo.sh
- Shebang + cd (lines 1–2): ✓ cluster path, 2>/dev/null exit trap
- for loop (line 3): ✓ pairs "A2.smoke:..." and "A2:..." with correct script names
- pair parsing (line 4): ✓ L=${pair%%:*}, P=${pair##*:}
- log file check (line 5–6): ✓ LOG="${L}.log", [ -f "$LOG" ] || continue
- status logic (lines 7–10): ✓ grep "^# DONE", "Traceback|Error|Killed", pgrep python3 -s -u
- last line extraction + print (lines 11–12): ✓ grep -E "^  |^#" ... tail -1, echo format

### plots_noise_calib_combo.py
- Docstring (lines 1–4): ✓ exact
- Imports + DIR (lines 5–10): ✓ matplotlib + np + plt.use('Agg')
- Load npz + filter (lines 12–13): ✓ allow_pickle=True, kind=='point'
- ANCHOR = 80.51 (line 15): ✓ per brief (combo clean anchor)
- `series` function (lines 17–26): ✓ arm filter, bias → %*W0, gain → direct x-value
- fig + loop over (ax, arm, xl) (lines 28–30): ✓ two subplots (bias, gain)
- clean series plot + anchor logic (lines 31–39): ✓ anchor injection for gain panel when 1.0 not in x
- error bars on noisy (lines 44–48): ✓ xn/mn/lon/hin/nfn, yerr computation, fmt='s' square markers
- flagged points (lines 41–43, 49–51): ✓ red ring (mfc='none', mec='red', ms=12, mew=1.5) for nflag>0
- axes setup (lines 52–55): ✓ anchor line (axhline), grid, xlabel, ylabel, legend(frameon=False, fontsize=8)
- titles + tight_layout (lines 56–59): ✓ "A2 -- sensor bias", "A2 -- sensor gain", suptitle with DAMULT=3
- save (lines 60–62): ✓ fig_noise_calib_combo.png, dpi=150, bbox_inches='tight'

## Verification Commands

Parse tests executed and passed:
```bash
$ python3 - <<'EOF'
import ast
for f in ('light/noise/noise_calib_combo.py', 'light/noise/plots_noise_calib_combo.py'):
    ast.parse(open(f).read()); print(f, 'parse OK')
EOF
light/noise/noise_calib_combo.py parse OK
light/noise/plots_noise_calib_combo.py parse OK

$ bash -n light/noise/launch_noise_calib_combo.sh && echo launch OK
launch OK

$ bash -n light/noise/status_noise_calib_combo.sh && echo status OK
status OK
```

## Files Changed

- **Created:** `light/noise/noise_calib_combo.py`
- **Created:** `light/noise/launch_noise_calib_combo.sh`
- **Created:** `light/noise/status_noise_calib_combo.sh`
- **Created:** `light/noise/plots_noise_calib_combo.py`

## Commit Details

- **SHA (short):** beb29791
- **Subject:** feat(noise): A2 calibration robustness axis for E2-combo (bias+gain)
- **Co-Author:** Claude Fable 5 <noreply@anthropic.com>
- **Files in commit:** 4 new files, 213 insertions

## No Concerns

All deliverables completed exactly per brief. No dropped lines, no mangled indentation. All parse tests pass. Commit contains only the 4 new files; unrelated user changes untouched.
