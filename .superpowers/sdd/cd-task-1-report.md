# Task 1 Report: Axis C2 — spatial per-shot jitter robustness script + cluster wrappers + plot

## Status
**DONE**

## Commit
- **SHA:** `8e9867d9`
- **Message:** `feat(noise): C2 spatial per-shot jitter robustness axis for E2-combo`
- **Files changed:** 4
  - `light/noise/noise_jitter_combo.py` (193 lines)
  - `light/noise/launch_noise_jitter_combo.sh` (13 lines)
  - `light/noise/status_noise_jitter_combo.sh` (14 lines)
  - `light/noise/plots_noise_jitter_combo.py` (56 lines)

## Parse Tests Output
```
light/noise/noise_jitter_combo.py parse OK
light/noise/plots_noise_jitter_combo.py parse OK
launch OK
status OK
```

All four files passed syntax validation. No TensorFlow imports attempted (verified with ast.parse only).

## Self-Review Findings

### JitterSensor class (noise_jitter_combo.py, lines 75-120)
- ✓ `__init__` correctly initializes `self.k = int(k)` (line 89)
- ✓ `_eta` helper method: correct logic for k=0 (zero-draw) and k>0 (jitter draw) (lines 76-79)
- ✓ Prewarm loop with `mt` correctly uses `_eta` to perturb sampled index (line 90: `mt = np.clip(mm[keep] + self._eta(int(keep.sum())), 0, Nsteps - 1)`)
- ✓ Main wc_fun body: exact duplicate of parent with `m_reg` (registration index) and `m_true` (true sampled index) correctly paired (lines 95-99)
- ✓ Indentation preserved throughout

### Plot file (plots_noise_jitter_combo.py)
- ✓ xticks block correctly combines frac=0.0 and frac=0.02 series (lines 44-46)
- ✓ Label format: `f'{int(t)}\n({t*DX_M:.2f} m)'` produces multi-line tick labels with jitter value and meter equivalent
- ✓ All axis labels, title, and legend correctly set

### Shell scripts
- ✓ `launch_noise_jitter_combo.sh`: cluster path (`/work/u10677113/LDNet_GLA`), nohup redirection, pid echo
- ✓ `status_noise_jitter_combo.sh`: dual-pair loop (smoke + full), status detection (DONE/ERROR/RUNNING/DEAD), last-line tail
- ✓ Both scripts use correct python invocation pattern matching in pgrep: `python3 -s -u noise_jitter_combo.py`

### Line Endings
- ✓ Normalized both .sh files to LF using `sed -i 's/\r$//'`

## Key Design Elements Verified
- **JitterSensor bit-exactness at k=0:** preserves parent rng stream (returns all-zeros from _eta)
- **m_reg vs m_true distinction:** sensor registers at m_reg (unchanged), samples from m_true (jittered) — captured correctly in lines 95-96
- **Prewarm phase:** applies jitter to backlog accumulation (line 90), enabling fusion to converge to low-pass convolution
- **Config fixed:** W0=30, Tg=0.4, Jmax=50, N=8, R=3e-4, R_du=0, lam=0 — all hardcoded
- **Smoke gate:** k=0 clean + k=2 jitter-only (2 seeds) only
- **Full run:** k in {0,1,2,5} jitter-only (6 seeds each) + k=2 compound σ=2% (6 seeds)
- **Output schema:** records keyed by axis='C2', arm='jitter', value (k), frac, sigma_del, bias_del

## No Concerns
- All content faithfully transcribed from brief (line-by-line, indentation preserved)
- Parse tests all pass
- Commit restricted to exactly 4 files created by this task
- Co-author trailer correctly appended
