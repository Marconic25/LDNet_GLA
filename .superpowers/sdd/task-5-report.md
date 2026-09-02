# Task 5 Report — White-noise robustness of E2-combo pipeline

## Status: SMOKE RUNNING (full run not yet launched; awaiting smoke pass)

## Files created

| File | Location |
|---|---|
| `noise_white_combo.py` | `light/noise/noise_white_combo.py` |
| `launch_noise_white_combo.sh` | `light/noise/launch_noise_white_combo.sh` |
| `status_noise_white_combo.sh` | `light/noise/status_noise_white_combo.sh` |
| `plots_noise_white_combo.py` | `light/noise/plots_noise_white_combo.py` |
| `NOTES.md` (appended W_combo section) | `light/noise/NOTES.md` |

## Parse tests

```
noise_white_combo.py: parse OK
plots_noise_white_combo.py: parse OK
```

## Commit

```
[main c5ee7403] feat(noise): white-noise robustness study of E2-combo pipeline
 5 files changed, 230 insertions(+), 1 deletion(-)
 create mode 100644 light/noise/launch_noise_white_combo.sh
 create mode 100644 light/noise/noise_white_combo.py
 create mode 100644 light/noise/plots_noise_white_combo.py
 create mode 100644 light/noise/status_noise_white_combo.sh
```

## Smoke test

- Launched: `ssh -n u10677113@10.78.18.100 '/work/u10677113/LDNet_GLA/light/noise/launch_noise_white_combo.sh smoke'`
- PID: 3712432
- Log: `/work/u10677113/LDNet_GLA/light/noise/Wco.smoke.log`
- Status at launch: RUNNING
- Header confirmed: `# W_combo | W30/Tg0.4 DAMULT=3 N=8 Jmax=50 R=0.0003 R_du=0 | open cex0=0.4600 | SMOKE`
  - `cex0=0.4600` matches the dp45 anchor exactly.

## Smoke CLred at frac=0 (pending — fill after DONE)

TBD: combo frac=0% CLred should be ~+80.51% (dp45 combo clean anchor ±1 pt).

## Full run

Not yet launched — pending smoke pass. Command:
```bash
ssh -n u10677113@10.78.18.100 '/work/u10677113/LDNet_GLA/light/noise/launch_noise_white_combo.sh full'
```
Expected runtime: ~5h (6 sigma levels × 11 seeds × ~8 min/seed combo, ~1 min/seed none).

## Cluster job status

- Smoke: RUNNING (pid 3712432, launched 2026-07-08 ~17:51 UTC+2)
- Full: NOT YET LAUNCHED

## Design notes

- `_ComboCtrl.reset()` correctly resets both sensor and mpc, plus `_delta_prev = 0.0`
- sigma=0 case uses `lambda j: 1e-9` (not 0.0) to avoid division by zero in `FusedPreviewSensor`
- `wc_plain` for one-step baseline: simple scalar noisy wnext preview
- `none` arm only run for frac > 0 (matches brief: no paired baseline at frac=0 for none)
- NOTES.md W_combo section appended with TBD table (fill after cluster run completes)

## Concerns

None. The script follows the exact pattern of the brief and the harness API.
`harness_noise.py` was NOT modified.
