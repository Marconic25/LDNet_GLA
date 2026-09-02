# Task 4 Report — CS-25 Combo Plots Script

## Status: COMPLETE

## Commit

```
e1789628 feat(cs25): cs25_combo_plots.py — side-by-side comparison plots and summary
```

## Test Output

```
parse OK
```

Script successfully validates as syntactically correct Python via `ast.parse()`.

## What Was Done

Created `/home/marco/LDNet_OF/light/tests/cs25_combo_plots.py` with exact implementation from task-4-brief.md:

- **Loads** optimal and combo results from `results_cs25/` and `results_cs25_combo/` respectively
- **Gracefully handles missing data** — uses `float('nan')` and `'?'` for missing traces (e.g., if combo not yet done)
- **Generates three outputs** in `results_cs25_combo/`:
  - `heatmap_clred_combo.png` — side-by-side pcolormesh (optimal left, combo right) with CLred%, R* values, and explosion flags
  - `summary_lines_combo.png` — CLred vs Tg line plot, dashed for optimal, solid for combo, three W0 values in separate colors
  - `summary.md` — markdown table with per-cell comparison of CLred, R*, flap, pitch, flags
- **No dependencies** on TensorFlow; uses only numpy, matplotlib, os
- **Safe directory creation** — `os.makedirs(DIR_COMBO, exist_ok=True)`

## Concerns

None. Script is ready to run after cluster results arrive via:

```bash
scp -r u10677113@10.78.18.100:/work/u10677113/LDNet_GLA/light/results_cs25_combo/ /home/marco/LDNet_OF/light/
python3 light/tests/cs25_combo_plots.py
```

## Next Steps

Wait for cluster jobs (Task 0–3) to complete (~4h), then scp results and run the script. Results files can then be staged for final commit in Step 3 of the brief.
