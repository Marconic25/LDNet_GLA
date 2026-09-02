## Task 0 â€“ Fix harness_noise.py + document dp45 anchor discrepancy in NOTES.md

**Files:**
- Modify: `light/noise/harness_noise.py:118` (step integrator line)
- Modify: `light/noise/NOTES.md` (append integrator-migration section)

**Context:** `harness_noise.py` line 118 calls `structure.step_dp45` directly. The spec requires a `getattr` guard so the file works on a cluster that still has the old `step_rk4`. After syncing the local dp45 tree to the cluster, all results will use dp45 â€” this is acceptable and must be documented.

- [ ] **Step 1: Apply getattr guard to harness_noise.py**

In `light/noise/harness_noise.py`, find line 118:
```python
        x = structure.step_dp45(x, Fy, Mz, DT)
```
Replace with:
```python
        # f92d8975: step_rk4 removed locally; guard for cluster trees that still have it
        _step = getattr(structure, 'step_rk4', structure.step_dp45)
        x = _step(x, Fy, Mz, DT)
```

Note: the `_step` lookup should be done ONCE outside the loop. Move it to just after `aero.reset(dt=DT)` in `rollout()` (line ~93), before the for-loop:
```python
    _step_struct = getattr(structure, 'step_rk4', structure.step_dp45)
```
Then in the loop: `x = _step_struct(x, Fy, Mz, DT)`

- [ ] **Step 2: Append integrator-migration note to NOTES.md**

Append the following section at the end of `light/noise/NOTES.md`:

```markdown
---

# Integrator migration note (f92d8975 â†’ dp45, 2026-07-08)

`structure.py` at commit f92d8975 replaced `step_rk4` with `step_dp45`
(Dormand-Prince RK45 via scipy.integrate.solve_ivp).  After syncing the local
tree to the cluster the cluster also uses dp45; the recorded E2/E2CC/E2-combo
results (all in `results/E2_*.npz`) were computed with `rk4_batch` in the MPC
horizon and `step_rk4` in the plant.  The new dp45 tree will not reproduce
those numbers bit-exactly â€” this is expected and **not a bug**.

## dp45 baseline anchors (home cell W30/Tg0.4, DAMULT=3) â€” TO BE FILLED

After running the smoke regression on cluster post-sync, fill in the dp45 values:

| check | rk4 value | dp45 value | delta |
|---|---|---|---|
| open cex0 | 0.4600 | TBD | TBD |
| one-step optimal R=3e-4 | +76.58% | TBD | TBD |
| one-step optimal R=1e-4 | +80.67% | TBD | TBD |
| combo oracle clean R=3e-4 | +80.5% | TBD | TBD |

Use the dp45 values as the new anchors for all subsequent studies.
If any non-chaotic number (open cex0, combo clean) differs by >1 pt, debug.
```

- [ ] **Step 3: Commit harness fix + NOTES update**
```bash
git add light/noise/harness_noise.py light/noise/NOTES.md
git commit -m "fix(harness_noise): getattr guard step_rk4â†’step_dp45; note integrator migration"
```

---

