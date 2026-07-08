"""
Honest equal-information comparison at the home cell (W30/Tg0.4, DAMULT=3):

    open  vs  prop-W(t+dt)  vs  one-step optimal wnext

Both controllers see the SAME information set {x(t), C_L measurement, W(t),
W(t+dt)} and run in the SAME scalar B=1 rollout (harness.scalar_rollout), with
the same actuator limits and the same pick rule:

    best = MAX CLred among runs with NO explosion flag
           (alpha_dot / alpha_ddot / h_ddot < 3x open loop); fallback min pitch.

Tuning budgets: prop sweeps the full 5x6 gain grid of propw_baseline.py
(g_CL x g_W, feedforward on W(t+dt)); optimal sweeps R over the standard
5-value grid. The full sweep tables are printed and saved — the prop table IS
the gain-fragility information.

Run on the cluster:
    DAMULT=3 python3 -s -u honest_home.py
Outputs: results_honest/honest_home.npz (+ full tables in the log).
"""
import os
import numpy as np
import harness as H
from controllers import PropW, OptGrid

W0, Tg = 30.0, 0.4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_honest')
os.makedirs(OUT, exist_ok=True)

R_GRID   = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
GCL_GRID = [-40., -60., -80., -120., -160.]
GW_GRID  = [0.0, -0.1, -0.2, -0.3, -0.5, -0.8]


def pick(ms):
    """No-flag pick rule (== cs25_study): max CLred among unflagged, else min pitch."""
    crs = np.array([m['clred'] for m in ms])
    prs = np.array([m['pitchpk'] for m in ms])
    idx = np.where(np.array([m['flag'] == '' for m in ms]))[0]
    return int(idx[np.argmax(crs[idx])]) if len(idx) else int(np.argmin(prs))


OL = H.scalar_rollout(None, W0, Tg)
cex0 = H.cex_of(OL['CL'], OL['_t'], Tg)
print(f'# honest_home W{W0:g}/Tg{Tg:g} DAMULT={os.environ.get("DAMULT", "1")} | '
      f'open cex0={cex0:.4f} (regression: 0.4600)', flush=True)

# ---- prop-W(t+dt): full gain sweep ----------------------------------------
print(f'\n=== prop-W(t+dt) gain sweep ({len(GCL_GRID)}x{len(GW_GRID)}) ===', flush=True)
print(f'{"gCL":>6s} {"gW":>6s} {"CLred":>7s} {"flap":>5s} {"pitch":>6s} {"flag":>8s}', flush=True)
pw_runs, pw_ms, pw_gains = [], [], []
for gCL in GCL_GRID:
    for gW in GW_GRID:
        r = H.scalar_rollout(PropW(gain_CL=gCL, gain_W=gW, use_wnext=True), W0, Tg)
        m = H.metrics(r, OL, Tg)
        pw_runs.append(r); pw_ms.append(m); pw_gains.append((gCL, gW))
        print(f'{gCL:6.0f} {gW:6.2f} {m["clred"]:+6.1f}% {m["flap_max"]:5.1f} '
              f'{m["pitchpk"]*180/np.pi:6.3f} {m["flag"]:>8s}', flush=True)
jp = pick(pw_ms)
PW, mp = pw_runs[jp], pw_ms[jp]
print(f'# prop-wnext BEST (no-flag pick): CLred={mp["clred"]:+.1f}%  '
      f'gCL={pw_gains[jp][0]:g} gW={pw_gains[jp][1]:g}  flap={mp["flap_max"]:.1f}  '
      f'pitch={mp["pitchpk"]*180/np.pi:.3f}deg  {mp["flag"]}', flush=True)

# ---- optimal wnext: R sweep -------------------------------------------------
print(f'\n=== optimal wnext R sweep ({len(R_GRID)}) ===', flush=True)
print(f'{"R":>7s} {"CLred":>7s} {"flap":>5s} {"pitch":>6s} {"flag":>8s}', flush=True)
op_runs, op_ms = [], []
for R in R_GRID:
    ctrl = OptGrid(R=R, G=161, gate='hard', use_wnext=True, refine=True)
    r = H.scalar_rollout(ctrl, W0, Tg)
    m = H.metrics(r, OL, Tg)
    op_runs.append(r); op_ms.append(m)
    print(f'{R:7g} {m["clred"]:+6.1f}% {m["flap_max"]:5.1f} '
          f'{m["pitchpk"]*180/np.pi:6.3f} {m["flag"]:>8s}', flush=True)
jo = pick(op_ms)
OP, mo = op_runs[jo], op_ms[jo]
print(f'# optimal-wnext BEST (no-flag pick): CLred={mo["clred"]:+.1f}%  '
      f'R*={R_GRID[jo]:g}  flap={mo["flap_max"]:.1f}  '
      f'pitch={mo["pitchpk"]*180/np.pi:.3f}deg  {mo["flag"]}', flush=True)

# ---- summary + npz ----------------------------------------------------------
print(f'\n# SUMMARY W{W0:g}/Tg{Tg:g} (equal info, equal pick rule):', flush=True)
print(f'#   open          cex0 = {cex0:.4f}', flush=True)
print(f'#   prop-W(t+dt)  CLred = {mp["clred"]:+.1f}%  (gCL={pw_gains[jp][0]:g}, gW={pw_gains[jp][1]:g})', flush=True)
print(f'#   optimal-wnext CLred = {mo["clred"]:+.1f}%  (R*={R_GRID[jo]:g})', flush=True)

out = dict(ts=OL['_t'], Wt=OL['_Wt'], CLTRIM=H.CLTRIM, cex0=cex0, W0=W0, Tg=Tg)
for name, r in [('open', OL), ('pw', PW), ('opt', OP)]:
    for k in ['CL', 'de', 'al', 'ad']:
        out[f'{name}_{k}'] = r[k]
out.update(
    pw_clred=mp['clred'], pw_gCL=pw_gains[jp][0], pw_gW=pw_gains[jp][1],
    pw_flap=mp['flap_max'], pw_pitch=mp['pitchpk'], pw_flag=mp['flag'],
    opt_clred=mo['clred'], opt_R=R_GRID[jo],
    opt_flap=mo['flap_max'], opt_pitch=mo['pitchpk'], opt_flag=mo['flag'],
    pw_tab_gCL=np.array([g[0] for g in pw_gains]),
    pw_tab_gW=np.array([g[1] for g in pw_gains]),
    pw_tab_clred=np.array([m['clred'] for m in pw_ms]),
    pw_tab_flap=np.array([m['flap_max'] for m in pw_ms]),
    pw_tab_pitch=np.array([m['pitchpk'] for m in pw_ms]),
    pw_tab_flag=np.array([m['flag'] for m in pw_ms]),
    opt_tab_R=np.array(R_GRID),
    opt_tab_clred=np.array([m['clred'] for m in op_ms]),
    opt_tab_flap=np.array([m['flap_max'] for m in op_ms]),
    opt_tab_pitch=np.array([m['pitchpk'] for m in op_ms]),
    opt_tab_flag=np.array([m['flag'] for m in op_ms]),
)
fn = os.path.join(OUT, 'honest_home.npz')
np.savez_compressed(fn, **out)
print(f'# saved {fn}', flush=True)
print('# DONE', flush=True)
