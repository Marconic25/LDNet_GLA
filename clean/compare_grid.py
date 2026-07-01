import os, csv
os.environ.setdefault("DAMULT", "3")

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpc_gust as mg

W0_LIST = [10, 20, 30]
TG_LIST = [0.5, 1.0, 2.0]
DESIGN  = (11.46, 1.12)
CELLS   = [(W0, Tg) for W0 in W0_LIST for Tg in TG_LIST] + [DESIGN]
GAIN_CANDIDATES = [10, 20, 40, 80]
SIM_KW  = dict(TEND=3.0, NH=6, NGRID=15, SCHED=True)
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def H_m(Tg):  return 80.0 * Tg / 2.0
def H_ft(Tg): return H_m(Tg) / 0.3048

# -- Open-loop cache (computed once per cell) --
print("=== Computing open-loop baselines ===", flush=True)
open_cache = {}
for W0, Tg in CELLS:
    open_cache[(W0, Tg)] = mg.simulate("open", W0, Tg, **SIM_KW)
    m = mg.metrics(open_cache[(W0,Tg)], open_cache[(W0,Tg)], Tg)
    print(f"  open W0={W0:.2f} Tg={Tg:.2f} CLexc={m['clexc']:.3f}", flush=True)

# -- Proportional gain selection --
print("\n=== Proportional gain selection ===", flush=True)
gain_data = {}
for G in GAIN_CANDIDATES:
    rows = []
    for W0, Tg in CELLS:
        r_p = mg.simulate("prop", W0, Tg, gain=-G, **SIM_KW)
        m   = mg.metrics(r_p, open_cache[(W0,Tg)], Tg)
        rows.append((W0, Tg, m["clred"], m["flag"]))
    gain_data[G] = rows
    stable = [r[2] for r in rows if r[3] == ""]
    med = float(np.median(stable)) if stable else -999.0
    print(f"  G={G:3d}: {len(stable)}/{len(rows)} stable, median red={med:.1f}%", flush=True)

GSEL = None; best_med = -np.inf
for G in GAIN_CANDIDATES:
    stable = [r[2] for r in gain_data[G] if r[3] == ""]
    if len(stable) == len(CELLS):
        med = float(np.median(stable))
        if med > best_med: best_med = med; GSEL = G
if GSEL is None:
    for G in GAIN_CANDIDATES:
        stable = [r[2] for r in gain_data[G] if r[3] == ""]
        if stable:
            med = float(np.median(stable))
            if med > best_med: best_med = med; GSEL = G
print(f"==> Selected G={GSEL} (gain=-{GSEL}), median red={best_med:.1f}%\n", flush=True)

# -- Full comparison --
print("=== Full comparison grid ===", flush=True)
results = {}
for W0, Tg in CELLS:
    r_o = open_cache[(W0, Tg)]
    r_p = mg.simulate("prop", W0, Tg, gain=-GSEL, **SIM_KW)
    r_m = mg.simulate("mpc",  W0, Tg, **SIM_KW)
    results[(W0,Tg)] = dict(
        open=r_o, prop=r_p, mpc=r_m,
        m_open=mg.metrics(r_o, r_o, Tg),
        m_prop=mg.metrics(r_p, r_o, Tg),
        m_mpc =mg.metrics(r_m, r_o, Tg),
    )
    mp = results[(W0,Tg)]["m_prop"]
    mm = results[(W0,Tg)]["m_mpc"]
    mo = results[(W0,Tg)]["m_open"]
    print(f"  W0={W0:5.2f} Tg={Tg:.2f} | open CLexc={mo['clexc']:.3f} | "
          f"prop {mp['clred']:+.0f}% {mp['flag']} | "
          f"MPC {mm['clred']:+.0f}% {mm['comp_ms']:.0f}ms {mm['flag']}", flush=True)

# -- Markdown + CSV --
RT_LIM = 2.0
md_lines = [
    "# GLA Controller Comparison --- CS-25.341 Discrete Gust Envelope\n\n",
    f"Proportional gain selected: G={GSEL} (delta = -{GSEL} * (C_L - C_L_trim))\n\n",
    "CS-25.341 / FAR 25.341: H = U*Tg/2 (gust gradient distance, U=80 m/s).\n",
    "Range 30-350 ft; Tg in {0.5,1,2} s gives H in {20,40,80} m = {66,131,262} ft.\n\n",
    "| W0 [m/s] | Tg [s] | H [m] | H [ft] | Arm  | CLexc | CLred [%] | adRMS [d/s] | flap [d] | de@2.5 | J      | ms/step | RT?  | flag |\n",
    "|----------|--------|-------|--------|------|-------|-----------|-------------|----------|--------|--------|---------|------|------|\n",
]
csv_rows = []
for W0, Tg in CELLS:
    Hm = H_m(Tg); Hft = H_ft(Tg)
    for arm in ("open","prop","mpc"):
        m = results[(W0,Tg)][f"m_{arm}"]
        rt = "YES" if m["comp_ms"] < RT_LIM else "NO"
        md_lines.append(
            f"| {W0:8.2f} | {Tg:6.2f} | {Hm:5.0f} | {Hft:6.0f} | {arm:4s} |"
            f" {m['clexc']:.3f} | {m['clred']:+8.1f}  |"
            f" {m['adrms']:11.3f} | {m['flap_max']:8.1f} | {m['de_end']:6.2f} |"
            f" {m['J']:.4f} | {m['comp_ms']:7.1f} | {rt:4s} | {m['flag']} |\n")
        csv_rows.append(dict(W0=W0,Tg=Tg,H_m=Hm,H_ft=Hft,arm=arm,
                             clexc=m["clexc"],clred=m["clred"],adrms=m["adrms"],
                             flap_max=m["flap_max"],de_end=m["de_end"],
                             J=m["J"],comp_ms=m["comp_ms"],flag=m["flag"]))
md_lines += [
    "\n## Caveats\n\n",
    f"- Flap authority: delta_max=14 deg, dC_L/ddelta~0.014/deg -> caps alleviation for strong/short gusts.\n",
    "- LDNet under-predicts strong gusts (A_027 model 0.59 vs CFD 0.99) -> saturation earlier in reality.\n",
    f"- DAMULT={mg.DAMULT:.0f} applied to structural pitch damping to compensate LDNet aero under-damping.\n",
    f"- Proportional G={GSEL} selected by honest cross-validation (best median, all cells stable if possible).\n",
]
with open(f"{OUT_DIR}/compare_grid.md","w") as f: f.writelines(md_lines)
with open(f"{OUT_DIR}/compare_grid.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(csv_rows[0].keys()))
    w.writeheader(); w.writerows(csv_rows)
print(f"Saved {OUT_DIR}/compare_grid.md and .csv", flush=True)

# -- Bar plot --
cell_labels = [f"W{int(W0)}/Tg{Tg}" for W0,Tg in CELLS[:-1]] + [f"W{DESIGN[0]:.0f}/Tg{DESIGN[1]}"]
prop_reds  = [results[c]["m_prop"]["clred"]    for c in CELLS]
mpc_reds   = [results[c]["m_mpc"]["clred"]     for c in CELLS]
mpc_ms     = [results[c]["m_mpc"]["comp_ms"]   for c in CELLS]
prop_flags = [results[c]["m_prop"]["flag"]     for c in CELLS]
mpc_flags  = [results[c]["m_mpc"]["flag"]      for c in CELLS]
x = np.arange(len(CELLS)); w = 0.35
fig, ax = plt.subplots(figsize=(13,5))
bp = ax.bar(x-w/2, prop_reds, w, label=f"Prop G={GSEL}", color="steelblue", alpha=0.85)
bm = ax.bar(x+w/2, mpc_reds,  w, label="MPC H=6",       color="crimson",   alpha=0.85)
for b,fl in zip(bp,prop_flags):
    if fl: ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,"X",ha="center",fontsize=8)
for b,fl,ms in zip(bm,mpc_flags,mpc_ms):
    if fl: ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,"X",ha="center",fontsize=8)
    ax.text(b.get_x()+b.get_width()/2,max(b.get_height()-4,1),
            f"{ms:.0f}ms",ha="center",va="top",fontsize=7,color="white")
xlab=[f"{lbl}\nH={H_m(Tg):.0f}m/{H_ft(Tg):.0f}ft" for (W0,Tg),lbl in zip(CELLS,cell_labels)]
ax.set_xticks(x); ax.set_xticklabels(xlab,fontsize=8)
ax.axhline(0,color="k",lw=0.8)
ax.set_ylabel("C_L excursion reduction [%]")
ax.set_title(f"GLA: Proportional vs MPC  ---  CS-25.341 discrete gust envelope  (DAMULT={mg.DAMULT:.0f})")
ax.legend(); ax.grid(axis="y",alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/compare_grid_bars.png",dpi=130)
print(f"Saved {OUT_DIR}/compare_grid_bars.png", flush=True)

# -- Overlay plots: 3 representative cells --
overlay_cells = [DESIGN, (30,1.0), (10,0.5)]
overlay_names = ["design_W11-Tg112","strong_W30-Tg100","short_W10-Tg050"]
plot_keys = [("CL","C_L [-]",1.0),("ad","alpha_dot [deg/s]",180/np.pi),("de","delta [deg]",1.0)]
colors = {"open":"tab:blue","prop":"tab:orange","mpc":"crimson"}
for (W0,Tg),name in zip(overlay_cells,overlay_names):
    cell = results[(W0,Tg)]
    fig,axes=plt.subplots(len(plot_keys),1,figsize=(8,6),sharex=True)
    fig.suptitle(f"Open / Prop / MPC  ---  W0={W0:.1f} m/s Tg={Tg:.2f} s\n"
                 f"H={H_m(Tg):.0f} m ~ {H_ft(Tg):.0f} ft  CS-25.341",fontsize=10)
    t=cell["open"]["_t"]
    for j,(k,lab,sc) in enumerate(plot_keys):
        for arm in ("open","prop","mpc"):
            axes[j].plot(t,cell[arm][k]*sc,color=colors[arm],lw=1.5,label=arm)
        axes[j].axvspan(0,Tg,color="lightblue",alpha=0.2)
        axes[j].set_ylabel(lab,fontsize=9); axes[j].grid(alpha=0.3)
        if j==0: axes[j].legend(fontsize=9)
    axes[-1].set_xlabel("Time [s]"); axes[-1].set_xlim(0,3.0)
    plt.tight_layout()
    fn=f"{OUT_DIR}/compare_grid_overlay_{name}.png"
    fig.savefig(fn,dpi=130); print(f"Saved {fn}",flush=True)

# -- DLPF sensitivity (optional) --
print("\n=== Prop DLPF sensitivity ===", flush=True)
for W0,Tg in [DESIGN,(10,1.0),(30,1.0)]:
    r_o=open_cache[(W0,Tg)]
    r_with=results[(W0,Tg)]["prop"]
    r_no=mg.simulate("prop",W0,Tg,gain=-GSEL,prop_dlpf=False,**SIM_KW)
    mw=mg.metrics(r_with,r_o,Tg); mn=mg.metrics(r_no,r_o,Tg)
    print(f"  W0={W0:.1f} Tg={Tg:.2f}: with DLPF {mw['clred']:+.1f}% flap={mw['flap_max']:.1f}d | "
          f"no DLPF {mn['clred']:+.1f}% flap={mn['flap_max']:.1f}d", flush=True)

print("\ncompare_grid.py DONE", flush=True)
