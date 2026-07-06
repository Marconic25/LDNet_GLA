"""
Phase 1 — characterize the good vs bad closed-loop branch at W30/Tg0.4, DAMULT=3.

Two runs of the batched single-step-optimal oracle:
  A) R_GRID = [3e-4]*5   -> IDENTICAL config in every row. Any trajectory split
     is PURE batch-position rounding. Isolates the good/bad branches cleanly.
  B) R_GRID = [1e-2,3e-3,1e-3,3e-4,1e-4]  -> the spec's reference sweep (row 3
     = R=3e-4 is claimed to land good on the cluster).

For each row we print final CLred / flap_max / pitch peak, and we locate the
FIRST time step where the chosen flap index diverges between a good row and a
bad row, dumping the local argmin cost landscape + near-tie margins there.

Saves everything to results_char.npz for offline plotting.
"""
import numpy as np
import harness as H

Tg = 0.4; W0 = 30.0

# open-loop reference
OL = H.scalar_rollout(None, W0, Tg)
ts = OL['_t']
cex0 = H.cex_of(OL['CL'], ts, Tg)
print(f"# W0={W0} Tg={Tg} DAMULT env-scaled | CLTRIM={H.CLTRIM:.5f}", flush=True)
print(f"# open-loop cex0 = {cex0:.4f}  (ref 0.4600)", flush=True)
print(f"# LAM(z_leak) = {H.LAM}", flush=True)


def summarize(T, label):
    ts = T['_t']; R = T['_R']; B = T['de'].shape[0]
    print(f"\n=== {label} ===", flush=True)
    cexs = []
    for b in range(B):
        cex = H.cex_of(T['CL'][b], ts, Tg)
        clred = (cex0 - cex) / cex0 * 100.0
        mw = H.window(ts, Tg)
        fmax = float(np.max(np.abs(T['de'][b][mw])))
        ppk  = float(np.max(np.abs(T['al'][b][mw])))
        cexs.append(cex)
        print(f"  row {b}: R={R[b]:.1e}  cex={cex:.4f}  CLred={clred:+6.1f}%  "
              f"flap_max={fmax:5.1f}  pitchpk={ppk*180/np.pi:6.3f}deg", flush=True)
    return np.array(cexs)


def find_split(T, gb, bb):
    """First step where chosen index differs between good row gb and bad row bb."""
    dj = T['jarg'][gb] != T['jarg'][bb]
    idx = np.where(dj)[0]
    if len(idx) == 0:
        print(f"  rows {gb}&{bb} never diverge in chosen index.", flush=True)
        return None
    i = int(idx[0])
    ts = T['_t']
    print(f"\n  --- split of rows {gb}(good) vs {bb}(bad) at step i={i} t={ts[i]:.4f}s ---", flush=True)
    for b in (gb, bb):
        print(f"    row{b}: j={T['jarg'][b,i]:3d} de={T['de'][b,i]:+7.3f}  "
              f"cl0={T['cl0'][b,i]:+.4f} gate_up={T['gate'][b,i]}  "
              f"nfeas={T['nfeas'][b,i]:3d} margin={T['margin'][b,i]:.3e}", flush=True)
    # a few steps of context before/after
    print("    context (step: de_good de_bad  CLg CLb  margin_good margin_bad):", flush=True)
    for k in range(max(0, i-3), min(len(ts), i+6)):
        print(f"      i={k} t={ts[k]:.4f}: de {T['de'][gb,k]:+7.3f} {T['de'][bb,k]:+7.3f} | "
              f"CL {T['CL'][gb,k]:+.4f} {T['CL'][bb,k]:+.4f} | "
              f"mrg {T['margin'][gb,k]:.2e} {T['margin'][bb,k]:.2e} | "
              f"gate {T['gate'][gb,k]}{T['gate'][bb,k]}", flush=True)
    return i


# --- Run A: identical config, isolate rounding branches ---------------------
print("\nRunning A: R_GRID=[3e-4]*5 (identical rows) ...", flush=True)
TA = H.batch_optW_trace(W0, Tg, [3e-4]*5)
cexA = summarize(TA, "A) R=[3e-4]x5")
# classify rows
good_rows = [b for b in range(5) if cexA[b] < 0.5*(cexA.min()+cexA.max())]
bad_rows  = [b for b in range(5) if b not in good_rows]
print(f"\n  good rows (low cex): {good_rows}   bad rows: {bad_rows}", flush=True)
splitA = None
if good_rows and bad_rows:
    splitA = find_split(TA, good_rows[0], bad_rows[0])

# --- Run B: reference R sweep -----------------------------------------------
print("\nRunning B: R_GRID=[1e-2,3e-3,1e-3,3e-4,1e-4] (reference sweep) ...", flush=True)
RB = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
TB = H.batch_optW_trace(W0, Tg, RB)
cexB = summarize(TB, "B) reference sweep")

# pick the good-branch row from whichever run has the lowest cex
allcex = list(cexA) + list(cexB)
if cexA.min() <= cexB.min():
    Tgood = TA; gr = int(np.argmin(cexA)); srcg = 'A'
else:
    Tgood = TB; gr = int(np.argmin(cexB)); srcg = 'B'
print(f"\n# GOOD branch source: run {srcg} row {gr}  cex={H.cex_of(Tgood['CL'][gr],ts,Tg):.4f}", flush=True)

np.savez_compressed('results_char.npz',
    ts=ts, Wt=OL['_Wt'], cex0=cex0, CLTRIM=H.CLTRIM,
    OL_CL=OL['CL'], OL_al=OL['al'], OL_ad=OL['ad'],
    A_de=TA['de'], A_CL=TA['CL'], A_al=TA['al'], A_ad=TA['ad'],
    A_cl0=TA['cl0'], A_jarg=TA['jarg'], A_margin=TA['margin'],
    A_gate=TA['gate'], A_R=TA['_R'], A_cex=cexA,
    B_de=TB['de'], B_CL=TB['CL'], B_al=TB['al'], B_ad=TB['ad'],
    B_cl0=TB['cl0'], B_jarg=TB['jarg'], B_margin=TB['margin'],
    B_gate=TB['gate'], B_R=TB['_R'], B_cex=cexB,
    good_src=srcg, good_row=gr)
print("\nSaved results_char.npz", flush=True)
print("# DONE", flush=True)
