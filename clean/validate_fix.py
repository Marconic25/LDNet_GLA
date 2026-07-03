"""Validate fix #1: receding-horizon MPC (N=6, z rolled forward, pitch cost, leak-patched)
vs the one-step optimal (the loser) vs proportional-W, at W48 / high-k. All know W.
Does horizon>1 recover the gap the one-step optimal lost?"""
import os, numpy as np
import mpc_gust as M
import smoothprop as SP   # batch_prop_dlpf(W0,Tg,G1,GW,DLPF), batch_opt(W0,Tg) -> (cex,tv)

CLTRIM = M.CLTRIM; TEND = 3.0
CELLS = {'W30/T0.30': (30., 0.30), 'W30/T0.40': (30., 0.40), 'W30/T0.50': (30., 0.50)}
g1s = [-10., -20., -40., -60., -80., -120., -160.]
gws = [-1.0, -0.6, -0.4, -0.2, 0.0]
G1 = np.array([a for a in g1s for _ in gws]); GW = np.array([b for _ in g1s for b in gws])


def cl_pk(r, Tg):
    mw = r['_t'] <= (Tg + 0.5)
    return (float(np.max(np.abs(r['CL'][mw] - CLTRIM))), float(np.max(np.abs(r['al'][mw]))))


def main():
    print(f'# validate_fix (MPC N=6 z-aware, leak-patched) vs one-step vs prop-W   DAMULT={os.environ.get("DAMULT","1")}', flush=True)
    print(f'\n{"cell":10s} {"k":>6s} | {"prop-W raw":>10s} | {"prop-W dlpf":>11s} | {"one-step":>9s} | {"MPC N=6":>18s}', flush=True)
    print('-' * 78, flush=True)
    for name, (W0, Tg) in CELLS.items():
        OL = M.simulate('open', W0, Tg, TEND=TEND, DLPF=0.0)
        t = OL['_t']; cex0, pk0 = cl_pk(OL, Tg); k = np.pi / (80 * Tg)
        # prop-W raw and smoothed (best CLred over gains)
        cr = {}
        for dl in [0.0, 0.7]:
            ce, _ = SP.batch_prop_dlpf(W0, Tg, G1, GW, dl)
            cr[dl] = float(np.max((cex0 - ce) / cex0 * 100))
        # one-step optimal (oracle W)
        ceo, _ = SP.batch_opt(W0, Tg); cr_one = float(np.max((cex0 - ceo) / cex0 * 100))
        # MPC N=6 (oracle W), DLPF=0, no gust-gating (RQUIET=RW), sweep (RW,QAD)
        best = (-1e9, None, None)
        for RW in [5e-4, 1e-3, 3e-3, 1e-2]:
            for QAD in [30., 100.]:
                r = M.simulate('mpc', W0, Tg, TEND=TEND, NH=6, NGRID=15, QAD=QAD, RW=RW,
                               DLPF=0.0, RQUIET=RW, SCHED=False)
                ce, pk = cl_pk(r, Tg); clred = (cex0 - ce) / cex0 * 100
                if clred > best[0]: best = (clred, pk / max(pk0, 1e-12), (RW, QAD))
        crm, prm, (RWb, QADb) = best
        print(f'{name:10s} {k:6.3f} | {cr[0.0]:9.1f}% | {cr[0.7]:10.1f}% | {cr_one:8.1f}% | '
              f'{crm:7.1f}% pitch={prm:.2f} (R={RWb:g},Q={QADb:g})', flush=True)
    print('\n# does MPC N=6 beat one-step and match/beat prop-W?', flush=True)
    print('# DONE', flush=True)


if __name__ == '__main__':
    main()
