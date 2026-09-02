"""
Fast, cluster-free screening: sweep Q_alpha_dot for MPCPreviewControllerQadot
on the ROM (LDNet surrogate plant), targeting the W30/Tg0.4 cell where the
real FOM shows flap chatter at the pitch mode. Reuses light/run.py's aero
model, structural integrator, gust profile, and metrics() UNCHANGED (only
imported) — this script duplicates run.py's simulate() loop because that
function hardcodes light/optimal.py's MPCPreviewController inline; swapping
the controller class is the only thing that differs below.

Usage:
    python3 rom_screen.py                    # default sweep, W30/Tg0.4
    python3 rom_screen.py --w0 30 --tg 0.4 --r 0.0003
"""
import argparse
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..'))  # light/
import run as Rn  # noqa: E402  (aero, structure, CLTRIM, gust(), metrics(), _gust_window())
from controller_qadot import MPCPreviewControllerQadot  # noqa: E402

U, RHO, C, DT, S = Rn.U, Rn.RHO, Rn.C, Rn.DT, Rn.S


def simulate_qadot(W0, Tg, R, Q_alpha_dot, TEND=3.0, DMAX=14., NGRID=161, NH=8, R_du=0.0):
    """Mirrors light/run.py::simulate(mode='combo', ...) but with the
    pitch-rate-penalized controller. Returns the same trajectory dict shape."""
    Nsteps = int(round(TEND / DT)) + 1
    ts = np.arange(Nsteps) * DT
    Wt = np.array([Rn.gust(t, W0, Tg) for t in ts])

    Rn.aero.reset(dt=DT)
    ctrl = MPCPreviewControllerQadot(
        Rn.aero, U=U, dt=DT, rho=RHO, S=S, C=C,
        C_L_trim=Rn.CLTRIM, N=NH, R=R, R_du=R_du, Q_alpha_dot=Q_alpha_dot,
        G=NGRID, delta_max=DMAX, delta_dot_max=300.)
    ctrl.reset()

    x = Rn.X0.copy()
    rec = {k: [] for k in ['h', 'hd', 'al', 'ad', 'hdd', 'add', 'de', 'CL', 'CM', 'Fy']}

    for i in range(Nsteps):
        Wi = float(Wt[i])
        lo = i + 1
        hi = min(i + 1 + NH, Nsteps)
        w_seq = np.zeros(NH)
        w_seq[:hi - lo] = Wt[lo:hi]
        de = ctrl.compute(x, w_seq, Wi)

        cl, cm = Rn.aero.predict(x, de, Wi, U)
        Fy = ctrl.q * cl
        Mz = ctrl.q * cm * C
        der = Rn.structure.rhs(x, Fy, Mz)
        Rn.aero.advance(x, de, Wi, U, DT)
        x = Rn.structure.step_dp45(x, Fy, Mz, DT)
        for k, v in zip(['h', 'hd', 'al', 'ad', 'hdd', 'add', 'de', 'CL', 'CM', 'Fy'],
                        [x[0], x[1], x[2], x[3], der[1], der[3], de, float(cl), float(cm), Fy]):
            rec[k].append(v)

    out = {k: np.array(v) for k, v in rec.items()}
    out['_t'] = ts
    out['_Wt'] = Wt
    return out


def count_oscillation(de, min_run=3):
    """Count sign changes in the flap-command first difference — same
    diagnostic used in this session's FOM log analysis."""
    diffs = np.diff(de)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w0', type=float, default=30.0)
    ap.add_argument('--tg', type=float, default=0.40)
    ap.add_argument('--r', type=float, default=0.0003)
    ap.add_argument('--qgrid', type=float, nargs='+',
                     default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument('--ngrid', type=int, nargs='+', default=None,
                     help='if given, sweep the argmin grid resolution G '
                          '(H1: quantization test) at Q_alpha_dot=0 instead '
                          'of sweeping --qgrid')
    args = ap.parse_args()

    OL = Rn.simulate('open', args.w0, args.tg, TEND=3.0)

    if args.ngrid:
        print(f"=== ROM NGRID sweep (H1, Q_alpha_dot=0): W0={args.w0} Tg={args.tg} R={args.r} ===")
        print(f"{'NGRID':>12} {'step_deg':>9} {'CLred%':>8} {'flap_max':>9} {'osc_count':>10} {'time_s':>8}")
        results = []
        for NG in args.ngrid:
            t0 = time.time()
            RES = simulate_qadot(args.w0, args.tg, args.r, 0.0, TEND=3.0, NGRID=NG)
            dt_s = time.time() - t0
            m = Rn.metrics(RES, OL, args.tg)
            osc = count_oscillation(RES['de'][Rn._gust_window(RES['_t'], args.tg)])
            step_deg = 2 * 14.0 / (NG - 1)
            print(f"{NG:12d} {step_deg:9.4f} {m['clred']:8.1f} {m['flap_max']:9.2f} {osc:10d} {dt_s:8.1f}")
            results.append(dict(NGRID=NG, step_deg=step_deg, clred=m['clred'],
                                flap_max=m['flap_max'],
                                pitch_max_deg=float(np.degrees(m['pitchpk'])), osc_count=osc))
        import json
        out_path = os.path.join(_THIS, f'rom_screen_ngrid_W{args.w0:g}_Tg{args.tg:.2f}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {out_path}")
        return

    print(f"=== ROM Q_alpha_dot sweep: W0={args.w0} Tg={args.tg} R={args.r} ===")
    print(f"{'Q_alpha_dot':>12} {'CLred%':>8} {'flap_max':>9} {'osc_count':>10} {'time_s':>8}")
    results = []
    for Q in args.qgrid:
        t0 = time.time()
        RES = simulate_qadot(args.w0, args.tg, args.r, Q, TEND=3.0)
        dt_s = time.time() - t0
        m = Rn.metrics(RES, OL, args.tg)
        osc = count_oscillation(RES['de'][Rn._gust_window(RES['_t'], args.tg)])
        print(f"{Q:12.4g} {m['clred']:8.1f} {m['flap_max']:9.2f} {osc:10d} {dt_s:8.1f}")
        results.append(dict(Q_alpha_dot=Q, clred=m['clred'], flap_max=m['flap_max'],
                            pitch_max_deg=float(np.degrees(m['pitchpk'])), osc_count=osc))

    import json
    out_path = os.path.join(_THIS, f'rom_screen_W{args.w0:g}_Tg{args.tg:.2f}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
