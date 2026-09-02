#!/usr/bin/env python3
"""
FOM controller server using MPCPreviewControllerStable (full validated-stable
cost recipe ported from clean/controller.py). Same stdin/stdout JSON-RPC
protocol as light/tests/mpc_fom_server.py.

Usage (inside tensorflow_gpu.sif):
    python3 -u mpc_fom_server_stable.py --model MODEL_DIR \
        [--Qh 1e4] [--Qalpha 1e4] [--Qadot 1e4] [--QCL 1e3] [--R 1.0] \
        [--N 8] [--damult 1.0] [--dt 0.002] [--U 80.0]
"""
import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..'))  # light/
import structure  # noqa: E402
from ldnet_aero import LDNetAero  # noqa: E402
from controller_stable import MPCPreviewControllerStable  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--Qh', type=float, default=1e4)
    ap.add_argument('--Qalpha', type=float, default=1e4)
    ap.add_argument('--Qadot', type=float, default=1e4)
    ap.add_argument('--QCL', type=float, default=1e3)
    ap.add_argument('--R', type=float, default=1.0)
    ap.add_argument('--Rdu', type=float, default=0.0)
    ap.add_argument('--N', type=int, default=8)
    ap.add_argument('--damult', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=0.002)
    ap.add_argument('--U', type=float, default=80.0)
    args = ap.parse_args()

    structure.D_ALPHA *= args.damult

    aero = LDNetAero(args.model)
    X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
    CLTRIM = float(aero.predict(X0, 0.0, 0.0, args.U)[0])

    mpc = MPCPreviewControllerStable(
        aero, U=args.U, dt=args.dt, C_L_trim=CLTRIM, N=args.N,
        Q_h=args.Qh, Q_alpha=args.Qalpha, Q_alpha_dot=args.Qadot,
        Q_CL=args.QCL, R=args.R, R_du=args.Rdu)

    print(json.dumps({"ready": True, "CLTRIM": CLTRIM, "lam": mpc.lam,
                       "Qh": args.Qh, "Qalpha": args.Qalpha, "Qadot": args.Qadot,
                       "QCL": args.QCL, "R": args.R}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get('cmd')
        if cmd == 'quit':
            print(json.dumps({"ok": True}), flush=True)
            break
        if cmd == 'reset':
            mpc.reset()
            print(json.dumps({"ok": True}), flush=True)
            continue
        state = tuple(float(v) for v in req['state'])
        wseq = [float(v) for v in req['wseq']]
        wnow = float(req['wnow'])
        delta = mpc.compute(state, wseq, wnow)
        print(json.dumps({"delta": delta}), flush=True)


if __name__ == '__main__':
    main()
