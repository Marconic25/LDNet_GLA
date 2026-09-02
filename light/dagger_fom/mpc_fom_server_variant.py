#!/usr/bin/env python3
"""
Copy of light/tests/mpc_fom_server.py that uses MPCPreviewControllerQadot
(this study's pitch-rate-penalized controller) instead of
light/optimal.py::MPCPreviewController. Same stdin/stdout JSON-RPC protocol,
same CLI, plus --qadot for the new penalty weight. See
light/tests/mpc_fom_server.py's docstring for the protocol; unchanged here.

Usage (inside tensorflow_gpu.sif):
    python3 -u mpc_fom_server_variant.py --model MODEL_DIR --R 3e-4 \
        --qadot 0.01 [--N 8] [--damult 1.0] [--dt 0.002] [--U 80.0]
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
from controller_qadot import MPCPreviewControllerQadot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='LDNet model dir (config.json + weights)')
    ap.add_argument('--R', type=float, required=True, help='MPC flap-effort weight (R*)')
    ap.add_argument('--qadot', type=float, default=0.0, help='pitch-rate move-suppression weight')
    ap.add_argument('--N', type=int, default=8, help='MPC preview horizon length')
    ap.add_argument('--damult', type=float, default=1.0, help='structure.D_ALPHA multiplier')
    ap.add_argument('--dt', type=float, default=0.002, help='controller horizon step [s]')
    ap.add_argument('--U', type=float, default=80.0, help='freestream velocity [m/s]')
    args = ap.parse_args()

    structure.D_ALPHA *= args.damult

    aero = LDNetAero(args.model)
    X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
    CLTRIM = float(aero.predict(X0, 0.0, 0.0, args.U)[0])

    mpc = MPCPreviewControllerQadot(aero, U=args.U, dt=args.dt, C_L_trim=CLTRIM,
                                     N=args.N, R=args.R, Q_alpha_dot=args.qadot)

    print(json.dumps({"ready": True, "CLTRIM": CLTRIM, "lam": mpc.lam,
                       "qadot": args.qadot}), flush=True)

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
