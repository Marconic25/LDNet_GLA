#!/usr/bin/env python3
"""
Persistent MPC controller server for the FOM closed-loop verification harness
(recon/cluster/cosim_driver_extract.py --controller mpc).

Neither Python environment that runs cosim_driver.py on the cluster
(~/cosim_env, NACA2312/my_venv) has TensorFlow installed — only the
tensorflow_gpu.sif apptainer image does. Spawning a fresh apptainer+TF
process per co-simulation window (potentially hundreds for a single gust
case at the production window size of ~3.5ms) would waste minutes of
container-startup + TF-import overhead per window. Instead this script runs
ONCE inside the TF container, loads the LDNet model + MPCPreviewController a
single time, and then serves one controller decision per co-simulation
window over stdin/stdout — a line-delimited JSON RPC protocol, synchronous
and single-threaded (there is exactly one caller, the FSI driver).

Request  (one JSON object per line on stdin):
    {"state": [h, hd, a, ad], "wseq": [w_1..w_N], "wnow": w}
        -> returns {"delta": <float, deg>}
    {"cmd": "reset"}
        -> clears the controller's internal latent + rate-limit memory,
           returns {"ok": true}. Call once at the start of a run.
    {"cmd": "quit"}
        -> returns {"ok": true} and exits.

Usage (inside tensorflow_gpu.sif):
    python3 -u mpc_fom_server.py --model MODEL_DIR --R 3e-4 [--N 8]
        [--damult 1.0] [--dt 0.002] [--U 80.0]

--damult scales structure.D_ALPHA the same way light/run.py's
`structure.D_ALPHA *= DAMULT` does, so the controller's internal horizon
prediction uses the same (possibly non-physical) pitch damping the R* grid
search in light/results_cs25_combo was calibrated against. The FOM driver
must apply the identical --damult to its OWN structural integration
(cosim_driver_extract.py's D_ALPHA) for the comparison to be apples-to-apples
— see cosim_driver_extract.py's --damult flag.
"""
import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..'))  # light/
import structure
from ldnet_aero import LDNetAero
from optimal import MPCPreviewController


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='LDNet model dir (config.json + weights)')
    ap.add_argument('--R', type=float, required=True, help='MPC flap-effort weight (R*)')
    ap.add_argument('--N', type=int, default=8, help='MPC preview horizon length')
    ap.add_argument('--damult', type=float, default=1.0, help='structure.D_ALPHA multiplier')
    ap.add_argument('--dt', type=float, default=0.002, help='controller horizon step [s]')
    ap.add_argument('--U', type=float, default=80.0, help='freestream velocity [m/s]')
    args = ap.parse_args()

    structure.D_ALPHA *= args.damult

    aero = LDNetAero(args.model)
    X0 = np.array([-6.49179e-3, 0.0, -8.76338e-4, 0.0])
    CLTRIM = float(aero.predict(X0, 0.0, 0.0, args.U)[0])

    mpc = MPCPreviewController(aero, U=args.U, dt=args.dt, C_L_trim=CLTRIM,
                                N=args.N, R=args.R)

    print(json.dumps({"ready": True, "CLTRIM": CLTRIM, "lam": mpc.lam}), flush=True)

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
