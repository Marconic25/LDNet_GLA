#!/usr/bin/env python3
"""
Persistent MPC controller server for the FOM closed-loop verification harness
(recon/cluster/cosim_driver_extract.py --controller mpc), using the LINEAR
UNSTEADY aerodynamic model (light/ablation/linear_aero.py) as the internal
model instead of the LDNet.

This is the FOM-side counterpart of light/ablation/run_ablation.py's ROM-vs-ROM
internal-model comparison (see light/latex/appendixA.tex, app:modelcomp): there
the linear model's own predictions stood in for the plant too, so the R*=0.01
Test-2 result (CLred=-45.6%) still carries the self-consistency effect the
LDNet-vs-LDNet comparison also has. This server lets the SAME MPC loop drive
the flap using the linear model as internal model while the real co-simulation
(recon/cluster/cosim_driver_extract.py) supplies the true structural state and
plant response each window — eliminating that asymmetry entirely, the same way
mpc_fom_server.py already does for the LDNet.

See mpc_fom_server.py's docstring for why this runs as a persistent server
inside the TF container rather than being spawned per co-simulation window
(container+TF import overhead). LinearUnsteadyAero itself does not need
TensorFlow, but it is launched from the same tensorflow_gpu.sif container and
LDNet_GLA/light/tests working directory as the LDNet server so that
cosim_driver_extract.py's start_mpc_server() can select between the two with
a single flag (--mpc-server-script) rather than needing two different
containers/paths wired through the FOM driver.

Request  (one JSON object per line on stdin):
    {"state": [h, hd, a, ad], "wseq": [w_1..w_N], "wnow": w}
        -> returns {"delta": <float, deg>}
    {"cmd": "reset"}
        -> clears the controller's internal lag-state + rate-limit memory,
           returns {"ok": true}. Call once at the start of a run.
    {"cmd": "quit"}
        -> returns {"ok": true} and exits.

Usage (inside tensorflow_gpu.sif):
    python3 -u mpc_fom_server_linear.py --coeffs light/ablation/linear_coeffs.json \
        --R 0.01 [--N 8] [--damult 3.0] [--dt 0.002] [--U 80.0]

--damult scales structure.D_ALPHA the same way mpc_fom_server.py does, so the
controller's internal horizon prediction uses the same (possibly non-physical)
pitch damping the FOM driver applies to ITS OWN structural integration via
cosim_driver_extract.py's --damult flag — see that script's docstring.
"""
import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..'))            # light/
sys.path.insert(0, os.path.join(_THIS, '..', 'ablation'))  # light/ablation/
import structure
from linear_aero import LinearUnsteadyAero
from optimal import MPCPreviewController

_DEFAULT_COEFFS = os.path.join(_THIS, '..', 'ablation', 'linear_coeffs.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--coeffs', default=_DEFAULT_COEFFS,
                     help='linear model coefficient JSON (fit_linear.py output)')
    ap.add_argument('--R', type=float, required=True, help='MPC flap-effort weight (R*)')
    ap.add_argument('--N', type=int, default=8, help='MPC preview horizon length')
    ap.add_argument('--damult', type=float, default=1.0, help='structure.D_ALPHA multiplier')
    ap.add_argument('--dt', type=float, default=0.002, help='controller horizon step [s]')
    ap.add_argument('--U', type=float, default=80.0, help='freestream velocity [m/s]')
    args = ap.parse_args()

    structure.D_ALPHA *= args.damult

    aero = LinearUnsteadyAero(coeff_path=args.coeffs)
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
