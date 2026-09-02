#!/usr/bin/env python3
"""
courant_sweep.py — maxCo sweep at fixed coupling window for the reference gust case.

The coupling window is held at the production value (50 steps x 7e-5 s = 3.5 ms,
same as generate_dataset_v5 / submit_gust) while pimpleFoam runs with
adjustTimeStep yes and maxCo in {0.5, 1, 2, 4}.  Window chaining relies on
writeControl adjustableRunTime: OF7 aligns writes to (startTime + k*writeInterval),
and each window restarts from latestTime, so writes land exactly on window ends.
The copied cosim_driver.py is patched so its per-window writeInterval is the
window DURATION in seconds instead of a step count.

Reference gust (as submit_gust.pbs): W0=40 m/s, t=[0.1, 0.9] s run-relative,
warm start --from-checkpoint.  t-end 1.2 s covers gust + CL peak.

Usage:
    python3 courant_sweep.py setup            # build case dirs + PBS scripts
    python3 courant_sweep.py run Co_0.5       # run one case (used inside PBS)
    python3 courant_sweep.py report           # write courant_sweep_report.txt

Every subcommand accepts --study-dir (default "courant_sweep") and --w0
(default 40.0); with the defaults the behaviour is identical to the original
single-study script.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKDIR   = Path("/work/u10677113/NACA2312")
SRC_CASE  = WORKDIR / "cosim_main"
VENV_PY   = WORKDIR / "my_venv" / "bin" / "python3"

MAXCO_LEVELS = [0.5, 1.0, 2.0, 4.0]
FIXED_NAME   = "Co_fixed"    # reference: exact production config (adjustTimeStep no, dt=7e-5)
WINDOW_STEPS = 50        # production coupling window
DT_NOMINAL   = 7e-5      # production nominal dt -> window = 3.5e-3 s
T_END        = 1.2       # run-relative [s]
MAX_DELTA_T  = 5e-4      # cap so every window keeps >= 7 CFD steps
N_PROCS      = 16

# defaults reproduce the original single-study behaviour byte-for-byte;
# overridden by the global --study-dir / --w0 options in _configure()
DEFAULT_STUDY = "courant_sweep"
DEFAULT_W0    = 40.0
STUDY_NAME = DEFAULT_STUDY
STUDY_DIR  = WORKDIR / DEFAULT_STUDY
W0         = DEFAULT_W0
GUST_ARGS  = "--gust-w0 40.0 --gust-t-start 0.1 --gust-t-end 0.9"
JOB_PREFIX = "CO"


def _configure(study_dir, w0):
    """Apply the global --study-dir / --w0 options to the module globals."""
    global STUDY_NAME, STUDY_DIR, W0, GUST_ARGS, JOB_PREFIX
    STUDY_NAME = study_dir
    STUDY_DIR  = WORKDIR / study_dir
    W0         = w0
    GUST_ARGS  = "--gust-w0 %s --gust-t-start 0.1 --gust-t-end 0.9" % w0
    tag = re.sub(r"^courant_sweep_?", "", study_dir)
    JOB_PREFIX = "CO" if not tag else "CO" + tag[:1].upper()


def _is_default():
    return STUDY_NAME == DEFAULT_STUDY and W0 == DEFAULT_W0


def _cli_opts():
    """Options to embed in the generated PBS run command ('' for defaults)."""
    if _is_default():
        return ""
    return " --study-dir %s --w0 %s" % (STUDY_NAME, W0)


def _pbs_path(name):
    if STUDY_NAME == DEFAULT_STUDY:
        return WORKDIR / ("submit_courant_%s.pbs" % name)
    tag = re.sub(r"^courant_sweep_?", "", STUDY_NAME) or STUDY_NAME
    return WORKDIR / ("submit_courant_%s_%s.pbs" % (tag, name))


def case_name(co):
    return "Co_%g" % co


def _restore_checkpoint(dst):
    # the copytree ignore patterns (processor*, cosim_state.json) apply at
    # every depth and hollow out checkpoint/ — restore it verbatim
    chk_dst = dst / "checkpoint"
    if chk_dst.exists():
        shutil.rmtree(chk_dst)
    shutil.copytree(SRC_CASE / "checkpoint", chk_dst)


def build_case(co):
    dst = STUDY_DIR / case_name(co)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC_CASE, dst, ignore=shutil.ignore_patterns(
        "processor*", "postProcessing*", "log.*", "dynamicCode",
        "*.csv", "validation_fsi", "figures", "cosim_state.json",
        "cosim_state_full.json"))
    # drop stale reconstructed time dirs (keep 0, 0.orig)
    for d in dst.iterdir():
        if d.is_dir() and re.match(r"^\d+\.?\d*$", d.name) and d.name != "0":
            shutil.rmtree(d)
    _restore_checkpoint(dst)

    # ---- controlDict: adaptive dt with this maxCo ----
    ctrl = dst / "system" / "controlDict"
    txt = ctrl.read_text()
    txt = re.sub(r"adjustTimeStep\s+no;", "adjustTimeStep  yes;", txt)
    txt = re.sub(r"(maxCo\s+)[\d.eE+\-]+;", "\\g<1>%g;" % co, txt)
    if "maxDeltaT" not in txt:
        txt = re.sub(r"(maxCo\s+[\d.eE+\-]+;)",
                     "\\1\nmaxDeltaT       %g;" % MAX_DELTA_T, txt)
    else:
        txt = re.sub(r"(maxDeltaT\s+)[\d.eE+\-]+;",
                     "\\g<1>%g;" % MAX_DELTA_T, txt)
    # top-level writeControl -> adjustableRunTime (functions block untouched)
    pre, post = txt.split("functions", 1)
    pre = re.sub(r"writeControl\s+timeStep;",
                 "writeControl    adjustableRunTime;", pre, count=1)
    pre = re.sub(r"(writeInterval\s+)[\d.eE+\-]+;",
                 "\\g<1>%.10e;" % (WINDOW_STEPS * DT_NOMINAL), pre, count=1)
    # force sampling every CFD step: with large adapted dt, 5-step sampling
    # can leave <2 force samples per coupling window (interp1d would fail)
    post = re.sub(r"(writeInterval\s+)5\s*;", "\\g<1>1;", post)
    ctrl.write_text(pre + "functions" + post)

    # ---- driver: per-window writeInterval as duration [s], not step count ----
    drv = dst / "cosim_driver.py"
    txt = drv.read_text()
    anchor = "    if write_interval is not None:\n"
    assert anchor in txt, "driver write_interval anchor not found"
    txt = txt.replace(anchor, anchor +
        "        # adjustableRunTime: interval is the window duration in seconds\n"
        "        write_interval = \"%.10e\" % (end_time - start_time)\n", 1)
    old_rx = 'r"(writeInterval\\s+)\\d+\\s*;"'
    new_rx = 'r"(writeInterval\\s+)[\\d.eE+\\-]+\\s*;"'
    n = txt.count(old_rx)
    assert n == 2, "expected 2 writeInterval regexes in driver, found %d" % n
    txt = txt.replace(old_rx, new_rx)
    drv.write_text(txt)
    print("  built %s" % dst)
    return dst


def build_case_fixed():
    """Reference case: byte-identical to production (adjustTimeStep no,
    fixed dt=7e-5, stock driver) — only the gust CLI args differ from
    submit_gust.pbs by t_end."""
    dst = STUDY_DIR / FIXED_NAME
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC_CASE, dst, ignore=shutil.ignore_patterns(
        "processor*", "postProcessing*", "log.*", "dynamicCode",
        "*.csv", "validation_fsi", "figures", "cosim_state.json",
        "cosim_state_full.json"))
    for d in dst.iterdir():
        if d.is_dir() and re.match(r"^\d+\.?\d*$", d.name) and d.name != "0":
            shutil.rmtree(d)
    _restore_checkpoint(dst)
    print("  built %s (production config, no patches)" % dst)
    return dst


def run_driver(case_dir, t_end):
    cmd = ("export PATH=$HOME/bin_of7:$PATH && cd %s && "
           "%s cosim_driver.py --np %d --window %d --dt %g --t-end %g "
           "--from-checkpoint %s"
           % (case_dir, VENV_PY, N_PROCS, WINDOW_STEPS, DT_NOMINAL, t_end, GUST_ARGS))
    log = case_dir / "log.cosim_gust"
    with open(log, "a") as lf:
        r = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def n_time_dirs(case_dir):
    p0 = case_dir / "processor0"
    if not p0.exists():
        return 0
    n = 0
    for d in p0.iterdir():
        if d.is_dir() and re.match(r"^\d+\.?\d*(e-?\d+)?$", d.name) and d.name != "0":
            n += 1
    return n


def cmd_run(name):
    case_dir = STUDY_DIR / name
    # probe: 2 windows to verify adjustableRunTime window chaining before
    # committing to the full run
    print("[%s] probe (2 windows)..." % name, flush=True)
    rc = run_driver(case_dir, 2 * WINDOW_STEPS * DT_NOMINAL)
    nd = n_time_dirs(case_dir)
    if rc != 0 or nd < 3:   # checkpoint dir + 2 window ends
        print("[%s] PROBE FAILED rc=%d time_dirs=%d — window chaining broken, aborting"
              % (name, rc, nd))
        sys.exit(1)
    print("[%s] probe OK (%d time dirs) — full run t_end=%gs" % (name, nd, T_END),
          flush=True)
    rc = run_driver(case_dir, T_END)
    if rc != 0:
        print("[%s] FULL RUN FAILED rc=%d" % (name, rc))
        sys.exit(1)
    print("[%s] done" % name)


def parse_cl(case_dir, t_skip=7e-4):
    # one forceCoeffs.dat per window (pimpleFoam restart): the first samples of
    # each carry the restart pressure transient (CL spikes of O(10)) — drop the
    # first t_skip seconds of each window (10 nominal production steps), the
    # same cut the driver applies to the structural forcing
    times, cl = [], []
    for f in (case_dir / "postProcessing").glob("**/forceCoeffs.dat"):
        rows = []
        for line in f.read_text().splitlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 4:
                try:
                    rows.append((float(cols[0]), float(cols[3])))
                except ValueError:
                    pass
        if not rows:
            continue
        t0 = rows[0][0]
        for tt, cc in rows:
            if tt >= t0 + t_skip:
                times.append(tt)
                cl.append(cc)
    if not times:
        return None, None
    t = np.array(times)
    c = np.array(cl)
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    return t[keep], c[keep]


def cmd_report():
    lines = []
    lines.append("NACA 2312 — Courant sweep at fixed coupling window")
    lines.append("window = %d steps x %g s = %.1f ms (production value), "
                 "adjustTimeStep yes, maxDeltaT = %g s"
                 % (WINDOW_STEPS, DT_NOMINAL, WINDOW_STEPS * DT_NOMINAL * 1e3, MAX_DELTA_T))
    lines.append("gust: W0=%g m/s, t=[0.1,0.9] s (run-relative), t_end=%g s, "
                 "warm start from checkpoint" % (W0, T_END))
    lines.append("=" * 82)
    lines.append("%-8s %-12s %-12s %-10s %-10s %-10s %-11s %-10s"
                 % ("maxCo", "Co_max real", "dt_avg (s)", "CL_base",
                    "CL_peak", "dCL_amp", "t_peak (s)", "wall (s)"))
    lines.append("-" * 82)
    amps = {}
    rows = [("fixed", FIXED_NAME)] + [("%g" % co, case_name(co)) for co in MAXCO_LEVELS]
    for label, dirname in rows:
        case_dir = STUDY_DIR / dirname
        t, c = parse_cl(case_dir)
        log = case_dir / "log.pimpleFoam"
        co_max = -1.0
        dt_avg = -1.0
        wall = -1.0
        if log.exists():
            ltxt = log.read_text()
            vals = [float(v) for v in re.findall(
                r"Courant Number mean: [\d.eE+\-]+ max: ([\d.eE+\-]+)", ltxt)]
            if vals:
                # skip the first window's samples: it starts from the
                # checkpoint dt (7e-5, Co~8.7 scaled) before adjustment kicks in
                k = min(200, max(1, len(vals) // 20))
                co_max = max(vals[k:]) if len(vals) > k else max(vals)
            dts = re.findall(r"^deltaT = ([\d.eE+\-]+)", ltxt, re.M)
            if dts:
                dt_avg = float(np.mean([float(x) for x in dts]))
            # one pimpleFoam session per window, each restarting ClockTime:
            # total wall = sum over sessions of the session's last ClockTime
            wall = 0.0
            for chunk in ltxt.split("Starting time loop")[1:]:
                clk = re.findall(r"ClockTime = ([\d.]+) s", chunk)
                if clk:
                    wall += float(clk[-1])
        if t is None:
            lines.append("%-8s RUN MISSING OR NO forceCoeffs DATA" % label)
            continue
        t0 = t[0]
        t_rel = t - t0
        pre = t_rel < 0.1                       # before gust onset
        base = float(np.mean(c[pre])) if pre.sum() >= 3 else float(c[0])
        i_pk = int(np.argmax(np.abs(c - base)))
        amp = float(c[i_pk] - base)
        amps[label] = amp
        lines.append("%-8s %-12.3f %-12.3e %-10.4f %-10.4f %-+10.4f %-11.4f %-10.0f"
                     % (label, co_max, dt_avg, base, float(c[i_pk]), amp,
                        t_rel[i_pk], wall))
    lines.append("")
    lines.append("Relative change in |dCL_amp| between consecutive maxCo levels:")
    labels = ["%g" % co for co in MAXCO_LEVELS if ("%g" % co) in amps]
    for a, b in zip(labels, labels[1:]):
        rel = abs(abs(amps[b]) - abs(amps[a])) / abs(amps[a]) * 100
        lines.append("  maxCo %s -> %s: %.2f%%" % (a, b, rel))
    if "fixed" in amps:
        lines.append("")
        lines.append("Relative deviation of |dCL_amp| vs fixed-dt production reference:")
        for lab in labels:
            rel = abs(abs(amps[lab]) - abs(amps["fixed"])) / abs(amps["fixed"]) * 100
            lines.append("  maxCo %s vs fixed(7e-5): %.2f%%" % (lab, rel))
    out = STUDY_DIR / "courant_sweep_report.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nReport saved to %s" % out)


def cmd_setup():
    STUDY_DIR.mkdir(exist_ok=True)
    for co in MAXCO_LEVELS:
        build_case(co)
        pbs = _pbs_path(case_name(co))
        pbs.write_text("""#!/bin/bash
#PBS -N %s_%g
#PBS -q cpu
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=48:00:00
#PBS -j oe

cd %s
source %s/my_venv/bin/activate
python3 courant_sweep.py run %s%s 2>&1 | tee %s/log_%s.txt
""" % (JOB_PREFIX, co, WORKDIR, WORKDIR, case_name(co), _cli_opts(),
       STUDY_DIR, case_name(co)))
        print("  wrote %s" % pbs)


def cmd_setup_fixed():
    STUDY_DIR.mkdir(exist_ok=True)
    build_case_fixed()
    pbs = _pbs_path(FIXED_NAME)
    pbs.write_text("""#!/bin/bash
#PBS -N %s_fixed
#PBS -q cpu
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=48:00:00
#PBS -j oe

cd %s
source %s/my_venv/bin/activate
python3 courant_sweep.py run %s%s 2>&1 | tee %s/log_%s.txt
""" % (JOB_PREFIX, WORKDIR, WORKDIR, FIXED_NAME, _cli_opts(),
       STUDY_DIR, FIXED_NAME))
    print("  wrote %s" % pbs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--study-dir", default=DEFAULT_STUDY,
                        help="study directory under WORKDIR (default: %(default)s)")
    common.add_argument("--w0", type=float, default=DEFAULT_W0,
                        help="gust amplitude W0 [m/s] (default: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", parents=[common],
                   help="build adaptive case dirs + PBS scripts")
    sub.add_parser("setup-fixed", parents=[common],
                   help="build fixed-dt reference case + PBS script")
    p_run = sub.add_parser("run", parents=[common],
                           help="run one case (used inside PBS)")
    p_run.add_argument("name")
    sub.add_parser("report", parents=[common], help="write the sweep report")
    args = ap.parse_args()
    _configure(args.study_dir, args.w0)
    if args.cmd == "setup":
        cmd_setup()
    elif args.cmd == "setup-fixed":
        cmd_setup_fixed()
    elif args.cmd == "run":
        cmd_run(args.name)
    elif args.cmd == "report":
        cmd_report()


if __name__ == "__main__":
    main()
