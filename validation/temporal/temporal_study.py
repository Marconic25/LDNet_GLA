"""
temporal_study.py
Temporal discretization study for NACA 2312 pimpleFoam co-simulation.
Runs 4 timestep levels and writes a summary report.

Usage:
    python3 temporal_study.py [--workdir /path/to/NACA2312] [--np 16]
"""

import argparse
import subprocess
import shutil
import re
import json
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
WORKDIR       = Path("/work/u10677113/NACA2312")
CONTAINER     = "/work/u10677113/of7.sif"
PIMPLE_CASE   = WORKDIR / "cosim_main"
STUDY_DIR     = WORKDIR / "temporal_study"
END_TIME      = 1.0        # s — enough for 5 oscillation cycles
DT_PHYS       = 7e-5       # physical CFD dt — same as production (dataset_v5, submit_gust)
N_PROCS       = 16

# Coupling-window levels at the PRODUCTION physical dt (7e-5 s): DT2 is the
# exact production window (3.5 ms), bracketed by one coarser and two finer.
DT_LEVELS = {
    "DT1": 100,   # window=100  Δt_coupling=7e-3s
    "DT2": 50,    # window=50   Δt_coupling=3.5e-3s  (production)
    "DT3": 25,    # window=25   Δt_coupling=1.75e-3s
    "DT4": 10,    # window=10   Δt_coupling=7e-4s
}


# ── Helpers ────────────────────────────────────────────────────────────────
def run(cmd: str, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    full = (
        f"apptainer exec {CONTAINER} /bin/bash -c "
        f"'source /opt/openfoam7/etc/bashrc && cd {cwd} && {cmd}'"
    )
    with open(log, "w") as lf:
        r = subprocess.run(full, shell=True, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def run_python(cmd: str, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def set_controldict_value(case_dir: Path, key: str, value: str):
    ctrl = case_dir / "system" / "controlDict"
    txt  = ctrl.read_text()
    txt  = re.sub(rf"({key}\s+)[^\s;]+;", rf"\g<1>{value};", txt)
    ctrl.write_text(txt)


def parse_cl_history(case_dir: Path):
    """Return (time, CL) arrays from postProcessing forceCoeffs, concatenating
    all per-window files (one per pimpleFoam restart)."""
    times, cl = [], []
    for f in (case_dir / "postProcessing").glob("**/forceCoeffs.dat"):
        for line in f.read_text().splitlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 4:
                try:
                    times.append(float(cols[0]))
                    cl.append(float(cols[3]))   # col3 = Cl
                except ValueError:
                    pass
    if not times:
        return None, None
    t = np.array(times)
    c = np.array(cl)
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    return t[keep], c[keep]


def max_co_from_log(log: Path) -> float:
    """Parse maximum Courant number from pimpleFoam log."""
    vals = re.findall(r"Courant Number mean: [\d.eE+\-]+ max: ([\d.eE+\-]+)", log.read_text())
    return max(float(v) for v in vals) if vals else -1.0


def signal_amplitude_phase(t: np.ndarray, cl: np.ndarray):
    """Estimate CL amplitude and phase (deg) via FFT on last half of signal."""
    if t is None or len(t) < 20:
        return -1.0, -1.0
    # Use last 60% to avoid transient
    n  = int(0.6 * len(t))
    tc = t[-n:]
    yc = cl[-n:]
    dt = np.mean(np.diff(tc))
    N  = len(yc)
    fft_vals = np.fft.rfft(yc - yc.mean())
    freqs    = np.fft.rfftfreq(N, d=dt)
    idx      = np.argmax(np.abs(fft_vals[1:])) + 1   # skip DC
    amp      = 2 * np.abs(fft_vals[idx]) / N
    phase    = np.angle(fft_vals[idx], deg=True)
    return float(amp), float(phase)


def wall_clock_per_second(log: Path, sim_time: float) -> float:
    """Wall-clock time per simulated second."""
    matches = re.findall(r"ClockTime = ([\d.]+) s", log.read_text())
    if matches:
        total_wall = float(matches[-1])
        return total_wall / sim_time if sim_time > 0 else -1.0
    return -1.0


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(WORKDIR))
    parser.add_argument("--np",      type=int, default=N_PROCS)
    parser.add_argument("--levels",  default=None,
                        help="Comma-separated subset of DT levels to run (e.g. DT1,DT2)")
    args = parser.parse_args()

    global DT_LEVELS
    if args.levels:
        keep = [s.strip() for s in args.levels.split(",") if s.strip()]
        DT_LEVELS = {k: DT_LEVELS[k] for k in keep}

    workdir   = Path(args.workdir)
    study_dir = workdir / "temporal_study"
    study_dir.mkdir(exist_ok=True)

    results = {}

    for name, dt in DT_LEVELS.items():
        case_dir = study_dir / name
        print(f"\n{'='*60}")
        print(f"  {name}  (dt = {dt:.1e} s)")
        print(f"{'='*60}")

        # ── Check if already done ──────────────────────────────────────
        result_file = study_dir / f"{name}_result.json"
        if result_file.exists():
            with open(result_file) as f:
                results[name] = json.load(f)
            print(f"  Already completed, loading results...")
            continue

        # ── Copy pimpleFoam case ───────────────────────────────────────
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(PIMPLE_CASE, case_dir,
                        ignore=shutil.ignore_patterns(
                            "processor*", "cosim_state.json",
                            "log.*", "[0-9]*.[0-9]*", "postProcessing"
                        ))

        # Keep only 0/ and 0.orig/ — remove all numeric timesteps and processor dirs
        for d in case_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name in ("0", "0.orig", "constant", "system", "__pycache__"):
                continue
            shutil.rmtree(d)
        # Reset 0/ from 0.orig/
        if (case_dir / "0").exists():
            shutil.rmtree(case_dir / "0")
        shutil.copytree(case_dir / "0.orig", case_dir / "0")

        # ── Patch controlDict ──────────────────────────────────────────
        set_controldict_value(case_dir, "deltaT",    f"{dt:.2e}")
        set_controldict_value(case_dir, "startFrom", "startTime")
        set_controldict_value(case_dir, "startTime", "0")
        # Restore correct deltaT and disable adjustTimeStep for fixed dt
        set_controldict_value(case_dir, "deltaT", f"{DT_PHYS:g}")
        ctrl = case_dir / "system" / "controlDict"
        txt = ctrl.read_text()
        txt = re.sub(r"adjustTimeStep\s+yes;", "adjustTimeStep  no;", txt)
        # writeControl is timeStep: writeInterval must be an integer step
        # count (a float duration is a FOAM fatal at the first OF call);
        # the driver rewrites it per window anyway
        txt = re.sub(r"(writeInterval\s+)[^;\n]+;",
                     f"\\g<1>{dt};", txt, count=1)
        ctrl.write_text(txt)

        # ── Remove copied postProcessing ──────────────────────────────
        pp = case_dir / "postProcessing"
        if pp.exists():
            shutil.rmtree(pp)

        # ── Patch flap schedule to start immediately ──────────────────
        cosim_src = case_dir / "cosim_driver.py"
        if cosim_src.exists():
            txt = cosim_src.read_text()
            txt = txt.replace(
                "DELTA_TIMES  = [0.0,  0.8,  1.0,  2.0]",
                "DELTA_TIMES  = [0.0,  0.0,  0.2,  2.0]"
            )
            txt = txt.replace(
                "DELTA_ANGLES = [0.0,  0.0,  15.0, 15.0]",
                "DELTA_ANGLES = [0.0,  0.0,  15.0, 15.0]"
            )
            cosim_src.write_text(txt)

        # ── Reset motion tables to t=0 ────────────────────────────────
        motion_template = """// tabulated6DoFMotion — generated by cosim_driver.py
(
  (0.0000000000e+00  ((0.0000000000e+00 0.0000000000e+00 0) (0 0 0.0000000000e+00)))
  (9.9990000000e+03  ((0.0000000000e+00 0.0000000000e+00 0) (0 0 0.0000000000e+00)))
)
"""
        for dat in ["wingMotion.dat", "flapMotion.dat"]:
            dat_path = case_dir / "constant" / dat
            if dat_path.exists():
                dat_path.write_text(motion_template)

        # ── Run cosim_driver ───────────────────────────────────────────
        print(f"  cosim_driver (window={dt})...")
        venv_python = str(Path("/work/u10677113/NACA2312/my_venv/bin/python3"))
        cosim_log = case_dir / "log.cosim_driver"
        cosim_cmd = (
            f"export PATH=$HOME/bin_of7:$PATH && "
            f"cd {case_dir} && "
            f"{venv_python} cosim_driver.py --np {args.np} --window {dt} --dt {DT_PHYS:g} --t-end {END_TIME} "
            f"--delta-times 0.0 0.0 0.2 2.0 --delta-angles 0.0 0.0 15.0 15.0"
        )
        rc = run_python(cosim_cmd, case_dir, cosim_log)
        if rc != 0:
            print(f"  [WARN] cosim_driver rc={rc}")
        pimple_log = case_dir / "log.pimpleFoam"


        # ── Extract results ────────────────────────────────────────────
        t, cl = parse_cl_history(case_dir)
        amp, phase = signal_amplitude_phase(t, cl)
        co_max     = max_co_from_log(pimple_log) if pimple_log.exists() else -1.0
        wc_per_s   = wall_clock_per_second(pimple_log, END_TIME) if pimple_log.exists() else -1.0

        res = {
            "dt":          dt,
            "co_max":      co_max,
            "cl_amp":      amp,
            "cl_phase":    phase,
            "wc_per_s":    wc_per_s,
            "t":           t.tolist()  if t  is not None else [],
            "cl":          cl.tolist() if cl is not None else [],
        }
        results[name] = res
        with open(result_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Co_max={co_max:.2f}  CL_amp={amp:.4f}  phase={phase:.1f}°  wc/s={wc_per_s:.1f}s")
        # Remove case directory to free disk space
       # print(f"  Removing {case_dir} to free disk...")
        #shutil.rmtree(case_dir)

    # ── Write report ───────────────────────────────────────────────────────
    # Primary metrics (the flap-step response is quasi-static: the FFT
    # amplitude of the residual oscillation is noise-level and unusable as a
    # convergence indicator):
    #   CL_steady       = mean CL on t in [0.6, 1.0]  (post-step steady value)
    #   NRMSE vs finest = rms(CL - CL_finest)/|mean CL_finest| on t in [0.25, 1.0]
    finest = list(DT_LEVELS.keys())[-1]
    t_ref  = np.array(results.get(finest, {}).get("t",  []))
    cl_ref = np.array(results.get(finest, {}).get("cl", []))

    def cl_steady(r):
        t = np.array(r.get("t", []))
        c = np.array(r.get("cl", []))
        m = t >= 0.6
        return float(c[m].mean()) if m.sum() > 5 else float("nan")

    def nrmse_vs_finest(r):
        t = np.array(r.get("t", []))
        c = np.array(r.get("cl", []))
        if len(t_ref) < 10 or len(t) < 10:
            return float("nan")
        m = t_ref >= 0.25
        ci = np.interp(t_ref[m], t, c)
        return float(np.sqrt(np.mean((ci - cl_ref[m])**2))
                     / max(abs(cl_ref[m].mean()), 1e-12) * 100)

    report_path = study_dir / "temporal_study_report.txt"
    with open(report_path, "w") as f:
        f.write("NACA 2312 — Temporal (coupling window) Discretization Study\n")
        f.write(f"physical dt = {DT_PHYS:g} s (production), flap step 0->15deg on [0,0.2]s,\n")
        f.write(f"endTime = {END_TIME} s,  mesh = M3 (wingMotion2D_pimpleFoam)\n")
        f.write("=" * 86 + "\n\n")

        header = (f"{'Level':<6} {'window':<8} {'dt_coup (s)':<12} {'Co_max':<8} "
                  f"{'CL_steady':<11} {'NRMSE% vs ' + finest:<14} {'CL_amp':<9} {'wc/s (s)':<8}\n")
        f.write(header)
        f.write("-" * 86 + "\n")
        for name in DT_LEVELS:
            r = results.get(name, {})
            win = r.get('dt', 0)
            f.write(
                f"{name:<6} "
                f"{win:<8} "
                f"{win * DT_PHYS:<12.2e} "
                f"{r.get('co_max', -1):<8.2f} "
                f"{cl_steady(r):<11.5f} "
                f"{nrmse_vs_finest(r):<14.3f} "
                f"{r.get('cl_amp', -1):<9.4f} "
                f"{r.get('wc_per_s', -1):<8.1f}\n"
            )
        f.write("\n")

        names = list(DT_LEVELS.keys())
        f.write("Convergence check (relative change vs next finer level):\n")
        for i in range(1, len(names)):
            s1 = cl_steady(results.get(names[i-1], {}))
            s2 = cl_steady(results.get(names[i],   {}))
            if s1 == s1 and s2 == s2 and abs(s2) > 0:
                rel = abs(s2 - s1) / abs(s2) * 100
                f.write(f"  CL_steady {names[i-1]} -> {names[i]}: {rel:.3f}%\n")
        f.write("\n")

        # Selected: largest coupling window whose full CL(t) stays within 1%
        # NRMSE of the finest level
        selected = None
        for name in names:
            e = nrmse_vs_finest(results.get(name, {}))
            if e == e and e < 1.0:
                selected = name
                break
        if selected is None:
            selected = names[-1]
        f.write(f"Selected coupling window: {selected}  "
                f"(window={DT_LEVELS[selected]}, dt_coup={DT_LEVELS[selected]*DT_PHYS:.2e} s)\n")
        f.write(f"Criterion: largest coupling window with NRMSE < 1% vs {finest} on t in [0.25,1.0] s.\n")

    print(f"\nReport written: {report_path}")

    # ── Save all CL time histories to CSV ─────────────────────────────────
    csv_path = study_dir / "cl_histories.csv"
    max_len = max(len(results[n].get("t", [])) for n in results)
    with open(csv_path, "w") as f:
        header_cols = []
        for name in DT_LEVELS:
            header_cols += [f"t_{name}", f"CL_{name}"]
        f.write(",".join(header_cols) + "\n")
        for i in range(max_len):
            row = []
            for name in DT_LEVELS:
                t_arr  = results[name].get("t",  [])
                cl_arr = results[name].get("cl", [])
                row.append(str(t_arr[i])  if i < len(t_arr)  else "")
                row.append(str(cl_arr[i]) if i < len(cl_arr) else "")
            f.write(",".join(row) + "\n")
    print(f"CL histories CSV: {csv_path}")


if __name__ == "__main__":
    main()
