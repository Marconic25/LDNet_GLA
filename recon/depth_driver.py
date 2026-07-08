#!/usr/bin/env python3
"""H-ARCH depth-ladder driver (Thread D).

Runs train_fields.py over a geometric depth ladder (total hidden layers
6,12,24,...,98304; NNdyn:NNrec layer ratio fixed at 1:2, widths fixed 7/24) for
d_s=1 and d_s=10, sequentially, resumable across PBS jobs.

Stop rules (recorded in models/depth_study/STOP):
 (a) both d_s runs >=2x worse final train loss than the L=6 baseline (or NaN)
     for two consecutive rungs  -> untrainable
 (b) run crashes (rc!=0); retried once; second crash -> the wall (OOM / TF graph)
 (c) a single run exceeds ~10 h wall -> walltime wall
 (d) NaN/divergence: seeds are fixed and training is deterministic, so a blind
     retry would reproduce the same NaN; NaN counts as a "bad" run under (a)
     instead of burning hours on a futile retry (justification in notes).
Cap: ladder ends at 98304 total layers (< 100000 hard cap).

Usage (inside the TF container, cwd = recon/):
  python3 -u depth_driver.py --budget-s 84000
"""
import argparse, json, math, subprocess, sys, time
from pathlib import Path

LADDER = [6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304]
DS_LIST = [1, 10]
DYN_WIDTH, REC_WIDTH = 7, 24
RESTARTS, ADAM, BFGS = 2, 300, 4000
RUN_WALL_CAP_S = 11 * 3600      # kill a single run beyond this (stop rule c, ~10h + slack)
RUN_WALL_STOP_S = 10 * 3600     # completed run slower than this still stops escalation

BASE = Path(__file__).resolve().parent
STUDY = BASE / "models" / "depth_study"

TRAIN_CMD = ("python3 -u train_fields.py "
             "--train data/FIELDS_div_train.h5 --valid data/FIELDS_div_valid.h5 "
             "--test data/FIELDS_Cc060.h5 "
             "--out {out} --latents {ds} "
             "--dyn-layers {dyn} --rec-layers {rec} "
             "--dyn-width {dw} --rec-width {rw} "
             "--output-nl linear --restarts {restarts} --adam {adam} --bfgs {bfgs} "
             "--log-every 1")


def split_layers(total):
    dyn = total // 3            # 1:2 dyn:rec ratio; ladder values divisible by 3
    return dyn, total - dyn


def n_params(total, ds):
    dyn_l, rec_l = split_layers(total)
    n_inp = ds + 7              # ds + 1 param + 6 signals
    n_rec_in = ds + 8           # ds + 6 signals + 2 space
    p_dyn = (n_inp * DYN_WIDTH + DYN_WIDTH) \
        + (dyn_l - 1) * (DYN_WIDTH * DYN_WIDTH + DYN_WIDTH) \
        + (DYN_WIDTH * ds + ds)
    p_rec = (n_rec_in * REC_WIDTH + REC_WIDTH) \
        + (rec_l - 1) * (REC_WIDTH * REC_WIDTH + REC_WIDTH) \
        + (REC_WIDTH * 3 + 3)
    return p_dyn, p_rec


def run_dir(total, ds):
    return STUDY / f"L{total}_ds{ds}"


def read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def write_json(p, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def run_status(total, ds):
    """Returns (state, info): state in {todo, done, bad_loss, failed}."""
    rd = run_dir(total, ds)
    st = read_json(rd / "status.json")
    if st and st.get("status") in ("failed", "killed_walltime", "nan_diverged"):
        return st["status"], st
    ri = read_json(rd / f"latent_{ds}" / "run_info.json")
    mt = read_json(rd / f"latent_{ds}" / "metrics.json")
    if ri is not None and mt is not None:
        tl = ri.get("final_train_loss")
        if tl is None or (isinstance(tl, float) and math.isnan(tl)):
            return "nan_diverged", {"train_loss": tl}
        return "done", {"train_loss": tl, "run_info": ri}
    return "todo", st or {}


def is_bad(total, ds, base_tr):
    """Bad under stop rule (a): >=2x baseline train loss, NaN, or hard failure."""
    state, info = run_status(total, ds)
    if state in ("failed", "killed_walltime", "nan_diverged"):
        return True
    if state == "done" and base_tr.get(ds):
        return info["train_loss"] >= 2.0 * base_tr[ds]
    return False


def rebuild_index():
    try:
        subprocess.run([sys.executable, str(BASE / "depth_index.py")], check=False,
                       timeout=300)
    except Exception as e:
        print(f"[driver] index rebuild failed: {e}")


def launch(total, ds, timeout_s):
    dyn_l, rec_l = split_layers(total)
    rd = run_dir(total, ds)
    rd.mkdir(parents=True, exist_ok=True)
    attempts_f = rd / "attempts.txt"
    attempts = int(attempts_f.read_text()) if attempts_f.exists() else 0
    attempts += 1
    attempts_f.write_text(str(attempts))

    p_dyn, p_rec = n_params(total, ds)
    P = p_dyn + p_rec
    hess_gb = P * P * 8 / 1e9
    print(f"[driver] L={total} ds={ds} (dyn {dyn_l}x{DYN_WIDTH}, rec {rec_l}x{REC_WIDTH}) "
          f"P={P} -> BFGS dense Hessian ~{hess_gb:.2f} GB (peak w/ temporaries ~{4*hess_gb:.1f} GB) "
          f"attempt {attempts}, timeout {timeout_s/3600:.1f} h", flush=True)

    cmd = TRAIN_CMD.format(out=rd.as_posix(), ds=ds, dyn=dyn_l, rec=rec_l,
                           dw=DYN_WIDTH, rw=REC_WIDTH,
                           restarts=RESTARTS, adam=ADAM, bfgs=BFGS)
    t0 = time.time()
    status, rc = "done", None
    with open(rd / "train.log", "a") as log:
        log.write(f"\n===== attempt {attempts} @ {time.strftime('%F %T')} =====\n{cmd}\n")
        log.flush()
        try:
            r = subprocess.run(cmd.split(), cwd=str(BASE), stdout=log,
                               stderr=subprocess.STDOUT, timeout=timeout_s)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            status, rc = "killed_walltime", -9
    wall = time.time() - t0

    if status != "killed_walltime":
        state, _ = run_status(total, ds)
        if state == "nan_diverged":
            status = "nan_diverged"
        elif rc != 0 or state == "todo":
            status = "crashed"
        else:
            status = "done"
    write_json(rd / "status.json",
               {"status": status, "rc": rc, "wall_s": wall, "attempts": attempts,
                "n_params_pred": {"dyn": p_dyn, "rec": p_rec, "total": P},
                "pred_bfgs_hessian_gb": hess_gb,
                "finished": time.strftime("%F %T")})
    print(f"[driver] L={total} ds={ds} -> {status} rc={rc} wall={wall/3600:.2f} h", flush=True)
    return status, wall, attempts


def stop_ladder(reason, detail):
    write_json(STUDY / "STOP", {"reason": reason, "detail": detail,
                                "time": time.strftime("%F %T")})
    print(f"[driver] LADDER STOPPED: {reason} — {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=84000.0,
                    help="wall budget for this driver invocation (leave PBS slack)")
    args = ap.parse_args()
    t_start = time.time()
    STUDY.mkdir(parents=True, exist_ok=True)

    def remaining():
        return args.budget_s - (time.time() - t_start)

    if (STUDY / "STOP").exists():
        print(f"[driver] STOP file present: {(STUDY / 'STOP').read_text()}", flush=True)
        return

    last_wall = {1: None, 10: None}   # per-ds wall of the most recent completed run

    for i, total in enumerate(LADDER):
        if (STUDY / "STOP").exists():
            return
        for ds in DS_LIST:
            state, info = run_status(total, ds)
            if state == "done":
                ri = info.get("run_info") or {}
                if ri.get("total_wall_s"):
                    last_wall[ds] = ri["total_wall_s"]
                continue
            if state in ("failed", "killed_walltime", "nan_diverged"):
                continue    # permanently settled in a previous invocation
            # budget check: estimate ~2.2x the previous rung's wall for this ds
            est = 4 * 3600 if last_wall[ds] is None else min(2.2 * last_wall[ds],
                                                             RUN_WALL_CAP_S)
            if remaining() < est + 600:
                print(f"[driver] budget exhausted before L={total} ds={ds} "
                      f"(need ~{est/3600:.1f} h, have {remaining()/3600:.1f} h) — "
                      "exiting for chained resubmit", flush=True)
                rebuild_index()
                return
            timeout_s = min(RUN_WALL_CAP_S, remaining() - 300)
            status, wall, attempts = launch(total, ds, timeout_s)
            rebuild_index()
            if status == "done":
                last_wall[ds] = wall
                if wall > RUN_WALL_STOP_S:
                    stop_ladder("rung_walltime",
                                f"L={total} ds={ds} completed but took {wall/3600:.2f} h "
                                f"(> {RUN_WALL_STOP_S/3600:.0f} h cap) — stop rule (c)")
            elif status == "killed_walltime":
                rd = run_dir(total, ds)
                write_json(rd / "status.json",
                           {**(read_json(rd / 'status.json') or {}), "status": "killed_walltime"})
                stop_ladder("rung_walltime",
                            f"L={total} ds={ds} killed at {wall/3600:.2f} h — stop rule (c)")
            elif status == "crashed":
                if attempts >= 2:
                    rd = run_dir(total, ds)
                    write_json(rd / "status.json",
                               {**(read_json(rd / 'status.json') or {}), "status": "failed"})
                    stop_ladder("crash_wall",
                                f"L={total} ds={ds} crashed twice (rc != 0) — stop rule (b): "
                                "optimizer memory / TF graph wall")
                else:
                    print(f"[driver] L={total} ds={ds} crashed, retrying once...", flush=True)
                    if remaining() > 900:
                        status, wall, attempts = launch(total, ds,
                                                        min(RUN_WALL_CAP_S, remaining() - 300))
                        rebuild_index()
                        if status == "crashed" and attempts >= 2:
                            rd = run_dir(total, ds)
                            write_json(rd / "status.json",
                                       {**(read_json(rd / 'status.json') or {}),
                                        "status": "failed"})
                            stop_ladder("crash_wall",
                                        f"L={total} ds={ds} crashed twice — stop rule (b)")
                        elif status == "done":
                            last_wall[ds] = wall
            elif status == "nan_diverged":
                print(f"[driver] L={total} ds={ds} produced NaN (deterministic seeds — "
                      "counts as bad under stop rule (a), no blind retry)", flush=True)
            if (STUDY / "STOP").exists():
                return

        # stop rule (a): both ds bad at this rung AND at the previous rung
        base_tr = {}
        for ds in DS_LIST:
            st, inf = run_status(LADDER[0], ds)
            if st == "done":
                base_tr[ds] = inf["train_loss"]
        if i >= 1 and base_tr:
            bad_here = all(is_bad(total, ds, base_tr) for ds in DS_LIST)
            bad_prev = all(is_bad(LADDER[i - 1], ds, base_tr) for ds in DS_LIST)
            if bad_here and bad_prev:
                stop_ladder("untrainable",
                            f"rungs L={LADDER[i-1]} and L={total}: both d_s runs >=2x "
                            f"baseline train loss (baseline: {base_tr}) — stop rule (a)")
                return

    stop_ladder("ladder_cap_reached", "completed final rung L=98304 (< 100000 cap)")


if __name__ == "__main__":
    main()
