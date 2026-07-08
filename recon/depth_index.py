#!/usr/bin/env python3
"""Rebuild recon/models/depth_study/index.csv from run artifacts (idempotent).

One row per run dir L{total}_ds{K}. Reads latent_{K}/{config,metrics,run_info}.json
and status.json; never modifies run artifacts. Safe to run at any time.
"""
import csv, json, math, re
from pathlib import Path

BASE = Path(__file__).resolve().parent
STUDY = BASE / "models" / "depth_study"

COLS = ["run_id", "total_layers", "dyn_layers", "rec_layers", "dyn_width", "rec_width",
        "d_s", "n_params_dyn", "n_params_rec", "n_params_total",
        "restarts", "adam", "bfgs",
        "train_loss", "val_loss",
        "NRMSE_vx", "NRMSE_vy", "NRMSE_p", "NRMSE",
        "adam_wall_s", "bfgs_wall_s", "wall_seconds",
        "bfgs_nit", "bfgs_nfev", "bfgs_message", "bfgs_grad_inf_norm",
        "status", "stop_reason", "model_dir"]


def read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.6e}" if (abs(x) < 1e-3 or abs(x) >= 1e4) and x != 0 else f"{x:.6g}"
    return x


def main():
    rows = []
    stop = read_json(STUDY / "STOP") or {}
    for rd in sorted(STUDY.glob("L*_ds*")):
        m = re.match(r"L(\d+)_ds(\d+)$", rd.name)
        if not m:
            continue
        total, ds = int(m.group(1)), int(m.group(2))
        ri = read_json(rd / f"latent_{ds}" / "run_info.json") or {}
        mt = read_json(rd / f"latent_{ds}" / "metrics.json") or {}
        st = read_json(rd / "status.json") or {}
        np_ = ri.get("n_params") or {}
        npred = st.get("n_params_pred") or {}
        bf = ri.get("bfgs_result") or {}
        pw = ri.get("phase_wall_s") or {}
        status = st.get("status") or ("done" if mt else "todo")
        stop_reason = st.get("note", "")
        if stop and status != "done":
            stop_reason = stop_reason or f"{stop.get('reason', '')}"
        dyn_l = ri.get("dyn_layers", total // 3)
        rec_l = ri.get("rec_layers", total - total // 3)
        rows.append({
            "run_id": rd.name, "total_layers": total,
            "dyn_layers": dyn_l, "rec_layers": rec_l,
            "dyn_width": ri.get("dyn_width", 7), "rec_width": ri.get("rec_width", 24),
            "d_s": ds,
            "n_params_dyn": np_.get("NNdyn", npred.get("dyn", "")),
            "n_params_rec": np_.get("NNrec", npred.get("rec", "")),
            "n_params_total": np_.get("total", npred.get("total", "")),
            "restarts": ri.get("restarts", ""), "adam": ri.get("adam", ""),
            "bfgs": ri.get("bfgs", ""),
            "train_loss": fmt(ri.get("final_train_loss")),
            "val_loss": fmt(ri.get("final_valid_loss")),
            "NRMSE_vx": fmt(mt.get("NRMSE_vx")), "NRMSE_vy": fmt(mt.get("NRMSE_vy")),
            "NRMSE_p": fmt(mt.get("NRMSE_p")), "NRMSE": fmt(mt.get("NRMSE")),
            "adam_wall_s": fmt(sum(v for k, v in pw.items() if k.startswith("adam"))
                               if pw else None),
            "bfgs_wall_s": fmt(pw.get("bfgs")),
            "wall_seconds": fmt(ri.get("total_wall_s", st.get("wall_s"))),
            "bfgs_nit": bf.get("nit", ""), "bfgs_nfev": bf.get("nfev", ""),
            "bfgs_message": bf.get("message", ""),
            "bfgs_grad_inf_norm": fmt(bf.get("grad_inf_norm")),
            "status": status, "stop_reason": stop_reason,
            "model_dir": rd.as_posix(),
        })
    rows.sort(key=lambda r: (r["total_layers"], r["d_s"]))
    STUDY.mkdir(parents=True, exist_ok=True)
    with open(STUDY / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"index.csv: {len(rows)} rows -> {STUDY / 'index.csv'}")


if __name__ == "__main__":
    main()
