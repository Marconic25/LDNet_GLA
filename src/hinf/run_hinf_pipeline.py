"""
run_hinf_pipeline.py
====================
Orchestratore end-to-end per la pipeline H∞ GLA.

Esegue in sequenza:
  A. Stima dei pesi di incertezza W_ΔL, W_ΔM dai residui LDNet/FSI
  B. Sintesi H∞ mixed-sensitivity sulla pianta generalizzata
  C. Validazione closed-loop e confronto LQR vs H∞

Tutti i parametri di peso e di sintesi stanno in control/hinf_synthesis.py
(blocco costanti in cima al file).  I parametri di simulazione (U_INF,
gust nominale) sono nel blocco costanti in fondo a questo file.

Uso:
    cd src && python run_hinf_pipeline.py

I checkpoint (figure diagnostiche) vengono salvati in src/ con nomi
hinf_fig*.png; i pesi su hinf_weights.npz e il controllore su
hinf_controller.npz.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# ── Parametri pipeline ────────────────────────────────────────────────────────
CSV_DIR        = str(Path(__file__).parent.parent / "data" / "GLA_data" / "timeseries")
MODELS_DIR     = str(Path(__file__).parent.parent / "models")
WEIGHTS_FILE   = "hinf_weights.npz"
CTRL_FILE      = "hinf_controller.npz"

U_INF_SIM      = 75.0    # m/s — velocità simulazione nominale
W0_NOM         = 60.0    # m/s — ampiezza gust nominale
TG_NOM         = 1.0     # s   — durata gust nominale
T_END          = 3.0     # s   — durata simulazione
DT             = 0.01    # s   — passo controllore

W0_SWEEP       = [30.0, 60.0, 90.0, 120.0]   # m/s — sweep ampiezza
TG_SWEEP       = [0.3, 0.5, 1.0, 2.0]         # s   — sweep durata

# Parametri LQR (copiati da test_mpc.py per coerenza)
Q_H  = 1e4; Q_A = 1e2; Q_W = 0.0; R_LQR = 1e-2
Q_Y  = np.diag([10.0, 1.0])


# ─────────────────────────────────────────────────────────────────────────────

def _load_aero():
    from aerodynamics.model import LDNetModel
    return LDNetModel(MODELS_DIR)


def _load_lqr(aero):
    from control.lqr import LQRController
    import numpy as np
    z_trim = np.zeros(aero.num_latent_states)
    for _ in range(200):
        z_trim, CL_trim, CM_trim = aero.step(z_trim, 0., 0., 0., 0., 0., 0., U_INF_SIM, DT)
    x_trim = np.zeros(4)
    Q_lqr = np.diag([Q_H, 0.0, Q_A, 0.0, 0.0])
    R_lqr = np.array([[R_LQR]])
    lqr = LQRController(
        aero, U_INF_SIM, DT, x_trim, z_trim,
        Q_lqr, R_lqr,
        CL_trim=float(CL_trim), CM_trim=float(CM_trim),
        Q_y=Q_Y, Q_w=Q_W,
    )
    return lqr


def _sep(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ── Parte A ───────────────────────────────────────────────────────────────────

def run_part_A(aero, skip_if_exists: bool = False) -> None:
    from control.uncertainty_weights import (
        compute_uncertainty_envelope,
        fit_uncertainty_weights,
        plot_uncertainty_envelope,
        load_uncertainty_weights,
    )

    _sep("PARTE A — Pesi di incertezza W_ΔL, W_ΔM")

    if skip_if_exists and Path(WEIGHTS_FILE).exists():
        print(f"  {WEIGHTS_FILE} già presente — skip rollout.")
        W_dL, W_dM, omega_hz, env_L, env_M = load_uncertainty_weights(WEIGHTS_FILE)
    else:
        print("A1. Rollout LDNet su test set e stima envelope...")
        omega_hz, env_L, env_M = compute_uncertainty_envelope(
            CSV_DIR, aero, eps=1e-3)

        print("\nA2. Fit W_ΔL, W_ΔM...")
        W_dL, W_dM = fit_uncertainty_weights(
            omega_hz, env_L, env_M, save_path=WEIGHTS_FILE)

    print("\nCHECKPOINT A — Figura envelope + fit...")
    plot_uncertainty_envelope(
        omega_hz, env_L, env_M, W_dL, W_dM,
        save_path="hinf_fig0_weights.png",
    )
    print("  >> Verificare hinf_fig0_weights.png: fit deve stare SOPRA l'envelope.")


# ── Parte B ───────────────────────────────────────────────────────────────────

def run_part_B(skip_if_exists: bool = False) -> None:
    from control.hinf_synthesis import main as synth_main

    _sep("PARTE B — Sintesi H∞")

    if skip_if_exists and Path(CTRL_FILE).exists():
        print(f"  {CTRL_FILE} già presente — skip sintesi.")
        return

    synth_main(
        weights_path=WEIGHTS_FILE,
        ctrl_path=CTRL_FILE,
        sv_fig_path="hinf_fig1_sv.png",
    )
    print("  >> Verificare hinf_fig1_sv.png: S < 1/W_p, KS < 1/W_u sulla banda.")


# ── Parte C ───────────────────────────────────────────────────────────────────

def run_part_C(aero, lqr) -> None:
    from control.hinf_synthesis import load_controller
    from control.hinf_simulation import (
        HInfController,
        run_comparison,
        sweep_gust_amplitude,
        sweep_gust_duration,
        plot_comparison_timeseries,
        plot_sweep_summary,
        print_summary_table,
    )

    _sep("PARTE C — Validazione e confronto")

    print("C1. Carica controllore ridotto...")
    K_A, K_B, K_C, K_D, dt_ctrl, gamma = load_controller(CTRL_FILE)
    print(f"  γ = {gamma:.4f}, ordine = {K_A.shape[0]}, DT = {dt_ctrl} s")
    hinf = HInfController(K_A, K_B, K_C, K_D)

    print("\nC2. Confronto nominale (W₀=60, T_g=1 s)...")
    results_nom = run_comparison(
        aero, lqr, hinf,
        U_INF=U_INF_SIM, W0=W0_NOM, T_g=TG_NOM,
        T_END=T_END, DT=DT,
    )
    plot_comparison_timeseries(results_nom, save_prefix="hinf_fig",
                                W0=W0_NOM, T_g=TG_NOM)
    print("  >> Figure hinf_fig2..5: confronto OL/LQR/H∞.")

    print("\nC3. Sweep ampiezza gust...")
    sweep_amp = sweep_gust_amplitude(aero, lqr, hinf, W0_list=W0_SWEEP)

    print("\nC4. Sweep durata gust...")
    sweep_dur = sweep_gust_duration(aero, lqr, hinf, T_g_list=TG_SWEEP)

    plot_sweep_summary(sweep_amp, sweep_dur, save_path="hinf_fig6_sweep.png")
    print_summary_table(sweep_amp, sweep_dur)
    print("  >> Figura hinf_fig6_sweep.png: H∞ deve degradare meno di LQR.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline H∞ GLA")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Salta le parti già calcolate (weights/controller)")
    parser.add_argument("--part", choices=["A", "B", "C", "all"], default="all",
                        help="Esegui solo la parte specificata")
    args = parser.parse_args()

    aero = _load_aero()

    if args.part in ("A", "all"):
        run_part_A(aero, skip_if_exists=args.skip_existing)

    if args.part in ("B", "all"):
        run_part_B(skip_if_exists=args.skip_existing)

    if args.part in ("C", "all"):
        lqr = _load_lqr(aero)
        run_part_C(aero, lqr)

    print("\n" + "="*60)
    print("Pipeline completata.")
    print("Figure prodotte: hinf_fig0_weights.png  (Checkpoint A)")
    print("                 hinf_fig1_sv.png        (Checkpoint B)")
    print("                 hinf_fig2..5_*.png      (Checkpoint C nominale)")
    print("                 hinf_fig6_sweep.png     (Checkpoint D sweep)")
    print("="*60)