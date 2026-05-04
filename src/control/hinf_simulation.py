"""
hinf_simulation.py
==================
Implementa il controllore H∞ nel loop closed-loop e confronta con LQR.

Struttura:
  - HInfController: filtro stato-spazio discreto, interfaccia analoga a
    LQRController.solve()
  - run_hinf_simulation: loop closed-loop, pattern identico a run_mpc_simulation
  - run_comparison: esegue OL / LQR / H∞ sullo stesso gust
  - sweep_gust_amplitude / sweep_gust_duration: sweep parametrici
  - plot_comparison_timeseries: 4 pannelli identici allo stile LQR del cap. 3
  - plot_sweep_summary: picco ḧ vs W₀ e T_g
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from structural.smd import (
    structural_rhs, M_WING, M_FLAP, I_WING, I_FLAP_EA,
    D_H, D_ALPHA, K_H, K_ALPHA, _D_X,
)


# ── Palette colori coerente con il capitolo 3 ─────────────────────────────────
_C_OL   = dict(color='steelblue',    lw=1.5, alpha=0.85, label='Open loop')
_C_LQR  = dict(color='darkcyan',     lw=1.5, alpha=0.85, label='LQR')
_C_HINF = dict(color='mediumpurple', lw=1.5, alpha=0.85, label='H∞')


# ── Classe controllore ────────────────────────────────────────────────────────

class HInfController:
    """Controllore H∞ come filtro stato-spazio discreto.

    Interfaccia analoga a LQRController: un metodo solve() che riceve le
    misure correnti e restituisce la deflessione del flap δ.

    Parameters
    ----------
    K_A : (n,n) ndarray — matrice di stato del controllore
    K_B : (n,2) ndarray — matrice di ingresso  [ḧ_meas, α_meas]
    K_C : (1,n) ndarray — matrice di uscita    δ
    K_D : (1,2) ndarray — feedthrough
    delta_max : float   — saturazione flap [°]
    """

    def __init__(self,
                 K_A: np.ndarray,
                 K_B: np.ndarray,
                 K_C: np.ndarray,
                 K_D: np.ndarray,
                 delta_max: float = 20.0):
        self.K_A = np.atleast_2d(K_A)
        self.K_B = np.atleast_2d(K_B)
        self.K_C = np.atleast_2d(K_C)
        self.K_D = np.atleast_2d(K_D)
        self.delta_max = delta_max
        n = self.K_A.shape[0]
        self._xk = np.zeros(n)

    def reset(self) -> None:
        """Azzera lo stato interno (chiamare prima di ogni episodio)."""
        self._xk = np.zeros(self.K_A.shape[0])

    def solve(self, v_meas: np.ndarray) -> float:
        """Calcola δ dalle misure correnti.

        Parameters
        ----------
        v_meas : (2,) ndarray — [ḧ_meas, α_meas]

        Returns
        -------
        delta : float — deflessione flap [°], saturata a ±delta_max
        """
        v = np.asarray(v_meas, dtype=float).ravel()
        y = float((self.K_C @ self._xk + self.K_D @ v).ravel()[0])
        # Aggiorna stato
        self._xk = self.K_A @ self._xk + self.K_B @ v
        return float(np.clip(y, -self.delta_max, self.delta_max))


# ── Loop di simulazione ───────────────────────────────────────────────────────

def run_hinf_simulation(
    U_INF: float,
    T_END: float,
    DT: float,
    aero_model,
    hinf_controller: HInfController | None,
    gust_profile=None,
    sigma_h_ddot: float = 0.0,
    sigma_alpha: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Loop closed-loop con controllore H∞.

    Pattern identico a run_mpc_simulation (src/control/mpc.py) ma:
    - nessun observer per W_hat (H∞ è output-feedback puro)
    - misure: ḧ + rumore e α + rumore

    Parameters
    ----------
    U_INF : float — velocità di volo [m/s]
    T_END : float — durata simulazione [s]
    DT    : float — passo di campionamento [s]
    aero_model : LDNetModel
    hinf_controller : HInfController o None (open-loop se None)
    gust_profile : callable t→W [m/s] o None (usa 1-cosine default)
    sigma_h_ddot : float — std rumore accelerometro [m/s²]
    sigma_alpha  : float — std rumore inclinometro [rad]
    rng : np.random.Generator — generatore casuale per il rumore

    Returns
    -------
    dict con chiavi: t, h, hd, a, ad, delta, C_L, C_M,
                     h_ddot, a_ddot, W_gust, delta_rate
    """
    if rng is None:
        rng = np.random.default_rng(42)

    q_dyn = 0.5 * 1.225 * U_INF**2 * 0.05
    M_hh  = M_WING + M_FLAP
    M_aa  = I_WING + I_FLAP_EA
    M_ha  = M_FLAP * _D_X

    t_win = np.linspace(0.0, T_END, int(T_END / DT) + 1)
    N     = len(t_win)

    if gust_profile is None:
        def gust_profile(t):
            T_g = 1.0; W0 = 60.0
            return 0.5 * W0 * (1.0 - np.cos(2.0 * np.pi * t / T_g)) \
                   if 0.0 <= t <= T_g else 0.0

    W_gust_arr = np.array([gust_profile(t) for t in t_win])

    # Trim latente
    z_trim = np.zeros(aero_model.num_latent_states)
    for _ in range(200):
        z_trim, _, _ = aero_model.step(z_trim, 0., 0., 0., 0., 0., 0., U_INF, DT)

    x = np.zeros(4)
    z = z_trim.copy()

    if hinf_controller is not None:
        hinf_controller.reset()

    h_hist      = np.zeros(N)
    hd_hist     = np.zeros(N)
    a_hist      = np.zeros(N)
    ad_hist     = np.zeros(N)
    delta_hist  = np.zeros(N)
    C_L_hist    = np.zeros(N)
    C_M_hist    = np.zeros(N)
    h_ddot_hist = np.zeros(N)
    a_ddot_hist = np.zeros(N)

    delta_prev = 0.0
    _h_ddot_filt_prev = 0.0

    for i, t in enumerate(t_win):
        h_hist[i] = x[0]; hd_hist[i] = x[1]
        a_hist[i] = x[2]; ad_hist[i] = x[3]

        # ── True system step ──────────────────────────────────────────────
        z, C_L, C_M = aero_model.step(
            z, x[0], x[1], x[2], x[3],
            delta_prev, W_gust_arr[i], U_INF, DT
        )
        C_L_hist[i] = C_L
        C_M_hist[i] = C_M

        Fy = q_dyn * C_L
        Mz = q_dyn * C_M

        def _srhs(s):
            return np.array(structural_rhs(t, s, Fy, Mz, 0.0, 0.0))

        k1 = _srhs(x); k2 = _srhs(x + 0.5*DT*k1)
        k3 = _srhs(x + 0.5*DT*k2); k4 = _srhs(x + DT*k3)
        x  = x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        _, h_ddot, _, a_ddot = _srhs(x)
        h_ddot_hist[i] = h_ddot
        a_ddot_hist[i] = a_ddot

        # ── Misure con rumore ─────────────────────────────────────────────
        h_ddot_meas = h_ddot + sigma_h_ddot * rng.standard_normal()
        alpha_meas  = x[2]  + sigma_alpha   * rng.standard_normal()

        # ── Controllo ─────────────────────────────────────────────────────
        if hinf_controller is not None:
            # Filtro passa-basso anti-aliasing su ḧ (fc=20 Hz, τ=1/(2π·20))
            tau_aa = 1.0 / (2.0 * np.pi * 20.0)
            alpha_aa = DT / (tau_aa + DT)
            h_ddot_filt = alpha_aa * h_ddot_meas + (1 - alpha_aa) * (
                0.0 if i == 0 else _h_ddot_filt_prev)
            _h_ddot_filt_prev = h_ddot_filt

            n_in = hinf_controller.K_B.shape[1]
            if n_in == 1:
                v_ctrl = np.array([h_ddot_filt])
            else:
                v_ctrl = np.array([h_ddot_filt, alpha_meas])[:n_in]
            delta = hinf_controller.solve(v_ctrl)
        else:
            delta = 0.0
        delta_hist[i] = delta
        delta_prev     = delta

    delta_rate = np.gradient(delta_hist, DT)

    return {
        't':          t_win,
        'h':          h_hist,
        'hd':         hd_hist,
        'a':          a_hist,
        'ad':         ad_hist,
        'delta':      delta_hist,
        'delta_rate': delta_rate,
        'C_L':        C_L_hist,
        'C_M':        C_M_hist,
        'h_ddot':     h_ddot_hist,
        'a_ddot':     a_ddot_hist,
        'W_gust':     W_gust_arr,
    }


# ── Confronto e sweep ─────────────────────────────────────────────────────────

def run_comparison(
    aero_model,
    lqr_controller,
    hinf_controller: HInfController,
    U_INF: float = 75.0,
    W0: float = 60.0,
    T_g: float = 1.0,
    T_END: float = 3.0,
    DT: float = 0.01,
    sigma_h_ddot: float = 0.0,
    sigma_alpha: float = 0.0,
) -> dict:
    """Esegue OL / LQR / H∞ sullo stesso gust e restituisce i tre dict.

    Parameters
    ----------
    aero_model : LDNetModel
    lqr_controller : LQRController
    hinf_controller : HInfController
    U_INF : float — velocità di volo [m/s]
    W0 : float — ampiezza gust [m/s]
    T_g : float — durata gust [s]
    T_END : float — fine simulazione [s]
    DT : float — passo [s]
    sigma_h_ddot, sigma_alpha : std rumore

    Returns
    -------
    dict con chiavi 'ol', 'lqr', 'hinf'
    """
    from control.mpc import run_mpc_simulation
    from structural.smd import get_space_state_matrices

    def _gust(t):
        return 0.5 * W0 * (1.0 - np.cos(2.0 * np.pi * t / T_g)) \
               if 0.0 <= t <= T_g else 0.0

    A_s, B_s, *_ = get_space_state_matrices()

    print(f"  Open-loop (W0={W0}, T_g={T_g})...")
    res_ol = run_mpc_simulation(
        U_INF, T_END, DT, aero_model, mpc_controller=None,
        A_s=A_s, B_s=B_s, gust_profile=_gust, observer='true_state')

    print("  LQR...")
    res_lqr = run_mpc_simulation(
        U_INF, T_END, DT, aero_model, mpc_controller=lqr_controller,
        A_s=A_s, B_s=B_s, gust_profile=_gust, observer='true_state')

    print("  H∞...")
    res_hinf = run_hinf_simulation(
        U_INF, T_END, DT, aero_model, hinf_controller,
        gust_profile=_gust,
        sigma_h_ddot=sigma_h_ddot, sigma_alpha=sigma_alpha)

    return {'ol': res_ol, 'lqr': res_lqr, 'hinf': res_hinf}


def sweep_gust_amplitude(
    aero_model,
    lqr_controller,
    hinf_controller: HInfController,
    W0_list: list[float] = (30.0, 60.0, 90.0, 120.0),
    T_g: float = 1.0,
    U_INF: float = 75.0,
    T_END: float = 3.0,
    DT: float = 0.01,
) -> dict:
    """Sweep sull'ampiezza gust W₀.

    Returns
    -------
    dict con chiavi W0, peak_hddot_lqr, peak_hddot_hinf (array)
    """
    peaks_lqr = []; peaks_hinf = []
    for W0 in W0_list:
        print(f"  Sweep ampiezza W0={W0} m/s...")
        r = run_comparison(aero_model, lqr_controller, hinf_controller,
                           U_INF=U_INF, W0=float(W0), T_g=T_g,
                           T_END=T_END, DT=DT)
        peaks_lqr.append(np.max(np.abs(r['lqr']['h_ddot'])))
        peaks_hinf.append(np.max(np.abs(r['hinf']['h_ddot'])))
    return {
        'W0':             np.array(W0_list),
        'peak_hddot_lqr': np.array(peaks_lqr),
        'peak_hddot_hinf': np.array(peaks_hinf),
    }


def sweep_gust_duration(
    aero_model,
    lqr_controller,
    hinf_controller: HInfController,
    T_g_list: list[float] = (0.3, 0.5, 1.0, 2.0),
    W0: float = 60.0,
    U_INF: float = 75.0,
    T_END: float = 4.0,
    DT: float = 0.01,
) -> dict:
    """Sweep sulla durata gust T_g.

    Returns
    -------
    dict con chiavi T_g, peak_hddot_lqr, peak_hddot_hinf (array)
    """
    peaks_lqr = []; peaks_hinf = []
    for T_g in T_g_list:
        print(f"  Sweep durata T_g={T_g} s...")
        r = run_comparison(aero_model, lqr_controller, hinf_controller,
                           U_INF=U_INF, W0=W0, T_g=float(T_g),
                           T_END=max(T_END, T_g + 2.0), DT=DT)
        peaks_lqr.append(np.max(np.abs(r['lqr']['h_ddot'])))
        peaks_hinf.append(np.max(np.abs(r['hinf']['h_ddot'])))
    return {
        'T_g':            np.array(T_g_list),
        'peak_hddot_lqr': np.array(peaks_lqr),
        'peak_hddot_hinf': np.array(peaks_hinf),
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_comparison_timeseries(
    results: dict,
    save_prefix: str = "hinf_fig",
    W0: float = 60.0,
    T_g: float = 1.0,
) -> None:
    """Quattro pannelli identici allo stile del capitolo 3.

    Pannello 1: h(t), α(t)
    Pannello 2: ḧ(t), α̈(t)
    Pannello 3: δ(t), δ̇(t)
    Pannello 4: C_L(t), C_M(t)

    Parameters
    ----------
    results : dict — da run_comparison: {'ol': ..., 'lqr': ..., 'hinf': ...}
    save_prefix : str — prefisso file figure
    W0, T_g : float — parametri gust (per titolo)
    """
    ol   = results['ol']
    lqr  = results['lqr']
    hinf = results['hinf']

    def _panel(fig_n, rows_left, rows_right, ylabel_left, ylabel_right,
               title, fname_suffix, scale_left=1.0, scale_right=1.0,
               unit_left="", unit_right=""):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, rows, ylabel, scale, unit in [
            (axes[0], rows_left,  ylabel_left,  scale_left,  unit_left),
            (axes[1], rows_right, ylabel_right, scale_right, unit_right),
        ]:
            for r, style in zip(rows, [_C_OL, _C_LQR, _C_HINF]):
                ax.plot(r[0], r[1] * scale, **style)
            ax.set_xlabel("t [s]")
            ax.set_ylabel(f"{ylabel} [{unit}]" if unit else ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{title} — W₀={W0} m/s, T_g={T_g} s", fontsize=13)
        fig.tight_layout()
        path = f"{save_prefix}{fig_n}_{fname_suffix}.png"
        fig.savefig(path, dpi=150)
        print(f"  Figura salvata: {path}")
        plt.close(fig)

    t = ol['t']

    # Fig 2: stati strutturali h e α
    _panel(2,
           rows_left  = [(t, ol['h']*100), (t, lqr['h']*100), (t, hinf['h']*100)],
           rows_right = [(t, np.degrees(ol['a'])),
                         (t, np.degrees(lqr['a'])),
                         (t, np.degrees(hinf['a']))],
           ylabel_left="h", ylabel_right="α",
           unit_left="cm", unit_right="°",
           title="Risposta strutturale", fname_suffix="states")

    # Fig 3: accelerazioni
    _panel(3,
           rows_left  = [(t, ol['h_ddot']), (t, lqr['h_ddot']), (t, hinf['h_ddot'])],
           rows_right = [(t, np.degrees(ol['a_ddot'])),
                         (t, np.degrees(lqr['a_ddot'])),
                         (t, np.degrees(hinf['a_ddot']))],
           ylabel_left="ḧ", ylabel_right="α̈",
           unit_left="m/s²", unit_right="°/s²",
           title="Accelerazioni", fname_suffix="accels")

    # Fig 4: deflessione e rate flap
    d_rate_ol   = np.gradient(ol['delta'],   t)
    d_rate_lqr  = np.gradient(lqr['delta'],  t)
    d_rate_hinf = np.gradient(hinf['delta'], t)
    _panel(4,
           rows_left  = [(t, ol['delta']), (t, lqr['delta']), (t, hinf['delta'])],
           rows_right = [(t, d_rate_ol), (t, d_rate_lqr), (t, d_rate_hinf)],
           ylabel_left="δ", ylabel_right="δ̇",
           unit_left="°", unit_right="°/s",
           title="Deflessione flap", fname_suffix="control")

    # Fig 5: coefficienti aerodinamici
    _panel(5,
           rows_left  = [(t, ol['C_L']), (t, lqr['C_L']), (t, hinf['C_L'])],
           rows_right = [(t, ol['C_M']), (t, lqr['C_M']), (t, hinf['C_M'])],
           ylabel_left="C_L", ylabel_right="C_M",
           title="Coefficienti aerodinamici", fname_suffix="aero")


def plot_sweep_summary(
    sweep_amp: dict,
    sweep_dur: dict,
    save_path: str = "hinf_fig6_sweep.png",
) -> None:
    """Picco ḧ vs W₀ e vs T_g per LQR e H∞.

    Parameters
    ----------
    sweep_amp : dict — da sweep_gust_amplitude
    sweep_dur : dict — da sweep_gust_duration
    save_path : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(sweep_amp['W0'], sweep_amp['peak_hddot_lqr'],
            color='darkcyan', marker='o', lw=1.5, label='LQR')
    ax.plot(sweep_amp['W0'], sweep_amp['peak_hddot_hinf'],
            color='mediumpurple', marker='s', lw=1.5, label='H∞')
    ax.set_xlabel("W₀ [m/s]")
    ax.set_ylabel("max |ḧ| [m/s²]")
    ax.set_title("Sweep ampiezza gust (T_g = 1 s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(sweep_dur['T_g'], sweep_dur['peak_hddot_lqr'],
            color='darkcyan', marker='o', lw=1.5, label='LQR')
    ax.plot(sweep_dur['T_g'], sweep_dur['peak_hddot_hinf'],
            color='mediumpurple', marker='s', lw=1.5, label='H∞')
    ax.set_xlabel("T_g [s]")
    ax.set_ylabel("max |ḧ| [m/s²]")
    ax.set_title("Sweep durata gust (W₀ = 60 m/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.suptitle("Confronto LQR vs H∞ — gust fuori dal training set", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Figura salvata: {save_path}")
    plt.close(fig)


def print_summary_table(
    sweep_amp: dict,
    sweep_dur: dict,
) -> None:
    """Stampa una tabella riassuntiva in stile LaTeX-ready."""
    print("\n── Tabella: sweep ampiezza gust ──────────────────────────")
    print(f"{'W₀ [m/s]':>10}  {'LQR peak ḧ':>12}  {'H∞ peak ḧ':>11}  {'Δ%':>6}")
    for W0, lqr, hinf in zip(sweep_amp['W0'],
                               sweep_amp['peak_hddot_lqr'],
                               sweep_amp['peak_hddot_hinf']):
        delta_pct = 100.0 * (hinf - lqr) / max(lqr, 1e-9)
        print(f"{W0:>10.0f}  {lqr:>12.3f}  {hinf:>11.3f}  {delta_pct:>+6.1f}%")

    print("\n── Tabella: sweep durata gust ────────────────────────────")
    print(f"{'T_g [s]':>10}  {'LQR peak ḧ':>12}  {'H∞ peak ḧ':>11}  {'Δ%':>6}")
    for T_g, lqr, hinf in zip(sweep_dur['T_g'],
                                sweep_dur['peak_hddot_lqr'],
                                sweep_dur['peak_hddot_hinf']):
        delta_pct = 100.0 * (hinf - lqr) / max(lqr, 1e-9)
        print(f"{T_g:>10.1f}  {lqr:>12.3f}  {hinf:>11.3f}  {delta_pct:>+6.1f}%")
