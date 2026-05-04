"""
uncertainty_weights.py
======================
Stima i pesi di incertezza moltiplicativa W_ΔL(z), W_ΔM(z) dai residui
LDNet vs FSI sul test set del capitolo 2.

Pipeline:
  1. Rollout LDNet su ogni CSV di test  → C_L^LDNet, C_M^LDNet
  2. Residui moltiplicativi normalizzati → r_L(t), r_M(t)
  3. FFT con detrend + finestra Hann    → envelope 95% in frequenza
  4. Fit TF stabile a fase minima       → W_ΔL(s), W_ΔM(s)
  5. Discretizzazione Tustin 2.5 Hz     → salvataggio su .npz

Il file .npz prodotto viene caricato da hinf_synthesis.py; i pesi
non vanno mai scelti a mano.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import minimize

# ── Costanti di analisi ─────────────────────────────────────────────────────
_DT_CSV      = 0.002          # s — passo dei CSV FSI
_U_INF_TEST  = 80.0           # m/s — U_inf del test set
_BAND_HZ     = (0.1, 50.0)    # Hz — banda di interesse per il fit
_PREWARP_HZ  = 2.5            # Hz — prewarping Tustin (flutter mode)


# ── Funzioni pubbliche ───────────────────────────────────────────────────────

def compute_uncertainty_envelope(
    test_csv_dir: str,
    aero_model,
    U_INF: float = _U_INF_TEST,
    eps: float = 1e-3,
    quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcola l'envelope spettrale dei residui moltiplicativi LDNet/FSI.

    Parameters
    ----------
    test_csv_dir : str
        Directory con i file sim_*_test.csv.
    aero_model : LDNetModel
        Istanza del modello LDNet già caricata.
    U_INF : float
        Velocità di volo per il rollout [m/s].
    eps : float
        Soglia denominatore nella normalizzazione moltiplicativa.
        Scelto = 1e-3 ≈ 1% del C_L al trim (≈0.126); evita amplificazioni
        spurie nelle zone a portanza quasi nulla mantenendo sensibilità fisica.
        Sensitività mostrata nella figura di diagnostica.
    quantile : float
        Livello di probabilità per l'envelope (default 0.95 → 95°percentile).

    Returns
    -------
    omega_hz : (N_freq,) ndarray
        Griglia di frequenze [Hz], da _BAND_HZ[0] a _BAND_HZ[1].
    env_L : (N_freq,) ndarray
        Envelope 95% del canale C_L.
    env_M : (N_freq,) ndarray
        Envelope 95% del canale C_M.
    """
    csv_dir = Path(test_csv_dir)
    test_files = sorted(csv_dir.glob("sim_*_test.csv"))
    if not test_files:
        raise FileNotFoundError(f"Nessun CSV di test in {csv_dir}")

    print(f"  Rollout LDNet su {len(test_files)} traiettorie di test "
          f"(U_inf={U_INF} m/s, eps={eps})...")

    specs_L: list[np.ndarray] = []
    specs_M: list[np.ndarray] = []
    freq_ref: np.ndarray | None = None

    for csv_path in test_files:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        # colonne: t, h, h_dot, alpha, alpha_dot, delta, W_g, C_L, C_M, Fy, Mz
        h    = data[:, 1]
        hd   = data[:, 2]
        a    = data[:, 3]
        ad   = data[:, 4]
        delta = data[:, 5]
        W_g  = data[:, 6]
        CL_fsi = data[:, 7]
        CM_fsi = data[:, 8]

        N = len(h)
        z = np.zeros(aero_model.num_latent_states)
        CL_nn = np.empty(N)
        CM_nn = np.empty(N)

        for i in range(N):
            z, CL_nn[i], CM_nn[i] = aero_model.step(
                z, h[i], hd[i], a[i], ad[i],
                delta[i], W_g[i], U_INF, _DT_CSV
            )

        # Residui moltiplicativi
        r_L = (CL_nn - CL_fsi) / np.maximum(np.abs(CL_fsi), eps)
        r_M = (CM_nn - CM_fsi) / np.maximum(np.abs(CM_fsi), eps)

        # FFT con detrend + finestra Hann
        freq, mag_L = _onesided_spectrum(r_L, _DT_CSV)
        _,    mag_M = _onesided_spectrum(r_M, _DT_CSV)

        # Interpolazione su griglia comune (log-spaced nella banda)
        if freq_ref is None:
            freq_ref = freq  # prima traiettoria definisce la griglia

        specs_L.append(np.interp(freq_ref, freq, mag_L))
        specs_M.append(np.interp(freq_ref, freq, mag_M))

    specs_L = np.array(specs_L)   # (N_traj, N_freq)
    specs_M = np.array(specs_M)

    env_L = np.quantile(specs_L, quantile, axis=0)
    env_M = np.quantile(specs_M, quantile, axis=0)

    # Restituisce solo la banda di interesse
    mask = (freq_ref >= _BAND_HZ[0]) & (freq_ref <= _BAND_HZ[1])
    return freq_ref[mask], env_L[mask], env_M[mask]


def fit_uncertainty_weights(
    omega_hz: np.ndarray,
    env_L: np.ndarray,
    env_M: np.ndarray,
    DT: float = 0.01,
    prewarp_hz: float = _PREWARP_HZ,
    max_order: int = 2,
    save_path: str | None = None,
) -> tuple[signal.dlti, signal.dlti]:
    """Fitta i pesi W_ΔL(s), W_ΔM(s) e li discretizza.

    Cerca il TF di ordine minimo (1° poi 2°) che sia stabile, a fase minima
    e che maggiori l'envelope su tutta la banda.  Se il bound viene violato
    aumenta il guadagno DC del 20% e riprova (max 5 iterazioni).

    Parameters
    ----------
    omega_hz : (N,) ndarray
        Frequenze [Hz].
    env_L : (N,) ndarray
        Envelope C_L [adimensionale].
    env_M : (N,) ndarray
        Envelope C_M [adimensionale].
    DT : float
        Passo di campionamento del controllore [s].
    prewarp_hz : float
        Frequenza di prewarping Tustin [Hz].
    max_order : int
        Ordine massimo da provare (1 o 2).
    save_path : str or None
        Se specificato, salva num/den e envelope su .npz.

    Returns
    -------
    W_dL : scipy.signal.dlti
        Peso discreto canale C_L.
    W_dM : scipy.signal.dlti
        Peso discreto canale C_M.
    """
    omega_rad = 2.0 * np.pi * omega_hz

    W_dL = _fit_single_weight(omega_rad, env_L, "C_L", max_order, DT, prewarp_hz)
    W_dM = _fit_single_weight(omega_rad, env_M, "C_M", max_order, DT, prewarp_hz)

    if save_path is not None:
        _save_weights(save_path, omega_hz, env_L, env_M, W_dL, W_dM)
        print(f"  Pesi salvati su {save_path}")

    return W_dL, W_dM


def load_uncertainty_weights(path: str) -> tuple[signal.dlti, signal.dlti,
                                                   np.ndarray, np.ndarray,
                                                   np.ndarray]:
    """Carica i pesi da file .npz.

    Returns
    -------
    W_dL, W_dM : scipy.signal.dlti
    omega_hz, env_L, env_M : np.ndarray
    """
    d = np.load(path)
    W_dL = signal.dlti(d["num_L"], d["den_L"], dt=float(d["DT"]))
    W_dM = signal.dlti(d["num_M"], d["den_M"], dt=float(d["DT"]))
    return W_dL, W_dM, d["omega_hz"], d["env_L"], d["env_M"]


def plot_uncertainty_envelope(
    omega_hz: np.ndarray,
    env_L: np.ndarray,
    env_M: np.ndarray,
    W_dL: signal.dlti,
    W_dM: signal.dlti,
    eps_values: tuple[float, ...] = (1e-4, 1e-3, 5e-3),
    test_csv_dir: str | None = None,
    aero_model=None,
    save_path: str = "hinf_fig0_weights.png",
) -> None:
    """Figura diagnostica: envelope + fit sovrapposti per C_L e C_M.

    Se test_csv_dir e aero_model sono forniti, mostra anche la sensitività
    all'scelta di eps.

    Parameters
    ----------
    omega_hz : (N,) ndarray — frequenze [Hz]
    env_L, env_M : (N,) ndarray — envelope 95%
    W_dL, W_dM : scipy.signal.dlti — pesi discreti fittati
    eps_values : tupla di eps da confrontare per sensitività
    test_csv_dir, aero_model : usati solo per la sensitività
    save_path : path di salvataggio figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, channel, env, W_d, label in zip(
        axes, ["C_L", "C_M"], [env_L, env_M], [W_dL, W_dM], ["L", "M"]
    ):
        # Envelope principale
        ax.semilogy(omega_hz, env, color="steelblue", lw=1.5, alpha=0.85,
                    label=f"Envelope 95% — $C_{{{label}}}$")

        # Fit W_Δ valutato sulla stessa griglia
        _, h_fit = signal.freqz(W_d.num, W_d.den, worN=2 * np.pi * omega_hz * W_d.dt)
        ax.semilogy(omega_hz, np.abs(h_fit), color="crimson", lw=2.0,
                    label=f"$W_{{\\Delta {label}}}(e^{{j\\omega}})$")

        # Sensitività eps (se dati disponibili)
        if test_csv_dir is not None and aero_model is not None:
            for eps_v in eps_values:
                if eps_v == 1e-3:
                    continue  # già mostrato come envelope principale
                try:
                    f2, e2_L, e2_M = compute_uncertainty_envelope(
                        test_csv_dir, aero_model, eps=eps_v)
                    e2 = e2_L if label == "L" else e2_M
                    ax.semilogy(f2, e2, lw=0.8, alpha=0.5,
                                linestyle="--", label=f"ε={eps_v:.0e}")
                except Exception:
                    pass

        ax.set_xlabel("Frequenza [Hz]")
        ax.set_ylabel("|W_Δ| [adim]")
        ax.set_title(f"Canale {channel}")
        ax.set_xlim(_BAND_HZ)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Incertezza moltiplicativa LDNet/FSI — envelope 95% e fit $W_\\Delta$",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Figura salvata: {save_path}")
    plt.close(fig)


# ── Funzioni di supporto interne ─────────────────────────────────────────────

def _onesided_spectrum(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """FFT monolaterale con detrend e finestra Hann.

    Returns
    -------
    freq : (N//2+1,) ndarray  [Hz]
    mag  : (N//2+1,) ndarray  magnitudine normalizzata (media per finestra)
    """
    x = x - x.mean()
    N = len(x)
    # Zero-padding alla potenza di 2 successiva per efficienza FFT
    N_fft = 1 << (N - 1).bit_length()
    win = np.hanning(N)
    win_norm = np.sum(win)
    X = np.fft.rfft(x * win, n=N_fft)
    mag = np.abs(X) / win_norm
    freq = np.fft.rfftfreq(N_fft, d=dt)
    return freq, mag


def _fit_single_weight(
    omega_rad: np.ndarray,
    envelope: np.ndarray,
    name: str,
    max_order: int,
    DT: float,
    prewarp_hz: float,
) -> signal.dlti:
    """Fitta un singolo peso W_Δ sull'envelope dato.

    Prova ordine 1 poi 2; per ciascun ordine verifica il bound e aumenta
    il guadagno se necessario.
    """
    prewarp_rad = 2.0 * np.pi * prewarp_hz

    for order in range(1, max_order + 1):
        num_c, den_c = _invfreqs_fit(omega_rad, envelope, order)
        if num_c is None:
            continue

        # Verifica stabilità e fase minima in s
        if not _is_stable_minphase(num_c, den_c):
            print(f"    [{name}] ordine {order}: TF non stabile/fase minima, provo ordine superiore")
            continue

        # Verifica bound con margine del 5%
        num_c, den_c = _enforce_bound(num_c, den_c, omega_rad, envelope,
                                       name=name, order=order)

        # Discretizzazione Tustin con prewarping
        sys_c = signal.lti(num_c, den_c)
        sys_d = signal.cont2discrete(
            (sys_c.num, sys_c.den), DT, method="bilinear",
            alpha=prewarp_rad,
        )
        # sys_d è (num_d, den_d, DT)
        W_d = signal.dlti(sys_d[0].ravel(), sys_d[1], dt=DT)
        print(f"    [{name}] ordine {order} accettato — "
              f"polo(i)_c: {np.roots(den_c).round(3)}, "
              f"zero(i)_c: {np.roots(num_c).round(3)}")
        return W_d

    # Fallback: guadagno costante = 2× massimo dell'envelope
    gain = 2.0 * float(np.max(envelope))
    warnings.warn(
        f"[{name}] Nessun TF di ordine ≤{max_order} ha soddisfatto il bound; "
        f"uso guadagno costante K={gain:.4f}",
        stacklevel=2,
    )
    sys_c = signal.lti([gain], [1.0])
    sys_d = signal.cont2discrete((sys_c.num, sys_c.den), DT, method="bilinear",
                                  alpha=prewarp_rad)
    return signal.dlti(sys_d[0].ravel(), sys_d[1], dt=DT)


def _invfreqs_fit(
    omega_rad: np.ndarray,
    magnitude: np.ndarray,
    order: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit razionale con invfreqs (LS nel dominio della frequenza).

    Usa scipy.signal.invres come base: costruisce il problema di minimi
    quadrati su |H(jω)| = envelope.  Per robustezza usa un'ottimizzazione
    diretta del modulo.

    Returns
    -------
    num, den : ndarray o (None, None) se il fit fallisce
    """
    # Punto iniziale: TF costante al valor medio dell'envelope
    gain0 = float(np.median(magnitude))

    # Parametri: [b0, ..., b_{order}] in num, [a1, ..., a_{order}] in den
    # (den[0] = 1 normalizzato)
    x0 = np.zeros(2 * order + 1)
    x0[0] = gain0          # b0
    x0[order + 1] = 1.0    # a1 (polo lontano)

    def residual(x):
        num = x[:order + 1]
        den = np.concatenate([[1.0], x[order + 1:]])
        _, H = signal.freqs(num, den, worN=omega_rad)
        return np.sum((np.abs(H) - magnitude) ** 2)

    result = minimize(residual, x0, method="Nelder-Mead",
                      options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-12})

    if not result.success and result.fun > 1e-2 * np.sum(magnitude ** 2):
        return None, None

    num = result.x[:order + 1]
    den = np.concatenate([[1.0], result.x[order + 1:]])
    return num, den


def _is_stable_minphase(num: np.ndarray, den: np.ndarray) -> bool:
    """Verifica che tutti i poli e zeri abbiano parte reale negativa."""
    poles = np.roots(den)
    zeros = np.roots(num)
    return bool(np.all(poles.real < 0) and np.all(zeros.real < 0))


def _enforce_bound(
    num: np.ndarray,
    den: np.ndarray,
    omega_rad: np.ndarray,
    envelope: np.ndarray,
    name: str,
    order: int,
    max_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Moltiplica il guadagno DC del 20% finché |W(jω)| ≥ envelope ovunque."""
    for k in range(max_iter):
        _, H = signal.freqs(num, den, worN=omega_rad)
        ratio = np.max(envelope / np.maximum(np.abs(H), 1e-30))
        if ratio <= 1.0:
            return num, den
        # Moltiplica il numeratore per ratio * 1.1 (10% di margine)
        num = num * ratio * 1.1
        warnings.warn(
            f"[{name}] ordine {order}: bound violato (ratio={ratio:.3f}), "
            f"guadagno scalato ×{ratio * 1.1:.3f} (iterazione {k + 1})",
            stacklevel=3,
        )
    return num, den


def _save_weights(
    path: str,
    omega_hz: np.ndarray,
    env_L: np.ndarray,
    env_M: np.ndarray,
    W_dL: signal.dlti,
    W_dM: signal.dlti,
) -> None:
    np.savez(
        path,
        num_L=W_dL.num,
        den_L=W_dL.den,
        num_M=W_dM.num,
        den_M=W_dM.den,
        DT=np.array(W_dL.dt),
        omega_hz=omega_hz,
        env_L=env_L,
        env_M=env_M,
    )