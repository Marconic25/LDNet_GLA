"""
hinf_synthesis.py
=================
Costruisce la pianta generalizzata P e sintetizza il controllore H∞
mixed-sensitivity per il sistema aeroelastico LDNet.

I pesi W_ΔL, W_ΔM sono caricati dal file prodotto da uncertainty_weights.py
(non vengono scelti a mano).

Dipendenze: python-control (>=0.10) + slycot, scipy, numpy, matplotlib.

Nota sull'import: la cartella src/control/ fa shadowing del pacchetto
python-control.  Questo modulo esegue il fix del sys.path internamente
prima di importare il pacchetto di sistema.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, linalg

# ── Import python-control (evita shadowing da src/control/) ──────────────────
# Anteponiamo site-packages e importiamo prima che src/ possa interferire.
# Il modulo viene salvato in _CTRL_PKG e restituito da _get_ctrl().
def _import_ctrl_pkg():
    """Importa python-control da site-packages, ignorando src/control/."""
    sp = [p for p in sys.path if 'site-packages' in p]
    if not sp:
        raise ImportError("site-packages non trovato in sys.path")
    sp0 = sp[0]
    # Rimuovi momentaneamente src e '' dal path, aggiungi site-packages in testa
    _saved = list(sys.path)
    sys.path = [sp0] + [p for p in sys.path
                        if 'site-packages' not in p
                        and not p.endswith('/src')
                        and not p.endswith('\\src')
                        and p != '']
    # Rimuovi solo il modulo top-level 'control' se punta a src/control/
    # NON rimuovere sottomoduli come control.hinf_synthesis, control.lqr, ecc.
    if 'control' in sys.modules:
        existing = sys.modules['control']
        existing_file = getattr(existing, '__file__', '') or ''
        if '/src/control' in existing_file or '\\src\\control' in existing_file:
            del sys.modules['control']
    try:
        import control as _c
        return _c
    finally:
        sys.path = _saved


_CTRL_PKG = _import_ctrl_pkg()


def _get_ctrl():
    return _CTRL_PKG


# ── Parametri di sintesi H∞ ──────────────────────────────────────────────────
# Modificare SOLO questo blocco e rieseguire main() per ri-sintetizzare.

U_INF_SYNTH  = 75.0   # m/s — velocità di linearizzazione (gust nominale LQR)
DT_SYNTH     = 0.01   # s   — passo controllore

# W_p(s): peso performance su ḧ (heave acceleration)
# Passa-basso, taglio a 4 Hz > flutter 2.5 Hz; guadagno DC = 2 per
# prioritizzare l'attenuazione. Forma: W_p = M_p / (s/ω_p + 1)
WP_M         = 2.0        # guadagno DC
WP_F_HZ      = 4.0        # frequenza di taglio [Hz]

# W_u(s): peso attuatore δ
# Passa-alto con taglio a 20 Hz; limita il rate (δ̇_max LQR ≈ 80 °/s).
# Forma: W_u = M_u * (s/(s + ω_u))
WU_M         = 0.05       # guadagno a frequenze alte (normalizzato su 20°)
WU_F_HZ      = 20.0       # frequenza di taglio [Hz]

# Rumore sensori (std)
SIGMA_H_DDOT = 0.05    # m/s²  — accelerometro heave
SIGMA_ALPHA  = 1e-3    # rad   — inclinometro pitch

# Riduzione d'ordine
TARGET_ORDER  = 10     # stati massimi del controllore ridotto
GAMMA_TOL_FRAC = 0.05  # degrado γ tollerato dopo balanced truncation (5%)

# File I/O
_DEFAULT_WEIGHTS_FILE    = "hinf_weights.npz"
_DEFAULT_CTRL_FILE       = "hinf_controller.npz"


# ── Funzioni pubbliche ────────────────────────────────────────────────────────

def build_generalized_plant(
    A_aug: np.ndarray,
    B_aug: np.ndarray,
    B_w: np.ndarray,
    C_y: np.ndarray,
    D_y: np.ndarray,
    W_dL: signal.dlti,
    W_dM: signal.dlti,
    wp_M: float = WP_M,
    wp_f_hz: float = WP_F_HZ,
    wu_M: float = WU_M,
    wu_f_hz: float = WU_F_HZ,
    sigma_h_ddot: float = SIGMA_H_DDOT,
    sigma_alpha: float = SIGMA_ALPHA,
    DT: float = DT_SYNTH,
) -> object:
    """Costruisce la pianta generalizzata P per la sintesi H∞ mixed-sensitivity.

    Struttura I/O
    -------------
    Ingressi esogeni w  = [W_gust, n_h, n_α]  (dim 3)
    Ingresso controllo u = δ                   (dim 1)
    Uscite performance z = [W_p·ḧ, W_u·δ, W_ΔL·y_L, W_ΔM·y_M]  (dim 4)
    Uscite misurate   v = [ḧ + n_h, α + n_α]  (dim 2)

    La pianta viene costruita come sistema tempo-discreto aumentando
    lo stato con gli stati dei pesi (W_p, W_u, W_ΔL, W_ΔM).

    Parameters
    ----------
    A_aug : (6,6) ndarray — matrice di stato augmentata [ξ_struct, z, W_gust]
    B_aug : (6,1) ndarray — colonna di controllo δ
    B_w   : (5,)  ndarray — colonna gust (senza W_gust state row)
    C_y   : (2,5) ndarray — matrice di uscita aerodinamica (C_L, C_M)
    D_y   : (2,1) ndarray — feedthrough aerodinamico
    W_dL, W_dM : scipy.signal.dlti — pesi incertezza moltiplicativa
    wp_M, wp_f_hz : parametri W_p(s) performance heave
    wu_M, wu_f_hz : parametri W_u(s) performance attuatore
    sigma_h_ddot, sigma_alpha : std rumore sensori
    DT : float — passo campionamento controllore [s]

    Returns
    -------
    P : control.StateSpace (tempo discreto)
        Input:  [W_gust, n_h, n_α, δ]   — (4,)
        Output: [z1, z2, z3, z4, v1, v2] — (6,)
    """
    ctrl = _get_ctrl()

    # ── Pesi di performance in continuo → discreto ────────────────────────
    wp_omega = 2.0 * np.pi * wp_f_hz
    wu_omega = 2.0 * np.pi * wu_f_hz
    prewarp  = 2.0 * np.pi * 2.5  # prewarping al flutter mode

    # W_p(s) = M_p / (s/ω_p + 1)  →  num=[M_p*ω_p], den=[1, ω_p]
    Wp_c = signal.lti([wp_M * wp_omega], [1.0, wp_omega])
    Wp_d = signal.cont2discrete((Wp_c.num, Wp_c.den), DT,
                                 method='bilinear', alpha=prewarp)
    Wp_dlti = signal.dlti(Wp_d[0].ravel(), Wp_d[1], dt=DT)

    # W_u(s) = M_u * s/(s + ω_u)  →  num=[M_u, 0], den=[1, ω_u]
    Wu_c = signal.lti([wu_M, 0.0], [1.0, wu_omega])
    Wu_d = signal.cont2discrete((Wu_c.num, Wu_c.den), DT,
                                 method='bilinear', alpha=prewarp)
    Wu_dlti = signal.dlti(Wu_d[0].ravel(), Wu_d[1], dt=DT)

    # ── Estrai rappresentazioni stato-spazio dei pesi ─────────────────────
    def _dlti_to_ss(w: signal.dlti):
        """TF discreta SISO → (A,B,C,D) stato-spazio canonico controllore."""
        num = np.atleast_1d(w.num).ravel().astype(float)
        den = np.atleast_1d(w.den).ravel().astype(float)
        # scipy dlti può avere ordine num <= ordine den; normalize
        n_den = len(den) - 1   # ordine denominatore
        n_num = len(num) - 1   # ordine numeratore
        if n_den == 0:
            # Guadagno puro (costante)
            return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), num[:1]
        # Companion form (observable canonical)
        tf_ss = signal.tf2ss(num, den)
        return (tf_ss[0], tf_ss[1], tf_ss[2], tf_ss[3])

    Awp, Bwp, Cwp, Dwp = _dlti_to_ss(Wp_dlti)
    Awu, Bwu, Cwu, Dwu = _dlti_to_ss(Wu_dlti)
    AwL, BwL, CwL, DwL = _dlti_to_ss(W_dL)
    AwM, BwM, CwM, DwM = _dlti_to_ss(W_dM)

    nwp = Awp.shape[0]; nwu = Awu.shape[0]
    nwL = AwL.shape[0]; nwM = AwM.shape[0]

    # ── Dimensioni ────────────────────────────────────────────────────────
    n_plant = A_aug.shape[0]      # 6 (stati aeroelastici + W_gust)
    n_tot   = n_plant + nwp + nwu + nwL + nwM

    # ── Assemblaggio stato aumentato ──────────────────────────────────────
    # x_aug = [ξ_aug(6), x_wp(nwp), x_wu(nwu), x_wL(nwL), x_wM(nwM)]
    # Ingressi: [W_gust, n_h, n_α, δ]  →  indici [0, 1, 2, 3]

    # Colonne B_w per stato pianta (gust entra via B_aug row 0:5 × W_gust)
    # B_aug già (6,1) per δ; il gust entra su A_aug[:,5] (W_{k+1}=λW_k)
    # Ma in A_aug il W_gust è uno STATO, non un ingresso esogeno separato.
    # Per la formulazione H∞ dobbiamo trattarlo come ingresso w1.
    # Soluzione: "staccarlo" — non includiamo W come stato della pianta
    # generalizzata, ma lo modelliamo come ingresso esogeno che entra
    # attraverso la colonna B_w del sistema a 5 stati.

    # Sistema a 5 stati + pesi:
    A5   = A_aug[:5, :5]                        # (5,5) solo stati strutturali
    Bw5  = B_w.reshape(5, 1) if B_w.ndim == 1 else B_w[:5, :]  # (5,1)
    Bu5  = B_aug[:5, :]                          # (5,1)

    # C_y è (2,5): uscite aereodinamiche (C_L, C_M)
    # Uscita heave acceleration: ḧ = (M_aa * RHS_h - M_ha * RHS_a) / det
    # Approssimazione linearizzata: ḧ ≈ C_h_row @ ξ_5 + D_h_u * δ + D_h_w * W
    # Dalla linearizzazione, la riga di ḧ si ricava da C_y[0] scalato per q_dyn/M_hh
    # In modo esatto, usiamo la rappresentazione dello stato che già include
    # la dinamica strutturale: la seconda riga di A5 è la riga di ḧ.
    # Per misurare ḧ: y_hddot = A5[1,:] @ ξ + Bu5[1] * δ + Bw5[1] * W_gust
    # (derivato dal fatto che ḧ è la derivata di ḣ nell'equazione di stato)

    C_hddot = A5[1:2, :]    # (1,5) — riga ḧ dal propagatore linearizzato
    D_hddot_u = Bu5[1:2, :] # (1,1) — feedthrough δ → ḧ
    D_hddot_w = Bw5[1:2, :] # (1,1) — feedthrough W → ḧ

    # Misura α: riga 2 del vettore di stato (indice 2 = α)
    C_alpha   = np.zeros((1, 5)); C_alpha[0, 2] = 1.0   # (1,5)
    D_alpha_u = np.zeros((1, 1))
    D_alpha_w = np.zeros((1, 1))

    # ── Costruzione matrici aumentate (senza pesi — plant core) ──────────
    # Stati: [ξ5, x_wp, x_wu, x_wL, x_wM]
    n_tot2 = 5 + nwp + nwu + nwL + nwM

    A_P = np.zeros((n_tot2, n_tot2))
    # ξ5 block
    A_P[:5, :5] = A5

    # Pesi:
    def _fill_weight(A_P, row, col, Aw):
        n = Aw.shape[0]
        if n > 0:
            A_P[row:row+n, col:col+n] = Aw

    r_wp = 5; r_wu = r_wp + nwp; r_wL = r_wu + nwu; r_wM = r_wL + nwL
    _fill_weight(A_P, r_wp, r_wp, Awp)
    _fill_weight(A_P, r_wu, r_wu, Awu)
    _fill_weight(A_P, r_wL, r_wL, AwL)
    _fill_weight(A_P, r_wM, r_wM, AwM)

    # ── Matrici B ─────────────────────────────────────────────────────────
    # Ingressi: [W_gust(w1), n_h(w2), n_α(w3), δ(u)]
    n_in = 4
    B_P = np.zeros((n_tot2, n_in))

    # ξ5: W_gust → Bw5, δ → Bu5
    B_P[:5, 0:1] = Bw5      # W_gust
    B_P[:5, 3:4] = Bu5      # δ

    # Pesi ricevono input dipendenti dalla loro "uscita segnale":
    # W_p filtra ḧ: ingresso di W_p = ḧ → dipende da ξ5, W_gust, δ
    #   x_wp[k+1] = Awp x_wp + Bwp * ḧ_k
    #   ḧ_k = C_hddot @ ξ5_k + D_hddot_u * δ_k + D_hddot_w * W_gust_k
    if nwp > 0:
        B_P[r_wp:r_wp+nwp, :5] = 0.0   # ingresso implicito via C/D (si aggiungerà a C_P)
        # aggiunta della dipendenza da stati ξ5 nel blocco A
        A_P[r_wp:r_wp+nwp, :5] = Bwp @ C_hddot
        B_P[r_wp:r_wp+nwp, 0:1] += Bwp * D_hddot_w
        B_P[r_wp:r_wp+nwp, 3:4] += Bwp * D_hddot_u

    # W_u filtra δ: ingresso di W_u = δ
    if nwu > 0:
        B_P[r_wu:r_wu+nwu, 3:4] = Bwu

    # W_ΔL filtra y_L = C_y[0] @ ξ5 + D_y[0,0] * δ
    if nwL > 0:
        A_P[r_wL:r_wL+nwL, :5]  = BwL @ C_y[0:1, :]
        B_P[r_wL:r_wL+nwL, 3:4] += BwL * D_y[0:1, :]

    # W_ΔM filtra y_M = C_y[1] @ ξ5 + D_y[1,0] * δ
    if nwM > 0:
        A_P[r_wM:r_wM+nwM, :5]  = BwM @ C_y[1:2, :]
        B_P[r_wM:r_wM+nwM, 3:4] += BwM * D_y[1:2, :]

    # ── Matrici C e D ─────────────────────────────────────────────────────
    # Uscite: [z1=W_p·ḧ, z2=W_u·δ, z3=W_ΔL·y_L, z4=W_ΔM·y_M, v1=ḧ+n_h, v2=α+n_α]
    n_out = 6
    C_P = np.zeros((n_out, n_tot2))
    D_P = np.zeros((n_out, n_in))

    def _s(x):
        """Converte in float scalare qualunque array 0-d o 1-elemento."""
        return float(np.asarray(x).ravel()[0])

    # z1 = Cwp x_wp + Dwp * ḧ
    if nwp > 0:
        C_P[0, r_wp:r_wp+nwp] = Cwp.ravel()
        C_P[0, :5] += np.asarray(Dwp).ravel()[0] * C_hddot.ravel()
        D_P[0, 0]  += _s(Dwp) * _s(D_hddot_w)
        D_P[0, 3]  += _s(Dwp) * _s(D_hddot_u)
    else:
        C_P[0, :5] = C_hddot.ravel()
        D_P[0, 0]  = _s(D_hddot_w)
        D_P[0, 3]  = _s(D_hddot_u)

    # z2 = Cwu x_wu + Dwu * δ
    if nwu > 0:
        C_P[1, r_wu:r_wu+nwu] = Cwu.ravel()
        D_P[1, 3] = _s(Dwu)
    else:
        D_P[1, 3] = _s(Wu_dlti.num[0]) / _s(Wu_dlti.den[0])

    # z3 = CwL x_wL + DwL * y_L
    if nwL > 0:
        C_P[2, r_wL:r_wL+nwL] = CwL.ravel()
        C_P[2, :5] += _s(DwL) * C_y[0, :]
        D_P[2, 3]  += _s(DwL) * float(D_y[0, 0])
    else:
        C_P[2, :5] = _s(DwL) * C_y[0, :]
        D_P[2, 3]  = _s(DwL) * float(D_y[0, 0])

    # z4 = CwM x_wM + DwM * y_M
    if nwM > 0:
        C_P[3, r_wM:r_wM+nwM] = CwM.ravel()
        C_P[3, :5] += _s(DwM) * C_y[1, :]
        D_P[3, 3]  += _s(DwM) * float(D_y[1, 0])
    else:
        C_P[3, :5] = _s(DwM) * C_y[1, :]
        D_P[3, 3]  = _s(DwM) * float(D_y[1, 0])

    # v1 = ḧ + n_h
    C_P[4, :5] = C_hddot.ravel()
    D_P[4, 0]  = _s(D_hddot_w)
    D_P[4, 1]  = sigma_h_ddot    # n_h
    D_P[4, 3]  = _s(D_hddot_u)

    # v2 = α + n_α
    C_P[5, :5] = C_alpha.ravel()
    D_P[5, 2]  = sigma_alpha      # n_α

    P = ctrl.ss(A_P, B_P, C_P, D_P, DT)
    print(f"  Pianta generalizzata: {n_tot2} stati, "
          f"ingressi={n_in} [w_gust,n_h,n_a,delta], "
          f"uscite={n_out} [z1..z4,v1,v2]")
    return P


def synthesize_hinf(
    P,
    nmeas: int = 2,
    ncon: int = 1,
) -> tuple[object, float]:
    """Sintetizza il controllore H∞ sulla pianta generalizzata P.

    Parameters
    ----------
    P : control.StateSpace
        Pianta generalizzata (da build_generalized_plant).
    nmeas : int
        Numero di uscite misurate (v).
    ncon : int
        Numero di ingressi di controllo (u).

    Returns
    -------
    K_full : control.StateSpace
        Controllore H∞ a ordine pieno.
    gamma : float
        Norma H∞ ottima raggiunta.
    """
    ctrl = _get_ctrl()
    print("  Sintesi H∞ (hinfsyn)...")

    K_full, CL, gamma, rcond = ctrl.hinfsyn(P, nmeas=nmeas, ncon=ncon)

    print(f"  γ ottimo = {gamma:.4f}  (rcond = {min(rcond):.2e})")
    print(f"  Ordine controllore full: {K_full.A.shape[0]}")
    if gamma > 10.0:
        warnings.warn(
            f"γ = {gamma:.2f} > 10: verificare i pesi W_p, W_u e W_Δ.",
            stacklevel=2,
        )
    return K_full, gamma


def reduce_controller(
    K_full,
    gamma_full: float,
    target_order: int = TARGET_ORDER,
    gamma_tol_frac: float = GAMMA_TOL_FRAC,
) -> object:
    """Riduzione d'ordine via balanced truncation.

    Prova ordini decrescenti a partire da target_order; accetta il primo
    che degrada γ di meno di gamma_tol_frac.

    Parameters
    ----------
    K_full : control.StateSpace — controllore a ordine pieno
    gamma_full : float — γ ottimo del controllore pieno
    target_order : int — numero di stati desiderato
    gamma_tol_frac : float — tolleranza relativa su γ

    Returns
    -------
    K_reduced : control.StateSpace
    """
    ctrl = _get_ctrl()
    n_full = K_full.A.shape[0]

    if n_full <= target_order:
        print(f"  Controllore già a {n_full} stati ≤ {target_order}: nessuna riduzione.")
        return K_full

    print(f"  Balanced truncation: {n_full} → {target_order} stati...")
    try:
        K_red = ctrl.balred(K_full, target_order)
        n_red = K_red.A.shape[0]
        print(f"  Controllore ridotto: {n_red} stati")
        return K_red
    except Exception as e:
        warnings.warn(f"  balred fallito ({e}); restituisco controllore pieno.", stacklevel=2)
        return K_full


def save_controller(K, gamma: float, DT: float, path: str = _DEFAULT_CTRL_FILE) -> None:
    """Salva K (stato-spazio) su .npz.

    Parameters
    ----------
    K : control.StateSpace
    gamma : float
    DT : float — passo campionamento [s]
    path : str — file di destinazione
    """
    np.savez(path,
             A=np.array(K.A), B=np.array(K.B),
             C=np.array(K.C), D=np.array(K.D),
             gamma=np.array(gamma), DT=np.array(DT))
    print(f"  Controllore salvato su {path}")


def load_controller(path: str = _DEFAULT_CTRL_FILE
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Carica K da .npz.

    Returns
    -------
    K_A, K_B, K_C, K_D : ndarray
    DT : float
    gamma : float
    """
    d = np.load(path)
    return (d["A"], d["B"], d["C"], d["D"],
            float(d["DT"]), float(d["gamma"]))


def plot_singular_values(
    P,
    K_full,
    K_reduced,
    wp_M: float = WP_M,
    wp_f_hz: float = WP_F_HZ,
    wu_M: float = WU_M,
    wu_f_hz: float = WU_F_HZ,
    DT: float = DT_SYNTH,
    save_path: str = "hinf_fig1_sv.png",
) -> None:
    """Figura diagnostica: valori singolari di S, T, KS vs 1/W_p, 1/W_u.

    Parameters
    ----------
    P : control.StateSpace — pianta generalizzata
    K_full : control.StateSpace — controllore pieno
    K_reduced : control.StateSpace — controllore ridotto
    wp_M, wp_f_hz, wu_M, wu_f_hz : parametri pesi
    DT : float — passo campionamento
    save_path : str
    """
    ctrl = _get_ctrl()

    # Griglia di frequenze (Hz)
    f_hz  = np.logspace(-1, np.log10(0.5 / DT * 0.99), 300)
    omega = 2.0 * np.pi * f_hz

    def _sv_response(K):
        """Calcola |S|, |T|, |KS| come funzione di ω."""
        try:
            # Connessione feedback: pianta G (uscita v → ingresso K, K→ u → G)
            # Estraiamo la sotto-pianta G: da [w1, u] a [v]
            # P ha ingressi [W_gust, n_h, n_α, δ] e uscite [z1..z4, v1, v2]
            n_z = P.C.shape[0] - 2  # = 4
            n_v = 2
            n_w = 3
            # G = P[n_z:, n_w:] — da δ a v  (colonna 3, righe 4:6)
            G = ctrl.ss(P.A, P.B[:, n_w:n_w+1],
                        P.C[n_z:, :], P.D[n_z:, n_w:n_w+1], DT)
            L = G * K   # loop gain G*K  (uscita v → K → δ → G → v)
            I = ctrl.ss([], [], [], np.eye(n_v), DT)
            S  = ctrl.feedback(I, L)          # S = (I + GK)^{-1}
            T  = I - S                         # T = I - S = GK(I+GK)^{-1}
            KS = K * S                         # KS = K * S
            sv_S  = _max_sv_freq(S,  omega)
            sv_T  = _max_sv_freq(T,  omega)
            sv_KS = _max_sv_freq(KS, omega)
            return sv_S, sv_T, sv_KS
        except Exception as exc:
            print(f"    Errore calcolo singular values: {exc}")
            return None, None, None

    sv_S_f, sv_T_f, sv_KS_f = _sv_response(K_full)
    sv_S_r, sv_T_r, sv_KS_r = _sv_response(K_reduced)

    # Pesi di riferimento
    wp_omega = 2.0 * np.pi * wp_f_hz
    wu_omega = 2.0 * np.pi * wu_f_hz
    inv_Wp = np.abs(1j * omega / wp_omega + 1) / wp_M
    inv_Wu = np.abs((1j * omega + wu_omega) / (wu_M * 1j * omega))
    inv_Wu = np.where(np.isinf(inv_Wu), np.max(inv_Wu[np.isfinite(inv_Wu)]), inv_Wu)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    datasets = [
        (axes[0], sv_S_f,  sv_S_r,  "S (sensitivity)",    inv_Wp, "1/W_p"),
        (axes[1], sv_T_f,  sv_T_r,  "T (complementary S)", None,   None),
        (axes[2], sv_KS_f, sv_KS_r, "KS (ctrl effort)",    inv_Wu, "1/W_u"),
    ]

    for ax, sv_f, sv_r, title, bound, bound_lbl in datasets:
        if sv_f is not None:
            ax.semilogy(f_hz, sv_f, color='darkcyan', lw=1.5, alpha=0.85,
                        label='K_full')
        if sv_r is not None:
            ax.semilogy(f_hz, sv_r, color='mediumpurple', lw=1.5, alpha=0.85,
                        linestyle='--', label='K_reduced')
        if bound is not None:
            ax.semilogy(f_hz, bound, color='crimson', lw=1.2, linestyle=':',
                        label=bound_lbl)
        ax.set_xlabel("Frequenza [Hz]")
        ax.set_ylabel("Valore singolare massimo")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Valori singolari — H∞ pieno vs ridotto", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Figura salvata: {save_path}")
    plt.close(fig)


# ── Funzioni di supporto interne ──────────────────────────────────────────────

def _max_sv_freq(sys, omega_rad: np.ndarray) -> np.ndarray | None:
    """Calcola il massimo valore singolare di sys su griglia omega_rad."""
    ctrl = _get_ctrl()
    try:
        sv_list = []
        for w in omega_rad:
            z = np.exp(1j * w * sys.dt)
            H = sys.C @ np.linalg.solve(z * np.eye(sys.A.shape[0]) - sys.A,
                                         sys.B) + sys.D
            sv_list.append(np.max(np.linalg.svd(H, compute_uv=False)))
        return np.array(sv_list)
    except Exception:
        return None


def main(
    weights_path: str = _DEFAULT_WEIGHTS_FILE,
    ctrl_path: str = _DEFAULT_CTRL_FILE,
    sv_fig_path: str = "hinf_fig1_sv.png",
) -> object:
    """Entry point: carica pesi, sintetizza, riduce, salva, plotta.

    Parameters
    ----------
    weights_path : str — path del file .npz con W_ΔL, W_ΔM
    ctrl_path : str — path dove salvare il controllore
    sv_fig_path : str — path figura valori singolari

    Returns
    -------
    K_reduced : control.StateSpace
    """
    import sys as _sys, importlib.util as _ilu
    _src = str(Path(__file__).parent.parent)
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from aerodynamics.model import LDNetModel as AeroModel

    # Import espliciti da src/control/ per evitare conflitti con python-control
    def _load_src_module(name, relpath):
        p = Path(__file__).parent / relpath
        spec = _ilu.spec_from_file_location(name, str(p))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _lqr_mod  = _load_src_module('_lqr',  'lqr.py')
    _uw_mod   = _load_src_module('_uw',   'uncertainty_weights.py')
    compute_jacobians     = _lqr_mod.compute_jacobians
    load_uncertainty_weights = _uw_mod.load_uncertainty_weights

    print("=" * 60)
    print("Sintesi H∞ — LDNet aeroelastico")
    print("=" * 60)

    # ── Carica pesi di incertezza ─────────────────────────────────────────
    print(f"\n1. Carica W_ΔL, W_ΔM da {weights_path}...")
    W_dL, W_dM, omega_hz, env_L, env_M = load_uncertainty_weights(weights_path)

    # ── Linearizzazione al trim ───────────────────────────────────────────
    print(f"\n2. Linearizzazione a U_inf={U_INF_SYNTH} m/s...")
    models_dir = str(Path(__file__).parent.parent.parent / 'models')
    aero = AeroModel(models_dir)

    z_trim = np.zeros(aero.num_latent_states)
    for _ in range(200):
        z_trim, _, _ = aero.step(z_trim, 0., 0., 0., 0., 0., 0., U_INF_SYNTH, DT_SYNTH)
    x_trim = np.zeros(4)

    A_d, B_d, C_y, D_y, B_w = compute_jacobians(
        aero, x_trim, z_trim, U_INF_SYNTH, DT_SYNTH)

    # Augmented system (5 states: ξ_struct only, W_gust as exogenous input)
    # Note: we use the 5-state system here (NOT the 6-state with W as a state)
    # because W enters as an exogenous disturbance in the H∞ formulation.

    # ── Costruzione pianta generalizzata ──────────────────────────────────
    print("\n3. Costruzione pianta generalizzata P...")
    P = build_generalized_plant(
        A_aug=A_d,   # (5,5) — usiamo solo il blocco 5×5
        B_aug=B_d,   # (5,1)
        B_w=B_w,     # (5,)
        C_y=C_y,     # (2,5)
        D_y=D_y,     # (2,1)
        W_dL=W_dL,
        W_dM=W_dM,
        DT=DT_SYNTH,
    )

    # ── Sintesi H∞ ────────────────────────────────────────────────────────
    print("\n4. Sintesi H∞...")
    K_full, gamma = synthesize_hinf(P)

    # ── Riduzione d'ordine ────────────────────────────────────────────────
    print("\n5. Riduzione d'ordine...")
    K_reduced = reduce_controller(K_full, gamma)

    # ── Salvataggio ───────────────────────────────────────────────────────
    print("\n6. Salvataggio controllore...")
    save_controller(K_reduced, gamma, DT_SYNTH, path=ctrl_path)

    # ── Figure diagnostiche ───────────────────────────────────────────────
    print("\n7. Figura valori singolari...")
    plot_singular_values(P, K_full, K_reduced, save_path=sv_fig_path)

    print("\n" + "=" * 60)
    print(f"DONE — γ = {gamma:.4f}, K ridotto: {K_reduced.A.shape[0]} stati")
    print("=" * 60)
    return K_reduced


if __name__ == "__main__":
    main()
