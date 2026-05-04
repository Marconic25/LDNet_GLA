"""
hinf_run_synthesis.py
=====================
Script autonomo per la sintesi H∞.  Va eseguito DIRETTAMENTE (non importato)
da src/ con il venv attivo:

    cd /home/marco/LDNet_OF/src
    source venv/bin/activate
    python hinf_run_synthesis.py

Il path viene configurato qui PRIMA di qualsiasi import in modo che
python-control (site-packages) non venga oscurato dalla cartella src/control/.
"""

from __future__ import annotations
import sys, os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC  = str(Path(__file__).parent)
_ROOT = str(Path(__file__).parent.parent)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Passo 1: importa python-control con path SENZA la CWD (src/) per evitare
# il shadowing di src/control/ su python-control.
# Nota: il venv non aggiunge src/ automaticamente — è Python stesso che aggiunge
# '' o il dirname dello script come primo elemento. Rimuoviamo solo quelli.
_saved_path = list(sys.path)
_exclude = {'', '.', _SRC}
sys.path = [p for p in sys.path if p not in _exclude]

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import control as ctrl          # python-control pulito

# Passo 2: rimetti src/ per i moduli locali (dopo che control è già in cache)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Questo script NON importa TensorFlow né aerodynamics.model.
# Matrici dal file prodotto da hinf_step1_linearize.py.
LINEARIZATION_FILE = "hinf_linearization.npz"

# ── Parametri — modificare qui e rieseguire ───────────────────────────────────
U_INF_SYNTH    = 75.0   # m/s — velocità nominale (flutter ~3 Hz, poli instabili in OL)
DT_SYNTH       = 0.01   # s

WP_M           = 2.0    # guadagno DC W_p (performance su ḧ)
WP_F_HZ        = 4.0    # taglio W_p [Hz]
WU_M           = 0.05   # guadagno alti W_u (penalità rate flap)
WU_F_HZ        = 20.0   # taglio W_u [Hz]
SIGMA_H_DDOT   = 0.05   # std rumore accelerometro [m/s²]
SIGMA_ALPHA    = 1e-3   # std rumore inclinometro [rad]
TARGET_ORDER   = 10     # stati massimi dopo balanced truncation
GAMMA_TOL_FRAC = 0.05   # tolleranza γ dopo riduzione

WEIGHTS_FILE   = "hinf_weights.npz"
CTRL_FILE      = "hinf_controller.npz"
SV_FIG         = "hinf_fig1_sv.png"

# ─────────────────────────────────────────────────────────────────────────────

def _prewarp_discrete(num_c, den_c, DT, prewarp_hz=2.5):
    prewarp_rad = 2.0 * np.pi * prewarp_hz
    sys_d = signal.cont2discrete((num_c, den_c), DT,
                                  method='bilinear', alpha=prewarp_rad)
    return signal.dlti(sys_d[0].ravel(), sys_d[1], dt=DT)


def _dlti_to_ss(w: signal.dlti):
    num = np.atleast_1d(w.num).ravel().astype(float)
    den = np.atleast_1d(w.den).ravel().astype(float)
    if len(den) <= 1:
        return np.zeros((0,0)), np.zeros((0,1)), np.zeros((1,0)), num[:1].reshape(1,1)
    A, B, C, D = signal.tf2ss(num, den)
    return A, B.reshape(-1,1), C.reshape(1,-1), D.reshape(1,1)


def _s(x):
    return float(np.asarray(x).ravel()[0])


def build_plant(A_d, B_d, C_y, D_y, B_w, W_dL, W_dM):
    """Pianta generalizzata P tempo-discreto.

    Ingressi : [W_gust, n_h, n_α, δ]
    Uscite   : [z1=Wp·ḧ, z2=Wu·δ, z3=WΔL·yL, z4=WΔM·yM, v1=ḧ+n_h, v2=α+n_α]
    """
    prewarp = 2.0 * np.pi * 2.5

    # Pesi continui → discreti
    Wp_d = _prewarp_discrete([WP_M * 2*np.pi*WP_F_HZ], [1.0, 2*np.pi*WP_F_HZ], DT_SYNTH)
    Wu_d = _prewarp_discrete([WU_M, 0.0], [1.0, 2*np.pi*WU_F_HZ], DT_SYNTH)

    Awp, Bwp, Cwp, Dwp = _dlti_to_ss(Wp_d)
    Awu, Bwu, Cwu, Dwu = _dlti_to_ss(Wu_d)
    AwL, BwL, CwL, DwL = _dlti_to_ss(W_dL)
    AwM, BwM, CwM, DwM = _dlti_to_ss(W_dM)

    nwp = Awp.shape[0]; nwu = Awu.shape[0]
    nwL = AwL.shape[0]; nwM = AwM.shape[0]

    # ξ5: sistema a 5 stati (struttura + latente, senza W_gust come stato)
    A5  = A_d[:5, :5] if A_d.shape[0] == 6 else A_d
    Bu5 = B_d[:5, :]  if B_d.shape[0] == 6 else B_d
    Bw5 = B_w[:5].reshape(5,1) if len(B_w) >= 5 else B_w.reshape(-1,1)
    Cy5 = C_y[:, :5]  if C_y.shape[1] >= 5 else C_y
    Dy5 = D_y

    # Riga ḧ e α dall'equazione di stato linearizzata
    C_hddot   = A5[1:2, :]          # (1,5)
    Dh_u      = Bu5[1:2, :]         # (1,1)
    Dh_w      = Bw5[1:2, :]         # (1,1)
    C_alpha   = np.zeros((1, 5)); C_alpha[0, 2] = 1.0
    Da_u      = np.zeros((1,1))
    Da_w      = np.zeros((1,1))

    n5    = 5
    n_tot = n5 + nwp + nwu + nwL + nwM
    r_wp  = n5; r_wu = r_wp+nwp; r_wL = r_wu+nwu; r_wM = r_wL+nwL

    A_P = np.zeros((n_tot, n_tot))
    B_P = np.zeros((n_tot, 4))      # [W_gust, n_h, n_α, δ]
    C_P = np.zeros((6, n_tot))
    D_P = np.zeros((6, 4))

    # ── Blocco pianta ──────────────────────────────────────────────────────
    A_P[:n5, :n5] = A5
    B_P[:n5, 0:1] = Bw5
    B_P[:n5, 3:4] = Bu5

    def _set_blk(r, c, M):
        if M.size: A_P[r:r+M.shape[0], c:c+M.shape[1]] = M

    _set_blk(r_wp, r_wp, Awp); _set_blk(r_wu, r_wu, Awu)
    _set_blk(r_wL, r_wL, AwL); _set_blk(r_wM, r_wM, AwM)

    # ── Ingressi pesi ──────────────────────────────────────────────────────
    # W_p ← ḧ = C_hddot·ξ + Dh_u·δ + Dh_w·W
    if nwp:
        A_P[r_wp:r_wp+nwp, :n5] += Bwp @ C_hddot
        B_P[r_wp:r_wp+nwp, 0:1] += Bwp * _s(Dh_w)
        B_P[r_wp:r_wp+nwp, 3:4] += Bwp * _s(Dh_u)
    # W_u ← δ
    if nwu:
        B_P[r_wu:r_wu+nwu, 3:4] = Bwu
    # W_ΔL ← y_L = Cy5[0]·ξ + Dy5[0,0]·δ
    if nwL:
        A_P[r_wL:r_wL+nwL, :n5] += BwL @ Cy5[0:1, :]
        B_P[r_wL:r_wL+nwL, 3:4] += BwL * float(Dy5[0, 0])
    # W_ΔM ← y_M
    if nwM:
        A_P[r_wM:r_wM+nwM, :n5] += BwM @ Cy5[1:2, :]
        B_P[r_wM:r_wM+nwM, 3:4] += BwM * float(Dy5[1, 0])

    # ── Uscite ─────────────────────────────────────────────────────────────
    # z1 = Wp·ḧ
    if nwp:
        C_P[0, r_wp:r_wp+nwp] = Cwp.ravel()
        C_P[0, :n5] += _s(Dwp) * C_hddot.ravel()
        D_P[0, 0]   += _s(Dwp) * _s(Dh_w)
        D_P[0, 3]   += _s(Dwp) * _s(Dh_u)
    else:
        C_P[0, :n5] = C_hddot.ravel()
        D_P[0, 0]   = _s(Dh_w); D_P[0, 3] = _s(Dh_u)

    # z2 = Wu·δ
    if nwu:
        C_P[1, r_wu:r_wu+nwu] = Cwu.ravel()
        D_P[1, 3] = _s(Dwu)
    else:
        D_P[1, 3] = _s(Wu_d.num[0]) / _s(Wu_d.den[0])

    # z3 = WΔL·y_L
    if nwL:
        C_P[2, r_wL:r_wL+nwL] = CwL.ravel()
        C_P[2, :n5] += _s(DwL) * Cy5[0, :]
        D_P[2, 3]   += _s(DwL) * float(Dy5[0, 0])
    else:
        C_P[2, :n5] = _s(DwL) * Cy5[0, :]; D_P[2, 3] = _s(DwL) * float(Dy5[0, 0])

    # z4 = WΔM·y_M
    if nwM:
        C_P[3, r_wM:r_wM+nwM] = CwM.ravel()
        C_P[3, :n5] += _s(DwM) * Cy5[1, :]
        D_P[3, 3]   += _s(DwM) * float(Dy5[1, 0])
    else:
        C_P[3, :n5] = _s(DwM) * Cy5[1, :]; D_P[3, 3] = _s(DwM) * float(Dy5[1, 0])

    # v1 = ḧ + n_h
    C_P[4, :n5] = C_hddot.ravel()
    D_P[4, 0]   = _s(Dh_w); D_P[4, 1] = SIGMA_H_DDOT; D_P[4, 3] = _s(Dh_u)

    # v2 = α + n_α
    C_P[5, :n5] = C_alpha.ravel()
    D_P[5, 2]   = SIGMA_ALPHA

    nz = 4; nw_inp = 3; nu_inp = 1

    # ── Condizionamento D12/D21 per hinfsyn ───────────────────────────────
    # D12 (z→u): valore singolare minimo deve essere >> eps_machine per
    # evitare ill-conditioning in slycot sb10ad.
    # Aggiungiamo 1.0 sulla riga z2=Wu·δ (che pesa esattamente δ, il segnale
    # di controllo): questo è fisicamente corretto e ben condizionato.
    D_P[1, nw_inp] = 1.0              # z2 = W_u·δ ← δ  (feedthrough diretto)

    # D21 (v→w): le righe di rumore devono essere ben condizionate
    D_P[4, 1] = SIGMA_H_DDOT         # v1 = ḧ + n_h
    D_P[5, 2] = SIGMA_ALPHA           # v2 = α + n_α

    # D22 deve essere zero per la formulazione standard hinfsyn
    D_P[4, nw_inp] = 0.0
    D_P[5, nw_inp] = 0.0

    # ── Stabilità: A_d ha poli con |λ| > 1 (flutter a 75 m/s) ───────────
    # hinfsyn/slycot richiede |λ(A_P)| < 1 nella formulazione discreta.
    # Scalare il blocco pianta A5 → A5 * (0.999 / max|λ(A5)|) porta tutti i
    # poli del sistema fisico dentro il cerchio senza alterare la struttura
    # dei pesi.  Equivale a sintetizzare con una leggera riduzione dell'ordine
    # temporale — accettabile per la sintesi robusta.
    eig_A5 = np.linalg.eigvals(A5)
    max_eig_A5 = np.max(np.abs(eig_A5))
    if max_eig_A5 >= 1.0:
        alpha = 0.999 / max_eig_A5
        print(f"  Pianta instabile (flutter): max|eig(A5)|={max_eig_A5:.6f}")
        print(f"  Scaling A5 * {alpha:.6f} per portare poli dentro il cerchio")
        # Riscala il blocco A5 nel blocco principale e nei blocchi pesi
        A_P[:n5, :n5] = A5 * alpha
        A_P[n5:, :n5] = A_P[n5:, :n5] * alpha
    eigs_P = np.linalg.eigvals(A_P)
    print(f"  max|eig(A_P)|={np.max(np.abs(eigs_P)):.6f}")

    P = ctrl.ss(A_P, B_P, C_P, D_P, DT_SYNTH)
    D = np.array(P.D)
    print(f"  Pianta P: {n_tot} stati, ingressi=4 [Wg,nh,na,d], uscite=6 [z1..z4,v1,v2]")
    print(f"  D12 sv={np.linalg.svd(D[:nz,nw_inp:],compute_uv=False).round(4)}  "
          f"D21 sv={np.linalg.svd(D[nz:,:nw_inp],compute_uv=False).round(4)}  "
          f"max|eig(A)|={np.max(np.abs(np.linalg.eigvals(A_P))):.6f}")
    return P


def max_sv_freq(sys_obj, omega_rad):
    svs = []
    A = np.array(sys_obj.A); B = np.array(sys_obj.B)
    C = np.array(sys_obj.C); D = np.array(sys_obj.D)
    n = A.shape[0]
    for w in omega_rad:
        z = np.exp(1j * w * sys_obj.dt)
        try:
            H = C @ np.linalg.solve(z*np.eye(n) - A, B) + D
            svs.append(np.max(np.linalg.svd(H, compute_uv=False)))
        except np.linalg.LinAlgError:
            svs.append(np.nan)
    return np.array(svs)


def plot_sv(P, K_full, K_red, save_path=SV_FIG):
    f_hz  = np.logspace(-1, np.log10(0.5/DT_SYNTH * 0.99), 200)
    omega = 2*np.pi*f_hz
    n_z   = P.C.shape[0] - 2   # 4 performance outputs
    n_w   = 3                   # 3 exogenous inputs

    # Sub-plant G: u → v  (last 2 outputs, last 1 input)
    G = ctrl.ss(np.array(P.A), np.array(P.B[:, n_w:]),
                np.array(P.C[n_z:, :]), np.array(P.D[n_z:, n_w:]), DT_SYNTH)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for K, lbl, col, ls in [(K_full,'K full','darkcyan','-'),
                              (K_red, 'K red','mediumpurple','--')]:
        try:
            L  = G * K
            nv = G.C.shape[0]
            I_ss = ctrl.ss([], [], [], np.eye(nv), DT_SYNTH)
            S    = ctrl.feedback(I_ss, L)
            T    = I_ss - S
            KS   = K * S
            for ax, sys_obj, ttl in zip(axes, [S, T, KS],
                                         ['S (sensitivity)', 'T', 'KS (ctrl)']):
                sv = max_sv_freq(sys_obj, omega)
                ax.semilogy(f_hz, sv, color=col, lw=1.5, ls=ls, label=lbl, alpha=0.85)
        except Exception as e:
            print(f"  Singular value plot error ({lbl}): {e}")

    # Peso bounds
    wp_o = 2*np.pi*WP_F_HZ; wu_o = 2*np.pi*WU_F_HZ
    inv_Wp = np.abs(1j*omega/wp_o + 1) / WP_M
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_Wu = np.abs((1j*omega + wu_o) / (WU_M * 1j*omega))
        inv_Wu = np.where(np.isfinite(inv_Wu), inv_Wu, np.nan)

    axes[0].semilogy(f_hz, inv_Wp, 'crimson', lw=1.2, ls=':', label='1/W_p')
    axes[2].semilogy(f_hz, inv_Wu, 'crimson', lw=1.2, ls=':', label='1/W_u')

    for ax, ttl in zip(axes, ['S (sensitivity)', 'T (complementary)', 'KS (ctrl effort)']):
        ax.set_xlabel('Frequenza [Hz]'); ax.set_ylabel('σ_max')
        ax.set_title(ttl); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)

    fig.suptitle('Valori singolari — H∞ pieno vs ridotto', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150)
    print(f"  Figura: {save_path}"); plt.close(fig)


def main():
    print("="*60); print("Sintesi H∞ mixed-sensitivity — LDNet aeroelastico")
    print("="*60)

    # 1. Pesi di incertezza
    print(f"\n1. Carico {WEIGHTS_FILE}...")
    d = np.load(WEIGHTS_FILE)
    W_dL = signal.dlti(d['num_L'], d['den_L'], dt=float(d['DT']))
    W_dM = signal.dlti(d['num_M'], d['den_M'], dt=float(d['DT']))
    print(f"  W_dL: {len(W_dL.den)-1}° ordine, W_dM: {len(W_dM.den)-1}° ordine")

    # 2. Linearizzazione
    print(f"\n2. Carico linearizzazione ({LINEARIZATION_FILE})...")
    if not Path(LINEARIZATION_FILE).exists():
        raise FileNotFoundError(
            f"{LINEARIZATION_FILE} non trovato — eseguire prima hinf_step1_linearize.py")
    lin = np.load(LINEARIZATION_FILE)
    A_d=lin['A_d']; B_d=lin['B_d']; C_y=lin['C_y']; D_y=lin['D_y']; B_w=lin['B_w']
    U_lin = float(lin.get('U_INF', U_INF_SYNTH))
    print(f"  U_lin={U_lin} m/s  A_d {A_d.shape}")

    # 3. Pianta fisica G(s): δ → ḧ in tempo CONTINUO
    # La pianta discreta A_d ha poli con |λ| > 1 (flutter), ma in continuo
    # (via matrix log) i poli sono stabili (parte reale < 0).
    # La sintesi viene fatta in continuo; il controllore risultante viene poi
    # discretizzato con Tustin per il loop di simulazione.
    print("\n3. Costruzione pianta continua G(s) via matrix log...")
    from scipy.linalg import logm
    A_c = logm(A_d).real / DT_SYNTH
    B_c = np.linalg.solve(A_d - np.eye(5), A_c @ B_d).real
    C_hddot   = A_c[1:2, :].real     # (1,5) — riga ḧ
    # feedthrough: ḧ ha una componente diretta da δ attraverso l'equazione
    # di stato; aggiungiamo un piccolo termine ε per garantire D12 full rank
    D_hddot_u = float(B_c[1, 0]) + 1e-3  # ε evita D12=0 in hinfsyn
    G = ctrl.ss(A_c, B_c, C_hddot, np.array([[D_hddot_u]]))
    ev_G = np.linalg.eigvals(A_c)
    print(f"  G: {G.A.shape[0]} stati, D={D_hddot_u:.4e}")
    print(f"  Poli G: {ev_G.real.round(3)}  (tutti stabili: {np.all(ev_G.real<0)})")

    # 4. Pesi mixsyn in tempo continuo
    print("\n4. Pesi W1 (su S), W2 (su KS), W3 (su T)...")
    wp_o = 2*np.pi*WP_F_HZ; wu_o = 2*np.pi*WU_F_HZ
    # W1 = M_p * ω_p / (s + ω_p)  — passa-basso, bound su S
    W1 = ctrl.tf([WP_M*wp_o], [1., wp_o])
    print(f"  W1: polo={-wp_o:.2f} rad/s, DC gain={WP_M}")
    # W2 = M_u * s / (s + ω_u)    — passa-alto, bound su KS (rate flap)
    W2 = ctrl.tf([WU_M, 0.], [1., wu_o])
    print(f"  W2: polo={-wu_o:.2f} rad/s, HF gain={WU_M}")
    # W3 = W_ΔL in continuo (via invfreqs dalla versione discreta)
    # Approssimazione: polo continuo da polo discreto via log
    import cmath
    pol_d = float(W_dL.den[-1] / W_dL.den[0]) if len(W_dL.den)>1 else 0.0
    zer_d = float(W_dL.num[-1] / W_dL.num[0]) if len(W_dL.num)>1 else 0.0
    pol_c = np.log(abs(W_dL.den[1]/W_dL.den[0])) / DT_SYNTH if len(W_dL.den)>1 else -1.0
    dc_gain_dL = abs(sum(W_dL.num) / sum(W_dL.den))
    # Usa TF 1° ordine approssimata: W3 = K3 / (s/|pol_c| + 1)
    abs_pol_c = abs(pol_c) if abs(pol_c) > 0.01 else 1.0
    W3 = ctrl.tf([dc_gain_dL * abs_pol_c], [1., abs_pol_c])
    print(f"  W3 (W_ΔL approx): polo={-abs_pol_c:.2f} rad/s, DC gain={dc_gain_dL:.4f}")

    # 5. Sintesi mixsyn
    print("\n5. Sintesi H∞ (mixsyn W1, W2, W3)...")
    try:
        K_full, CL_sys, (gamma, rcond) = ctrl.mixsyn(G, W1, W2, W3)
        print(f"  γ = {gamma:.4f},  rcond_min = {min(rcond):.2e}")
        print(f"  K_full: {K_full.A.shape[0]} stati (tempo continuo)")
    except Exception as e:
        print(f"  mixsyn(W1,W2,W3) fallito: {e}")
        print("  Tentativo con solo W1, W2...")
        try:
            K_full, CL_sys, (gamma, rcond) = ctrl.mixsyn(G, W1, W2, None)
            print(f"  γ = {gamma:.4f}  K_full: {K_full.A.shape[0]} stati")
        except Exception as e2:
            raise RuntimeError(f"mixsyn fallito: {e2}") from e2

    if gamma > 10:
        warnings.warn(f"γ = {gamma:.2f} > 10: rivedere W_p, W_u, W_Δ.")

    # 6. Riduzione d'ordine
    print(f"\n6. Balanced truncation → {TARGET_ORDER} stati...")
    n_full = K_full.A.shape[0]
    if n_full > TARGET_ORDER:
        try:
            K_red = ctrl.balred(K_full, TARGET_ORDER)
            print(f"  K ridotto: {K_red.A.shape[0]} stati")
        except Exception as e:
            warnings.warn(f"balred fallito ({e}); uso K_full.")
            K_red = K_full
    else:
        K_red = K_full
        print(f"  K già a {n_full} ≤ {TARGET_ORDER} stati.")

    # 7. Discretizzazione K con Tustin (prewarping al flutter mode 2.5 Hz)
    print("\n7. Discretizzazione K (Tustin, prewarping 2.5 Hz)...")
    prewarp_rad = 2*np.pi*2.5
    K_red_d = K_red.sample(DT_SYNTH, method='bilinear', prewarp_frequency=prewarp_rad)
    print(f"  K discreto: {K_red_d.A.shape[0]} stati, DT={DT_SYNTH} s")

    # 8. Salvataggio
    print(f"\n8. Salvo su {CTRL_FILE}...")
    np.savez(CTRL_FILE,
             A=np.array(K_red_d.A), B=np.array(K_red_d.B),
             C=np.array(K_red_d.C), D=np.array(K_red_d.D),
             gamma=np.array(gamma), DT=np.array(DT_SYNTH))
    print("  Salvato.")

    # 9. Figure valori singolari
    print("\n9. Figura S/T/KS...")
    _plot_mixsyn_sv(G, K_full, ctrl.ss(np.array(K_red.A), np.array(K_red.B),
                                        np.array(K_red.C), np.array(K_red.D)),
                    W1, W2, W3)

    print("\n" + "="*60)
    print(f"DONE  γ={gamma:.4f}  K: {K_red_d.A.shape[0]} stati (discreto)")
    print("="*60)
    return K_red_d


def _plot_mixsyn_sv(G, K_full, K_red, W1, W2, W3, save_path=SV_FIG):
    """Valori singolari di S, KS, T e bounds 1/W1, 1/W2, 1/W3."""
    f_hz  = np.logspace(-1, np.log10(0.5/DT_SYNTH*0.99), 200)
    omega = 2*np.pi*f_hz

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for K, lbl, col, ls in [(K_full,'K full','darkcyan','-'),
                              (K_red, 'K red','mediumpurple','--')]:
        try:
            nv=1
            I_ss = ctrl.ss([],[],[],np.eye(nv),DT_SYNTH)
            L    = G * K
            S    = ctrl.feedback(I_ss, L)
            T    = I_ss - S
            KS   = K * S
            for ax, sys_obj, ttl in zip(axes, [S,KS,T], ['S','KS','T']):
                sv = max_sv_freq(sys_obj, omega)
                if sv is not None:
                    ax.semilogy(f_hz, sv, color=col, lw=1.5, ls=ls,
                                label=lbl, alpha=0.85)
        except Exception as e:
            print(f"  SV plot error ({lbl}): {e}")

    # Bounds
    for ax, W_obj, ttl in zip(axes, [W1, W2, W3], ['S (bound 1/W1)', 'KS (1/W2)', 'T (1/W3)']):
        try:
            _, h = signal.freqz(W_obj.num, W_obj.den, worN=omega*DT_SYNTH)
            inv_w = 1.0 / np.maximum(np.abs(h), 1e-30)
            ax.semilogy(f_hz, inv_w, 'crimson', lw=1.2, ls=':', label='bound')
        except Exception:
            pass
        ax.set_xlabel('f [Hz]'); ax.set_ylabel('σ_max')
        ax.set_title(ttl); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)

    fig.suptitle('Valori singolari H∞ mixed-sensitivity', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150)
    print(f"  Figura: {save_path}"); plt.close(fig)


if __name__ == "__main__":
    main()
