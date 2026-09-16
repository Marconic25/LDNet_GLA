# Stima di completamento — campagna dataset v6 (146 run)

Documento di calibrazione. Va **ritarato** dopo le prime run reali: vedi
"Ricalibrazione" in fondo.

## 1. Modello di costo

Il costo di una run FSI si scompone in due termini indipendenti: il costo dei passi
fluidi (integrazione di `pimpleFoam`) e il costo delle finestre di accoppiamento
(scrittura di `wingMotion.dat`/`flapMotion.dat`, restart del solutore, lettura delle
forze, integrazione strutturale in Python).

Dati di calibrazione — `light/dagger_fom/NOTES.md`, stessa cella W30/Tg0.40,
`TEND=1.25 s`, `dt=7e-5`, 16 core, quindi `N_passi = 17 857` costante e solo
`N_finestre` variabile:

| `--window` | `N_finestre` | wall misurato |
|---:|---:|---:|
| 50 | 357 | 45 min |
| 29 | 616 | 73 min |
| 15 | 1 190 | 112 min |

Regressione ai minimi quadrati su `T = a·N_passi + b·N_finestre`:

```
a = 1.129e-3  min / passo fluido
b = 7.84e-2   min / finestra di accoppiamento
```

Residui del fit sui tre punti: `+3.1 / −4.6 / +1.4` min, cioè 4–6% — il modello a due
termini non è esatto, ma lo scarto è ben sotto la variabilità di nodo che si osserva
in produzione.

### Validazione su un punto indipendente

La regressione è tarata su run a `TEND=1.25` con controllore MPC. La si verifica su
un punto che **non** è entrato nel fit: una run v5 completa da 3 s, `window=50`,
`dt=7e-5`, in modalità `schedule` (job 24216, `recon/RECON_NOTES.md`).

```
N_passi    = 3.0 / 7e-5        = 42 857
N_finestre = 3.0 / 3.5e-3      =    857
T_pred     = 1.129e-3·42 857 + 7.84e-2·857 = 48.4 + 67.2 = 116 min = 1.9 h
T_misurato ≈ 2 h
```

Errore ~5%. Il modello regge, e il fatto che regga su una run `schedule` mentre è
tarato su run `mpc` dice che l'overhead del server MPC non è dominante.

> Nota su come si leggono i tempi PBS in questo progetto: la colonna `time` del
> poller è **CPU-time ≈ 16 × wall**, non elapsed. `RECON_NOTES.md` documenta un
> caso in cui questo ha fatto scambiare una run da 2 h per una da 18 h. Confrontare
> sempre elapsed con elapsed.

## 2. Applicazione a v6

Parametri nuovi: `dt = 3.16e-5`, `window = 29` → finestra di accoppiamento
`29 × 3.16e-5 = 9.164e-4 s`.

Run piena da 3 s:

```
N_passi    = 3.0 / 3.16e-5      = 94 937      (×2.215 vs v5)
N_finestre = 3.0 / 9.164e-4     =  3 274      (×3.82  vs v5)

T_CFD = 1.129e-3·94 937 + 7.84e-2·3 274
      = 107 + 257
      = 364 min = 6.1 h
```

Il termine dominante è ora quello delle **finestre**, non dei passi fluidi: 257 min
contro 107. È la conseguenza diretta di `Nwin=29`, ed è il motivo per cui il costo
non scala col solo rapporto dei `dt`.

## 3. Effetto del taglio del tempo finale

v5 girava 3.0 s fissi per tutte le run. La finestra di valutazione di ogni studio a
valle è `t ≤ Tg + 0.5`, quindi la coda non viene mai usata. Regole in
`design/gen_matrix.py`:

| Famiglia | `t_end` | durata media | run |
|---|---|---:|---:|
| A | `min(3.0, Tg + 0.9)` | 1.648 s | 30 |
| B | `1.8` s fisso | 1.800 s | 46 |
| Cc | `min(3.0, Tg + 1.2)` | 1.937 s | 70 |

```
tempo simulato totale (dalla matrice reale) = 268 s
                 v5   = 146 · 3.0           = 438 s
                                              → −39%
```

Famiglia B è l'unica a durata fissa. La regola v5 — stroke veloce al rate limit, poi
rilascio lento «sul resto della run» — è ben definita solo perché la run durava 3.0 s
fissi; derivare `t_end` dallo schedule la rende circolare. Rendere il rilascio
simmetrico al rate limit chiuderebbe il cerchio, ma collasserebbe 2/3 della famiglia
su un impulso da ~0.2 s, buttando via proprio il contenuto a bassa frequenza del
canale flap→carichi che la famiglia B esiste per generare. La run resta quindi a
`T_B_NOMINAL = 1.8 s`, con il rilascio che arriva a zero a 1.5 s e 0.3 s di decadimento
libero: lo stroke dura al massimo `15/20 = 0.75 s`, quindi il rilascio è più lento
dello stroke ovunque tranne che nell'angolo estremo della scatola LHS. Riferimento
strutturale: il modo di heave decade con `1/(ζ·ω_h) = 1.37 s`.

## 4. Totale campagna

Durata media per run 1.84 s, cioè 61% di 3 s. Overhead di staging (rsync del caso +
16 processor dir di checkpoint, cold start del container, `--finalize`
dell'estrattore) ~1.0 h medi, di cui una parte scala con la durata.

| | per run | × 146 |
|---|---:|---:|
| CFD | 3.7 h | 541 h |
| staging + reconstruct + estrazione | 1.0 h | 146 h |
| **wall-time di job** | **~4.7 h** | **~687 h** |

Tempo di calendario, con `max_user_run = 4`:

| Scenario | Corsie | Calendario |
|---|---:|---:|
| Ottimistico — 4 corsie piene, nessuna contesa | 4 | ~7 giorni |
| **Realistico — contesa con altri utenti, qualche rilancio** | ~3 | **~10 giorni** |
| Conservativo — catena 2-wide (`submit_ladder.sh`) | 2 | ~14 giorni |

**Stima da usare: ~10 giorni.**

Senza il taglio del tempo finale: 146 × 7.06 h = 1 031 h → ~14 giorni nello scenario
realistico. Il taglio vale ~4.8 giorni di calendario.

### Ripartizione per famiglia

| Famiglia | run | h/run | totale | note |
|---|---:|---:|---:|---|
| A | 30 | 4.33 | 130 h | le più economiche, `schedule` |
| B | 46 | 4.64 | 213 h | `schedule`, durata fissa 1.8 s |
| Cc — `prop` | 13 | 4.78 | 62 h | nessun container TF in parallelo |
| Cc — `mpc` | 57 | 4.94 | 282 h | +2 127 chiamate JSON al server per run |
| | **146** | | **~687 h** | |

Ricavata dai `t_end` effettivi di `design/run_matrix.csv`, non da medie assunte.

## 5. Disco — il vincolo che dimensiona il design

Il pattern v5 (`recon/cluster/field_run.pbs`: `purgeWrite 0`, tieni tutto,
`reconstructPar` finale, poi estrai) applicato a v6:

```
3 274 finestre × 16 processor × ~8 MB = 409 GB su /scratch_local per singola run
```

Non è sostenibile: già in v5, con 857 finestre, erano ~110 GB. Da qui le due leve in
`cluster/cosim_driver_final.py`:

- **`--field-every K`** — la cadenza di scrittura dei campi è disaccoppiata dalla
  finestra di accoppiamento, tarata per ~860 snapshot/run come in v5.
  `build_fields_h5.py` ricampiona comunque a `n_times=150`, quindi a valle non si
  perde nulla.
- **reconstruct + slice + purge dentro al loop** — ogni snapshot viene ricostruito,
  affettato a mezza apertura e cancellato subito; le time-dir dei processor tornano a
  essere purgate a `numeric[:-3]` (le 3 che servono al restart).

Risultato:

| | v5 pattern su v6 | v6 con estrazione incrementale |
|---|---:|---:|
| picco `/scratch_local` per run | ~409 GB | **~1–2 GB** |
| `/work` a fine campagna | — | **~17 GB** (114 MB × 146) |

Quota `/work` = 100 GB, `/home` = 10 GB. `/scratch_local` è **per-nodo** e viene
ripulito dopo ~30 giorni: run ed estrazione devono stare **nello stesso job PBS**,
mai in due job separati (in v5 questo errore ha distrutto tutti i dati di campo).

## 6. Walltime da richiedere

Cap della coda `cpu` = 48 h.

| Righe | walltime | margine sul modello |
|---|---|---|
| A, B, Cc-`prop` | `12:00:00` | ×2.4 |
| Cc-`mpc` | `24:00:00` | ×4.8 |

Margine ampio di proposito: il modello è tarato su un nodo scarico, e il repo
documenta un episodio di rallentamento **×35** per un job finito su un nodo già
occupato da altri 3–4 job. Verificare `pbsnodes -aSj` prima di lanciare.

## 7. Ricalibrazione (da fare)

Il modello è una previsione, non una misura. Prima del lotto completo:

1. **Smoke test**, 1 job, `t_end = 0.05 s`: verifica `deltaT`, Courant effettivo dal
   log (`Courant Number max:` ≤ 1.0), numero di finestre ≈ 55, picco di scratch,
   `field_times.npy` coerente con `--field-every`.
2. **Una run A completa**: si misura `a` e `b` reali e si riscrive questa tabella.
   Se l'elapsed devia di più del **30%** da 4.3 h, ricalcolare tutto qui prima di
   sottomettere le altre 145.

Valori misurati.

| Data | Job | Run | `t_end` | Finestre | Snapshot | Picco scratch | Courant max |
|---|---|---|---|---|---|---|---|
| 2026-09-03 | 31041 | sim_A_001_train | 0.05 | 55 | 1/55 | 60 MB | — |
| 2026-09-03 | 31044 | sim_A_001_train | 0.05 | 55 | 55/55 | 64 MB | 0.978 |
| 2026-09-03 | 31047 | sim_Cc_041_train | 0.05 | 55 | 55/55 | 77 MB | 1.518 |
| 2026-09-03 | 31049 | sim_Cc_041_train | 0.25 | 273 | 273/273 | — | **2.609** |
| 2026-09-03 | 31050 | sim_A_006_train | 0.50 | 546 | 546/546 | — | **1.303** |
| 2026-09-03 | 31061 | sim_B_046_test | 0.10 | 109 | 110/110 | — | **0.976** |

Job 31041 è la prima versione della guardia sull'ordinamento dei punti, troppo
stretta: rifiutava 54 snapshot su 55. Vedi §8.

Il picco di scratch si conferma **O(10 MB)**, tre ordini di grandezza sotto i
409 GB che il pattern v5 avrebbe richiesto: l'estrazione incrementale funziona.

## 8. Il Courant non è imposto — va misurato

`adjustTimeStep no` significa che è il **passo temporale** a essere fisso, non il
Courant. `maxCo` nel `controlDict` è inerte con quella impostazione: `Co` è un
output, `|u − u_g|·Δt/Δx`, che varia nello spazio e nel tempo. Niente lo vincola a
run time, quindi `run_sim.pbs` lo raschia dal log di `pimpleFoam` prima di
cancellare lo scratch e lo scrive in `sim_info.txt`.

Da dove viene `dt = 3.16e-5`: è il `dt` medio della riga `maxCo 1` dello sweep
adattivo in `validation/courant_mild/courant_sweep_report.txt` (`dt_avg =
3.159e-05`). Scelta fondata, ma quello sweep gira su un caso **mite, W0 = 5 m/s**,
dove la produzione v5 a `dt` fisso misura `Co_max = 2.304`.

Sulle raffiche vere il numero è un altro. Misurato su v6:

| caso | raffica | flap | Co_max v6 |
|---|---|---|---|
| **B, W0=0** | assente | **3.16° a 199.5 °/s** | **0.976** |
| A, W0=24.3, solo avvio | debole | fermo | 0.978 |
| **A, W0=47.1, oltre il picco** | **la più forte della campagna** | fermo | **1.303** |
| Cc, W0=30 + MPC, avvio | debole | 0.2° | 1.518 |
| **Cc, W0=30 + MPC, oltre il picco** | forte | **8.4°** | **2.609** |

**Il Courant nasce dalla combinazione raffica + deflessione, non da nessuna delle due
da sola.** I tre casi isolati stanno tutti sotto 1.31: la raffica più violenta della
campagna con flap fermo dà 1.30, e lo slew più veloce della campagna senza raffica dà
0.98, cioè il livello di base. È solo quando le due cose coesistono — raffica W30 e
flap a 8.4° — che si arriva a 2.61. L'effetto è superadditivo.

Una prima lettura attribuiva il fenomeno alla velocità di griglia `u_g` del flap in
movimento (nella ALE il termine convettivo è `u − u_g`). La famiglia B la smentisce:
ha lo slew più veloce di tutta la campagna e il Courant più basso. La spiegazione
compatibile con i tre punti è l'**incidenza efficace**: la raffica alza l'incidenza, il
flap deflesso aggiunge camber, e insieme producono un picco di suzione molto più
intenso vicino al ginocchio del flap — coerente con la separazione incipiente ad alta
incidenza istantanea che l'Appendice A si aspetta, e con la sensibilità al `dt` già
documentata a `W/U = 0.5` in `validation/README.md`.

**Il meccanismo non è però dimostrato.** Per separarlo servirebbe una sonda con la
stessa raffica della cella Cc (W0 = 30, Tg = 0.4) e flap bloccato: se resta ~1.3 è la
combinazione, se sale a ~2.6 è la raffica da sola. Non è stata eseguita perché non
condiziona la campagna, ma è la misura da fare prima di scrivere il meccanismo in tesi.

Due conseguenze.

**La tesi sottostima il Courant di v5.** `light/latex/appendixA.tex` riga 95 dichiara
`Co_max ≃ 2` a `Δt = 7e-5`: è il caso mite (2.304), non la raffica di produzione, per
cui `validation/README.md` documenta ~3.5 — e la cella Cc severa arriva a 5.78
equivalenti. La stessa riga 82 parla di *"the step changes imposed by the adaptive
Courant control"*, ma in produzione `adjustTimeStep` è `no`: non c'è alcun controllo
adattivo. Entrambe le frasi vanno corrette a prescindere da v6.

**`dt = 3.16e-5` non dà Co ≤ 1 sulle celle severe.** Dà 2.61. Resta un miglioramento
di 2.2× su v5 (5.78 → 2.61), ma il claim difendibile è "Courant più che dimezzato",
non "Courant unitario". Per Co = 1 sulla cella peggiore servirebbe `dt ≈ 1.21e-5`,
cioè ×2.6 sul costo: **687 h → 1793 h, ~25 giorni**. La terza via è differenziare il
passo per famiglia, se la sonda senza flap mostra che è il moto di mesh del flap a
dominare.

## 9. Ricalibrazione sulle run reali (2026-09-03)

Le sonde 31049 e 31050 danno due punti a lunghezze diverse, abbastanza per separare
l'overhead fisso da quello che scala col numero di snapshot:

| job | run | `t_end` | finestre | CFD previsto | misurato | overhead |
|---|---|---:|---:|---:|---:|---:|
| 31049 | Cc + MPC | 0.25 | 273 | 30.3 min | 47 min | 16.7 min |
| 31050 | A | 0.50 | 546 | 60.6 min | 86 min | 25.4 min |

```
overhead = 8.0 min fissi  +  1.91 s per snapshot
```

Gli 8 minuti fissi sono rsync del caso, le 16 processor dir di checkpoint e il cold
start del container; l'1.91 s per snapshot è `reconstructPar -time` + slice + purge,
cioè il prezzo dell'estrazione incrementale. Il termine CFD del modello originale non
si tocca: era tarato su dati indipendenti e i due punti lo confermano.

Campagna ritarata:

| gruppo | n | h/run | totale |
|---|---:|---:|---:|
| A | 30 | 3.91 | 117 h |
| B | 46 | 4.29 | 197 h |
| Cc — `prop` | 13 | 4.38 | 57 h |
| Cc — `mpc` | 57 | 4.54 | 259 h |
| **totale** | **146** | **4.32** | **630 h** |

**6.6 giorni** a 4 corsie, **8.8** a 3, **13.1** a 2. Contro le 687 h stimate a priori:
**−8%**, dentro il gate del 30% fissato nel piano. La stima iniziale era conservativa
perché assumeva 60 min piatti di overhead contro i ~25 reali.

## 10. Throughput misurato in campagna (2026-09-03)

Prime 4 run di famiglia A, una per nodo su cpu02–05, elapsed 3h44m a 1538 finestre:

```
7.1 finestre/minuto        (famiglia A, nessun MPC, un job per nodo)
```

Le sonde danno lo stesso ritmo: 31050 (A, 546 finestre in 78 min al netto del setup)
7.0/min, 31049 (Cc con MPC, 273 in 39 min) 7.0/min. **L'overhead dell'MPC sul ritmo
per finestra è trascurabile**, al contrario di quanto suggeriva il modello tarato su
`dagger_fom`.

| gruppo | finestre medie | h/run | totale |
|---|---:|---:|---:|
| A | 1 798 | 4.35 | 131 h |
| B | 1 964 | 4.74 | 218 h |
| Cc — `prop` | 2 039 | 4.92 | 64 h |
| Cc — `mpc` | 2 128 | 5.13 | 292 h |
| **totale** | | | **~705 h** |

**~7.3 giorni** con 4 corsie su nodi dedicati. Il modello calibrato al §9 dava 630 h:
la misura è il 12% sopra, quindi il modello regge.

> Trappola in cui sono caduto: un tick del loop era partito su messaggio dell'utente
> e non al risveglio programmato, e avevo dedotto l'elapsed dalla schedulazione invece
> di leggerlo. Ne era uscito un throughput di 32.7 finestre/minuto e una stima di 3.5
> giorni, sbagliata di 4.6×. L'elapsed va letto da `qstat`, mai dedotto.

## 11. Contesa di nodo — misurata, non ipotizzata

Al primo lancio lo scheduler ha impacchettato due job su cpu01 (11 core liberi su 112,
4 job di altri utenti) lasciando cpu03/04/05 completamente vuoti. Dopo 25 minuti:

| nodo | core liberi | finestre completate |
|---|---|---|
| cpu01 | 11/112 | **3** |
| cpu02 | 80/112 | **180** |

**60×.** E non è lentezza recuperabile: a 3 finestre ogni 25 minuti una run completa
richiede ~208 h e sbatte contro il walltime di 12 h. Tutti i job finiti su nodi carichi
sarebbero morti.

Da qui il pinning corsia→nodo in `submit_campaign.sh` (`--nodes`, disattivabile con
`--no-pin`), con preflight che stampa i core liberi di ogni nodo bersaglio prima di
sottomettere. Una corsia per nodo, 16 core su 112: non monopolizza il cluster.

Nota secondaria ma utile: anche **due job miei sullo stesso nodo da 112 core** si
ostacolano a vicenda. Con 16 core ciascuno il collo di bottiglia non sono i core, è la
banda di memoria — il pinning serve anche in assenza di altri utenti.

## 12. Margine reale della guardia sull'ordinamento

`guard_max_farfield_disp` misurato su 272 snapshot della run Cc: **0.0204 corde**,
contro `ANCHOR_TOL = 0.05`. La guardia funziona, ma il commento nel codice prevedeva
uno spostamento di campo lontano O(1e-3): il valore vero è **20 volte più grande** e
il margine è 2.5×, non il divario ampio che il commento suggeriva. Discrimina ancora
in modo netto — una permutazione sposterebbe le ancore di O(1) corda su una finestra
di crop 5×2 — ma se il crop venisse allargato o il moto del flap crescesse, la
tolleranza andrebbe rivista sulla base di questo numero, non della stima.


## 13. Il ripple a 109 Hz sui carichi — artefatto dello schema di accoppiamento

**Risolto 2026-09-04.** I carichi `F_y` e `M_z` di v6 portano una sinusoide pulita a
~109 Hz che in v5 non c'era. È un **artefatto numerico**, non fisica.

### La prova

| run | W0 | picco | finestre per periodo | corr. con sinusoide |
|---|---:|---:|---:|---:|
| sim_A_001_train | 24.3 | 109.63 Hz | 9.954 | +1.000 |
| sim_A_002_train | — | 108.92 Hz | 10.019 | +0.998 |
| sim_A_004_train | — | 109.66 Hz | 9.951 | +0.998 |
| sim_A_006_train | 47.1 | 109.34 Hz | 9.981 | +0.997 |

Frequenza di accoppiamento `1/(29·3.16e-5) = 1091.23 Hz`; la sua **decima
subarmonica è 109.12 Hz**. Il picco misurato cade a 0.02 Hz da lì, cioè entro zero
bin di risoluzione, e il rapporto finestre/periodo è 10.00 su tutte e quattro le run.

Quattro raffiche diverse danno la stessa identica frequenza: una frequenza fisica
varierebbe con le condizioni, questa è fissata dallo schema. La forma d'onda è una
sinusoide quasi perfetta (correlazione +0.997..+1.000, seconda armonica al 6%, terza
e oltre a zero) — un modo lineare, non una struttura di flusso.

### Come ci sono arrivato, e le ipotesi sbagliate lungo la strada

Vale la pena registrarle perché tre erano plausibili e tutte e tre false.

1. **Distacco di vortici.** Smentito dal profilo spaziale: la pressione superficiale
   porta il segnale forte a metà corda (~20%) e debole al bordo d'uscita (~3%), e sei
   sonde valide in scia non lo vedono affatto. Una scia vorticosa farebbe l'opposto.
2. **Artefatto alla frequenza di accoppiamento.** Cercata energia a 1091 Hz: 0.00%.
   Conclusione affrettata "non è l'accoppiamento" — l'errore è stato non cercare le
   **subarmoniche**.
3. **Arrotondamento di `timePrecision 6`.** Il passo `3.16e-5` non è multiplo di 1e-5
   mentre il `7e-5` di v5 lo è, quindi sembrava che i nomi delle time-dir di v6
   venissero arrotondati. Falso: i tempi scritti hanno jitter dello 0.013%.

Due difetti di metodo hanno allungato l'indagine e sono documentati in §14.

### Perché v6 sì e v5 no

L'instabilità dello schema partizionato loose è debole e compete con la dissipazione
numerica dell'Eulero implicito, che scala col passo. A ~109 Hz lo smorzamento per
ciclo è ~14% a `dt=7e-5` e ~6.7% a `dt=3.16e-5`. v5 la teneva sotto soglia; v6,
dimezzando il passo, dimezza anche la dissipazione e il modo cresce fino a saturare.

Nota: in v5 la stessa subarmonica sarebbe caduta a `285.7/10 = 28.6 Hz`, cioè vicino
al doppio del modo di beccheggio (14.46 Hz) — se mai si risvegliasse lì sarebbe molto
più insidiosa che a 109 Hz, perché indistinguibile dalla dinamica strutturale.

### Cosa farne

L'ampiezza è ~12 N su ~100 N di media, cioè ~12% dell'escursione di raffica che il
dataset deve insegnare. Non trascurabile, ma è una riga spettrale stretta a frequenza
nota.

- **Filtrare i carichi prima del training.** Il ROM gira a passi di 0.002 s (500 Hz):
  a 109 Hz avrebbe 4.5 campioni per ciclo e non può rappresentarlo comunque. Un
  passa-basso lo rimuove senza toccare nulla di ciò che il modello può imparare.
  Più economico, non richiede di rigenerare niente.
- **Sotto-iterazioni per finestra** (accoppiamento strong). È il fix corretto sul
  piano fisico, ma costa e richiede di rifare la campagna.
- **Cambiare la finestra non serve**: l'artefatto è agganciato alla frequenza di
  accoppiamento, quindi si sposterebbe soltanto.

Decisione da prendere prima del training, non prima della fine della campagna: i dati
grezzi restano validi e filtrabili a posteriori.

## 14. Due difetti di metodo da non ripetere

**FFT su campionamento non uniforme.** `structural_trajectory.csv` è scritto
dall'integratore strutturale e alterna passi da 4 e 5 step CFD (uno lungo per
finestra). Una `rfft` con `dt` medio deforma il segnale. Verificare sempre
`np.diff(t)` prima, e in caso incrociare con Lomb-Scargle, che non assume
uniformità.

**Sonde di campo non validate.** `argmin` sulla distanza restituisce *sempre* un
punto: per una posizione dentro il profilo restituisce un punto di superficie con
`U = 0` per no-slip. Ne era uscito un `Uy rms = 0.0000` letto come "nessun distacco".
Controllare sempre la distanza dal punto richiesto e scartare le sonde troppo lontane
o identicamente nulle.

**Terzo, di processo:** un CSV residuo del template `cosim_main` è stato scambiato per
il risultato di un esperimento fallito. Corretto in `run_sim.pbs`, che ora lo cancella
subito dopo lo staging.


## 15. Il ripple NON era filtrabile: a finestra 29 la soluzione era sbagliata

**Correzione a §13, 2026-09-04.** In §13 avevo concluso che il ripple a 109 Hz fosse
«una riga spettrale stretta a frequenza nota» e raccomandato di **filtrare i carichi
prima del training**, dato che il ROM a 500 Hz non può rappresentarla comunque.

Quel consiglio era pericoloso. Sovrapponendo `F_y` della stessa run alle due finestre:

| | `F_y` primo campione | media pre-raffica |
|---|---|---|
| v5 (campagna validata) | 164.53 N | **164.33 ± 0.21 N** |
| v6 finestra 63 | 164.86 N | **165.30 ± 0.32 N** |
| v6 finestra 29 | 132.96 N | **104.21 ± 14.56 N** |

A finestra 29 il **trim è sbagliato del 37%** e la dispersione è 70 volte maggiore.
La differenza è già presente al primo campione, a `t = 6.3e-5 s`, con condizioni
iniziali identiche: non è un errore che si accumula, è sbagliato dall'istante zero.
Anche il resto è fuori scala — picco di `F_y` 203 contro 298 N, `h` 8.0 contro
12.1 mm, `α` 0.112 contro 0.180°.

L'instabilità non aggiungeva un'oscillazione sopra la fisica corretta: **sostituiva
la fisica**. Filtrare avrebbe prodotto un dataset visivamente pulito con carichi
falsi del 37%, e il LDNet sarebbe stato addestrato su quelli.

Finestra 63 riproduce il trim di v5 entro lo 0.6% — conferma indipendente che è
quella la configurazione giusta, non un compromesso.

### Perché l'analisi spettrale non l'ha visto

Guardavo l'energia nella banda 105–115 Hz, cioè una quantità **relativa**, e mai i
valori assoluti contro un riferimento. Tutte le metriche di §13 (rapporti di banda,
correlazione con una sinusoide, finestre per periodo) sono invarianti rispetto a un
errore sul livello medio. Sovrapporre le due tracce ha impiegato dieci secondi e ha
mostrato subito quello che tre giorni di spettri non avevano mostrato.

**Regola operativa**: prima di caratterizzare un artefatto, confrontare il segnale
con un riferimento validato in valore assoluto. Se il livello non torna, la forma
non conta.
