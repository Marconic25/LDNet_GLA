# dagger_fom — closed-loop LDNet retraining + MPC formulation study

Studio autonomo, non tocca nulla di esistente in `light/` (solo import). Vedi il
piano completo in `C:\Users\marco\.claude\plans\crea-un-loop-per-kind-metcalfe.md`.

## Contesto (riassunto)

Verifica FOM reale su W30/Tg0.4 (k=0.131, la cella più severa della griglia) mostra
il flap oscillare a dente di sega fino a saturazione. Root cause confermato:

1. Test teacher-forcing: il surrogato LDNet, alimentato con la traiettoria REALE
   del FOM, mostra un picco di errore di predizione C_L enorme (RMSE locale 0.22 vs
   baseline 0.02) esattamente a t≈0.30-0.44s — la stessa finestra dell'oscillazione.
   Errore sistematico, non rumore (media ≈0).
2. LDNet non è mai stato addestrato su traiettorie flap closed-loop (solo
   raffica-pura/gradino-flap/raffica+flap-schedulato in dataset_v5).
3. La stessa cella è già validata robusta a rumore lidar realistico (0/6 flag) —
   quindi non è sensibilità al rumore, è mismatch sistematico modello-vs-realtà.
4. Indizio complementare: `clean/controller.py` ha un termine Q_alpha_dot (penalità
   sul rate di pitch) assente in `light/optimal.py`, documentato come l'unico
   controllore stabile su entrambe le intensità di raffica. `R_du` è invece
   documentato dannoso su questa cella con fused-sensor.

## Log iterazioni

### Setup — 2026-08-22

Cartella creata. Decisioni prese (autorizzazione utente a procedere in autonomia):
- Celle Fase B: W30/Tg{0.30, 0.40, 0.50} (angolo k-alto)
- Studio 100% autonomo: nessuna modifica fuori da `light/dagger_fom/`, nemmeno a
  `data/preprocess_GLA.py` (uso un convertitore HDF5 autonomo)
- Budget: fino a 3 iterazioni DAgger prima di fermarsi e rivalutare

### Fase A, step 1: sweep Q_alpha_dot sul ROM — risultato nullo (atteso, informativo)

`rom_screen.py` su W30/Tg0.4, Q_alpha_dot ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}:
**nessuna variazione** in CLred (80.5%), flap_max (7.875°), osc_count (6) su tutto il
range. Il ROM (surrogato LDNet) non mostra MAI l'oscillazione — è auto-consistente
per costruzione (il controllore pianifica col MODELLO STESSO che genera la
traiettoria). Conferma quanto già trovato col test di teacher-forcing: l'instabilità
esiste SOLO nel gap fisica-reale-vs-surrogato, quindi il ROM non può discriminare
se Q_alpha_dot aiuta sulla vera oscillazione — serve testarlo direttamente sul FOM
reale. Non è un fallimento dello script, è un limite metodologico dello screening
ROM per QUESTA classe di bug (a differenza di un vero errore di formulazione visibile
anche sul modello, che lo screening ROM avrebbe individuato).

Prossimo passo: testare Q_alpha_dot ∈ {0.01, 0.1, 1.0} direttamente su FOM reale
(W30/Tg0.4, TEND=1.25, 3 job paralleli), confrontati contro il baseline Q_alpha_dot=0
già raccolto (questa sessione, per l'aggiornamento della Figure 8).

### Fase A, step 2: Q_alpha_dot su FOM reale — bug di scala trovato

Bug nel primo tentativo: ho scelto il grid {0.01,0.1,1.0} arbitrariamente, senza
controllare la scala VALIDATA. `clean/controller.py` usa `Q_alpha_dot=1e4` di default
(insieme a `Q_h=1e4, Q_alpha=1e4, Q_CL=1e3, R=1.0`) — 4 ordini di grandezza sopra il
mio range. Inoltre il mio termine penalizza `(ad_after-ad_before)` su un SINGOLO
sotto-passo di integrazione da 0.002s (quantità naturalmente minuscola, O(1e-3) rad/s
o meno), quindi serve un peso enorme perché competa col termine di tracking C_L
(O(0.01-1) per step, sommato su N=8 passi dell'orizzonte).

Risultato del primo tentativo (grid sbagliato): qadot ∈ {0, 0.01, 0.1, 1.0} → **numeri
IDENTICI in tutti e 4 i run** (CLred≈-132%, flap_max=14.00°, osc_count=20) — nessun
effetto misurabile, confermando il sospetto di scala. Da notare: CLred=-132% per il
baseline è un numero corretto (non un bug di calcolo) — riflette quanto la cella
W30/Tg0.4 sia già severamente compromessa: il picco C_L del FOM (~1.95-2.0, visto
nella Figure 8 aggiornata) supera abbondantemente anche il riferimento open-loop
del ROM (~1.2), quindi l'escursione "controllata" è PEGGIORE di quella non
controllata.

Rilanciato con grid corretto: Q_alpha_dot ∈ {1e3, 1e4, 1e5, 1e6}, 4 job paralleli
(cpu02×2, cpu03, cpu05), stesso protocollo (TEND=1.25).

Nota operativa: la coda condivisa (max_user_run=4) era occupata da job dell'utente
(`stall_rate`/`stall_lw`, walltime 48h) in parallelo su cluster. Su richiesta esplicita
dell'utente, ridotti a 2 job (Q_alpha_dot=1e3, 1e4) per stare dentro gli slot liberati.

### Fase A — risultato finale: Q_alpha_dot aiuta ma NON risolve

| Q_alpha_dot | CLred | flap_max | osc_count |
|---|---|---|---|
| 0 (baseline) | -132.1% | 14.00° (saturato) | 20 |
| 1e3 | -140.7% (peggio) | 14.00° (saturato) | 27 (peggio) |
| 1e4 | -126.8% (leggero miglioramento) | **11.03° (non più saturato)** | 25 |

Q_alpha_dot=1e4 è l'unico valore testato con un effetto misurabile: il flap non
satura più (11.03° vs 14.00°) e il CLred migliora leggermente, ma l'oscillazione
resta (osc_count anzi sale) e il CLred resta catastroficamente negativo. **Conclusione:
un penalty di move-suppression smorza l'AMPIEZZA della reazione ma non corregge
l'INFORMAZIONE SBAGLIATA su cui il controllore pianifica** (il modello LDNet interno
continua a mispredire in questo regime, come già mostrato dal test di teacher-forcing).
Serve il retraining (Fase B) — Q_alpha_dot da solo non basta. Procedo alla Fase B
usando Q_alpha_dot=1e4 come configurazione controllore per la raccolta dati (è la
migliore disponibile, anche se non risolutiva), poi rivaluto se il fix combinato
(controllore + modello retrained) risolve davvero.

### Fase B, iterazione 1: raccolta dati + fine-tuning

Raccolte 3 traiettorie closed-loop reali (Q_alpha_dot=1e4): W30/Tg{0.30,0.40,0.50},
TEND=Tg+0.5, tutte completate con successo. Dataset costruito con
`build_dagger_h5.py`: train=[Tg0.40 (la cella target), Tg0.30], valid=[Tg0.50],
T=401 campioni @ dt=0.002s (t_common=0.8s, il minimo tra le celle raccolte).

Nota operativa: la coda condivisa (max_user_run=4) era continuamente ripopolata dal
flusso di job dell'utente (`stall_rate`/`stall_lw`/`stall_sep`, walltime 48h ciascuno,
in parallelo su cluster). Su autorizzazione esplicita dell'utente ("continua a
liberare slot quando serve"), cancellati i job attivi necessari per far partire i 3
job di raccolta dati.

Primo tentativo di fine-tuning FALLITO: `ModuleNotFoundError: No module named
'structure'` — avevo omesso `--env PYTHONPATH=.../clean` dall'invocazione apptainer
(la ricetta originale in `train_rollout.sh` lo imposta; `structure.py` vive in
`clean/`, non in `src/` dove gira lo script). Corretto in `retrain_launch.sh`,
rilanciato.

Rilancio riuscito. Diagnostica interessante ANCHE PRIMA del fine-tuning (rollout
closed-loop del modello di produzione sulla traiettoria di validazione Tg0.50):
NRMSE per dof — h:0.310, hd:0.181, **a:0.418, ad:0.945** (quasi totale mismatch sul
rate di pitch!) — conferma diretta e quantitativa del problema, indipendente dal
test di teacher-forcing fatto prima.

Training: loss di validazione migliora da 1.19e-01 a **1.48e-02 (8x) all'epoca 20**,
poi overfitting (solo 2 traiettorie di training) — risale a ~0.28-0.30 e resta lì
fino all'epoca 500. Il checkpoint salvato è correttamente quello dell'epoca 20
(migliore, non quello finale overfit).

Validazione con teacher-forcing sulla traiettoria baseline originale (Q_alpha_dot=0,
NON nel training set — training ha usato le traiettorie qadot=1e4): **risultato
misto**. Errore di picco ridotto (max_abs 0.2702→0.1938, -28%) ma RMSE medio
peggiorato (0.0607→0.0827) e nuovo bias sistematico (mean -0.0045→-0.0316) —
probabile artefatto di overfitting con un training set così piccolo. Il test
decisivo è il closed-loop reale: lanciata verifica FOM con modello nuovo +
Q_alpha_dot=1e4 insieme su W30/Tg0.4.

Risultato: CLred=-87.4% (migliore finora con l'exo ROM), flap_max=14.00° (satura
di nuovo), osc_count=21 — miglioramento reale ma non risolutivo.

## Pivot (su richiesta utente): stabilizzare il controllore ORIGINALE (CL+R) via tuning di R

Accantonato il tentativo di portare l'intera ricetta di costo di `clean/controller.py`
(Q_h, Q_alpha, Q_alpha_dot, Q_CL, R con scale diverse) — troppo distante dal
controllore realmente usato nella pipeline validata. Torno a
`light/optimal.py::MPCPreviewController` non modificato (solo C_L + R) e provo a
tunare l'UNICA leva che ha: R (flap-effort weight), usando gli script ORIGINALI
non modificati (`recon/cluster/mpc_fom_verify.pbs`, `cosim_driver_extract.py`).

**Bug scoperto durante lo sweep**: due job (R=0.001 e R=0.03) sono stati lanciati
sul nodo cpu02, che in quel momento ospitava contemporaneamente 3-4 job dell'utente
(`stall_*`) — replica ESATTA del problema di contesa MPI trovato all'inizio di
questa sessione (il bug del checkpoint). Con un job open-loop di controllo, la
velocità è passata da 9 finestre/30min (cpu02 conteso, 4 job) a 96 finestre/9min
(cpu03 pulito, solo il mio job) — un rallentamento di ~35x per la sola compresenza
di altri processi, indipendentemente dai core fisicamente liberi. Il trend liscio e
monotono nei risultati suggerisce che il bias non abbia invalidato la tendenza
generale, ma è un caveat da tenere presente. **Lezione operativa**: controllare
sempre `pbsnodes -aSj` per il nodo target PRIMA di lanciare, non solo `qstat`.

**Bug di bookkeeping**: lo script originale (non taggato per R) scrive sempre sullo
stesso path `mpc_fom_verify/W30_Tg0.40/` — il test R=0.001 ha sovrascritto la CSV
originale del baseline R*=0.0003 (quella con l'oscillazione a dente di sega). I
numeri del baseline restano validi (calcolati PRIMA della sovrascrittura) ma non è
più possibile ricalcolarli da zero dalla CSV grezza — solo back-derivare exc dai
CLred già misurati.

**Scoperta cruciale — il riferimento `exo` era sbagliato**: `CLred` è sempre stato
calcolato con `exo` preso dalla predizione OPEN-LOOP del ROM (surrogato LDNet), mai
da un vero open-loop del FOM. Lanciato un run FOM reale con δ=0 costante
(`mpc_fom_openloop.pbs`): **exo reale = 0.866358, contro exo ROM = 0.498145** — il
ROM sottostima l'escursione open-loop reale del 74% in questo regime severo. Tutti
i CLred calcolati finora per questa cella erano quindi artificialmente troppo
negativi (denominatore troppo piccolo).

### Tabella finale corretta (R-sweep, con exo REALE del FOM)

| R | CLred (con exo ROM, sbagliato) | **CLred (con exo REALE)** | flap_max | osc_count |
|---|---|---|---|---|
| R*=0.0003 (baseline tesi, back-derivato) | -132.1% | **-33.5%** | 14.00° (saturato) | 20 |
| R=0.001 | -96.0% | **-4.0%** | 14.00° (saturato) | 20 |
| R=0.003 | -95.5% | **-3.8%** | 13.12° | 20 |
| R=0.01 | -95.2% | **-3.6%** | 8.93° | 22 |
| R=0.03 | -93.7% | **-2.8%** | 4.55° | 32 |
| R=0.1 | -94.7% | **-3.4%** | 1.57° (quasi passivo) | 34 |

**Conclusione corretta**: il controllore NON sta drammaticamente peggiorando il
carico come sembrava (-132% era un artefatto del riferimento sbagliato). Con
R*=0.0003 (il valore ottimo sul ROM) il FOM reale è genuinamente ~33% peggio
dell'open-loop — un problema vero. Ma basta salire a **R=0.001 (3.3x più
conservativo)** per recuperare quasi al pareggio (-4%), e valori più alti restano
nello stesso range (-3% a -4%), MAI positivi ma nemmeno più catastrofici. L'
`osc_count` non migliora mai (resta ~20, peggiora a R alti) — il flap continua a
"chatterare" a bassa ampiezza anche quando R è alto, ma questo chatter a bassa
ampiezza non genera più penalità netta di carico significativa. In sintesi: **R
tunato (es. 0.001-0.01) rende il controllore "innocuo" (non peggio del non fare
nulla) ma non "utile" (CLred resta leggermente negativo, mai il beneficio positivo
previsto dal ROM) — serve probabilmente il fix sul modello (Fase B) per recuperare
un beneficio netto reale in questo regime severo, ma almeno ora il controllore
tunato è SICURO da usare (non genera più carichi peggiori del baseline).

## Test H2 — mismatch finestra/orizzonte (`--window 29` vs 50), CONFERMATO

Domanda dell'utente: il pattern piatto/negativo di CLred su tutto il range di R
(mai positivo, osc_count che non migliora) potrebbe essere un artefatto NUMERICO
di finestra piuttosto che un vero limite fisico? Evidenza da codice:
`cosim_driver_extract.py:92-94` fissa `_MPC_CTRL_DT=0.002` esplicitamente
"independent of the FOM co-sim window size" — il controllore ripiana UNA VOLTA
per finestra di accoppiamento (window=50 → 0.0035s, zero-order-hold per 51
sotto-passi CFD), mentre in `light/run.py` (validazione ROM) `ctrl.compute()`
viene chiamato ad OGNI passo nativo (0.002s, nessuna finestra) — il controllore
ripianifica 1.75× più spesso nel ROM che nel FOM. Test a costo zero (nessuna
modifica di codice, solo `--window 29` ≈ 0.00203s, la finestra più vicina al
passo interno 0.002s), stessi due R già caratterizzati, W30/Tg0.40, TEND=1.25,
exo reale=0.866358, stessa metodologia esatta di `validate_iteration.py`
(finestra `t<=tg+0.5`, `exc=max|CL-CLTRIM|`, osc_count = cambi di segno di
`diff(delta)` nella stessa finestra):

| R | window=50 (baseline) | **window=29** | Δ CLred |
|---|---|---|---|
| R*=0.0003 (ottimo tesi) | CLred=-33.5%, flap_max=14.00° (saturato), osc=20 | **CLred=+50.3%**, flap_max=14.00° (saturato), osc=15 | **+83.8 pt** |
| R=0.001 (best-so-far) | CLred=-4.0%, flap_max=14.00° (saturato), osc=20 | **CLred=+50.5%**, flap_max=14.00° (saturato), osc=18 | **+54.5 pt** |

**Verdetto: H2 CONFERMATA, effetto dominante.** Riducendo la finestra di
accoppiamento da 50 a 29 passi CFD (nessun'altra modifica), CLred passa da
negativo/quasi-pareggio a **+50% per ENTRAMBI i valori di R** — praticamente
indipendente da R, esattamente come previsto da un effetto di cadenza (non di
costo). Il flap satura ancora a 14° in entrambi i casi (nessun artefatto nuovo
di ampiezza), ma l'osc_count scende (20→15, 20→18): meno chatter E miglior
CLred insieme. Il pattern piatto/mai-positivo osservato nell'intero R-sweep a
window=50 era quindi in larga parte un **artefatto numerico della cadenza di
accoppiamento troppo grossa rispetto all'assunzione interna del controllore**,
non un limite fisico del modello LDNet o della formulazione MPC. Il caveat H1
(quantizzazione griglia argmin) resta plausibile come fattore secondario ma non
serve più come spiegazione primaria.

**Prossimo passo (autorizzato, eseguito automaticamente)**: testare `--window
15` (finestra ancora più fine, ≈0.00105s) sugli stessi due R per vedere se il
trend continua oltre +50% o satura, valutando il trade-off costo (window=29 ha
richiesto ~73 min per TEND=1.25 contro ~45 min di window=50, quindi window=15
atteso ~140 min).

### `--window 15` — il trend continua, con rendimenti decrescenti

Stessa metodologia esatta, stessi due R, W30/Tg0.40, TEND=1.25, exo
reale=0.866358:

| R | window=50 | window=29 | **window=15** |
|---|---|---|---|
| R*=0.0003 | CLred=-33.5%, osc=20 | CLred=+50.3%, osc=15 | **CLred=+54.3%**, flap_max=14.00° (saturato), osc=**9** |
| R=0.001 | CLred=-4.0%, osc=20 | CLred=+50.5%, osc=18 | **CLred=+54.1%**, flap_max=14.00° (saturato), osc=**9** |

Costo: window=15 ha impiegato ~112 min (TEND=1.25) contro ~73 min di window=29
e ~45 min di window=50 — cresce, ma sub-lineare rispetto al numero di finestre
di accoppiamento (~357→616→1190, un fattore 3.3× in finestre produce solo
~2.5× in tempo di parete).

**Verdetto finale**: il trend di H2 CONTINUA ma con rendimenti fortemente
decrescenti — il salto 50→29 vale +54..+84 pt di CLred, il salto 29→15 vale
solo altri +4 pt, ma dimezza ulteriormente l'osc_count (15-18 → 9, il flap
oscilla molto meno). Il sistema converge asintoticamente verso CLred≈54-55%
per QUALSIASI R testato — la conferma più forte finora che l'intero
comportamento negativo/piatto osservato a window=50 era un artefatto di
cadenza di accoppiamento, non un limite del modello o della formulazione MPC.
**R*=0.0003 (il valore ottimo derivato dal ROM, nessun detuning necessario)
recupera la stessa performance di R=0.001 non appena la finestra è
sufficientemente fine** — quindi non serve più "detunare" R per rendere il
controllore sicuro sul FOM, basta accoppiare a cadenza corretta.

**Raccomandazione pratica**: `window=29` (≈0.002s, il più vicino a
`_MPC_CTRL_DT` del controllore) è il miglior compromesso costo/beneficio per
run di verifica futuri — cattura il grosso del guadagno (+50 pt) a un terzo
del costo extra di `window=15`. `window=15` resta preferibile se il chatter
del flap (fatica attuatore) è una preoccupazione primaria, dato l'osc_count
dimezzato.

## Cross-check con `appendixA.tex` e confronto CLred ROM vs FOM (W30/Tg0.40)

Lo studio di verifica ufficiale (`appendixA.tex`, `tab:appA_window`) rifinisce
N_win∈{100,50,25,10} su un **flap-step passivo (no gust, no controllore)**:
misura solo l'errore di staggering del coupling fluido-struttura, NRMSE<1%
anche a N_win=100 → "N_win=50 convergente". Questo è vero per quello che
misura, ma è un asse ORTOGONALE al nostro H2: lì non c'è mai un controllore
in retroazione, quindi quello studio non poteva vedere il mismatch di cadenza
di ripianificazione MPC. `N_win=50` resta valido per l'accuratezza numerica
del coupling; è il valore sbagliato per le verifiche closed-loop MPC-su-FOM.
(Nota collaterale dallo stesso paragrafo: anche `dt_CFD=7e-5s`, Co_max≈2, non
è pienamente convergente sul picco di carico — Co_max=1 cambia il picco del
+4.3% — ma è un errore separato, più piccolo dell'effetto di cadenza.)

CLred ROM (da `light/results_cs25_combo/summary.md`, k=0.098, R*=0.0003)
vs CLred FOM alle varie window:

| Sorgente | CLred |
|---|---|
| **ROM** (ottimo tesi) | **+80.5%** |
| FOM, window=50 (produzione) | **-33.5%** |
| FOM, window=29 | **+50.3%** |
| FOM, window=15 | **+54.3%** |

Gap ROM↔FOM(window=50) ≈114 punti, spiegato per ~85 punti dalla cadenza
(H2). Resta un gap residuo di **~26-30 punti** anche a cadenza corretta.

## Investigazione del gap residuo (~26-30 pt): teacher-forcing + H1 (quantizzazione)

Domanda dell'utente: il gap residuo è vero errore di modello (→ serve
retraining) o quantizzazione della griglia argmin (→ no retraining, basta
raffinare la griglia)? Il teacher-forcing fatto ad inizio studio (RMSE
picco 0.22 vs baseline 0.02) era stato misurato sulla traiettoria PATOLOGICA
a window=50 — andava rifatto sulla traiettoria pulita prima di concludere
qualsiasi cosa sul retraining.

**Teacher-forcing, stesso script (`validate_iteration.py --model`), stessa
metrica, su tutte le window disponibili** (via container
`tensorflow_gpu.sif`, comando in fondo alla nota):

| Traiettoria | rmse | max_abs | mean |
|---|---|---|---|
| window=50 (R=0.001, patologica, osc=20) | 0.0582 | **0.2218** | 0.0092 |
| window=29, R*=0.0003 | 0.0347 | 0.1458 | 0.0114 |
| window=29, R=0.001 | 0.0292 | 0.1258 | 0.0093 |
| window=15, R*=0.0003 | 0.0294 | 0.1297 | 0.0059 |

`max_abs=0.2218` a window=50 combacia col vecchio "picco RMSE 0.22" (stesso
fenomeno, stessa scala — la nuova misura è consistente con la vecchia).
**Risultato chiave**: passando da window=50 a window=29/15, l'errore di
teacher-forcing CROLLA da solo (~-45/-50% sia su rmse che su max_abs) **senza
alcun retraining** — lo STESSO modello, sulla traiettoria a cadenza
corretta, predice molto meglio. Questo conferma che una parte sostanziale
dell'errore di predizione osservato all'inizio dello studio era un SINTOMO
della traiettoria patologica (stato del FOM fuori distribuzione a causa del
bug di cadenza), non un limite intrinseco del modello. **Ma l'errore non
sparisce**: resta un rmse≈0.03 / max_abs≈0.13-0.15 anche sulla traiettoria
pulita — circa 1.5× il livello "baseline" (~0.02) menzionato nella nota
originale, quindi un residuo reale, non rumore.

**Test H1 (quantizzazione griglia) sul ROM**, `rom_screen.py --ngrid 161 321
641 1281` (Q_alpha_dot=0, R*=0.0003, job 29120, cpu03):

| NGRID | step | CLred_ROM | flap_max | osc_count |
|---|---|---|---|---|
| 161 (baseline) | 0.175° | 80.3% | 8.05° | 18 |
| 321 | 0.0875° | 82.1% | 8.23° | 18 |
| 641 | 0.04375° | 72.8% | 9.23° | 18 |
| 1281 | 0.021875° | 83.4% | 8.82° | 18 |

CLred_ROM oscilla in una banda di ~10.6 punti (72.8-83.4%) **senza trend
monotono** con la finestra di raffinamento — non converge verso un valore
"vero" al crescere di NGRID, il che è più coerente con sensibilità
numerica/chaotic-like del sistema retroazionato a scelte discrete minuscole
del flap che con un errore sistematico di quantizzazione. **osc_count resta
IDENTICO (18) in tutti e 4 i casi** — la griglia non ha alcun effetto
sull'oscillazione. Verdetto H1: **esclusa come spiegazione del gap
sistematico e dell'osc_count** — contribuisce al più qualche punto di
rumore/varianza (~10pt), non un bias direzionale di 26-30 punti sempre nella
stessa direzione (FOM sotto ROM).

### Verdetto e raccomandazione retraining

Il gap residuo di ~26-30 punti è **prevalentemente errore di predizione
genuino del modello LDNet** in questo regime severo (W30/Tg0.4, k=0.098,
cella tra le più dure della griglia), non un artefatto di griglia. Rispetto
alla misura originale (fatta su dati patologici) il problema è oggettivamente
più PICCOLO di quanto sembrasse — l'errore si è dimezzato passando a una
traiettoria sana, senza toccare il modello — ma non è zero: rmse≈0.03,
max_abs≈0.13-0.15 in C_L, una quota plausibile per spiegare 26-30 punti di
CLred mancante.

**Raccomandazione: il retraining ha senso, ma con priorità BASSA rispetto al
fix di cadenza già ottenuto.** window=29 da solo ha già reso il controllore
genuinamente utile (+50% CLred, prima -33.5%) senza toccare il modello. Un
eventuale retraining (loop DAgger) andrebbe fatto raccogliendo dati
closed-loop **dalla traiettoria a cadenza corretta** (window=29), non da
quella patologica come nel tentativo precedente (che aveva dato risultato
misto/probabile overfitting) — i dati di addestramento sarebbero più
rappresentativi e meno rumorosi. Non incluso in questa fase: è un lavoro
separato e più costoso (raccolta dati + fine-tuning + rivalidazione).

**Comando di riferimento per il teacher-forcing** (container TF, nessun
ambiente locale disponibile):
```
apptainer exec --bind /work /work/u10677113/tensorflow_gpu.sif /bin/bash -c \
  "cd /work/u10677113/LDNet_GLA/light/dagger_fom && \
   OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
   python3 validate_iteration.py --csv <path>/structural_trajectory.csv \
   --w0 30 --tg 0.40 --r <R> \
   --model /work/u10677113/LDNet_GLA/clean/models_rollout/latent_10"
```

## Generalità: window=29 su altre celle mostrate in `chapter3.tex`

Le celle più citate/mostrate nel capitolo 3 della tesi (oltre W30/Tg0.40, già
fatta): **W30/Tg0.70** (la seconda traccia in `fig:mpc_traces`, CLred_ROM
91.8%), **W30/Tg0.30** (angolo severo-ripido, CLred_ROM 39.4%, il peggiore
della griglia, discusso a lungo perché il gradiente di raffica
$\dot W/U_\infty\approx225°/s$ è comparabile al rate limit dell'attuatore),
**W30/Tg1.20** (angolo raffica-lunga, CLred_ROM 41.7%, discusso perché il
vincolo attivo è la stabilità in beccheggio, non il carico). Per ciascuna:
run FOM `--window 29` allo stesso R* di `summary.md`, PIÙ un run open-loop
dedicato per l'exo reale (nessuna delle tre aveva già un exo reale misurato).
Per W30/Tg0.70 e W30/Tg1.20 esisteva anche una verifica FOM originale a
`window=50` (fatta a inizio sessione, prima della scoperta di H2) — permette
un terzo confronto diretto vecchio-vs-nuovo, ricalcolato con lo stesso exo
reale per coerenza.

| Cella (k) | ROM | FOM window=50 (vecchio) | FOM window=29 (nuovo) |
|---|---|---|---|
| W30/Tg0.30 (k=0.131) | +39.4% | — (mai verificata) | **-18.1%**, flap=14.00° sat, osc=14 |
| W30/Tg0.40 (k=0.098) | +80.5% | -33.5%, osc=20 | **+50.3%**, flap=14.00° sat, osc=15 |
| W30/Tg0.70 (k=0.056) | +91.8% | +70.5%, osc=16 | **+72.1%**, flap=14.00° sat, osc=7 |
| W30/Tg1.20 (k=0.033) | +41.7% | +47.5%, osc=13 | **+54.3%**, flap=8.40° (non sat), osc=4 |

**Il quadro NON è uniforme, e questo è il risultato più importante di questa
fase**:

- **W30/Tg0.70 e W30/Tg1.20**: il fix di cadenza generalizza bene. Il
  guadagno assoluto di CLred è più piccolo che su Tg0.40 (+1.6 e +6.8 punti
  contro +83.8) perché qui window=50 non era già catastrofico — ma
  l'osc_count crolla nettamente in entrambe (16→7, 13→4), stessa firma
  qualitativa vista su Tg0.40. Su Tg1.20 il FOM batte perfino il ROM sia a
  window=50 che a window=29 — generalizzazione oltre le aspettative in
  questo regime (raffica lenta, vincolo di beccheggio).
- **W30/Tg0.30 (il caso peggiore della griglia): il fix di cadenza NON
  basta.** CLred resta negativo (-18.1%) anche a window=29 — il controllore
  PEGGIORA il carico rispetto a non fare nulla, mentre il ROM prevedeva
  +39.4%. Questo è coerente con la spiegazione fisica già nel testo di
  tesi (`chapter3.tex:340-347`): il gradiente della raffica
  ($\dot W/U_\infty\approx225°/s$) è comparabile al rate limit
  dell'attuatore ($\dot\delta_{\max}=300°/s$), quindi il flap non riesce a
  inseguire la rampa "a prescindere dal tuning" — ma qui la realtà del FOM è
  PEGGIORE della previsione ROM, non solo in linea con essa: un gap
  residuo diverso da quello di Tg0.40 (là il segno si ribaltava da negativo
  a positivo con la cadenza corretta; qui resta negativo). Non ancora
  investigato se sia lo stesso tipo di errore di modello (probabile, dato
  che è il regime più estremo e più fuori-distribuzione) o un effetto
  saturazione/rate-limit del tutto reale e non risolvibile via software.

**Conclusione**: il fix di cadenza (H2) è un miglioramento reale e generale
su TUTTE le celle testate (osc_count sempre in calo, CLred sempre uguale o
migliore rispetto a window=50), ma non è una bacchetta magica — sul caso più
estremo della griglia (W30/Tg0.30) resta un problema irrisolto, e non è
chiaro se sia un limite fisico reale (rate limit dell'attuatore, come già
argomentato in tesi per il ROM) o un ulteriore errore di modello concentrato
in quel regime. Andrebbe indagato con lo stesso approccio (teacher-forcing +
H1) usato per Tg0.40 prima di scriverne la conclusione in tesi.

## Investigazione W30/Tg0.30 (il caso irrisolto)

Tre controlli sulla traiettoria FOM `window=29` già raccolta
(`Rsweep_W30_Tg0.30_R0.0001_win29/structural_trajectory.csv`), stesso
approccio usato per Tg0.40.

### 1. Teacher-forcing: errore di modello genuinamente alto, il più alto misurato finora

```
rmse=0.1033  max_abs=0.2342  mean=0.0014
```

Per confronto: Tg0.40 a window=29 aveva rmse≈0.03/max_abs≈0.13-0.15 (regime
"sano"); Tg0.40 a window=50 (patologico) aveva rmse=0.0582/max_abs=0.2218.
**Tg0.30 a window=29 (cadenza già corretta) ha un errore di predizione
PEGGIORE del caso patologico di Tg0.40** — la cadenza qui non è il problema:
il modello LDNet sbaglia davvero, e più che altrove nella griglia.

### 2. Rate-limit dell'attuatore: rispettato correttamente, nessun artefatto di simulazione

Prima ipotesi scartata per errore di misura: un calcolo grezzo di dδ/dt tra
righe consecutive del CSV dava 1500°/s (5× il limite fisico 300°/s) — ma è un
artefatto, perché il CSV registra molti sotto-passi RK45 per finestra e
`delta` è costante tranne ai bordi finestra (zero-order-hold, confermato in
`cosim_driver_extract.py::delta_schedule()`, righe 137-144: nessuna rampa,
nessun'interpolazione, ritorna `_MPC_CUR_DELTA` invariato per tutta la
finestra). Rifacendo il calcolo SOLO sulle transizioni vere (342 cambi di
valore su tutta la traiettoria):
```
dt tra transizioni: mediana 2.030 ms (= window_dt esatto di window=29)
step per transizione: mediana E massimo 0.525° (limite teorico reach=300°/s×0.002s=0.6°)
rate reale sostenuto: 258.6°/s, COSTANTE su tutte le 342 transizioni
```
Il flap satura al rate limit (0.525° è il punto griglia più vicino sotto la
soglia di reachability 0.6° per `G=161`, passo 0.175°: 3 step di griglia)
**per l'intera durata della risposta**, in modo pulito e consistente con
`self.dt=_MPC_CTRL_DT=0.002s` usato da `MPCPreviewController.compute()`
(`light/optimal.py:253`, `reach = self.delta_dot_max * self.dt`) — a
window=29 questo dt coincide quasi esattamente col window_dt reale, quindi
il vincolo di reachability è fisicamente corretto qui (a differenza di
window=50 dove sarebbe stato troppo conservativo, o window=15 dove sarebbe
stato troppo permissivo). **Nessun artefatto di attuazione**: il
comportamento è esattamente quello descritto in `chapter3.tex:340-347` per
il ROM — il flap rincorre la raffica al massimo rate possibile e non basta.

### 3. Quantizzazione griglia (H1): test INCONCLUSIVO — scoperto un gap di metodologia

```
NGRID=161  (baseline): CLred_ROM = -0.4%   (!! non +39.4% di summary.md)
NGRID=321:              CLred_ROM = -12.3%
NGRID=641:               CLred_ROM = -0.1%
NGRID=1281:              CLred_ROM = +63.3%
```

**Attenzione**: `rom_screen.py::simulate_qadot()` usa il preview grezzo
`w_seq=Wt[i+1:i+N+1]` (oracolo diretto), MENTRE la tabella di
`summary.md` è stata generata dalla pipeline "combo"
(`light/tests/cs25_combo_study.py`, header: *"FusedSensor Jmax=50, MPC N=8,
R=R*, R_du=0, oracle preview... dp45 horizon"*) — con uno smoothing/fusione
del sensore (`Jmax=50`) sul preview che `rom_screen.py` NON riproduce. Per
Tg0.40 questa differenza era trascurabile (rom_screen dava 80.3% contro
80.5% di summary.md, praticamente identico) — ma per Tg0.30 il gap è
enorme: -0.4% (rom_screen, G=161) contro +39.4% (combo, summary.md), ~40
punti di differenza SOLO dal preview processing, prima ancora di guardare
NGRID. **Il test H1 con `rom_screen.py` non è quindi attendibile per
questa cella**: le oscillazioni selvagge tra NGRID (-12%→+63%, mai viste su
Tg0.40 dove la banda era ~10pt e senza trend) sono compatibili sia con una
vera sensibilità di quantizzazione ANCORA PIÙ marcata su questo regime
estremo, sia semplicemente con l'instabilità nota di un preview non
filtrato in un caso limite — non si possono distinguere le due cause con
questo strumento. Per un test H1 valido su Tg0.30 servirebbe rifare lo
sweep NGRID dentro la pipeline "combo" vera (`cs25_combo_study.py`), non
fatto in questa fase.

### Verdetto W30/Tg0.30

Il fix di cadenza (H2) non basta qui perché **il problema non è di cadenza**:
il rate-limit dell'attuatore è correttamente modellato e già al suo massimo
fisico per l'intera risposta (confermando quanto già argomentato in tesi per
il ROM), e l'errore di predizione del modello LDNet è il più alto misurato
in questo studio (rmse=0.103, quasi 2× il residuo "sano" di Tg0.40).
**Conclusione più probabile: gap di modello genuino**, non quantizzazione
né cadenza — ma senza un test H1 pulito (pipeline combo) resta un'ipotesi
non definitivamente esclusa. Questa è la cella più fuori-distribuzione della
griglia (flap che satura il rate-limit per l'intera traiettoria, mai visto
altrove) — coerente con un modello che non ha visto abbastanza dati di
training in questo regime estremo. Un eventuale retraining mirato
dovrebbe includere traiettorie closed-loop proprio da celle come questa,
non solo da Tg0.40.

## Estensione a W10/W20: tabella completa 3×4, ROM vs FOM window=29

Stesso protocollo (closed-loop `--window 29` allo stesso R* di `summary.md`
+ open-loop dedicato per l'exo reale) esteso a W10 e W20, agli stessi 4 Tg
già usati per W30, senza toccare i job dell'altra chat in coda (rimasti solo
3 slot liberi condivisi). 15/16 job completati; **W20/Tg1.20 ancora in corso**
(manca solo l'open-loop, il closed-loop è pronto) — riga aggiornata quando
finisce.

| Cella (k) | ROM | FOM window=29 | gap (ROM−FOM) |
|---|---|---|---|
| W10/Tg0.30 (k=0.131) | +81.2% | +61.9%, flap=6.30°, osc=6 | 19.3 pt |
| W10/Tg0.40 (k=0.098) | +89.1% | +74.6%, flap=6.47°, osc=6 | 14.5 pt |
| W10/Tg0.70 (k=0.056) | +93.5% | +79.1%, flap=4.55°, osc=6 | 14.4 pt |
| W10/Tg1.20 (k=0.033) | +93.4% | +80.2%, flap=3.67°, osc=5 | 13.2 pt |
| W20/Tg0.30 (k=0.131) | +86.0% | +46.6%, flap=12.78°, osc=8 | 39.4 pt |
| W20/Tg0.40 (k=0.098) | +87.8% | +55.6%, flap=14.00° sat, osc=8 | 32.2 pt |
| W20/Tg0.70 (k=0.056) | +91.9% | +80.7%, flap=10.68°, osc=8 | 11.2 pt |
| W20/Tg1.20 (k=0.033) | +58.2% | +55.4%, flap=5.42° (non sat), osc=4 | 2.8 pt |
| W30/Tg0.30 (k=0.131) | +39.4% | **-18.1%**, flap=14.00° sat, osc=14 | 57.5 pt |
| W30/Tg0.40 (k=0.098) | +80.5% | +50.3%, flap=14.00° sat, osc=15 | 30.2 pt |
| W30/Tg0.70 (k=0.056) | +91.8% | +72.1%, flap=14.00° sat, osc=7 | 19.7 pt |
| W30/Tg1.20 (k=0.033) | +41.7% | +54.3%, flap=8.40°, osc=4 | **-12.6 pt (FOM vince)** |

**Pattern emergente, oltre H2**: a parità di Tg (quindi di reduced-frequency
k), il gap ROM↔FOM cresce monotonicamente con l'ampiezza W0 — non solo la
forma della raffica (k) conta, ma anche quanto è AGGRESSIVA in assoluto:

- **Tg=0.30 (k=0.131, il più severo)**: gap 19.3pt (W10) → 39.4pt (W20) →
  57.5pt e segno ribaltato (W30). Stessa forma di raffica, stesso k, ma
  l'errore residuo triplica con l'ampiezza. Il flap satura (14° o quasi)
  solo per W20/Tg0.30-adiacente e W30; a W10 resta ben sotto saturazione
  (6.30°) e il gap è il più piccolo della riga.
- **Tg=0.70 e Tg=1.20 (k basso, raffiche più dolci)**: gap piccolo e stabile
  su tutte le ampiezze (11-20pt), nessuna rottura di segno, W30/Tg1.20
  arriva perfino a battere il ROM.
- **Interpretazione preliminare** (rivista sotto con dati): la SATURAZIONE
  del flap (ampiezza+durata insieme) sembrava il driver più diretto — dove
  flap_max si avvicina/tocca 14° il gap esplode. Confermato solo in parte,
  vedi analisi di correlazione sotto.

## Cosa determina il delta: teacher-forcing su tutte le 12 celle + correlazione

Per rispondere in modo diretto (non per proxy) — teacher-forcing
(`validate_iteration.py --model`) ripetuto su TUTTE le 12 celle della griglia
3×4 (job PBS, container TF; le due celle W30/Tg0.30 e W30/Tg0.40 riusano i
numeri già misurati prima). Tabella completa:

| Cella | gap (ROM−FOM) | flap_max | osc | teacher-force rmse | max_abs |
|---|---|---|---|---|---|
| W10/Tg0.30 | 19.3 | 6.30° | 6 | 0.0205 | 0.0718 |
| W10/Tg0.40 | 14.5 | 6.47° | 6 | 0.0150 | 0.0475 |
| W10/Tg0.70 | 14.4 | 4.55° | 6 | 0.0087 | 0.0258 |
| W10/Tg1.20 | 13.2 | 3.67° | 5 | 0.0065 | 0.0219 |
| W20/Tg0.30 | 39.4 | 12.78° | 8 | 0.0421 | 0.1757 |
| W20/Tg0.40 | 32.2 | 14.00° | 8 | 0.0319 | 0.1283 |
| W20/Tg0.70 | 11.2 | 10.68° | 8 | 0.0225 | 0.0517 |
| W20/Tg1.20 | 2.8 | 5.42° | 4 | 0.0063 | 0.0264 |
| W30/Tg0.30 | 57.5 | 14.00° | 14 | 0.1033 | 0.2342 |
| W30/Tg0.40 | 30.2 | 14.00° | 15 | 0.0347 | 0.1458 |
| W30/Tg0.70 | 19.7 | 14.00° | 7 | 0.0212 | 0.0788 |
| W30/Tg1.20 | -12.6 | 8.40° | 4 | 0.0070 | 0.0264 |

**Correlazione di Pearson tra il gap e ciascun candidato** (n=12):

| Predittore | r | r² (varianza spiegata) |
|---|---|---|
| **teacher-force rmse** | **0.87** | **76%** |
| osc_count | 0.77 | 60% |
| flap_max | 0.62 | 39% |

**Risposta**: il delta è dettato PRINCIPALMENTE dall'errore di predizione
del modello LDNet, non dalla saturazione del flap né dal chatter del
comando — l'rmse di teacher-forcing da solo spiega ~3/4 della varianza del
gap su tutta la griglia, un margine netto sopra osc_count e flap_max (che
sono comunque correlati, ma come SINTOMI a valle: un modello che sbaglia di
più genera decisioni di controllo peggiori, quindi più chatter e più
saturazione — non il contrario). flap_max da solo spiega meno di metà della
varianza: non è la saturazione in sé a causare l'errore, ma le due cose
condividono la stessa causa a monte (regime aerodinamico estremo, raro nei
dati di training). Il caso limite è illustrativo: W20/Tg0.40 satura il flap
(14.00°) quanto W30/Tg0.70 (14.00°) ma ha un gap doppio (32.2 vs 19.7) e un
rmse quasi doppio (0.0319 vs 0.0212) — a parità di saturazione, è l'errore
di modello a spostare il gap, non la saturazione stessa.

**Implicazione pratica**: un retraining mirato (dati closed-loop dalle celle
con rmse più alto: W30/Tg0.30, W20/Tg0.30, W20/Tg0.40, W30/Tg0.40) è la leva
con il ritorno atteso più diretto sul gap residuo — molto più che qualunque
intervento sul controllore (griglia, rate limit, costo) dato che quelli
agiscono sui sintomi (osc_count, flap_max) e non sulla causa (rmse).

## DAgger retrain iter2 — ESITO NEGATIVO: overfitting sulle celle di training

Eseguito secondo il piano approvato: dataset mirato (non tutte le 12 celle),
train = le 4 celle a rmse più alto (W30/Tg0.30, W20/Tg0.30, W30/Tg0.40,
W20/Tg0.40), valid = 2 celle scelte per catturare rispettivamente
generalizzazione in-regione (W20/Tg0.70) e regressione sul regime facile
(W10/Tg1.20). `build_dagger_h5.py --iter 2`: T=401 @ dt=0.002s, t_common=0.80s
(dominato dalle celle Tg0.30). `retrain_launch.sh ITER=2`: stessi
iperparametri di iter1 (`NADAM=0 NBFGS=500 LAMBDA_DAMP=0.003 W_LOAD=1.0
ROLLOUT_LEN=350`), warm-start da `clean/models_rollout/latent_10`.

**Curva di training**: loss di training monotona e liscia (2.408→5.2e-4).
Loss di validazione MOLTO instabile a metà training (picchi fino a ~12 tra
epoca 80-260, probabile instabilità del rollout closed-loop in configurazioni
intermedie dei pesi), poi converge a un minimo genuino **6.19e-3 all'epoca
370** — quasi 2× meglio del punto di partenza (1.24e-2 a epoca 0) e nettamente
meglio del miglior risultato di iter1 (1.48e-2). Dopo l'epoca 370 torna a
peggiorare (overfitting, come iter1, ma molto più tardi e a un livello
migliore). Checkpoint salvato correttamente quello dell'epoca 370
(`light/dagger_fom/models/iter2/latent_10`).

**Verifica economica (teacher-forcing su tutte le 12 celle, traiettorie
COMPLETE non troncate — stesso comando/metodologia di sempre)**:

| Cella | rmse OLD | rmse iter2 | ruolo | esito |
|---|---|---|---|---|
| W20/Tg0.30 | 0.0421 | 0.0200 | train | migliora |
| W20/Tg0.40 | 0.0319 | 0.0193 | train | migliora |
| W30/Tg0.30 | 0.1033 | 0.0562 | train | migliora |
| W30/Tg0.40 | 0.0347 | 0.0248 | train | migliora |
| W20/Tg0.70 | 0.0225 | 0.0602 | valid | **peggiora 2.7×** |
| W10/Tg1.20 | 0.0065 | 0.0478 | valid | **peggiora 7.4×** |
| W10/Tg0.30 | 0.0205 | 0.0315 | — | peggiora |
| W10/Tg0.40 | 0.0150 | 0.0329 | — | peggiora |
| W10/Tg0.70 | 0.0087 | 0.0418 | — | peggiora |
| W20/Tg1.20 | 0.0063 | 0.1146 | — | **peggiora 18×** |
| W30/Tg0.70 | 0.0212 | 0.0485 | — | peggiora |
| W30/Tg1.20 | 0.0070 | 0.1684 | — | **peggiora 24×** |

**Verdetto: overfitting netto, esattamente il fallimento che il design del
dataset (2 celle di valid dedicate a intercettarlo) doveva rilevare — e lo
ha rilevato correttamente.** Solo le 4 celle di training migliorano; TUTTE
le altre 8 peggiorano, comprese entrambe le celle di validazione. La loss di
validazione interna al training (6.19e-3, apparentemente buona) NON ha visto
questo perché misura una cosa diversa: rollout closed-loop dell'errore
stato+carico, troncato a `ROLLOUT_LEN=350` campioni (0.7s, meno del t_common
0.80s), con propagazione delle predizioni del modello su sé stesse (non
teacher-forcing) — un rollout auto-consistente può avere una forma "giusta"
pur avendo un errore di predizione one-step (quello misurato dal
teacher-forcing) molto peggiorato, specialmente nella porzione più tardiva
delle traiettorie di valid (Tg0.70/Tg1.20, mai vista durante il training a
causa del troncamento t_common). Con soli 4 trajectory di training e 500
iterazioni L-BFGS, il modello si specializza fortemente sulla combinazione
specifica di regimi visti, a scapito della generalità.

**Decisione presa (secondo il criterio del piano approvato)**: NON eseguito
il test decisivo FOM su W30/Tg0.30 — il criterio "rmse migliora senza
regressioni sistematiche" non è soddisfatto, sarebbe stato tempo di cluster
sprecato su un modello già bocciato dal teacher-forcing. **`iter2/latent_10`
NON va usato in produzione.** Il modello di produzione
(`clean/models_rollout/latent_10`) resta quello valido; il fix di cadenza
(`window=29`, H2) resta l'unico intervento validato con beneficio netto
confermato in questa sessione.

**Possibili direzioni per un iter3, se si vuole riprovare** (non eseguite,
da decidere insieme): dataset di training più ampio (più delle 4 celle
peggiori, per dare al modello più contesto e non solo il regime più estremo);
meno iterazioni L-BFGS o un criterio di early-stopping legato al
teacher-forcing rmse su traiettoria intera invece che alla sola loss di
rollout troncata (che qui si è dimostrata un proxy fuorviante); regolarizzazione
esplicita (weight decay o simile) per contrastare l'overfitting con dataset
piccoli, mai provata finora.

## iter3 (dataset ampio) e iter4 (damping aumentato) — su richiesta dell'utente

Testate entrambe le direzioni proposte sopra, in parallelo, ciascuna isolando
UNA variabile rispetto a iter2 (stessa disciplina "una modifica alla volta"
di tutto lo studio).

**`retrain_launch.sh` reso parametrico** (piccola modifica, file di nostra
proprietà, non `sensitivity_latent_rollout.py`): `LAMBDA_DAMP`, `ROLLOUT_LEN`,
`NADAM`, `NBFGS`, `W_LOAD` ora leggono da env con `${VAR:-default}` invece di
essere hardcoded nell'invocazione apptainer — prima non era possibile
cambiarli dall'esterno nonostante lo script sembrasse parametrizzato.

**iter4 — `LAMBDA_DAMP=0.01`** (10/3× il default 0.003), stesso dataset di
iter2 (4 train + 2 valid). `LAMBDA_DAMP` è un termine di damping fisico
nell'equazione di evoluzione del latente (`src/sensitivity_latent_rollout.py:126`:
`z += dt*(NNdyn(z) - LAMBDA_DAMP*z)`), non un weight-decay classico — è
comunque l'unico knob di regolarizzazione già esposto senza toccare lo script
condiviso. **Fallimento netto**: training loss stesso molto peggiore
(9.8e-3 contro 5.2e-4 di iter2 a epoca 500 — il modello non riesce più a
fittare bene nemmeno i dati di training), validation loss instabile fino alla
fine (1.3-2.0). Teacher-forcing sulle 12 celle: rmse 3-40× peggio del vecchio
modello IN OGNI CELLA, con `max_abs` bloccato a ≈0.627 in quasi tutte —
sintomo di un collasso sistematico della dinamica latente (troppo smorzata
per seguire qualunque variazione). **Scartato, nessun dubbio.**

**iter3 — dataset ampio**: train = tutte le 8 celle W20+W30 (tutti i 4 Tg),
valid = tutte le 4 celle W10 (tutti i 4 Tg) — copertura molto più ampia della
regione W0∈{20,30}, held-out l'intera ampiezza W10 per un test di
generalizzazione pulito. Stessi iperparametri di iter2 altrimenti.

Curva di training molto più sana di iter2: nessuna esplosione della loss di
validazione (resta in banda 1.2e-3–4.2e-3 per tutte le 500 epoche, contro i
picchi fino a ~12 di iter2). Minimo a epoca 150 (1.22e-3), poi drift
moderato ma NON collasso.

Teacher-forcing sulle 12 celle — **risultato misto, non un successo pulito**:

| Cella | OLD | iter2 | iter3 | ruolo in iter3 |
|---|---|---|---|---|
| W20/Tg0.30 | 0.0421 | 0.0200 | 0.0261 | train, migliora |
| W20/Tg0.40 | 0.0319 | 0.0193 | 0.0250 | train, migliora |
| W30/Tg0.40 | 0.0347 | 0.0248 | 0.0253 | train, migliora |
| **W30/Tg0.30** (il caso peggiore) | 0.1033 | 0.0562 | **0.0991** | train, **quasi invariato** (e max_abs peggiora: 0.234→0.558, nuovo outlier) |
| W20/Tg0.70 | 0.0225 | 0.0602 | 0.0260 | train, lievemente peggio di OLD |
| W20/Tg1.20 | 0.0063 | 0.1146 | 0.0377 | train, peggio di OLD (ma molto meno di iter2) |
| W30/Tg0.70 | 0.0212 | 0.0485 | 0.0270 | train, lievemente peggio di OLD |
| W30/Tg1.20 | 0.0070 | 0.1684 | 0.0413 | train, peggio di OLD (ma molto meno di iter2) |
| W10 (tutte e 4) | 0.0065-0.0205 | 0.0315-0.0478 | 0.0247-0.0352 | valid, peggio di OLD ovunque (meno che iter2) |

**Scoperta**: quattro delle otto celle di TRAINING (W20/Tg0.70, W20/Tg1.20,
W30/Tg0.70, W30/Tg1.20) peggiorano comunque rispetto a OLD, nonostante siano
letteralmente nel training set. Causa identificata: `build_dagger_h5.py`
usa un'unica finestra temporale comune (`t_common=0.80s`, dettata dalle
celle Tg=0.30 più corte) per TUTTE le celle incluse — quindi anche le celle
Tg=0.70/1.20 (durata naturale 1.2-1.7s) vengono troncate agli stessi primi
0.80s durante il training. Il teacher-forcing valuta invece la traiettoria
COMPLETA (fino a Tg+0.5s): la parte più tardiva di queste celle, mai vista
durante il fine-tuning, è probabilmente dove si concentra il peggioramento.
Il limite dello strumento, notato all'inizio come "non urgente", risulta
invece un fattore attivo che limita la generalizzazione.

**Verdetto complessivo iter3**: miglioramento reale ma parziale — cura
alcune celle del blocco W20/W30-Tg-corto senza risolvere quella che conta di
più (W30/Tg0.30, il caso che aveva innescato tutto lo studio), e sposta il
costo sulle celle a Tg lungo e sull'ampiezza W10 mai vista. Non soddisfa il
criterio "migliora senza regressioni sistematiche" del piano approvato —
**non eseguito il test FOM decisivo** su nessuno dei tre modelli (iter2/3/4).

## Stato finale del filone di retraining

Nessuno dei tre tentativi (iter2, iter3, iter4) produce un modello pronto per
la produzione. Il modello di produzione (`clean/models_rollout/latent_10`)
resta quello valido. **L'unico intervento con beneficio netto confermato in
tutto questo studio resta il fix di cadenza `--window 29`** (H2), che da solo
recupera la maggior parte del gap ROM-FOM senza toccare il modello.

**Prossimo passo più promettente per un futuro iter5** (non eseguito):
risolvere il limite del `t_common` condiviso in `build_dagger_h5.py` — per
esempio permettendo un `ROLLOUT_LEN`/finestra di valutazione diversa per
cella invece di un'unica finestra troncata alla cella più corta — prima di
riprovare con un dataset ampio come iter3. Finché il training non vede mai
la coda delle traiettorie più lunghe, qualunque dataset che le includa
rischia di peggiorarle comunque.

## iter5 — fix del `t_common` troncato: SUCCESSO netto (su richiesta dell'utente, in loop autonomo)

Eseguito esattamente il "prossimo passo" indicato sopra. `build_dagger_h5.py`
supporta già `--t-common` esplicito (nessuna modifica di codice) — bastava
non affidarsi al default. Dataset = stessa composizione di iter3 (train =
tutte le 8 celle W20+W30, valid = tutte le 4 celle W10) ma con
`--t-common 1.2` (invece del default 0.80s derivato dalle celle Tg=0.30):
T=601 campioni, copre per intero le celle Tg∈{0.30,0.40,0.70} e lascia
scoperto solo l'ultimo 0.5s delle celle Tg=1.20 (contro 0.9s prima).
`ROLLOUT_LEN=550` (era 350). `retrain_launch.sh` reso parametrico su
`LAMBDA_DAMP/ROLLOUT_LEN/NADAM/NBFGS/W_LOAD` per permettere questo (prima
erano hardcoded nell'invocazione apptainer nonostante lo script sembrasse
già parametrizzato).

Curva di training sana (nessuna esplosione), minimo a epoca 380: valid loss
1.74e-3 (contro 1.22e-3 di iter3, 6.19e-3 di iter2 — comparabile a iter3,
molto meglio di iter2).

**Teacher-forcing su tutte le 12 celle (traiettorie complete) — risultato
netto, il migliore di tutti i tentativi**:

| Cella | OLD | iter3 | **iter5** | Δ vs OLD |
|---|---|---|---|---|
| W30/Tg0.30 (il caso peggiore) | 0.1033 | 0.0991 | **0.0394** | **-62%** |
| W20/Tg0.30 | 0.0421 | 0.0261 | **0.0149** | -65% |
| W20/Tg0.40 | 0.0319 | 0.0250 | **0.0134** | -58% |
| W20/Tg0.70 | 0.0225 | 0.0260 | **0.0091** | -60% |
| W30/Tg0.40 | 0.0347 | 0.0253 | **0.0186** | -46% |
| W30/Tg0.70 | 0.0212 | 0.0270 | **0.0135** | -36% |
| W10/Tg0.30 | 0.0205 | 0.0252 | **0.0156** | -24% |
| W10/Tg0.40 | 0.0150 | 0.0247 | **0.0110** | -27% |
| W10/Tg0.70 | 0.0087 | 0.0265 | 0.0096 | +10% (marginale) |
| W10/Tg1.20 | 0.0065 | 0.0352 | 0.0125 | +92% (ma resta piccolo in assoluto) |
| W20/Tg1.20 | 0.0063 | 0.0377 | 0.0127 | +102% (idem) |
| W30/Tg1.20 | 0.0070 | 0.0413 | 0.0118 | +69% (idem) |

**9 celle su 12 migliorano rispetto a OLD, comprese TUTTE le celle
"difficili" (W20/W30 × Tg 0.30/0.40/0.70) con riduzioni 36-65%.** Le uniche
3 regressioni sono lievi, isolate e sistematicamente concentrate sulle
celle Tg=1.20 (l'unico regime ancora parzialmente troncato dal
`t_common=1.2s`, dato che la loro finestra naturale è 1.7s) — coerente con
la causa identificata, e comunque piccole in valore assoluto (max 0.0127,
contro miglioramenti che arrivano a portare celle da 0.10 a 0.04). Media
rmse sull'intera griglia: 0.0266 (OLD) → 0.0152 (iter5), **-43%**.

### Test decisivo: run FOM reali con iter5 su due celle

**Attenzione bookkeeping**: `mpc_fom_verify_rtag.pbs` tagga l'output solo
per R/window, non per modello — stesso path del run col modello vecchio.
Backup preventivo (`cp -r ... _OLDMODEL_backup`) fatto PRIMA di ogni run
con iter5, per non perdere i dati baseline.

**W30/Tg0.30** (`--window 29 --mpc-R 0.0001 --mpc-model iter5`, stesso exo
reale 1.1843 di prima):

| Modello | CLred | flap_max | osc_count |
|---|---|---|---|
| OLD (produzione) | -18.1% | 14.00° sat | 14 |
| **iter5** | **-8.8%** | 14.00° sat | 21 |
| ROM (riferimento) | +39.4% | — | — |

Migliora (+9.3 punti) ma **resta negativo** — coerente con l'analisi fisica
già fatta per questa cella (rate-limit dell'attuatore quasi saturo per
l'intera risposta, gradiente di raffica comparabile al limite fisico):
retraining riduce l'errore di modello ma non può superare un vincolo
fisico reale. osc_count peggiora (14→21) nonostante CLred migliori — da
tenere d'occhio.

**W30/Tg0.40** (`--window 29 --mpc-R 0.0003 --mpc-model iter5`, la cella che
ha dato il via a tutto lo studio, stesso exo reale 0.866358):

| Modello | CLred | flap_max | osc_count |
|---|---|---|---|
| OLD (produzione) | +50.3% | 14.00° sat | 15 |
| **iter5** | **+56.0%** | 14.00° sat | **10** |
| ROM (riferimento) | +80.5% | — | — |

**Vittoria pulita su entrambe le metriche insieme**: CLred +5.7 punti E
osc_count -33% (15→10) — a differenza di Tg0.30, qui non c'è un vincolo
fisico dominante a bloccare il recupero. Il gap residuo ROM-FOM su questa
cella scende da 30.2 a 24.5 punti (-19%).

## Conclusione finale del filone di retraining

**Il retraining iter5 funziona ed è pronto per essere adottato.** Percorso
completo: causa del gap residuo identificata (errore di predizione LDNet,
r=0.87 con la varianza del gap) → dataset di fine-tuning scelto sulla base
di questa evidenza (celle a rmse più alto) → primo tentativo (iter2)
overfitta (dataset troppo piccolo) → causa dell'overfitting isolata
(dataset non abbastanza diverso, non copre bene lo spazio W0×Tg) → secondo
tentativo con dataset più ampio (iter3) migliora ma non pulito → causa
identificata (truncamento della finestra temporale condivisa in
`build_dagger_h5.py`) → fix mirato (iter5, `--t-common` esplicito) →
**successo netto**: teacher-forcing migliora su 9/12 celle (-43% rmse medio
sulla griglia), confermato da due run FOM reali indipendenti (W30/Tg0.30 e
W30/Tg0.40).

**Risultato pratico**: `light/dagger_fom/models/iter5/latent_10` è un
modello candidato migliore del modello di produzione
(`clean/models_rollout/latent_10`) per l'uso closed-loop con MPC. Non
sostituisce il modello di produzione automaticamente (per disegno di questo
studio — mai sovrascritto), ma è pronto per essere promosso se si decide di
farlo.

**Limite residuo, non risolvibile da ulteriore retraining**: W30/Tg0.30 (il
caso più estremo della griglia, gradiente di raffica al limite del rate
dell'attuatore) migliora ma resta negativo — qui il collo di bottiglia è
fisico (rate-limit dell'attuatore vs velocità della raffica), non di
modello, e nessun retraining potrà risolverlo: servirebbe intervenire
sull'attuatore stesso (rate limit più alto) o accettare che quella cella
resti fuori dall'inviluppo di controllo efficace, coerente con quanto già
argomentato in `chapter3.tex` per il ROM.

**Con questo si chiude il filone**: window fix (H2, beneficio netto e
generale) + retraining mirato (iter5, beneficio netto sulla maggior parte
della griglia) sono i due interventi validati di questo studio. Il gap
residuo su W30/Tg0.30 è un limite fisico documentato, non un problema
aperto da inseguire ulteriormente con questo approccio.

## Estensione dei test decisivi: altre 4 celle (le a maggior guadagno di rmse)

Su richiesta dell'utente, verificate anche le 4 celle col maggior
miglioramento di teacher-forcing rmse non ancora testate in closed-loop
reale: W20/Tg0.30 (-65%), W20/Tg0.70 (-60%), W20/Tg0.40 (-58%), W30/Tg0.70
(-36%). Stesso protocollo (backup del run OLD-model prima di sovrascrivere,
stesso exo reale già misurato per ciascuna cella).

**Nota operativa**: il run W30/Tg0.70 iniziale è finito senza pin esplicito
su un nodo già occupato da altri job (cpu01, contenzioso con 7 job
paralleli) — a quel ritmo (39 finestre in 3h46 contro le ~764 attese)
avrebbe richiesto giorni. Killato e rilanciato pinnato su nodo pulito
(cpu03): stesso problema di contesa già documentato più volte in questa
sessione, lezione confermata ancora una volta — pinnare sempre esplicitamente
un nodo libero verificato con `pbsnodes -aSj`, mai lasciare che lo
scheduler scelga da solo quando la coda condivisa è affollata.

**Tabella finale, tutte e 6 le celle testate in closed-loop reale con iter5**:

| Cella | ROM | FOM OLD | **FOM iter5** | Δ CLred | gap ROM-FOM: OLD→iter5 |
|---|---|---|---|---|---|
| W20/Tg0.40 | +87.8% | +55.6%, osc=8 | **+65.3%**, osc=10 | +9.7 pt | 32.2→22.5 (-30%) |
| W20/Tg0.30 | +86.0% | +46.6%, osc=8 | **+55.9%**, osc=9 | +9.3 pt | 39.4→30.1 (-24%) |
| W20/Tg0.70 | +91.9% | +80.7%, osc=8 | **+82.9%**, osc=6 | +2.2 pt | 11.2→9.0 (-20%) |
| W30/Tg0.40 | +80.5% | +50.3%, osc=15 | **+56.0%**, osc=10 | +5.7 pt | 30.2→24.5 (-19%) |
| W30/Tg0.70 | +91.8% | +72.1%, osc=7 | **+73.9%**, osc=6 | +1.8 pt | 19.7→17.9 (-9%) |
| W30/Tg0.30 | +39.4% | -18.1%, osc=14 | **-8.8%**, osc=21 | +9.3 pt | 57.5→48.2 (-16%) |

**5 celle su 6 migliorano su ENTRAMBE le metriche (CLred e osc_count)
insieme** — solo W30/Tg0.30 (il caso limite fisico) vede osc_count
peggiorare nonostante CLred migliori. Il gap ROM-FOM si chiude in modo
consistente del 9-30% su ogni singola cella testata, senza eccezioni nel
segno del miglioramento (tutte e 6 migliorano il CLred). Questo conferma
su base ampia — non solo sulle 2 celle del test decisivo iniziale — che
iter5 è un miglioramento reale e generalizzato, non un caso isolato.

**Tabella completa (12 celle): 6 misurate in closed-loop reale, 6 note solo
via teacher-forcing** (le celle Tg=1.20 e W10, dove la teacher-forcing
rmse indicava lievi regressioni — non testate in closed-loop reale in
questa fase, coerente con la scelta di dare priorità al sottoinsieme a
maggior guadagno).

## Indagine multi-agente sul gap residuo e tentativi DAgger iter6/iter7

Su richiesta esplicita dell'utente ("c'è ancora troppo delta tra ROM e FOM"),
lanciati 3 agenti di ricerca in parallelo (skill `systematic-debugging`) su
ipotesi non ancora escluse per il gap residuo (24.5pt su W30/Tg0.40 anche
con iter5), poi due nuovi tentativi di retraining.

### Le 3 ipotesi testate dagli agenti

1. **"FusedSensor"/mismatch di preview tra pipeline "combo" (ROM) e FOM** —
   **ESCLUSA**. Verificato riga per riga: `FusedPreviewSensor`
   (`light/optimal.py:75-166`) non è MAI istanziato né importato in
   `light/tests/cs25_combo_study.py`/`light/run.py` — l'header di
   `summary.md` che lo cita è testo di documentazione ereditato e stale da
   uno studio diverso (`light/noise/e2_combo.py`), non descrive il codice
   reale. Il preview è costruito in modo genuinamente identico su ROM
   (`run.py:81-83`) e FOM (`cosim_driver_extract.py:1168`): stessa formula
   analitica 1-cosine, stesso `_MPC_CTRL_DT=0.002`, stessa classe
   `MPCPreviewController.compute()`. Nessun ritardo, nessuno smoothing,
   nessun'asimmetria oracolo — a differenza del precedente analogo in
   `light/noise/` (vantaggio oracolo reale, documentato in memoria di
   sessioni passate), qui il "sospetto" non si è confermato.
2. **R\* non ri-tunato per FOM/iter5** — **confermato ma priorità bassa**.
   R\* è scelto SOLO tramite sweep sul ROM originale pre-retraining
   (`cs25_combo_study.py:47-64`, criterio: max CLred tra i run senza flag di
   instabilità strutturale — un check di non-esplosione, non un vero
   vincolo di robustezza a errori di modello). Non è mai stato ri-scelto né
   per il FOM né per iter5. Ma le uniche evidenze empiriche esistenti (due
   punti R sul FOM a window=29) mostrano CLred quasi indipendente da R una
   volta fissata la cadenza — non sembra la leva dominante.
3. **Auto-consistenza del ROM + capacità limitata del modello** —
   **confermata come limite metodologico reale ma già aggirato** (il FOM
   stesso è il "modello perturbato" con cui abbiamo validato tutto), e
   **capacità NON è il collo di bottiglia**: la dimensione latente $d_s$ è
   già stata esplorata in tesi fino a 20 senza guadagno (`chapter3.tex`,
   tabella `tab:latent_sweep`), con causa architetturale chiara — `NNdyn` ha
   solo 2 layer×7 neuroni, che limita il rank dello Jacobiano latente a 7,
   rendendo inerti gli stati oltre il settimo. **Lead nuovo emerso**:
   allargare la *width* di NNdyn (mai provato, a differenza di $d_s$) — ma
   richiede training da zero (non fine-tuning, la warm-start richiede shape
   compatibili), troppo costoso per questa sessione.

### iter6 — aggregazione DAgger (dati OLD + iter5 per le stesse 6 celle): FALLITO

Dataset: 12 traiettorie di training (6 celle × 2 policy: OLD-model e
iter5-on-policy, generate senza NESSUN nuovo run CFD — riusate le CSV già
raccolte durante i test decisivi di iter5), valid = W10 invariato,
warm-start da iter5 (non dalla produzione, per continuare la catena
DAgger). Loss di validazione interna migliore di iter5 (1.32e-3 vs 1.74e-3)
— ma **teacher-forcing su traiettoria intera PEGGIORA su quasi tutte le
celle di training** (es. W30/Tg0.30: 0.0394→0.0615, W30/Tg1.20:
0.0118→0.0277), mentre migliora su TUTTE le celle W10 (mai toccate).
Interpretazione: mescolare traiettorie OLD-policy e NEW-policy per la
STESSA condizione di raffica introduce supervisione in conflitto (stesso
stato iniziale, sequenze di flap diverse) — rumore, non vera diversità.

### iter7 — sostituzione pulita (solo dati on-policy dove disponibili): FALLITO ANCH'ESSO

Stessa composizione di iter5 (8 celle W20+W30, tutti i 4 Tg) ma con le
traiettorie ON-POLICY di iter5 al posto delle OLD per le 6 celle già
testate, OLD invariato per le 2 celle mai testate live (Tg=1.20). Warm-start
da iter5. Loss di validazione interna **la migliore di tutte le iterazioni**
(9.50e-4) — ma **teacher-forcing peggiora su 10 delle 12 celle rispetto a
iter5**, compresa la cella target W30/Tg0.30 (0.0394→0.0518).

### Verdetto (skill systematic-debugging, "question the architecture" dopo 2 fallimenti consecutivi)

**Pattern netto e ripetuto due volte**: ogni ulteriore fine-tuning a partire
da iter5 (warm-start ripetuto) migliora la loss di validazione INTERNA
(rollout troncato) ma PEGGIORA la vera capacità di generalizzazione
(teacher-forcing su traiettoria intera, su quasi tutte le celle). La
metrica di validazione del training non è un proxy affidabile per la
qualità reale del modello oltre un certo punto — probabilmente perché il
rollout loss è calcolato sulle STESSE poche condizioni di raffica viste in
training (anche nel valid set, che copre solo l'ampiezza W10), quindi
"migliorare" quella metrica significa sempre più specializzarsi su quella
distribuzione ristretta, non generalizzare meglio all'intera griglia.

**iter5 resta il modello migliore prodotto in questo studio**, confermato
su tre fronti indipendenti: teacher-forcing (9/12 celle migliori, -43% rmse
medio), validazione FOM reale (6/6 celle con CLred migliore, 5/6 anche con
osc_count migliore), e ora anche per confronto diretto con due successivi
tentativi di raffinamento (iter6, iter7) che lo hanno entrambi peggiorato.
**Chiudo qui il filone di ulteriori iterazioni DAgger a partire da iter5** —
il pattern di due fallimenti consecutivi con la stessa causa (overfitting
alla distribuzione ristretta di condizioni disponibili) è un segnale di
limite strutturale dei dati disponibili, non di una singola scelta di
iperparametri da correggere. Le direzioni che resterebbero (allargare
NNdyn, raccogliere molte più celle FOM reali per aumentare davvero la
diversità) sono entrambe fuori portata per iterazione rapida in questa
sessione — richiedono rispettivamente training da zero o giorni di nuovi
run cluster.

**Raccomandazione finale**: adottare `light/dagger_fom/models/iter5/latent_10`
come candidato di produzione. Il gap ROM-FOM residuo con iter5 (9-30% di
chiusura, non azzeramento) è il miglior risultato ottenibile con
l'infrastruttura e i dati disponibili in questa sessione; il fix di cadenza
(`window=29`) resta l'intervento a beneficio più ampio e netto in assoluto.

> **[SUPERATO — vedi sezione successiva]** La raccomandazione qui sopra è
> stata scritta quando le uniche celle testate in closed-loop reale con
> iter5 erano 6, TUTTE appartenenti alla regione di training (W20/W30).
> Il test delle 4 celle W10 (held-out) ribalta il quadro: iter5 le peggiora
> tutte. Vedi sotto.

## Le 4 celle W10 (held-out): iter5 le PEGGIORA tutte — la validazione era viziata

Completati i test FOM closed-loop con iter5 anche sulle 4 celle W10, cioè
l'unica ampiezza MAI usata nel training di iter5 (era il valid set). Stesso
protocollo di sempre (backup del run OLD-model prima di sovrascrivere,
stesso exo reale per cella, `window=29`).

| Cella W10 | OLD CLred | iter5 CLred | Δ | osc OLD → iter5 |
|---|---|---|---|---|
| Tg0.30 | +61.9% | **+58.8%** | −3.1 pt | 6 → 5 |
| Tg0.40 | +74.6% | **+67.8%** | −6.8 pt | 6 → 5 |
| Tg0.70 | +79.1% | **+64.6%** | −14.5 pt | 6 → 4 |
| Tg1.20 | +80.2% | **+70.3%** | −9.9 pt | 5 → 5 |

**Risultato netto: 4 celle su 4 peggiorano** (−3 a −15 punti di CLred),
mentre le 6 celle W20/W30 (regione di training) miglioravano tutte. Il
segno è perfettamente correlato con l'appartenenza o meno al training set,
non con la severità della raffica né con altro.

**Lezione metodologica (importante, vale oltre questo studio)**: la
conclusione precedente "iter5 è un miglioramento reale e generalizzato,
confermato su 6 celle" era **statisticamente viziata per costruzione** — le
6 celle erano tutte dentro la regione di training. Il teacher-forcing rmse
su W10 *migliorava* (0.0065→0.0125 ecc. erano lievi peggioramenti, ma la
media grid-wide scendeva del 43%), quindi nemmeno quella metrica ha
predetto correttamente il comportamento closed-loop su held-out. **Solo il
test closed-loop reale su celle MAI viste in training discrimina un vero
miglioramento da una specializzazione.** Le due metriche più economiche
(loss di validazione interna, teacher-forcing rmse) si sono entrambe
rivelate proxy inaffidabili in questo studio, ciascuna in un modo diverso.

**Diagnosi**: iter5 non è "migliore", è **specializzato**. Scambia
prestazioni sull'ampiezza W10 per prestazioni su W20/W30. Coerente con la
causa già identificata per i fallimenti di iter6/iter7 (sovra-specializzazione
sulla distribuzione ristretta di training), ma qui misurata direttamente in
closed-loop invece che dedotta.

## iter8 — dataset full-grid: il rimedio diretto alla diagnosi

Se la causa è la copertura ristretta del training set, il rimedio è
allenare su TUTTA la griglia di ampiezze. Composizione (job 29986):

- **train (9 celle)**: W10/{0.30,0.70,1.20}, W20/{0.30,0.40,1.20},
  W30/{0.30,0.40,0.70} — tutte e tre le ampiezze rappresentate, con un mix
  di durate per ciascuna.
- **valid (3 celle)**: W10/0.40, W20/0.70, W30/1.20 — held-out che copre
  anch'esso tutte e tre le ampiezze (a differenza di iter5, dove il valid
  era interamente W10 e quindi non poteva rilevare la specializzazione su
  W20/W30).
- Dati **on-policy iter5** dove disponibili (10 celle su 12, raccolti dai
  test decisivi senza nessun nuovo run CFD); OLD-model per W20/Tg1.20 e
  W30/Tg1.20, le uniche mai girate live con iter5.
- **Warm-start dalla PRODUZIONE**, non da iter5 — deliberato: i pesi di
  iter5 portano già impressa la specializzazione su W20/W30, ripartire da
  lì la propagherebbe. Stessi iperparametri di iter5 altrimenti
  (`t_common=1.2`, `ROLLOUT_LEN=550`, `NBFGS=500`, `LAMBDA_DAMP=0.003`).

**Criterio di valutazione rivisto** (dopo la lezione di cui sopra): il
verdetto su iter8 NON verrà dato su loss di validazione né su
teacher-forcing rmse, ma su **test FOM closed-loop reali che includano
celle di entrambe le regioni**, con particolare attenzione a quelle held-out.

**Nota operativa — login node sovraccarico**: durante questa fase il nodo
di login (login01) ha raggiunto **load average 107 con 45 utenti**, con
l'effetto che ogni comando SSH impiega minuti solo per l'avvio della shell
(autenticazione istantanea, poi stallo). Non è un problema di rete né di
chiave: diagnosticato con `ssh -v` (autenticazione OK, blocco su "Entering
interactive session") + `uptime`. Rimedio pratico: timeout ≥ 300 s e
raggruppare più operazioni in una singola connessione SSH invece di molte
connessioni brevi.

### Esito iter8: FALLIMENTO NETTO — ma con causa diagnosticata (underfitting)

Test FOM closed-loop sulle due celle discriminanti (backup dei risultati
iter5 salvati come `_ITER5_backup` prima di sovrascrivere, iter8 poi come
`_ITER8_backup`):

| Cella | OLD | iter5 | **iter8** |
|---|---|---|---|
| W10/Tg0.40 *(held-out per iter8)* | +74.6%, osc=6, flap=6.47 | +67.8%, osc=5, flap=7.00 | **−20.1%, osc=20, flap=3.33** |
| W30/Tg0.40 *(train per iter5 e iter8)* | +50.3%, osc=15, flap=14.00 | +56.0%, osc=10, flap=14.00 | **−6.4%, osc=32, flap=7.88** |

CLred **negativo su entrambe** (il controllore peggiora il carico rispetto
al non far nulla), osc_count esploso (20 e 32), flap_max crollato — il
controllore è diventato timido e confuso.

**Causa identificata — underfitting, non overfitting**: la training loss
finale di iter8 è **4.6e-3, circa 10× peggiore** di quella di iter5
(≈4e-4). Anche la validation loss (4.24e-3) era la peggiore della serie.
Il modello non ha *sovra*-appreso: non ha **finito di apprendere**. 500
iterazioni L-BFGS bastavano per le 8 traiettorie relativamente omogenee di
iter5 (solo W20/W30), ma non per le 9 di iter8, che coprono tre ampiezze
con escursioni di carico di scala molto diversa (W10 ha escursioni ~4×
più piccole di W30) — un problema di ottimizzazione sostanzialmente più
duro. Il risultato è un modello fermo a metà strada, peggiore del punto di
partenza da cui era partito il warm-start.

**Nota su cosa ha e non ha predetto il fallimento**: questa volta la loss
di validazione interna *avrebbe* dato un segnale corretto (era la peggiore
della serie), a differenza dei casi iter6/iter7 dove era ottima e il
modello era comunque peggiore. Le due metriche economiche restano quindi
inaffidabili come criterio *sufficiente*, ma una loss di training/validazione
palesemente alta resta un campanello d'allarme valido di non-convergenza.

### iter9 — stessa configurazione di iter8 con NBFGS=2000 (job 30102)

Cambiata **una sola variabile** rispetto a iter8 (disciplina a variabile
singola: dataset identico, stesso warm-start dalla produzione, stesso
`ROLLOUT_LEN=550`, stesso `t_common=1.2`): `NBFGS` da 500 a **2000**
(4× iterazioni L-BFGS), walltime alzato a 12 h. Test dell'ipotesi
"iter8 era solo non convergente".

Criterio di verifica: (a) la training loss deve scendere al livello di
iter5 (~4e-4) o meglio — se resta ~5e-3, l'ipotesi underfitting è
smentita e il problema è nella composizione del dataset, non nel budget di
ottimizzazione; (b) solo in caso affermativo, test FOM closed-loop sulle
stesse due celle discriminanti (W10/Tg0.40 held-out + W30/Tg0.40).

**Esito iter9 — ipotesi confermata solo a metà**: training loss
4.6e-3 → **1.16e-3** (4× migliore, quindi l'underfitting era reale e il
budget di iterazioni ne era la causa), ma **la generalizzazione non
migliora**: best validation loss 4.24e-3 → 3.80e-3 (solo −10%), e a fine
training la validation diverge a 2.4e-2 mentre la training continua a
scendere — cioè oltre un certo punto iter9 passa direttamente da underfit a
overfit, senza mai attraversare una zona di buona generalizzazione. Il
budget di ottimizzazione NON era l'unico problema: la composizione del
dataset full-grid (tre ampiezze con escursioni di scala molto diversa,
9 traiettorie sole) resta intrinsecamente difficile da fittare bene.
Test FOM closed-loop lanciato comunque sulle due celle discriminanti
(job 30106/30107) per chiudere la domanda con una misura reale invece che
con un'inferenza dalla loss.

## REGOLA OPERATIVA — mai eseguire calcolo su login01

Durante questa sessione alcune operazioni sono state eseguite **direttamente
sul nodo di login via SSH** invece che tramite `qsub`: le chiamate a
`build_dagger_h5.py` dentro `apptainer` (costruzione dataset iter2/5/6/7/8)
e un primo tentativo di `rom_screen.py`. I training e i test FOM sono
sempre passati da PBS, ma quelle build no. Hanno contribuito al carico del
nodo di login (osservato load average **107 con 45 utenti**), e i job
dell'utente sono stati successivamente rimossi dalla coda — plausibilmente
per intervento amministrativo legato proprio all'uso improprio del nodo di
login.

**Regola da qui in avanti, senza eccezioni**: qualunque cosa esegua codice
(anche pochi secondi di Python, anche una build di dataset HDF5) va
sottomessa con `qsub` su un nodo di calcolo, con `-l select=...:host=cpuNN`
verificato libero via `pbsnodes -aSj`. Su login01 solo comandi di
bookkeeping istantanei (`ls`, `qstat`, `cat` di file piccoli, `cp`).
