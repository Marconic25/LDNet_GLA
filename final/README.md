# `final/` — campagna dataset v6

Rigenerazione completa del dataset FSI e ri-esecuzione dell'intera catena di studi
della tesi su nuove numeriche. `light/` resta l'archivio congelato della campagna v5
e **non va modificata**: i suoi risultati sostengono il testo attuale di
`light/latex/`.

## Perché

Tre limiti di `dataset_v5`:

1. **Courant alto e mal documentato.** `adjustTimeStep` è `no`: è il passo temporale
   a essere fisso, non il Courant, e `maxCo` nel `controlDict` è inerte. Il Co reale
   di v5 non è il 2 dichiarato in Appendice A — quello è il caso mite; sulla raffica
   di produzione `validation/README.md` documenta ~3.5, e la cella Cc severa misura
   5.78 equivalenti. Vedi `ESTIMATE.md` §8.
2. **La finestra di accoppiamento a 50 passi è un artefatto.**
   `light/dagger_fom/NOTES.md` documenta che la verifica MPC-sul-FOM alla cella
   W30/Tg0.40 crolla a CLred ≈ −33% contro il +80.5% del ROM, e che **~85 punti** di
   quel divario sono un artefatto del rate di re-plan, corretto da `--window 29`.
   `window=29` è verificato ma non è mai stato usato per generare un dataset.
3. **Nessun dato di campo esiste per v5.** Le time-dir furono escluse dagli rsync e
   `/scratch_local` le ha cancellate dopo 30 giorni (`recon/RECON_NOTES.md`). Ogni
   studio di field reconstruction ha dovuto rigenerare la CFD a posteriori, e solo
   per ~7 sim.

v6 dimezza il Courant, porta `Nwin = 29` e **estrae i campi per tutte e 146 le run**,
così che loads e field reconstruction condividano la stessa base.

## Parametri

| | v5 | v6 |
|---|---|---|
| `deltaT` | `7e-5` s | **`3.16e-5` s** |
| Courant max misurato | ~3.5 (fino a 5.78) | **0.98 – 2.61** |
| `Nwin` | 50 | **29** |
| Finestra di accoppiamento | 3.5 ms | **0.916 ms** |
| Passi fluidi / 3 s | 42 857 | 94 937 |
| Finestre / 3 s | 857 | 3 274 |
| Campi estratti | ~7 sim | **146 sim** |
| `t_end` | 3.0 s fissi | per-run, `Tg`-dipendente |

`3.16e-5` è il `dt` medio della riga `maxCo 1` dello sweep adattivo in
`validation/courant_mild/courant_sweep_report.txt`. Su quel caso mite dà Co ≈ 1, ma
sulle raffiche reali no: **misurato 1.30 su famiglia A e 2.61 su Cc con flap MPC**.

Il Courant alto nasce dalla **combinazione** raffica + flap deflesso, non da nessuna
delle due da sola: la raffica più forte della campagna con flap fermo dà 1.30, lo slew
più veloce senza raffica dà 0.98, ma raffica W30 con flap a 8.4° dà 2.61. Meccanismo
probabile l'incidenza efficace; non dimostrato — vedi `ESTIMATE.md` §8.

**Decisione (2026-09-03): si accetta.** Portare Co ≤ 1 ovunque richiederebbe
`dt ≈ 1.21e-5`, cioè ~25 giorni di campagna invece di ~9. Il claim difendibile per la
tesi diventa quindi **"Courant più che dimezzato rispetto a v5"** (5.78 → 2.61 sulla
cella peggiore misurata), non "Courant unitario".

## Finestra di accoppiamento: 63 passi, non 29

La finestra è **63 passi = 1.991 ms**. Il criterio non è un conteggio di passi ma un
**tempo**: `_MPC_CTRL_DT = 0.002 s`, il passo con cui `MPCPreviewController` dimensiona
ogni movimento del flap (`reach = 300°/s × 0.002 s = 0.6°`), poi tenuto per una
finestra. `light/dagger_fom` sceglie 29 perché a `dt=7e-5` sono 2.030 ms; trasportare
il **numero** al `dt` di v6 dà 0.916 ms e un rate implicito di 655°/s contro un limite
fisico di 300.

A 0.916 ms l'accoppiamento è instabile e **la soluzione è sbagliata, non solo
rumorosa**: trim di `F_y` a 104 N invece di 165 (−37%), picco 203 invece di 298 N,
affondamento 8.0 invece di 12.1 mm. A 63 passi il trim coincide con v5 entro lo 0.6%.
Dettagli e la correzione al consiglio sbagliato di filtrare in `ESTIMATE.md` §15.

## Struttura

```
final/
  README.md          questo file
  ESTIMATE.md        modello di costo, misure di Courant, vincolo di disco
  design/            gen_matrix.py -> run_matrix.csv (146 righe, fonte di verita')
  cluster/           driver adattato, controlDict, PBS, submit/status/pull
  data/              extract_fields_step.py (incrementale) + builder HDF5
  studies/           studi light/ riadattati, puntati a v6
```

## Le 146 run

| Famiglia | Run | train/val/test | Eccitazione | Controllore |
|---|---:|---|---|---|
| A | 30 | 20/5/5 | solo raffica | `schedule` (flap a 0) |
| B | 46 | 30/8/8 | solo flap | `schedule` (raffica spenta) |
| Cc | 70 | 50/10/10 | raffica + flap | **57 `mpc`** + 13 `prop` |

A e B restano identiche a v5 (stessi intervalli LHS). **Cc è ridisegnata**: non più
feed-forward LHS ma copertura sistematica delle celle CS-25, con priorità al
controllore MPC — `W0 ∈ {10,20,30}` × `Tg` da 0.30 a 1.20 passo 0.05 = 57 run MPC, più
13 run col proporzionale `PropWRef` come braccio di confronto dentro al dataset.

Conseguenza utile: **le 57 run Cc-MPC *sono* la verifica MPC-sul-FOM** su tutte le
celle, a `Nwin=29`. Quello studio diventa un sottoprodotto del dataset.

Il totale di Cc è un parametro: `gen_matrix.py --n-cc 100` se si vuole portarlo a 100
(campagna totale 176). Le righe in più vanno al proporzionale, la griglia MPC resta
prioritaria.

## Le tre modifiche tecniche non banali

**1. Estrazione incrementale.** Con `purgeWrite 0` e 3 274 finestre il pattern v5
chiederebbe ~409 GB di scratch per singola run. Il driver ora scrive i campi ogni
`--field-every K` finestre e fa reconstruct + slice + purge *dentro* al loop: scratch
~1–2 GB costanti, `/work` ~17 GB a fine campagna. Dettagli in `ESTIMATE.md` §5.

**2. `--controller prop`.** Nuova modalità del driver che calcola il comando flap
in-process con la formula di `light/noise/controllers_ref.py::PropWRef`, senza server
TensorFlow. Le 13 run proporzionali costano quanto una `schedule`.

**3. `valid_mask` nei dataset HDF5.** I `t_end` per-run (che fanno risparmiare il 39%
di CFD) hanno un costo a valle: su una griglia temporale condivisa il 23.6% dei
campioni è padding a valore costante, e per la famiglia A si arriva al 31% (49% sulla
run più corta). Quel padding è fabbricato — il sistema reale sta ancora decadendo, non
si appiattisce. Entrambi i builder scrivono una maschera di validità e stampano la
frazione di padding; il training deve pesare la loss con quella. Dettagli e
alternative scartate in `data/README.md`.

**4. Health-check del server MPC.** Con `window=29` una run Cc-MPC fa ~3 274 chiamate
JSON al server persistente: il driver ora rileva un server morto, lo rilancia e
ritenta prima di abortire.

## Ordine di esecuzione

```
1. design/gen_matrix.py                  -> run_matrix.csv        (locale, secondi)
2. cluster/submit_campaign.sh            -> 146 job PBS           (~9 giorni)
3. cluster/pull_results.sh               -> dataset_v6 in locale
4. data/preprocess_GLA_v6.py             -> GLA_{train,valid,test}.h5
   data/build_fields_h5_v6.py            -> FIELDS_*.h5
5. train_l003_full.sh -> train_rollout.sh -> modello di produzione v6  (~7 h)
6. studies/run_all.sh                    -> tutti gli studi e le figure
```

Il passo 5 è bloccante per tutto il 6: gli studi ROM girano contro il modello, non
contro il dataset.

## Regole operative sul cluster — vincolanti

- **Solo `qsub`.** Nessun calcolo su `login01`, nemmeno smoke test o assemblaggio
  HDF5. La regola nasce da un incidente reale: load average 107 con 45 utenti, job
  dell'utente rimossi dalla coda (`light/dagger_fom/NOTES.md`). `login01` serve solo
  per `ls`, `qstat`, `cp`.
- **Controllare il nodo prima**: `pbsnodes -aSj`. Un job finito su un nodo già carico
  è andato ×35 più lento.
- **`max_user_run = 4`**, e questo scheduler manda in stato **E** i job in eccesso
  invece di accodarli: `submit_campaign.sh` usa 4 corsie `-W depend=afterany` con gli
  altri job `-h` (held), rilasciati da `status_campaign.sh`.
- **Wrapper OpenFOAM per-job** in `$SCRATCH/bin_of7`, mai `$HOME/bin_of7` condiviso
  (race di corruzione su un batch di 7 job).
- **Run ed estrazione nello stesso job.** `/scratch_local` è per-nodo e sparisce dopo
  ~30 giorni: separarli è esattamente l'errore che ha distrutto i campi di v5.
- Un'altra sessione che sottomette lavoro può preemptare i job: lanciare a lotti.
  Un job PBS che riparte da zero (SessID/elapsed cambiano) non è un job evicted — non
  dedurlo da un singolo `ssh` fallito.

## Stima

**~9 giorni** di calendario (6.6–13 a seconda della contesa), ~630 h di wall-time di
job. Modello di costo, validazione su un punto indipendente e ricalibrazione in
`ESTIMATE.md`.

## Verifica prima del lotto completo

1. `python3 design/gen_matrix.py --dry-run` — 146 righe, 30/46/70, split corretti.
2. **Smoke test**, 1 job, `t_end = 0.05 s`: `deltaT` = 3.16e-05, Courant ≤ 1.0 dal log,
   ~55 finestre, picco di scratch O(GB), `field_times.npy` coerente.
3. **Una run A completa** per ritarare `ESTIMATE.md`. Scarto > 30% dal previsto ⇒ si
   rifà la stima prima di sottomettere le altre 145.
4. In corsa: `status_campaign.sh` marca corrotta ogni run con `field_times.npy` sotto
   i 100 campioni e la rilancia.

A valle, il confronto che vale: la cella W30/Tg0.40 fra ROM v6 e run Cc-MPC. Se il
residuo di model-mismatch scende sotto i ~26–30 punti lasciati aperti da
`dagger_fom`, il passaggio a Co=1 / `Nwin=29` ha pagato, ed è un risultato da tesi.
