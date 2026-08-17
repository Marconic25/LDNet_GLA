# Linee guida per la cartella LaTeX

Regole vincolanti per qualunque agente che modifichi i file di questa cartella
(`chapter*.tex`, `Introduction.tex`, `Conclusion.tex`, `appendixA.tex`,
`bibliography.bib`).

## Contesto
- Tesi magistrale in Ingegneria Aeronautica.
- Formato articolo, lunghezza obiettivo circa 30 pagine. Ogni aggiunta va
  pesata contro questo limite: se un contenuto non è necessario, non si scrive.

## Struttura del documento
Il paper segue il modello Introduzione → Model → Methods → Results →
Conclusion, con la verifica e validazione del full-order model in appendice.
Mappa file → sezioni (il main che fa gli `\input` non è in questo repo, è sul
progetto Overleaf `Thesis_article_format`; i nuovi file vanno inclusi lì):

- `Introduction.tex` — Sez. 1 Introduction (`sec:introduction`): gust loads,
  stato dell'arte GLA (feedback/feedforward/lidar/MPC), gap modelli-controllo,
  surrogati data-driven, contributo, paragrafo con la struttura del paper.
- `chapter1.tex` — Sez. 2 Model (`sec:model`):
  - 2.1 Airfoil aeroelastic model (`subsec:aeroelastic_airfoil`): sistema
    completo (URANS + condizioni al contorno + equazioni strutturali +
    accoppiamento) in un unico blocco `subequations` (`eq:coupled_system`),
    matrici e forzanti, cinematica della cerniera, raffica, adimensionale.
  - 2.2 LDNet (`subsec:ldnet_arch`): formulazione, ingressi/uscite,
    coefficienti $C_L$, $C_M$ (`eq:coefficients`), sistema ridotto
    (`eq:reduced_system`).
  - 2.3 GLA controller (`subsec:controller`, alias `subsec:mpc_arch`):
    obiettivo CS-25, costo MPC (`eq:mpc_cost`), vincoli, modello di misura
    lidar (`eq:meas_model`), diagramma a blocchi (`fig:control_loop`).
    Niente strategie implementative: quelle stanno nei Methods.
- `chapter2.tex` — Sez. 3 Methods (`sec:methods`):
  - 3.1 Discretizzazione temporale del closed loop (`subsec:discretisation`):
    stepper strutturale RK45 (`eq:rk45_update`), update latente
    (`eq:latent_dyn`), rollout di inferenza, soluzione receding-horizon
    (`eq:mpc_recursion`, enumerazione su griglia), fusione del preview
    (`eq:invvar`, `eq:tikhonov`).
  - 3.2 Dataset generation (`subsec:dataset`): paragrafo che nomina schema di
    accoppiamento, solutore fluido e driver rimandando all'appendice; 3.2.1
    campagna di simulazioni e preprocessing (`subsubsec:campaign`,
    `tab:dataset_families`) con le tre famiglie A, B, Cc, il campionamento
    Latin hypercube e le logiche di movimento del flap. Gli schemi numerici del
    FOM stanno in Appendice A (`app:fom`), non qui.
  - 3.3 LDNet training (`subsec:training`): teacher forcing vs rollout, loss
    (`eq:rollout_loss`), algoritmo (BPTT, Adam, L-BFGS), metodo di selezione
    degli iperparametri e metriche (`eq:acc_metrics`). Solo metodo: i
    risultati del tuning stanno nei Results.
- `chapter3.tex` — Sez. 4 Results (`sec:results`): 4.1 accuratezza del
  surrogato e tuning (`subsec:training_results`), 4.2 GLA sull'inviluppo
  CS-25 (`subsec:mpc_results`), 4.3 robustezza al rumore lidar
  (`subsec:noise_results`), 4.4 errori sistematici e model mismatch
  (`subsec:systematic_results`).
- `Conclusion.tex` — Sez. 5 Conclusion (`sec:conclusion`).
- `appendixA.tex` — contiene due appendici. Appendice A (`app:fom`): metodi
  numerici e setup di co-simulazione del FOM (schema partizionato
  `subsec:partitioned`, che ospita anche lo schema del flusso di informazione fra
  i due solutori `fig:fsi_scheme`; solutore fluido, mesh e mesh motion
  `openfoam`, driver di co-simulazione `python-struct`). Appendice B (`app:vv`): verifica e
  validazione del FOM (convergenza di griglia, passo temporale, schema di
  accoppiamento). Lo switch `\appendix` è nel main su Overleaf
  (`\startappendices`), non in questo file.

Ogni macro-sezione (2, 3, 4) apre con un breve paragrafo che ne anticipa i
contenuti; mantenerlo aggiornato se si spostano sottosezioni.

## Modifiche manuali dell'utente
- L'utente modifica il testo mentre lo rilegge. Un agente che riprende un file e
  trova modifiche rispetto all'ultima versione prodotta da un agente deve
  preservarle: sono intenzionali. Non riscriverle, non annullarle e non
  riallineare il testo a una versione precedente; integrarle e costruirci sopra.

## Contenuto
- Non ripetere concetti già introdotti altrove nel testo. Prima di scrivere,
  controllare se l'argomento è già trattato; in tal caso rimandare con `\cref`
  invece di riscriverlo.
- Ogni simbolo nuovo va spiegato brevemente alla prima comparsa (grandezza fisica
  e, se serve, unità o convenzione di segno). Se è già stato definito in un
  capitolo precedente, non ridefinirlo. Vale senza eccezioni anche per i simboli
  intuibili dal contesto: vedi "Simboli e variabili".
- Ogni elemento esterno non derivato nel testo (modelli, metodi, teoremi, dati,
  standard, correlazioni) va corredato da citazione: aggiungere la voce in
  `bibliography.bib` e richiamarla con `\cite` nel punto pertinente. Niente
  attribuzioni informali del tipo "(Autore, anno)" nel corpo del testo.
- Materiale di terzi (figure, tabelle, dati pubblicati da altri) va usato solo con
  i permessi necessari e citando esplicitamente autore e fonte.

## Forma
- Evitare `\paragraph`, `\textbf`, `\emph` e corsivi se non strettamente
  necessari alla comprensione.
- Linguaggio sobrio e tecnico. Niente formule retoriche, enfasi o giri di frase
  tipici del testo generato automaticamente.
- Niente parentesi esplicative accanto a un termine — "stale (delayed)" è uno
  stilema da testo generato. Se un termine richiede una glossa tra parentesi,
  usare direttamente il termine più chiaro ("delayed"). Le parentesi restano
  per valori numerici, unità, sigle e riferimenti.
- Lessico vietato (scelta dell'autore): mai "plant" (usare "controlled system",
  "aeroelastic system", "reduced system" secondo il contesto) e mai
  "traction(s)" (usare "stresses" e formulazioni equivalenti).
- Preferire la forma sintetica: mostrare equazioni, algoritmi e risultati
  (formule, pseudo-codice, tabelle, figure) invece di descriverli a parole. La
  prosa serve a spiegare e collegare quando necessario, non a sostituire ciò che
  un'equazione o un algoritmo esprime in modo più compatto e preciso. A parità di
  chiarezza, la versione più breve è quella corretta.

## Simboli e variabili
Regole non negoziabili, valide su tutto il documento (corpo, appendici, caption,
label degli assi nelle figure).

- Un simbolo, una grafia. Ogni grandezza ha un solo simbolo in tutto il
  documento e si scrive sempre allo stesso modo: mai una variante nel testo e
  un'altra nell'equazione. La velocità del fluido è $\mathbf{u}$ ovunque, mai
  $\mathbf{u}_f$ in prosa e $\mathbf{u}$ in formula. Il pedice si usa solo per
  distinguere grandezze diverse, non per decorare la stessa grandezza in
  contesti diversi: $\dot{\mathbf{u}}_s$ è la velocità della superficie
  strutturale, quindi il pedice porta informazione ed è legittimo.
- Definire tutto, anche l'ovvio. Ogni simbolo che compare in un'equazione va
  definito esplicitamente alla prima comparsa, nel testo che introduce o segue
  l'equazione, anche quando il significato è deducibile dal contesto o dalla
  figura ($\mathbf{x}_\mathrm{EA}$, $\mathbf{x}_\mathrm{hinge}$, $d_x$, $d_y$,
  $I_{f,\mathrm{EA}}$, $\hat{\mathbf e}_z$, $p_\infty$, ...). Se un simbolo non
  merita una definizione, non merita di comparire.
- Scelte già fissate, da non ribaltare: velocità del fluido $\mathbf{u}$, velocità
  della superficie strutturale $\dot{\mathbf{u}}_s$, velocità di griglia ALE
  $\mathbf{w}$; il vettore degli ingressi del surrogato è $\boldsymbol{\eta}(t)$,
  $\boldsymbol{\eta}_n$ e mai $\mathbf{u}$, che resta riservato alla velocità del
  fluido (vale anche per le figure: l'etichetta in `generate_ldnet_arch.py` è
  $\boldsymbol{\eta}(t)$); picco di raffica $W_0$ (mai $W_\mathrm{max}$, le figure dei
  risultati usano già $W_0$) e ampiezza adimensionale $r_g = W_0/U_\infty$;
  $R$ è solo il peso dello sforzo di controllo dell'MPC e $R^\star$ il valore
  selezionato caso per caso nello sweep dei Results, la matrice di rotazione
  è $\mathbf{R}(\alpha)$; corda $c_\mathrm{ref}$ (mai $c$); profilo di raffica
  lisciato $\mathbf{W}^\ast$ e preview $\widehat{W}_k$; inerzie del flap
  $I_{f,\mathrm{CG}}$, $I_{f,\mathrm{EA}}$, $I_{f,\mathrm{hinge}}$; nello sweep di
  architettura $L$ è il numero totale di hidden layer delle due reti e $P$ il numero
  di pesi allenabili. Le posizioni
  vettoriali sono $\mathbf{x}_\mathrm{EA}$, $\mathbf{x}_\mathrm{hinge}(t)$; le
  corrispondenti stazioni scalari lungo la corda sono $x_\mathrm{EA}$,
  $x_{\mathrm{hinge},0}$.
- Definizione dove serve: subito prima o subito dopo l'equazione che introduce
  il simbolo, non tre paragrafi più avanti e non in appendice. Per i blocchi con
  molti simboli nuovi (matrici strutturali, vettori di forzamento) elencarli in
  un unico periodo dopo l'equazione, nell'ordine in cui compaiono.

## Equazioni
- Qualificatore di dominio: a destra dell'equazione si indica dove vale, nella
  forma `\text{in } \Omega_f(t)\times(0,T]`, `\text{on } \Gamma_\mathrm{in}\times(0,T]`,
  `\text{in } (0,T]` per le equazioni senza dipendenza spaziale. Non usare la
  forma `\mathbf{x}\in\Omega_f(t),\; t\in(0,T]`: una variabile non può comparire
  nel qualificatore se non compare nel membro sinistro dell'equazione. La scelta
  fra `in` e `on` segue l'oggetto geometrico: `in` per domini di volume e
  intervalli temporali, `on` per bordi e interfacce.
- Notazione temporale: il tempo continuo $(t)$ si usa solo nella sezione Model e
  per le storie temporali fisiche; la griglia discreta $t_n = n\Delta t$ è
  introdotta all'inizio dei Methods (§3.1) e ogni segnale campionato usa il
  pedice $n$ ($\boldsymbol{\zeta}_n$, $\mathbf{s}_n$, $\boldsymbol{\eta}_n$, $W_n$); l'apice
  $(k)$, $k=1,\dots,N$, è riservato ai passi predetti nell'orizzonte MPC; gli
  indici di iterazione degli ottimizzatori usano $i$, mai $t$; $j$ nodi
  spaziali lidar, $m$ nodi fusi.
- Simboli riservati: lo stato latente è sempre $\mathbf{s}$ (mai $\mathbf{z}$,
  anche nell'MPC: $\mathbf{s}^{(k)}$); lo stato strutturale è
  $\boldsymbol{\zeta} = [h,\dot h,\alpha,\dot\alpha]^T$ e mai $\mathbf{x}$, che
  resta la coordinata spaziale ($\mathbf{x}_\mathrm{EA}$,
  $\mathbf{x}_\mathrm{hinge}$, $\mathbf{x}_\mathrm{query}$);
  $\boldsymbol{\xi}_n = (\mathbf{s}_n,
  \boldsymbol{\zeta}_n^\mathrm{ro})$ è lo stato aumentato del rollout di training; i
  candidati della griglia MPC sono $\delta_g$, $g=1,\dots,G$; $(r)$ indicizza le
  traiettorie del dataset; le finestre di co-simulazione (§3.2) vivono sulla
  griglia propria $\tau_l = l\,\Delta t_\mathrm{win}$, distinta da $t_n$; la
  lunghezza del rollout di training è $N_\mathrm{rollout}$ (non
  $T_\mathrm{rollout}$). Altri usi legittimi di $\tau$: $\boldsymbol{\tau}$ in
  grassetto è il tensore viscoso (Model), $\tau$ scalare è il time shift
  dell'errore sistematico di preview (§4.4); non introdurne altri. Nella
  co-simulazione (§3.2) gli stati ai bordi finestra si scrivono come valutazioni
  continue, $\boldsymbol{\zeta}(\tau_l)$, $\mathbf{f}(\tau_{l+1})$: niente pedici o apici
  discreti, che appartengono solo al closed loop di §3.1.
- Simboli di relazione (`\approx`, `=`, `\le`, `\sim`) vogliono sempre un membro a
  sinistra e uno a destra: mai "reaching $\approx 4\cdot10^{-5}$", ma "reaching a
  training loss $\mathcal{L}_\mathrm{train}\approx 4\cdot10^{-5}$". Se la grandezza a
  sinistra non ha ancora un simbolo, o glielo si dà, o si scrive il valore in prosa
  ("about", "near") senza il simbolo di relazione.
- Righe di un sistema: quando si commenta un blocco `subequations` riga per riga,
  usare sempre lo stesso formato, "Row \labelcref{...}" al singolare e "Rows
  \labelcref{...}" al plurale, con gli intervalli scritti "Rows (1c) to (1h)".
  Mai alternare il richiamo numerato a perifrasi tipo "the first two rows" o
  "the wall conditions": se una riga serve nel testo, va etichettata e richiamata.
- Numerare un'equazione solo se richiamata nel testo; altrimenti usare le varianti
  non numerate (`equation*`, `align*`, `\nonumber`).
- I riferimenti incrociati sono gestiti da `cleveref`: usare `\cref`/`\Cref`, che
  generano da soli il nome dell'oggetto ("Equation", "Figure", "Table", ...). Non
  scrivere il prefisso a mano né usare `\eqref`.
- Sistemi di equazioni: `subequations` con label di gruppo e sub-label per i
  singoli righi; graffa a sinistra con `align` e `[left=\empheqlbrace]`, oppure
  `\left\{\dots\right.` intorno a un `aligned` quando serve una sola label.

## Figure, tabelle e algoritmi
- Usare gli ambienti già forniti dal template di tesi (PoliMi); non caricare
  pacchetti nuovi nel preambolo.
- Ogni figura, tabella o algoritmo ha una caption che ne descrive il contenuto,
  un `\label`, ed è richiamato nel testo con `\cref`.
- Posizionamento: figure e tabelle stanno sempre in cima alla pagina. Si scrive
  `\begin{figure}[t]` e `\begin{table}[t]`, con `figure*`/`table*` per quelle a
  piena larghezza. Mai `[h]`, `[H]`, `[htbp]`, `[b]`, `[bp]` o altre
  combinazioni: un float non deve spezzare il testo né finire a piè di pagina.
  Di conseguenza il testo non può mai dipendere dalla posizione del float sulla
  pagina — niente "the figure below" o "as shown above", si richiama solo con
  `\cref`.
- Figure: TikZ per le figure vettoriali (citare `\cite{tikz}`) oppure
  `\includegraphics[...]{...}` (`.png`, `.jpg`, `.eps`); le immagini raster vanno
  nella cartella `Images/`. `\subfloat` per le sotto-figure, ciascuna con caption
  e label propri.
- Tabelle: ambienti `table`/`tabular` nello stile del template (intestazione con
  `\rowcolor{bluePoli!40}`, macro di spaziatura `\T\B`, righe separate da
  `\hline`). Titolo opzionale sopra la tabella con `\caption*{...}`; per una tabella
  a piena larghezza usare `table*`, sempre con `[t]`. L'uso di `\textbf` è ammesso
  solo nelle intestazioni di tabella e nel titolo `\caption*`.
- Algoritmi: ambienti `algorithm` + `algorithmic` con i comandi `\STATE`,
  `\FOR`/`\ENDFOR`, `\IF`/`\ENDIF`, `\WHILE`/`\ENDWHILE`.

## Linee guida per i plot
- Ogni figura di risultati è generata da un unico script archiviato nel repo
  (`light/tests/` o `light/noise/`): niente figure ritoccate a mano, ogni plot
  deve essere rigenerabile con una sola esecuzione.
- Font serif con `mathtext.fontset='cm'` (coerente con il corpo della tesi),
  dimensione ~9 pt alla larghezza finale di stampa. Dimensionare la figura al
  target LaTeX (`\textwidth` ≈ 6.3 in, `0.48\textwidth` ≈ 3 in) così i font non
  vengono riscalati; esportare PNG a 300 dpi nella cartella `Images/`.
- Nessun titolo dentro la figura (`suptitle`/`set_title`): la descrizione va
  nella caption LaTeX.
- Label degli assi con la stessa notazione del testo ($W_0$, $T_g$, $H$, $k$,
  $\delta$, CLred, ...) e unità tra parentesi quadre.
- Nelle figure dei risultati compare solo il controllore MPC del capitolo:
  nessuna serie di controllori legacy (one-step optimal, proporzionale).
- Legende senza jargon interno di sviluppo (niente codenames come "E2-combo",
  "A2/B2", "home cell"): etichette descrittive in inglese sobrio.
- Studi con seed multipli: linea/barra della media più banda o whisker min–max;
  il numero di seed è dichiarato in caption. I punti che violano il criterio di
  stabilità (flag) vanno evidenziati con un marker dedicato.
- Palette sobria e coerente tra tutte le figure: stesso colore per la stessa
  quantità ovunque (es. open-loop grigio, closed-loop blu); griglia leggera.
- Sotto-figure affiancate (`\subfloat`) devono avere la stessa dimensione
  fisica (stessa `figsize`, export senza crop variabile: niente
  `bbox_inches='tight'` se altera le proporzioni tra i pannelli) e, quando
  mostrano le stesse quantità, gli stessi limiti e la stessa scala degli assi,
  così da essere confrontabili a colpo d'occhio; la legenda condivisa compare
  in un solo pannello.

## Teoremi, proposizioni ed elenchi
- Teoremi e proposizioni: ambienti `theorem` e `proposition` del template. La
  dimostrazione si apre con `\textit{Proof.}` e si chiude con `\vspace{0.3cm}`.
- Elenchi: `itemize` per liste puntate, `enumerate` per liste numerate.

## Fonti — non inventare
- Non inventare nulla. Per la sezione della simulazione high-fidelity (FSI,
  modello strutturale, caso OpenFOAM, parametri e schemi numerici) estrarre
  sempre ogni dato — parametri, valori, tolleranze, schemi — dal codice sorgente
  in `/LDNet_OF/light` e in `NACA2312_cluster/cosim_main` (in particolare
  `cosim_driver.py` e i file del caso in `constant/`, `system/`, `0.orig/`).
  Se un valore non è nel codice, non scriverlo.

## Validazione e repository
- Gli studi di verifica e validazione (convergenza di griglia, convergenza
  temporale, validazione dell'integratore strutturale, convergenza dello schema
  di accoppiamento) sono riportati integralmente nel repository GitHub linkato in
  tesi. Per questi studi è ammesso rimandare al repository e riportare nel testo
  solo il risultato sintetico (valore numerico o verdetto), senza riprodurre la
  procedura completa, le figure di dettaglio o le tabelle intermedie.
- Vale comunque il vincolo "non inventare": il risultato riportato deve
  corrispondere all'output effettivo dello script o dei dati nel repository. Se un
  risultato numerico non è archiviato in forma verificabile, descrivere lo studio
  e il suo criterio di accettazione (la tolleranza codificata nel test), senza
  fabbricare valori misurati.
