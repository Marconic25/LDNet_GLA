# Linee guida per la cartella LaTeX

Regole vincolanti per qualunque agente che modifichi i file di questa cartella
(`chapter*.tex`, `Introduction.tex`, `bibliography.bib`).

## Contesto
- Tesi magistrale in Ingegneria Aeronautica.
- Formato articolo, lunghezza obiettivo circa 30 pagine. Ogni aggiunta va
  pesata contro questo limite: se un contenuto non è necessario, non si scrive.

## Contenuto
- Non ripetere concetti già introdotti altrove nel testo. Prima di scrivere,
  controllare se l'argomento è già trattato; in tal caso rimandare con
  `\ref`/`\eqref` invece di riscriverlo.
- Ogni elemento esterno non derivato nel testo (modelli, metodi, teoremi, dati,
  standard, correlazioni) va corredato da citazione: aggiungere la voce in
  `bibliography.bib` e richiamarla con `\cite` nel punto pertinente. Niente
  attribuzioni informali del tipo "(Autore, anno)" nel corpo del testo.

## Forma
- Evitare `\paragraph`, `\textbf`, `\emph` e corsivi se non strettamente
  necessari alla comprensione.
- Linguaggio sobrio e tecnico. Niente formule retoriche, enfasi o giri di frase
  tipici del testo generato automaticamente.

## Fonti — non inventare
- Non inventare nulla. Per la sezione della simulazione high-fidelity (FSI,
  modello strutturale, caso OpenFOAM, parametri e schemi numerici) estrarre
  sempre ogni dato — parametri, valori, tolleranze, schemi — dal codice sorgente
  in `/LDNet_OF/light` e in `NACA2312_cluster/cosim_main` (in particolare
  `cosim_driver.py` e i file del caso in `constant/`, `system/`, `0.orig/`).
  Se un valore non è nel codice, non scriverlo.
