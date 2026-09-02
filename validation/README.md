# Verifica e validazione del full-order model (NACA 2312 FSI)

Script e risultati della campagna di verifica del solutore partizionato
(OpenFOAM 7 `pimpleFoam` + driver Python) usata nell'appendice A della tesi.
I run girano sul cluster HPC Polimi in `/work/u10677113/NACA2312`; questa
cartella archivia gli script cosi come eseguiti e i report finali
(campagna del 2026-07-14/16).

## Contenuto

- `grid_convergence/` — studio di convergenza di griglia su 4 mesh annidate
  (36k / 76k / 375k / 1.9M celle), RANS stazionario al trim, GCI con
  estrapolazione di Richardson (Fs = 1.25). Lancio: `qsub submit_gci.pbs`
  dopo aver copiato `grid_convergence_study.py` nel workdir. Output:
  `grid_convergence_report.txt`, `results.json`.
- `temporal/` — raffinamento della window di accoppiamento a dt fisico di
  produzione (7e-5 s): window 100/50/25/10 step = 7/3.5/1.75/0.7 ms, step di
  flap 0-15 gradi su [0, 0.2] s, metriche CL_steady e NRMSE sul segnale CL(t)
  rispetto al livello piu fine. Un job PBS per livello
  (`submit_temporal_DT*.pbs`, argomento `--levels`); il report completo si
  rigenera da cache con `python3 temporal_study.py`. Output:
  `temporal_study_report.txt`, `cl_histories.csv`.
- `courant/` — sweep di maxCo (0.5/1/2/4, `adjustTimeStep yes`,
  `maxDeltaT 5e-4`) a window di accoppiamento fissa 3.5 ms sul caso raffica di
  riferimento (W_g0 = 40 m/s, 1-cos su [0.1, 0.9] s, warm start da
  checkpoint), piu caso `Co_fixed` identico alla produzione (dt fisso 7e-5).
  `python3 courant_sweep.py setup|run <caso>|report`. Output:
  `courant_sweep_report.txt`.
- `coupling/` — validazione dello schema di accoppiamento loose: replay delle
  858 finestre del run di raffica di produzione con i carichi registrati,
  bilancio energetico di interfaccia, confronto loose vs strong (punto fisso
  per finestra). `python3 run_coupling_validation.py` dentro `cosim_main/`.
  Output: `summary.txt`.
- `cluster/campaign_status.sh` — snapshot di stato della campagna PBS con
  rilascio automatico a fasce dei job in hold.

## Risultati chiave

- Mesh di produzione M3 (375k celle): CL entro 0.61% e Cd entro 1.02% dalla
  M4 (1.9M), GCI medium 0.84% su CL.
- Window di accoppiamento di produzione (3.5 ms) convergente con fattore 2 di
  margine: CL_steady varia meno dello 0.05% su un range 10x, NRMSE < 0.7%.
- Passo temporale: coarsificare non cambia il picco (Co4 vs fixed: 0.3%);
  raffinare lo alza di ~5% per dimezzamento senza asintoto nel range testato
  (fino a +17% a dt 1.4e-5). Sul caso estremo W/U = 0.5 il picco entra in
  stallo dinamico ed e' dt-sensibile: limite dichiarato del FOM a bordo
  inviluppo. Co_max effettivo di produzione nel gust: ~3.5.
- Accoppiamento loose: bilancio energetico globale chiuso allo 0.78%,
  858/858 finestre a punto fisso in una iterazione, differenza loose-strong
  nulla sul replay a carichi registrati.

## Avvertenze operative (lezioni apprese)

- `patch_snappy` ora fallisce rumorosamente se una regex non matcha il
  template: un no-op silenzioso produce mesh tutte uguali al template.
- Il driver di co-simulazione purga le time-dir dei processori piu vecchie
  delle ultime 3: senza purge ~8 MB/finestra/processore saturano la quota.
- Lo scarto del transitorio di restart nelle finestre e' time-based (7e-4 s):
  con campionamento forze a ogni step, uno scarto a conteggio fisso lascia
  entrare gli spike di restart nel forzamento strutturale.
