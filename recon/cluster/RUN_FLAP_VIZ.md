# Importare una run con flap grande → mesh deformata per ParaView

Obiettivo: portare in locale la mesh **deformata** (flap che si muove) di una run con
`delta` e velocità di attuazione grandi, in formato ParaView. Le immagini le fai poi tu
in ParaView.

Le famiglie: **A** = solo raffica (`delta=0`, inutile qui), **B** = solo flap,
**Cc** = raffica + flap. Cerca una **B** o **Cc**.

⚠️ `/scratch_local` è **locale al nodo** e ripulito dopo ~30 giorni: il caso OpenFOAM
ricostruito (le time-dir `[0-9]*`) di solito **non c'è più** e va rigenerato, e l'export
va lanciato **sullo stesso nodo** dove sta il caso.

---

## 0. Connessione

```bash
# VPN gp-dmat-saml.vpn.polimi.it attiva, poi:
ssh u10677113@10.78.18.100
. /etc/profile.d/pbs.sh
cd /work/u10677113/NACA2312/recon/cluster   # dove stanno questi script
```

Copia gli script aggiornati dal repo locale, se serve:
```bash
# dal PC locale
scp recon/cluster/scan_flap.py recon/cluster/export_deformed.py \
    recon/cluster/export_foamtovtk.sh \
    u10677113@10.78.18.100:/work/u10677113/NACA2312/recon/cluster/
```

## 1. Scegliere la run (login node, legge solo i CSV)

```bash
python3 scan_flap.py --top 25
# solo le famiglie con flap:
python3 scan_flap.py --top 25 --family Cc
python3 scan_flap.py --top 25 --family B
```

Colonne: `d_range` (peak-to-peak di delta, deg), `|d|max`, `rate` (deg/s),
`W_max` (raffica), `case?` = `Nt` se un caso OF ricostruito con N time-dir è già su
disco (pronto per l'export), `-` se va rigenerato.

Scegli una sim in cima con `d_range` grande (e `rate` grande se vuoi un flap veloce).
Esempio sotto: `SIM=sim_Cc_060_test`.

```bash
SIM=sim_Cc_060_test
```

## 2. Avere il caso OpenFOAM ricostruito

**Se `case?` era `Nt`** (già presente): salta al punto 3, usando quel path come `CASE`.

**Se era `-`**: rigenera il caso (replay open-loop con la raffica + il flap registrato).
Il job scrive il caso in `/scratch_local/$USER/fldrun_$SIM`:

```bash
qsub -v SIM=$SIM field_run_flap.pbs
qstat -u u10677113                       # attendi la fine
tail -f /work/u10677113/NACA2312/recon_fields/$SIM/run.log
```

⚠️ Annota **su quale nodo** gira il job (`qstat -n1 -u u10677113`): il caso in
`/scratch_local` è visibile solo lì. Per l'export lancia un job interattivo su quel nodo,
oppure aggiungi i comandi di export in coda a `field_run_flap.pbs` prima che scada.

```bash
CASE=/scratch_local/$USER/fldrun_$SIM
ls -d $CASE/[0-9]* | wc -l               # deve essere > 0
```

## 3. Esportare la mesh deformata

### Opzione A — pyvista, slice 2D (consigliata: leggera, mostra bene il flap)

```bash
source ~/cosim_env/bin/activate          # stesso env di extract_fields.py
OUT=/work/u10677113/NACA2312/recon_fields/$SIM/vtk

# 3 istantanee: delta minimo, ~0, massimo
python3 export_deformed.py --case $CASE --out $OUT --name $SIM --auto

# oppure una serie ridotta per animazione (ogni 10 time-dir):
python3 export_deformed.py --case $CASE --out $OUT --name $SIM --stride 10

# oppure tempi espliciti:
python3 export_deformed.py --case $CASE --out $OUT --name $SIM --times 1.2 1.4 1.6
```

Produce `$OUT/$SIM.pvd` + `$SIM_XXXX.vtp`. `--full3d` per l'intera mesh interna (.vtu).

### Opzione B — foamToVTK nativo (3D completo + patch di parete flap/profilo)

```bash
bash export_foamtovtk.sh $CASE            # tutti i tempi
bash export_foamtovtk.sh $CASE "1.2:1.6"  # solo un range (timeSelector OF)
# -> $CASE/VTK/  (mesh interna + sottocartelle per le patch)
```

Nota: `foamToVTK` legge dal container `/work/u10677113/of7.sif`. Se `$CASE` è in
`/scratch_local`, copia prima `VTK/` in `/work` per poterlo scaricare (scratch è node-local):
```bash
cp -r $CASE/VTK /work/u10677113/NACA2312/recon_fields/$SIM/VTK
```

## 4. Scaricare in locale

```bash
# dal PC locale, dentro il repo
mkdir -p recon/data/${SIM}_vtk
# Opzione A:
scp -r u10677113@10.78.18.100:/work/u10677113/NACA2312/recon_fields/$SIM/vtk/* \
    recon/data/${SIM}_vtk/
# Opzione B:
scp -r u10677113@10.78.18.100:/work/u10677113/NACA2312/recon_fields/$SIM/VTK \
    recon/data/${SIM}_vtk/
```

## 5. ParaView (locale — lo fai tu)

Apri `recon/data/${SIM}_vtk/$SIM.pvd` (opzione A) o la serie `*.vtk` (opzione B).
Representation → **Surface With Edges** per vedere la griglia; colora per `p` o `U`;
scorri i time step: il flap si muove. `recon/data/${SIM}_vtk/` è ignorato da git se
pesante — tienilo fuori dal commit.
