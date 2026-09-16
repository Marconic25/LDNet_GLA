#!/bin/bash
# final/studies/run_all.sh — the post-dataset_v6 chain, in dependency order.
#
# dataset_v6 (dt=3.16e-5, coupling window 29, Courant~1) sits at the root of
# everything below: preprocess -> LDNet training -> accuracy studies ->
# depth ladder -> field reconstruction -> CS-25 grid -> noise axes. This
# script documents and sequences that chain; it does not reimplement any
# step that belongs to a tree outside final/studies/ (src/, recon/, the
# root train_*.sh scripts) -- those are emitted as the exact command to run,
# with v6 paths substituted, and a comment. Only the CS-25 grid and noise
# axes (steps 6-7) live in final/studies/ and are things this script can
# actually submit.
#
# --dry-run is the DEFAULT (no compute is launched). Pass --go to actually
# qsub steps 6-7. Steps 0-5 are ALWAYS printed only, never submitted by this
# script, even with --go: they belong to other trees/agents and this script
# cannot verify their preconditions (dataset landed, previous stage's
# checkpoint exists, etc.) from final/studies/.
#
# HARD RULE (every launcher in this repo must follow it): compute goes
# through qsub on a compute node, never bare `nohup`/inline execution on the
# login node. light/noise/run_axis.sh does the latter and is explicitly
# NOT the pattern to copy here -- see light/dagger_fom/NOTES.md, "REGOLA
# OPERATIVA -- mai eseguire calcolo su login01" (login01 hit load average
# 107 with 45 users; the user's jobs were administratively removed as a
# result). Steps 6-7 below submit through final/studies/run_study.pbs.
#
# Cluster politeness: max_user_run=4 on this account (see the cluster
# endpoint notes) and the scheduler preempts everything if another session
# submits concurrently. Prefer launching one job at a time over blasting the
# whole grid; the qsub calls below are deliberately left as separate lines
# you can run one at a time rather than a background-parallel fan-out.
#
# Usage:
#   ./run_all.sh              # dry run: print every command, submit nothing
#   ./run_all.sh --dry-run    # same, explicit
#   ./run_all.sh --go         # actually qsub steps 6-7 (CS-25 grid + noise axes)
set -euo pipefail
cd "$(dirname "$0")"

MODE="dry-run"
[ "${1:-}" = "--go" ] && MODE="go"

# v6 paths (mirrors paths.py's defaults; override the same way paths.py does)
V6_MODEL_DIR="${LDNET_V6_MODEL_DIR:-/work/u10677113/LDNet_GLA/clean/models_rollout_v6/latent_10}"
V6_DATASET_ROOT="${LDNET_V6_DATASET_ROOT:-/work/u10677113/NACA2312/dataset_v6}"
V6_TRAINING_DATA_ROOT="${LDNET_V6_TRAINING_DATA_ROOT:-/work/u10677113/LDNet_GLA/data_v6}"
V6_DAMPED_DIR="/work/u10677113/LDNet_GLA/clean/models_damped_l003_full_v6"

say() { echo; echo "# $*"; }
emit() { echo "    $*"; }
run_qsub() {
    if [ "$MODE" = "go" ]; then
        echo "    + $*"
        eval "$*"
    else
        echo "    (dry-run, not submitted) $*"
    fi
}

echo "=== final/studies/run_all.sh : MODE=$MODE ==="

# ---------------------------------------------------------------------------
say "STEP 0/8 [external, NOT in this repo] -- preprocess: regenerate dataset_v6"
# The FOM co-simulation campaign driver lives in the NACA2312_cluster repo
# (cosim_main/, cosim_driver.py -- see light/latex/AGENTS.md, "Fonti -- non
# inventare"), not in LDNet_OF. Regenerate the campaign with the new
# numerics and land it at V6_DATASET_ROOT:
emit "dt_CFD = 3.16e-5 s, coupling window N_win = 29 steps (~9.16e-4 s window), Courant ~ 1"
emit "output root: $V6_DATASET_ROOT   (v6 sibling of dataset_v5 in recon/cluster/field_run.pbs)"
emit "then preprocess the raw campaign into the HDF5 training set consumed by src/:"
emit "  GLA_train.h5 / GLA_valid.h5 / GLA_test.h5  ->  $V6_TRAINING_DATA_ROOT"
emit "(same preprocessing step that produced light/../data/GLA_*.h5 for dataset_v5;"
emit " script not identified from final/studies/ -- confirm with whoever owns the"
emit " NACA2312_cluster preprocessing before wiring this step up for real)"

# ---------------------------------------------------------------------------
say "STEP 1/8 [external: root train_*.sh + src/] -- LDNet training (loads model)"
# train_l003_full.sh and train_rollout.sh are plain apptainer-wrapper scripts
# (not PBS) that hardcode OUTDIR/DATA_OVERRIDE for the v5 run; they are NOT
# forked here. src/sensitivity_latent_damped_ckpt.py and
# src/sensitivity_latent_rollout.py already read DATA_OVERRIDE/OUTDIR/
# WARMSTART from the environment (confirmed by reading both files), so the
# v6 run is the SAME underlying command with v6 env vars, wrapped in PBS
# instead of the bare apptainer invocation train_l003_full.sh uses:
emit "qsub -N ldnet_v6_l003 -q gpu -l select=1:ncpus=8:ngpus=1 -l walltime=24:00:00 -j oe \\"
emit "     -v DATA_OVERRIDE=$V6_TRAINING_DATA_ROOT,OUTDIR=$V6_DAMPED_DIR,LAMBDA_DAMP=0.003,NADAM=400,NBFGS=4000 \\"
emit "     -- apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \\"
emit "        /work/u10677113/tensorflow_gpu.sif bash -c \\"
emit "        'pip install -q scipy matplotlib h5py; cd /work/u10677113/LDNet_GLA && python3 -u src/sensitivity_latent_damped_ckpt.py'"
emit "# then, warm-started from the above (mirrors train_rollout.sh with v6 paths):"
emit "qsub -N ldnet_v6_rollout -q gpu -l select=1:ncpus=8:ngpus=1 -l walltime=16:00:00 -j oe \\"
emit "     -v DATA_OVERRIDE=$V6_TRAINING_DATA_ROOT,WARMSTART=$V6_DAMPED_DIR/latent_10,LAMBDA_DAMP=0.003,ROLLOUT_LEN=800,NADAM=0,NBFGS=500,W_LOAD=1.0,OUTDIR=$(dirname "$V6_MODEL_DIR") \\"
emit "     -- apptainer exec --writable-tmpfs --bind /work/u10677113:/work/u10677113 \\"
emit "        /work/u10677113/tensorflow_gpu.sif bash -c \\"
emit "        'pip install -q scipy matplotlib h5py; cd /work/u10677113/LDNet_GLA && python3 -u src/sensitivity_latent_rollout.py'"
emit "# result: $V6_MODEL_DIR  (consumed by final/studies/run_v6.py via LDNET_V6_MODEL_DIR / MD_OVERRIDE)"

# ---------------------------------------------------------------------------
say "STEP 2/8 [external: src/, already PBS] -- accuracy studies (latent/input sensitivity)"
# run_rollout_sweep.pbs is ALREADY a proper PBS file with DATA_OVERRIDE/
# RESULTS_OVERRIDE hooks; only the v6 paths change. Backs subsec:training_results.
emit "qsub -v INPUT_SET=6,DATA_OVERRIDE=$V6_TRAINING_DATA_ROOT,RESULTS_OVERRIDE=/work/u10677113/LDNet_GLA/results_v6/sensitivity_rollout,WARMSTART_ROOT=/work/u10677113/LDNet_GLA/results_v6/sensitivity ../../run_rollout_sweep.pbs"
emit "# repeat for INPUT_SET in {2,4} for full grid coverage, as sync_rollout_sweep.sh does for v5"

# ---------------------------------------------------------------------------
say "STEP 3/8 [external: recon/loads_depth/] -- depth ladder (loads-model depth sweep)"
emit "qsub -v DATA_OVERRIDE=$V6_TRAINING_DATA_ROOT,RESULTS_OVERRIDE=/work/u10677113/LDNet_GLA/results_v6/loads_depth \\"
emit "     ../../recon/loads_depth/... (see recon/cluster/loads_depth.pbs for the current v5 invocation; substitute the v6 data/results roots above)"
emit "# NOTE: recon/ is a separate thesis thread from light/ (field reconstruction,"
emit "# not loads/MPC -- see the 'recon-lives-in-recon-not-light' memory note); this"
emit "# step and step 4 retrain that thread against dataset_v6 too, they do not feed"
emit "# steps 6-7 directly."

# ---------------------------------------------------------------------------
say "STEP 4/8 [external: recon/train_fields.py] -- field reconstruction"
# recon/train_fields.py is a plain argparse CLI (--train/--valid/--test/--out/
# --latents/...), not env-driven; the v6 command just points --train/--valid/
# --test at the v6 FIELDS_*.h5 (built by recon/build_fields_h5.py from the v6
# campaign) and --out at a fresh models dir so v5's recon/models/ is untouched.
emit "python3 recon/build_fields_h5.py ... --out $V6_TRAINING_DATA_ROOT/fields   # from $V6_DATASET_ROOT"
emit "qsub -q cpu -l select=1:ncpus=8 -l walltime=24:00:00 -j oe -- apptainer exec --writable-tmpfs \\"
emit "     --bind /work/u10677113:/work/u10677113 /work/u10677113/tensorflow_gpu.sif bash -c \\"
emit "     'pip install -q h5py scipy matplotlib pandas; cd /work/u10677113/NACA2312/recon && \\"
emit "      python3 -u train_fields.py --train $V6_TRAINING_DATA_ROOT/fields/FIELDS_train.h5 \\"
emit "        --valid $V6_TRAINING_DATA_ROOT/fields/FIELDS_valid.h5 \\"
emit "        --test $V6_TRAINING_DATA_ROOT/fields/FIELDS_test.h5 \\"
emit "        --out models_v6 --latents 1,5,10 --mean-split'"

# ---------------------------------------------------------------------------
say "STEP 5/8 [depends on step 1] -- sanity-check the v6 model loads"
emit "LDNET_V6_MODEL_DIR=$V6_MODEL_DIR python3 -s -u run_v6.py   # prints MD, CLTRIM, LAM"

# ---------------------------------------------------------------------------
say "STEP 6/8 [this repo, final/studies/] -- CS-25 grid (tab:cs25_grid, fig_ch3_trace_*)"
for w0 in 10 20 30; do
    run_qsub "qsub -v SCRIPT=cs25_combo_study.py,ENVS=\"W0=$w0\" run_study.pbs"
done
say "  -- after all three rows finish (check with: qstat -u \$USER):"
emit "python3 -s -u cs25_combo_plots.py     # summary.md -> transcribe into tab:cs25_grid"
emit "python3 -s -u cs25_thesis_figs.py     # fig_ch3_trace_W30Tg04.png, fig_ch3_trace_W30Tg07.png, ..."

# ---------------------------------------------------------------------------
say "STEP 7/8 [this repo, final/studies/] -- noise axes (fig:noise_white, fig:noise_bias_timing)"
run_qsub "qsub -v SCRIPT=noise_white_combo.py run_study.pbs"
run_qsub "qsub -v SCRIPT=noise_calib_combo.py run_study.pbs"
run_qsub "qsub -v SCRIPT=noise_timing_combo.py run_study.pbs"
run_qsub "qsub -v SCRIPT=e2_combo.py,ARGS=flat run_study.pbs"
run_qsub "qsub -v SCRIPT=e2_combo.py,ARGS=dlr run_study.pbs"
run_qsub "qsub -v SCRIPT=e2_combo_cells.py,ARGS=W10T07 run_study.pbs"
run_qsub "qsub -v SCRIPT=e2_combo_cells.py,ARGS=W30T07 run_study.pbs"
say "  -- after these finish, run LOCALLY (matplotlib only, no cluster needed):"
emit "python3 -s -u plots_noise_white_combo.py     # fig_ch3_noise_white.png"
emit "python3 -s -u plots_thesis_bias_timing.py    # fig_ch3_bias_timing.png"

echo
echo "=== done (MODE=$MODE). Re-run with --go to actually submit steps 6-7. ==="
