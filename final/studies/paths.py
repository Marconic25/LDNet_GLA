"""
final/studies/paths.py — the single place where the v6 campaign differs from v5.

Every other script in final/studies/ imports this module first. It does two
things:

  1. Puts light/ and light/noise/ on sys.path so the dataset-independent
     control/integration code there (structure.py, ldnet_aero.py, optimal.py,
     harness_noise.py, controllers_ref.py) can be imported unchanged, exactly
     like light/tests/cs25_combo_study.py does today with
     `sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))`
     — here rooted at the repo, so it works regardless of the caller's cwd.

  2. Exposes the three v6-specific roots (model dir, results root, dataset
     root), each overridable by environment variable with a sane default.
     Nothing else in final/studies/ should hardcode a model path, a results
     directory, or a dataset path — that is exactly the exception the fork
     rule in the task carves out, and this module is where it is paid.

Design note — why harness_noise.py needs a patch function instead of an env
var: light/run.py already reads its model directory from MD_OVERRIDE (see
light/run.py lines 25-26), so retargeting it at v6 is a pure environment-
variable change (see run_v6.py) and light/run.py is never forked. But
light/noise/harness_noise.py computes its model directory from its OWN
__file__ with no override hook:

    MD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', '..', 'clean', 'models_rollout', 'latent_10')

Forking harness_noise.py just to parameterize MD would duplicate ~170 lines
of verified rollout/metrics logic (the exact thing "import, don't fork"
exists to prevent) for a one-line change. Instead, load_v6_harness() imports
it unchanged and then monkey-patches the already-imported module object's
`aero`, `CLTRIM`, `LAM`, `MD` globals. This is safe because rollout()/
metrics() in harness_noise.py read `aero`/`CLTRIM`/`LAM` as *module globals
at call time*, never as captured defaults — and because Python caches
modules by name, so every other file that does `import harness_noise as H`
in the same process (controllers_ref.py, e2_combo.py, ...) sees the same
patched object. The important ordering constraint is documented on the
function itself.
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout / sys.path
# ---------------------------------------------------------------------------
STUDIES_DIR = Path(__file__).resolve().parent            # final/studies
FINAL_DIR   = STUDIES_DIR.parent                          # final
REPO_ROOT   = FINAL_DIR.parent                             # repo root

LIGHT_DIR       = REPO_ROOT / 'light'
LIGHT_NOISE_DIR = LIGHT_DIR / 'noise'

for _p in (str(LIGHT_DIR), str(LIGHT_NOISE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# v6 roots — the only things that differ from the v5 campaign in light/
# ---------------------------------------------------------------------------

# LDNet model directory (NNdyn_weights.weights.h5 / NNrec_weights.weights.h5 /
# config.json), trained on dataset_v6. Default mirrors the existing
# models_rollout_L12 sibling-directory convention (see
# light/tests/launch_cs25_combo_L12.sh) rather than overwriting
# clean/models_rollout/latent_10, which is the v5 archive.
V6_MODEL_DIR = os.environ.get(
    'LDNET_V6_MODEL_DIR',
    str(REPO_ROOT / 'clean' / 'models_rollout_v6' / 'latent_10'))

# Raw FOM campaign root for dataset_v6 (co-simulation output per simulation
# folder, sim_params.csv etc.) — the v6 sibling of dataset_v5 referenced in
# recon/cluster/field_run.pbs. Lives on the cluster filesystem; only used
# here as a documented convention for run_all.sh, never dereferenced at
# import time.
V6_DATASET_ROOT = os.environ.get(
    'LDNET_V6_DATASET_ROOT', '/work/u10677113/NACA2312/dataset_v6')

# Preprocessed HDF5 training root (GLA_train.h5 / GLA_valid.h5 / GLA_test.h5)
# consumed by src/sensitivity_latent*.py via DATA_OVERRIDE — distinct from
# V6_DATASET_ROOT (that is the raw per-simulation campaign; this is what the
# preprocessing step derives from it).
V6_TRAINING_DATA_ROOT = os.environ.get(
    'LDNET_V6_TRAINING_DATA_ROOT', '/work/u10677113/LDNet_GLA/data_v6')

# Root under which each study gets its own results_<name>/ directory,
# mirroring light/results_cs25_combo/ and light/noise/results/ but never
# writing into either of them.
V6_RESULTS_ROOT = os.environ.get('LDNET_V6_RESULTS_ROOT', str(FINAL_DIR))

# Where regenerated thesis-style PNGs land. Deliberately NOT
# light/latex/Images/ — those PNGs back the current (v5) thesis text.
V6_IMG_DIR = os.environ.get('LDNET_V6_IMG_DIR', str(FINAL_DIR / 'images'))


def results_dir(name):
    """final/results_<name>/ (created on demand). E.g. results_dir('cs25_combo')
    -> final/results_cs25_combo, results_dir('noise') -> final/results_noise."""
    d = os.path.join(V6_RESULTS_ROOT, f'results_{name}')
    os.makedirs(d, exist_ok=True)
    return d


def img_dir():
    """v6 image output directory (created on demand)."""
    os.makedirs(V6_IMG_DIR, exist_ok=True)
    return V6_IMG_DIR


# ---------------------------------------------------------------------------
# harness_noise.py monkey-patch (see module docstring)
# ---------------------------------------------------------------------------

def load_v6_harness():
    """
    Import light/noise/harness_noise.py and retarget it at the v6 model.

    Call this immediately after importing this module and BEFORE building any
    controller or calling H.rollout(...). Some controllers read H.aero at
    *construction* time (e.g. controllers_ref.MPCConstRef.__init__ does
    `self._num_z = int(H.aero._num_z)`), so if a controller is constructed
    before this patch runs, it silently bakes in the v5 model's latent
    dimension. The safe pattern used by every forked study script here is:

        import paths
        H = paths.load_v6_harness()
        from optimal import FusedPreviewSensor, MPCPreviewController
        ... (build controllers only from this point on)

    DAMULT scaling (structure.D_ALPHA *= DAMULT) happens once, inside
    harness_noise.py, at its FIRST import in the process — set the DAMULT
    env var before that first `import harness_noise` (directly or via this
    function), exactly as light/noise/harness_noise.py's own docstring
    requires. This function does not touch DAMULT.
    """
    import harness_noise as H
    from ldnet_aero import LDNetAero

    H.aero = LDNetAero(V6_MODEL_DIR)
    H.aero.reset(dt=H.DT)
    H.MD = V6_MODEL_DIR
    H.CLTRIM = float(H.aero.predict(H.X0, 0., 0., H.U)[0])
    H.LAM = float(H.aero._z_leak)
    return H
