"""
final/studies/run_v6.py — v6 wrapper over light/run.py.

light/run.py ALREADY reads its model directory from the MD_OVERRIDE
environment variable, falling back to clean/models_rollout/latent_10 (the v5
model) only when it is unset:

    MD = os.environ.get('MD_OVERRIDE', os.path.join(..., 'clean',
                        'models_rollout', 'latent_10'))
    aero = LDNetAero(MD); aero.reset(dt=DT)

That is enough to retarget it at the v6 model — light/run.py is imported
unchanged, NOT forked. This wrapper's only job is to set MD_OVERRIDE to
paths.V6_MODEL_DIR *before* importing it (light/run.py builds its LDNetAero
at module import time, as top-level code, so the env var must be in place
before the `import run` line below), then re-export the same simulate()/
metrics()/gust() API so every downstream script that used to
`import run as Rn` can instead `import run_v6 as Rn` with no other changes.

Caveat: because module-level code only runs once per process (Python caches
imports by name), do not mix `import run` (pointing at v5) and `import
run_v6` (pointing at v6) in the same interpreter — whichever imports 'run'
first wins for the rest of the process. Every script in final/studies/ that
needs the run.py API imports run_v6, never run, so this never happens here.

MD_OVERRIDE is set with setdefault(), not assignment: if the caller already
exported MD_OVERRIDE (e.g. to compare two different v6 checkpoints without
editing paths.py), that value is honoured instead of being clobbered.
"""
import os

import paths  # noqa: F401  (side effect: light/ + light/noise/ on sys.path)

os.environ.setdefault('MD_OVERRIDE', paths.V6_MODEL_DIR)

from run import (  # noqa: E402,F401  (import after MD_OVERRIDE is set)
    simulate,
    metrics,
    gust,
    CLTRIM,
    LAM,
    SCORE_W,
    U,
    RHO,
    C,
    DT,
    S,
    q,
    MD,
)

if __name__ == '__main__':
    print(f'run_v6: MD={MD}')
    print(f'run_v6: CLTRIM={CLTRIM:.6f}  LAM={LAM:.6g}')
