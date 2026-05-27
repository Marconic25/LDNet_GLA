"""
Stateful LDNet aerodynamic model wrapper for use in clean/.

Loads NNdyn and NNrec from a model directory (NNdyn_weights.weights.h5,
NNrec_weights.weights.h5, config.json) and exposes the same interface as
clean/aero.py:

    predict(state, delta_deg, W, U) -> (C_L, C_M)

The latent state z is maintained internally. predict() is read-only
(does not modify z) so the controller's scalar optimizer can call it
repeatedly without corrupting state. advance() steps z forward once
with the true delta and true gust — call it once per timestep in run.py.
"""
import json
import shutil
import tempfile
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from pathlib import Path

tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float64')


class LDNetAero:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self._norm = config['normalization']
        self._problem = config['problem']
        self._num_z = config['num_latent_states']
        self._dt_ref = self._norm['time']['time_constant']

        # Detect output type: forces (F_y/M_z) or coefficients (C_L/C_M)
        out_names = [s if isinstance(s, str) else s['name']
                     for s in self._problem['output_signals']]
        self._output_is_forces = ('F_y' in out_names)

        n_signals = len(self._problem['input_signals'])      # 6
        n_params  = len(self._problem['input_parameters'])   # 1 (U_inf)
        n_space   = self._problem['space']['dimension']       # 2 spatial coords (always 0,0 at eval)

        dyn_in = self._num_z + n_params + n_signals
        self.NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(7, activation='tanh', input_shape=(dyn_in,)),
            tf.keras.layers.Dense(7, activation='tanh'),
            tf.keras.layers.Dense(self._num_z),
        ])

        # NNrec expects 4-D input (batch, x, y, features) to support spatial fields;
        # for scalar C_L/C_M we pass a single point at (0, 0).
        rec_in = self._num_z + n_signals + n_space
        self.NNrec = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='tanh', input_shape=(None, None, rec_in)),
            tf.keras.layers.Dense(24, activation='tanh'),
            tf.keras.layers.Dense(24, activation='tanh'),
            tf.keras.layers.Dense(24, activation='tanh'),
            tf.keras.layers.Dense(len(self._problem['output_signals'])),
        ])

        self._load_weights(model_dir)
        self._z  = np.zeros(self._num_z)
        self._dt = 0.01  # set by reset(dt) or advance(); default matches run.py DT

    def _load_weights(self, model_dir):
        try:
            self.NNdyn.load_weights(model_dir / 'NNdyn_weights.weights.h5')
            self.NNrec.load_weights(model_dir / 'NNrec_weights.weights.h5')
        except OSError as e:
            if 'lock' in str(e).lower() or 'Unable to synchronously open' in str(e):
                print('  [WARNING] h5py lock detected, using tempdir workaround...')
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    shutil.copy(model_dir / 'NNdyn_weights.weights.h5',
                                tmp_path / 'NNdyn_weights.weights.h5')
                    shutil.copy(model_dir / 'NNrec_weights.weights.h5',
                                tmp_path / 'NNrec_weights.weights.h5')
                    self.NNdyn.load_weights(str(tmp_path / 'NNdyn_weights.weights.h5'))
                    self.NNrec.load_weights(str(tmp_path / 'NNrec_weights.weights.h5'))
                    print('  [OK] weights loaded from temp location')
            else:
                raise

    def _normalize_signals(self, h, hd, a, ad, delta, W):
        s = self._norm['input_signals']
        def n(v, key):
            lo, hi = s[key]['min'], s[key]['max']
            return (2.0 * v - lo - hi) / (hi - lo)
        return np.array([n(h,'h'), n(hd,'hd'), n(a,'a'),
                         n(ad,'ad'), n(delta,'delta'), n(W,'W_gust')])

    def _normalize_U(self, U):
        p = self._norm['input_parameters']['U_inf']
        return (2.0 * U - p['min'] - p['max']) / (p['max'] - p['min'])

    def _denorm_output(self, v0_n, v1_n):
        """Denormalize first two output signals, return physical values."""
        o = self._norm['output_signals']
        keys = list(o.keys())
        def dn(v_n, key):
            lo, hi = o[key]['min'], o[key]['max']
            return 0.5 * float(v_n) * (hi - lo) + 0.5 * (hi + lo)
        return dn(v0_n, keys[0]), dn(v1_n, keys[1])

    def _forward(self, z, sigs_n, U_n, U):
        """Run NNdyn and NNrec, return (z_new, C_L, C_M). Does not mutate self._z."""
        dyn_inp = np.reshape(
            np.concatenate([z, [U_n], sigs_n]),
            (1, self._num_z + 1 + len(sigs_n))
        )
        dz = self.NNdyn(dyn_inp, training=False)
        z_new = z + (self._dt / self._dt_ref) * dz.numpy().flatten()

        rec_inp = np.reshape(
            np.concatenate([z_new, sigs_n, [0.0, 0.0]]),
            (1, 1, 1, self._num_z + len(sigs_n) + 2)
        )
        out_n = self.NNrec(rec_inp, training=False)

        # sensitivity sweep models use cubic output activation: (x^3 + alpha*x)/(1+alpha)
        if self._output_is_forces:
            alpha = 0.05
            o0 = out_n[0, 0, 0, 0]
            o1 = out_n[0, 0, 0, 1]
            o0 = (o0**3 + alpha * o0) / (1 + alpha)
            o1 = (o1**3 + alpha * o1) / (1 + alpha)
        else:
            o0 = out_n[0, 0, 0, 0]
            o1 = out_n[0, 0, 0, 1]

        v0, v1 = self._denorm_output(o0, o1)

        if self._output_is_forces:
            # F_y [N], M_z [N·m] -> C_L, C_M
            q_dyn = 0.5 * 1.225 * float(U)**2 * 0.05
            C_L = v0 / q_dyn
            C_M = v1 / (q_dyn * 1.0)   # C_REF = 1.0 m
        else:
            C_L, C_M = v0, v1

        return z_new, C_L, C_M

    def predict(self, state, delta_deg, W, U):
        """
        Predict (C_L, C_M) using current z — read-only, does NOT update z.

        Same signature as clean/aero.predict:
          state     : (h, hd, alpha, alpha_dot)
          delta_deg : flap deflection [degrees]
          W         : gust velocity [m/s]
          U         : freestream velocity [m/s]
        """
        h, hd, a, ad = state
        sigs_n = self._normalize_signals(h, hd, a, ad, delta_deg, W)
        U_n    = self._normalize_U(U)
        _, C_L, C_M = self._forward(self._z, sigs_n, U_n, U)
        return float(C_L), float(C_M)

    def advance(self, state, delta_deg, W, U, dt):
        """
        Advance latent state z one step using true (state, delta, W, U, dt).
        Call once per timestep in run.py after true forces are computed.
        """
        self._dt = float(dt)
        h, hd, a, ad = state
        sigs_n = self._normalize_signals(h, hd, a, ad, delta_deg, W)
        U_n    = self._normalize_U(U)
        z_new, _, _ = self._forward(self._z, sigs_n, U_n, U)
        self._z = z_new

    def reset(self, dt=None):
        """Reset latent state to zero. Pass dt to sync the timestep for predict()."""
        self._z = np.zeros(self._num_z)
        if dt is not None:
            self._dt = float(dt)
