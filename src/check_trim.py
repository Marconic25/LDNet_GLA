import sys
sys.path.insert(0, '.')
import numpy as np
from aerodynamics.model import LDNetModel
m = LDNetModel('../models')
z = np.zeros(m.num_latent_states)
for _ in range(200):
    z, CL, CM = m.step(z, 0.,0.,0.,0.,0.,0., 75.0, 0.01)
print(f'CL_trim={float(CL):.5f}  CM_trim={float(CM):.5f}  n_lat={m.num_latent_states}')
