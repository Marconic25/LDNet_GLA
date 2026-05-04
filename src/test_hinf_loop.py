"""Quick test: OL vs H-inf closed-loop."""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys; sys.path.insert(0, '/home/marco/LDNet_OF/src')
import numpy as np
from aerodynamics.model import LDNetModel
from control.hinf_simulation import HInfController, run_hinf_simulation

aero = LDNetModel('/home/marco/LDNet_OF/models')
d = np.load('/home/marco/LDNet_OF/src/hinf_controller.npz', allow_pickle=False)
hinf = HInfController(np.array(d['A']), np.array(d['B']),
                       np.array(d['C']), np.array(d['D']))
print('K: %d stati, gamma=%.4f' % (d['A'].shape[0], float(d['gamma'])))

def gust(t, W0=60., Tg=1.):
    return 0.5*W0*(1 - np.cos(2*np.pi*t/Tg)) if 0 <= t <= Tg else 0.

print('Open-loop...')
res_ol = run_hinf_simulation(75., 2.5, 0.01, aero, None, gust_profile=gust)
pk_ol = np.max(np.abs(res_ol['h_ddot']))
print('  OL  hddot_peak=%.2f m/s2' % pk_ol)

print('H-inf...')
res_hi = run_hinf_simulation(75., 2.5, 0.01, aero, hinf, gust_profile=gust)
pk_hi = np.max(np.abs(res_hi['h_ddot']))
pk_d  = np.max(np.abs(res_hi['delta']))
print('  Hinf hddot_peak=%.2f m/s2  delta_peak=%.2f deg' % (pk_hi, pk_d))
print('  Riduzione hddot: %.1f%%' % ((pk_ol - pk_hi) / pk_ol * 100))
print('DONE')
