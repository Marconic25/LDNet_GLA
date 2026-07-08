"""Verification script: TF fixed-step DP45 == numpy scipy DP45, backprop OK."""
import sys, os
sys.path.insert(0, '/home/marco/LDNet_OF/light')
sys.path.insert(0, '/home/marco/LDNet_OF/src')
import numpy as np

# ---- 1. TF vs numpy consistency check ----
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import structure

M_HH = tf.constant(structure.M_HH, tf.float64)
M_AA = tf.constant(structure.M_AA, tf.float64)
M_HA = tf.constant(structure.M_HA, tf.float64)
DET  = tf.constant(structure.DET,  tf.float64)
K_H  = tf.constant(structure.K_H,  tf.float64)
D_H  = tf.constant(structure.D_H,  tf.float64)
K_ALPHA = tf.constant(structure.K_ALPHA, tf.float64)
D_ALPHA = tf.constant(structure.D_ALPHA, tf.float64)

def struct_rhs(x, Fy, Mz):
    h=x[:,0]; hd=x[:,1]; a=x[:,2]; ad=x[:,3]
    rhs_h = -Fy - D_H*hd - K_H*h
    rhs_a =  Mz - D_ALPHA*ad - K_ALPHA*a
    hdd = (M_AA*rhs_h - M_HA*rhs_a)/DET
    add = (M_HH*rhs_a - M_HA*rhs_h)/DET
    return tf.stack([hd, hdd, ad, add], axis=-1)

def struct_step_tf(x, Fy, Mz, h):
    k1 = struct_rhs(x, Fy, Mz)
    k2 = struct_rhs(x + h*(1.0/5)*k1, Fy, Mz)
    k3 = struct_rhs(x + h*(3.0/40*k1 + 9.0/40*k2), Fy, Mz)
    k4 = struct_rhs(x + h*(44.0/45*k1 - 56.0/15*k2 + 32.0/9*k3), Fy, Mz)
    k5 = struct_rhs(x + h*(19372.0/6561*k1 - 25360.0/2187*k2 + 64448.0/6561*k3 - 212.0/729*k4), Fy, Mz)
    k6 = struct_rhs(x + h*(9017.0/3168*k1 - 355.0/33*k2 + 46732.0/5247*k3 + 49.0/176*k4 - 5103.0/18656*k5), Fy, Mz)
    return x + h*(35.0/384*k1 + 500.0/1113*k3 + 125.0/192*k4 - 2187.0/6784*k5 + 11.0/84*k6)

DT_PHYS = 0.002
np.random.seed(42)
xr = np.random.randn(3, 4) * np.array([0.01, 0.3, 0.005, 0.3])
Fy = np.array([100., -50., 200.])
Mz = np.array([5., -3., 10.])

tf_step = struct_step_tf(tf.constant(xr), tf.constant(Fy), tf.constant(Mz), DT_PHYS).numpy()
np_step = np.array([structure.step_dp45(xr[i], Fy[i], Mz[i], DT_PHYS) for i in range(3)])

err = np.max(np.abs(tf_step - np_step))
ok = 'OK' if err < 1e-8 else 'MISMATCH!'
print(f'TF-vs-numpy DP45 max|err| = {err:.2e}  [{ok}]')
assert err < 1e-8, f'TF-numpy mismatch: {err}'

# ---- 2. Backprop test ----
x_var = tf.Variable(xr, dtype=tf.float64)
Fy_t = tf.constant(Fy)
Mz_t = tf.constant(Mz)
with tf.GradientTape() as tape:
    x_next = struct_step_tf(x_var, Fy_t, Mz_t, DT_PHYS)
    loss = tf.reduce_sum(tf.square(x_next))
grads = tape.gradient(loss, x_var)
assert grads is not None, 'No gradient!'
grad_finite = np.all(np.isfinite(grads.numpy()))
print(f'Backprop gradient finite: {grad_finite}  max|grad|={np.max(np.abs(grads.numpy())):.4f}')
assert grad_finite, 'Gradient not finite!'

# ---- 3. One-step summary ----
print()
print('=== One-step comparison summary ===')
xa = structure.step_dp45(xr[0], Fy[0], Mz[0], DT_PHYS)   # new numpy
from scipy.integrate import solve_ivp
sol = solve_ivp(lambda t,s: structure.rhs(s,Fy[0],Mz[0]),
                [0.,DT_PHYS], xr[0], method='RK45', rtol=1e-8, atol=1e-10, max_step=2*DT_PHYS)
xb = sol.y[:,-1]
def rhs_np(s,F,M): return np.array(structure.rhs(s,F,M))
k1=rhs_np(xr[0],Fy[0],Mz[0]); k2=rhs_np(xr[0]+0.5*DT_PHYS*k1,Fy[0],Mz[0])
k3=rhs_np(xr[0]+0.5*DT_PHYS*k2,Fy[0],Mz[0]); k4=rhs_np(xr[0]+DT_PHYS*k3,Fy[0],Mz[0])
xc = xr[0] + (DT_PHYS/6.)*(k1+2*k2+2*k3+k4)

print(f'  (a) new dp45 (scipy):    {xa}')
print(f'  (b) cosim solve_ivp ref: {xb}')
print(f'  (c) old RK4:             {xc}')
print(f'  (a)vs(b) max|err| = {np.max(np.abs(xa-xb)):.2e}   [must be ~0]')
print(f'  (a)vs(c) max|err| = {np.max(np.abs(xa-xc)):.2e}   [RK45-RK4 diff]')
print()
print('ALL CHECKS PASSED')
