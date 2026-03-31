#!/usr/bin/env python3
#%% Import modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')
import json
import utils
import optimization

#%% Set some hyperparameters
dt = 0.2
dt_base = 5.4
num_latent_states = 1

#%% Define problem
problem = {
    "space": {
        "dimension" : 2
    },
    "input_parameters": [{ "name": "U_inf" }],

    "input_signals": [
        { "name": "h" },
        { "name": "hd" },
        { "name": "a" },
        { "name": "ad" },
        { "name": "delta" },
        { "name": "W_gust" },
        
    ],

    "output_signals": [
        { "name": "C_L" },
        { "name": "C_M" },

    ],

    "output_fields": [
        { "name": "ux" },
        { "name": "uy" }

    ]
}

normalization = {
    'space': {
        'min' : [0, 0],
        'max' : [1, 1],
    },
    'time': {
        'time_constant' : dt_base
    },

    'input_parameters': {
        'U_inf': { 'min': 0, 'max': 120 }
    },

    'input_signals': {
        'h': { 'min': -0.025, 'max': 0.025 },
        'hd': { 'min': -1, 'max': 1 },
        'a': { 'min': -0.1, 'max': 0.1 },
        'ad': { 'min': -1, 'max': 1 },
        'delta': { 'min': -20, 'max': 20 },
        'W_gust': { 'min': 0, 'max': 50 }

    },

    'output_signals': {
        'C_L': { 'min': -0.01, 'max': 0.5 },
        'C_M': { 'min': -0.05, 'max': 0.05 }
    },
    
    'output_fields': {
        'ux': { 'min': -50, "max": 150 },
        'uy': { 'min': -100, 'max': 100 },
    }
}

#%% Dataset

data_set_1_path = '../data/GLA_train.h5'
data_set_2_path = '../data/GLA_valid.h5'
data_set_3_path = '../data/GLA_test.h5'

dataset_train = utils.load_gla_h5(data_set_1_path)
dataset_valid = utils.load_gla_h5(data_set_2_path)
dataset_tests = utils.load_gla_h5(data_set_3_path)

# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset(dataset_train, problem, normalization, dt = dt, num_points_subsample = 1)
utils.process_dataset(dataset_valid, problem, normalization, dt = dt, num_points_subsample = 1)
utils.process_dataset(dataset_tests, problem, normalization, dt = dt)

#%% Define LDNet model

# dynamics network
input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(7, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(7, activation = tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states)
        ])
NNdyn.summary()

v = NNdyn.variables[0]
print(type(v))
print(type(v.value))
print(hasattr(v, '_variable'))
print(hasattr(v, 'handle'))

# reconstruction network
input_shape = (None, None, num_latent_states + len(problem['input_signals']) + problem['space']['dimension'])
NNrec = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(len(problem['output_signals']))
        ])
NNrec.summary()


def evolve_dynamics(dataset):
    num_samples = dataset['input_signals'].shape[0]
    num_times = dataset['input_signals'].shape[1]
    state = tf.zeros((num_samples, num_latent_states), dtype=tf.float64)
    state_history = tf.TensorArray(tf.float64, size=num_times)
    state_history = state_history.write(0, state)
    dt_ref = normalization['time']['time_constant']

    for i in tf.range(num_times - 1):
        # Concateno lo stato con input_parameters e input_signals al tempo i
        state = state + dt/dt_ref * NNdyn(tf.concat([state, tf.expand_dims(dataset['input_parameters'][:, 0], axis=-1), dataset['input_signals'][:, i, :]], axis=-1))
        state_history = state_history.write(i + 1, state)

    return tf.transpose(state_history.stack(), perm=(1,0,2))  # (num_samples, num_times, num_latent_states)

def reconstruct_output(dataset, states):    
    states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2), 
        [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states])
    inp_signals_expanded = tf.broadcast_to(tf.expand_dims(dataset['input_signals'], axis = 2),
        [dataset['num_samples'], dataset['num_times'], dataset['num_points'], len(problem['input_signals'])])
    output = NNrec(tf.concat([states_expanded, inp_signals_expanded, dataset['points_full']], axis = 3))
    # nonlinear transformation to compress long tails
    alpha = 0.05
    output = (output**3 + alpha*output)/(1+alpha)
    return output

def LDNet(dataset):
    states = evolve_dynamics(dataset)
    return reconstruct_output(dataset, states)

#%% Loss function
weight_direction = 0 #for cl and cm only, we can set to 0 the weight of the direction loss, since the velocity is already constrained in magnitude by the MSE loss. For other problems, it might be useful to set a positive weight to better constrain the direction of the velocity vector.
epsilon = 1e-4

def get_direction(velocity): 
    return tf.math.divide(velocity, (epsilon + tf.expand_dims(tf.norm(velocity, axis = 3), axis = -1)))

def loss(dataset, target_velocity, target_direction):
    velocity = LDNet(dataset)

    # Convert target arrays to tf.Tensor
    target_velocity_tf = tf.convert_to_tensor(target_velocity, dtype=tf.float64)
    target_direction_tf = tf.convert_to_tensor(target_direction, dtype=tf.float64)

    MSE_velocity = tf.reduce_mean(tf.square(velocity - target_velocity_tf))
    direction = get_direction(velocity)
    MSE_direction = tf.reduce_mean(tf.square(direction - target_direction_tf))

    return MSE_velocity + weight_direction * MSE_direction

target_direction_train = get_direction(dataset_train['output_signals'])
target_direction_valid = get_direction(dataset_valid['output_signals'])
loss_train = lambda: loss(dataset_train, dataset_train['output_signals'], target_direction_train)
loss_valid = lambda: loss(dataset_valid, dataset_valid['output_signals'], target_direction_valid)

#%% Training
trainable_variables = NNdyn.variables + NNrec.variables
opt = optimization.OptimizationProblem(trainable_variables, loss_train, loss_valid)

num_epochs_Adam = 200
num_epochs_BFGS = 10000

print('training (Adam)...')
opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
print('training (BFGS)...')
opt.optimize_BFGS(num_epochs_BFGS)

fig, axs = plt.subplots(1, 1)
axs.loglog(opt.iterations_history, opt.loss_train_history, 'o-', label = 'training loss')
axs.loglog(opt.iterations_history, opt.loss_valid_history, 'o-', label = 'validation loss')
axs.axvline(num_epochs_Adam)
axs.set_xlabel('epochs'), plt.ylabel('loss')
axs.legend()


NNdyn.save_weights('../models/NNdyn_weights.weights.h5')
NNrec.save_weights('../models/NNrec_weights.weights.h5')
import json
config = {'problem': problem, 'normalization': normalization, 'num_latent_states': num_latent_states}
with open('../models/config.json', 'w') as f:
    json.dump(config, f, indent=2)


#%% Testing 

# Compute predictions.
out_signals = LDNet(dataset_tests)

# Since the LDNet works with normalized data, we map back the outputs into the original ranges.
out_signals_FOM = utils.denormalize_output(dataset_tests['output_signals'], problem, normalization).numpy()
out_signals_ROM = utils.denormalize_output(out_signals                 , problem, normalization).numpy()

NRMSE = np.sqrt(np.mean(np.square(out_signals_ROM - out_signals_FOM))) / (np.max(out_signals_FOM) - np.min(out_signals_FOM))

import scipy.stats
R_coeff = scipy.stats.pearsonr(np.reshape(out_signals_ROM, (-1,)), np.reshape(out_signals_FOM, (-1,)))

print('Normalized RMSE:       %1.3e' % NRMSE)
print('Pearson dissimilarity: %1.3e' % (1 - R_coeff[0]))

#%% Postprocessing
i_sample = 0
time_axis = dataset_tests['times'][:] * dt_base

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# C_L
axs[0].plot(time_axis, out_signals_FOM[i_sample, :, 0, 0], 'b-', label='C_L FOM')
axs[0].plot(time_axis, out_signals_ROM[i_sample, :, 0, 0], 'r--', label='C_L ROM')
axs[0].set_ylabel('C_L')
axs[0].legend()

# C_M
axs[1].plot(time_axis, out_signals_FOM[i_sample, :, 0, 1], 'b-', label='C_M FOM')
axs[1].plot(time_axis, out_signals_ROM[i_sample, :, 0, 1], 'r--', label='C_M ROM')
axs[1].set_ylabel('C_M')
axs[1].legend()

# Errore assoluto
err_CL = np.abs(out_signals_FOM[i_sample, :, 0, 0] - out_signals_ROM[i_sample, :, 0, 0])
err_CM = np.abs(out_signals_FOM[i_sample, :, 0, 1] - out_signals_ROM[i_sample, :, 0, 1])
axs[2].plot(time_axis, err_CL, 'b-', label='|err C_L|')
axs[2].plot(time_axis, err_CM, 'r-', label='|err C_M|')
axs[2].set_ylabel('Errore assoluto')
axs[2].set_xlabel('Tempo [s]')
axs[2].legend()

# NRMSE per C_L e C_M separatamente
for i, name in enumerate(['C_L', 'C_M']):
    fom = out_signals_FOM[:, :, 0, i]
    rom = out_signals_ROM[:, :, 0, i]
    nrmse = np.sqrt(np.mean(np.square(rom - fom))) / (np.max(fom) - np.min(fom))
    print(f'NRMSE {name}: {nrmse:.3e}')

fig.tight_layout()
fig.savefig('TestCase2.png')