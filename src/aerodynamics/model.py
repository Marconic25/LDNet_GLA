import json
import tensorflow as tf
from pathlib import Path
import numpy as np

tf.keras.backend.set_floatx('float64')

class LDNetModel: #create a class cause all the functions share the state
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        self.config = config
        self.problem = config['problem']
        self.normalization = config['normalization']
        self.num_latent_states = config['num_latent_states']

        #Construct the model
        
        input_shape = (self.num_latent_states + len(self.problem['input_parameters']) + len(self.problem['input_signals']),)
        self.NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(7, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(7, activation = tf.nn.tanh),
            tf.keras.layers.Dense(self.num_latent_states)
        ])
        

        input_shape = (None, None, self.num_latent_states + len(self.problem['input_signals']) + self.problem['space']['dimension'])
        self.NNrec = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(24, activation = tf.nn.tanh),
            tf.keras.layers.Dense(len(self.problem['output_signals']))
        ])
        
        # Load the weights
        self.NNdyn.load_weights(model_dir / 'NNdyn_weights.weights.h5')
        self.NNrec.load_weights(model_dir / 'NNrec_weights.weights.h5')

    def normalize_input(self, h, hd, a, ad, delta, W_gust, U_inf):
        h_n=(2.0*h - self.normalization['input_signals']['h']['min'] - self.normalization['input_signals']['h']['max']) / (self.normalization['input_signals']['h']['max'] - self.normalization['input_signals']['h']['min'])
        hd_n = (2.0*hd - self.normalization['input_signals']['hd']['min'] - self.normalization['input_signals']['hd']['max']) / (self.normalization['input_signals']['hd']['max'] - self.normalization['input_signals']['hd']['min'])
        a_n = (2.0*a - self.normalization['input_signals']['a']['min'] - self.normalization['input_signals']['a']['max']) / (self.normalization['input_signals']['a']['max'] - self.normalization['input_signals']['a']['min'])
        ad_n = (2.0*ad - self.normalization['input_signals']['ad']['min'] - self.normalization['input_signals']['ad']['max']) / (self.normalization['input_signals']['ad']['max'] - self.normalization['input_signals']['ad']['min'])
        delta_n = (2.0*delta - self.normalization['input_signals']['delta']['min'] - self.normalization['input_signals']['delta']['max']) / (self.normalization['input_signals']['delta']['max'] - self.normalization['input_signals']['delta']['min'])
        W_gust_n = (2.0*W_gust - self.normalization['input_signals']['W_gust']['min'] - self.normalization['input_signals']['W_gust']['max']) / (self.normalization['input_signals']['W_gust']['max'] - self.normalization['input_signals']['W_gust']['min'])
        U_inf_n = (2.0*U_inf - self.normalization['input_parameters']['U_inf']['min'] - self.normalization['input_parameters']['U_inf']['max']) / (self.normalization['input_parameters']['U_inf']['max'] - self.normalization['input_parameters']['U_inf']['min'])
        return np.array([h_n, hd_n, a_n, ad_n, delta_n, W_gust_n]), np.array([U_inf_n])

    def denormalize_output(self, C_L_n, C_M_n):
        C_L = 0.5*C_L_n*(self.normalization['output_signals']['C_L']['max'] - self.normalization['output_signals']['C_L']['min']) + 0.5*(self.normalization['output_signals']['C_L']['max'] + self.normalization['output_signals']['C_L']['min'])
        C_M = 0.5*C_M_n*(self.normalization['output_signals']['C_M']['max'] - self.normalization['output_signals']['C_M']['min']) + 0.5*(self.normalization['output_signals']['C_M']['max'] + self.normalization['output_signals']['C_M']['min'])
        return C_L, C_M
    
    def step(self,z, h, hd, a, ad, delta, W_gust, U_inf, dt):
        # Normalize the input
        input_signals_n, input_parameters_n = self.normalize_input(h, hd, a, ad, delta, W_gust, U_inf)

        # Compute the latent state
        state = self.NNdyn(np.reshape(np.concatenate(([z, input_parameters_n, input_signals_n])), (1, len(input_signals_n) + len(input_parameters_n) + 1)))    
        #Update z
        dt_ref = self.normalization['time']['time_constant']
        z_new = (z + (dt/dt_ref) * state)
        z_new = z_new.numpy().flatten() #remove dimensions of size 1, so that z_new is a scalar and not an array of shape (1,)

        # Compute the output
        points_full = np.array([0.0, 0.0])
        output_signals_n = self.NNrec(np.reshape(np.concatenate([z_new, input_signals_n, points_full]), (1, 1, 1, len(input_signals_n) + self.num_latent_states + self.problem['space']['dimension'])))
        C_L_n, C_M_n = output_signals_n[0, 0, 0, 0], output_signals_n[0, 0, 0, 1]
        C_L, C_M = self.denormalize_output(C_L_n, C_M_n)

        return z_new, C_L, C_M

#Concatenare = mettere tutto in fila in un unico vettore:
#[0.3,  0.5,  0.1, -0.2, 0.0, 0.4, -0.1, 0.7]
 #^z    ^Uinf  ←────── 6 segnali ─────────────^
 #→ 8 numeri in fila, shape (8,)
#Reshape = stessa identica lista di numeri, ma "organizzati diversamente":
#Shape (8,) → 8 numeri in una riga sola:
#[[0.3, 0.5, 0.1, -0.2, 0.0, 0.4, -0.1, 0.7]]
#Shape (1, 1, 1, 8) → la stessa cosa ma con 3 "contenitori" annidati:
#[[[[0.3, 0.5, 0.1, -0.2, 0.0, 0.4, -0.1, 0.7]]]]
#Perché NNdyn vuole (1, 8) e NNrec vuole (1, 1, 1, 9)?Perché sono stati addestrati così. NNdyn è una rete "semplice" → batch + features. NNrec è una rete "spazio-temporale" → batch + time + points + features.