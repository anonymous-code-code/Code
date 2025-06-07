import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy.io
import os
from scipy.stats import skew
from scipy.stats import kurtosis
import copy
import warnings
import math
from scipy.linalg import svd
from sklearn.decomposition import PCA
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


class FourierTransformLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(FourierTransformLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, x):
        x_ft = tf.signal.fft(tf.cast(x, tf.complex64))
        return x_ft
    
    def get_config(self):
        config = super(FourierTransformLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

class EncoderLayer(Layer):
    def __init__(self, d_model, dff, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.w1 = self.add_weight(name="w1", shape=(d_model,), initializer="ones", trainable=True)
        self.w2 = self.add_weight(name="w2", shape=(d_model,), initializer="ones", trainable=True)

    
    def call(self, x, training):
        dft_seq = FourierTransformLayer(axis=1)(x)
        dft_hidden = FourierTransformLayer(axis=2)(dft_seq)
        dft_hidden = tf.math.real(dft_hidden)
        out1 = self.layernorm1(x + (tf.reshape(self.w1, (1, 1, -1)) * dft_hidden))        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + (tf.reshape(self.w2, (1, 1, -1))* ffn_output))
    
    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "d_model": self.ffn.dense2.units,
            "dff": self.ffn.dense1.units,
            "rate": self.dropout1.rate,
        })
        return config

class PointWiseFeedForwardNetwork(Layer):
    def __init__(self, d_model, dff, **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)
    
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)
    
    def get_config(self):
        config = super(PointWiseFeedForwardNetwork, self).get_config()
        config.update({
            "d_model": self.dense2.units,
            "dff": self.dense1.units,
        })
        return config

class Encoder(Layer):
    def __init__(self, num_layers, d_model, dff, input_dim, maximum_position_encoding, rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Dense(d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [EncoderLayer(d_model, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        
        return x
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "dff": self.enc_layers[0].ffn.dense1.units,
            "input_dim": self.embedding.units,
            "maximum_position_encoding": self.pos_encoding.shape[1],
            "rate": self.dropout.rate,
        })
        return config

