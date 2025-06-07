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
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Input
from tensorflow.keras.models import Model


def mean_square_error(array1, array2):

    if array1.shape != array2.shape:
        raise ValueError("The shapes of the input arrays must be the same.")

    difference = array1 - array2

    squared_difference = np.square(difference)

    mse = np.mean(squared_difference)
    
    return mse

def sample_creation(data_):
    input_size = 96
    output_size = 336

    total_windows = data_.shape[0] - input_size - output_size + 1

    input_sets = []
    output_sets = []

    for i in range(total_windows):
        input_set = data_[i:i + input_size]
        output_set = data_[i + input_size:i + input_size + output_size]
        input_sets.append(input_set)
        output_sets.append(output_set)

    input_sets = np.array(input_sets)
    output_sets = np.array(output_sets)    

    return input_sets, output_sets


data_name = 'ETTh1.csv' # change name here
input_seq_len = 96


if (data_name=='ETTh1.csv'):
    data_path = 'ETT-small/ETTh1.csv'
    border1s = [0, 12 * 30 * 24 - input_seq_len, 12 * 30 * 24 + 4 * 30 * 24 - input_seq_len]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

if (data_name=='ETTh2.csv'):
    data_path = 'ETT-small/ETTh2.csv'
    border1s = [0, 12 * 30 * 24 - input_seq_len, 12 * 30 * 24 + 4 * 30 * 24 - input_seq_len]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

if (data_name=='ETTm1.csv'):
    data_path = 'ETT-small/ETTm1.csv'
    border1s = [0, 12 * 30 * 24 * 4 - input_seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - input_seq_len]
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

if (data_name=='ETTm2.csv'):
    data_path = 'ETT-small/ETTm2.csv'
    border1s = [0, 12 * 30 * 24 * 4 - input_seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - input_seq_len]
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

if (data_name=='electricity.csv'):
    data_path = 'electricity/electricity.csv'
    data_len = pd.read_csv("./dataset/"+data_path)
    num_train = int(len(data_len) * 0.7)
    num_test = int(len(data_len) * 0.2)
    num_vali = len(data_len) - num_train - num_test
    border1s = [0, num_train - input_seq_len, len(data_len) - num_test - input_seq_len]
    border2s = [num_train, num_train + num_vali, len(data_len)]

if (data_name=='weather.csv'):
    data_path = 'weather/weather.csv'
    data_len = pd.read_csv("./dataset/"+data_path)
    num_train = int(len(data_len) * 0.7)
    num_test = int(len(data_len) * 0.2)
    num_vali = len(data_len) - num_train - num_test
    border1s = [0, num_train - input_seq_len, len(data_len) - num_test - input_seq_len]
    border2s = [num_train, num_train + num_vali, len(data_len)]



data = pd.read_csv("./dataset/"+data_path)

main_att = data.shape[1]-1







train_border1 = border1s[0]
train_border2 = border2s[0]

val_border1 = border1s[1]
val_border2 = border2s[1]

test_border1 = border1s[2]
test_border2 = border2s[2]


train_data_ = data.values[train_border1:train_border2,1:]
val_data_ = data.values[val_border1:val_border2,1:]
test_data_ = data.values[test_border1:test_border2,1:]



scaler.fit(train_data_)
train_data = scaler.transform(train_data_)
val_data = scaler.transform(val_data_)
test_data = scaler.transform(test_data_)


x_train, y_train = sample_creation(train_data)
x_val, y_val = sample_creation(val_data)
x_test, y_test = sample_creation(test_data)


from Model_FNet import Encoder
def build_transformer_model(input_shape, num_layers, d_model, dff, maximum_position_encoding, output_shape, rate=0.1):
    inputs = Input(shape=input_shape)
    encoder1 = Encoder(num_layers, d_model, dff, input_shape[-1], maximum_position_encoding, rate)
    enc_output1 = encoder1(inputs, training=True)
    outputs = Dense(output_shape[-1])(tf.tile(enc_output1, [1, 4, 1])[:,:336,:]) 
    return Model(inputs=inputs, outputs=outputs)


input_shape = (96, x_train.shape[-1])
num_layers = 4
d_model = 32
dff = 64
maximum_position_encoding = 10000
output_shape = (96, main_att)

F_Net_model = build_transformer_model(input_shape, num_layers, d_model, dff, maximum_position_encoding, output_shape)


x_train=np.asarray(x_train).astype(np.float32)
x_val=np.asarray(x_val).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)

y_train=np.asarray(y_train[:,:,:main_att]).astype(np.float32)
y_val=np.asarray(y_val[:,:,:main_att]).astype(np.float32)
y_test=np.asarray(y_test[:,:,:main_att]).astype(np.float32)



F_Net_model.load_weights('Weight_FNet_336_ETTh1.tf') # change name here

pred = F_Net_model.predict(x_test)

print("Mean Square Error [F-Net]:")
print(mean_square_error(y_test, pred))


input_layer = F_Net_model.input
custom_layer = F_Net_model.get_layer('encoder')
sublayer_output = custom_layer.output
submodel = Model(inputs=input_layer, outputs=sublayer_output)
sublayer_output_value = submodel.predict(x_train)



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense,MaxPooling1D,LSTM
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Lambda,MaxPooling1D,LSTM,concatenate, GlobalMaxPooling1D, Layer, Average,UpSampling1D,Conv1DTranspose
from tensorflow.keras.callbacks import EarlyStopping

input_size = (96,x_train.shape[-1])
inputs = Input(input_size)

conv1 = Conv1D(8, 3, activation='relu')(inputs)
conv2 = Conv1D(16, 3, activation='relu')(conv1)
pool1 = MaxPooling1D(2)(conv2)

conv3 = Conv1D(32, 3, activation='relu')(pool1)
pool2 = MaxPooling1D(2)(conv3) 

up1 = UpSampling1D(size = 2)(pool2)
concat1 = concatenate([conv3,up1]) 

deconv1 = Conv1DTranspose(32, 3)(concat1) 


up2 = UpSampling1D(size = 2)(deconv1)
concat2 = concatenate([conv2,up2]) 

deconv2= Conv1DTranspose(16, 3)(concat2)
concat3 = concatenate([conv1,deconv2])

deconv3= Conv1DTranspose(8, 3)(concat3)
conv4 = Conv1D(32, 1, activation='relu')(deconv3)

outputs = Conv1D(32, 1, activation='relu',name="out_mse")(conv4)


model = Model(inputs=[inputs], outputs=[outputs])

sublayer_output_value_test = submodel.predict(x_test) 
model.load_weights('Weight_EFNet_336_ETTh1.tf') # change name here
test_unet = model.predict(x_test)


sec_half_model = Model(inputs=F_Net_model.get_layer('tf.tile').input, outputs=F_Net_model.get_layer('dense_9').output) 

final_pred = sec_half_model.predict(test_unet)
print("Mean Square Error (EF-Net) :")
print(mean_square_error(y_test, final_pred))

