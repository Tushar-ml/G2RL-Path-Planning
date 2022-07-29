import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

def get_cnn_model(height, width, depth, nt):
    model = Sequential()
    model.add(layers.Conv3D(32,(1,3,3),activation='relu',input_shape=(1, height,width,depth)))
    #model.add(layers.Conv3D(64,(1,1,1),activation='relu'))
    #model.add(layers.Conv3D(128,(1,2,2),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(1))
    model.add(layers.LSTM(512))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation = 'linear'))

    model.compile(optimizer='rmsprop', loss = 'mse')
    return model