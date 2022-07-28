import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

def get_cnn_model(height, width, depth, nt):
    model = Sequential()
    model.add(layers.Conv3D(32,(3,3,3),activation='relu',input_shape=(height,width,depth,nt)))
    model.add(layers.Conv3D(64,(2,2,2),activation='relu'))
    model.add(layers.Conv3D(128,(1,1,1),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(1))
    model.add(layers.LSTM(512))
    model.add(layers.Dense(512))
    model.add(layers.Dense(5))

    print(model.summary())
    model.compile(optimizer='rmsprop')
    return model