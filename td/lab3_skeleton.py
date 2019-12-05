from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np


def save_model(model,filename):
    model_json = model.to_json()
    with open(filename+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename+".h5")
    print("Saved model to disk in  files:", filename)

def load_model(filename):
    # load json and create model
    json_file = open(filename+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename +".h5")
    print("Loaded model from disk")

####  EXERCICE 3

(x_train,y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32') #y: verite terrain
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')


x_train = x_train/ 255 #niveau de gris entre 0 et 255  (regionde 28*28*1)
x_test = x_test/255


'''  One hot encoding:
transposition en vecteur de taille 10

Valeur 6-> |0|0|0|0|0|0|1|0|0|0|
Valeur 2-> |0|0|1|0|0|0|0|0|0|0|
Valeur 9-> |0|0|1|0|0|0|0|0|0|1|

Fonction de perte (loss):L(
'''
y_train  = keras.utils.to_categorical(y_train)
y_test  = keras.utils.to_categorical(y_test)

filename_from_model="test"

# load_model(filename_from_model)

from keras.models import Sequential
from keras.layers import Conv2D,  MaxPooling2D, Flatten, Dense
def cnn1():
    cnn_model  =  Sequential()
    cnn_model.add(
        keras.layers.Conv2D(64,
                         kernel_size=(3,3),
                         activation='relu',
                         input_shape=(28,28,1)
                         )
        )
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    cnn_model.add(Flatten())
    cnn_model.add(Dense(10, activation='relu', kernel_initializer='normal')) #param1=nb neuronne, param2=activation
    numClasses=10
    cnn_model.add(Dense(numClasses, activation='softmax', kernel_initializer='normal')) #param1=nb neuronne, param2=activation
    
    cnn_model.compile(loss='categorical_crossentropy', optimizer='sgd',  metrics=['accuracy'])
    
    return cnn_model

epochs=10    #nombre d epoques
batch_size=64
cnn = cnn1()
cnn.summary()
cnn.fit(x_train, y_train, validation_data=(x_test,y_test),  epochs=epochs, batch_size=batch_size)
save_model(cnn, filename_from_model)
#possibilite de save un model pour le sauvegarder et eviter de repartir de 0



