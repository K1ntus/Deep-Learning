"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Conv2D, Activation, Flatten, MaxPooling2D
from keras.optimizers import RMSprop
import numpy as np


class TrainerCNN:
    def __init__(self, 
                numFilters = 64, 
                kernel_size = (3,3), 
                activation_mode = 'relu', 
                input_shape = (32,32,3),
                numClasses = 10,
                cnn_model = None
    ):
        self.numFilters = numFilters
        self.kernel_size = (4,4)#kernel_size
        self.activation_mode = activation_mode
        self.input_shape = input_shape
        self.numClasses = numClasses


        self.model = cnn_model
        if(self.model is None):
            self.model = Sequential()


    def generateModelCNN(self, x_train):
        self.model.add(
            keras.layers.Conv2D(32,
                kernel_size = self.kernel_size,
                activation  = self.activation_mode,
                input_shape = self.input_shape
            )
        )

        self.model.add(Conv2D(32, self.kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(self.numFilters, self.kernel_size, padding='same'))
        self.model.add(Activation(self.activation_mode))


        self.model.add(Conv2D(self.numFilters, self.kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())


        # self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        
        # self.model.add(Flatten())
        self.model.add(Dense(512, activation=self.activation_mode, kernel_initializer='normal')) #param1=nb neuronne, param2=activation
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.numClasses, activation='softmax', kernel_initializer='normal')) #param1=nb neuronne, param2=activation
        
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd',  metrics=['accuracy'])
        
        return self.model