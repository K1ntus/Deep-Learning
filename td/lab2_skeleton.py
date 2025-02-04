from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

#print('tensorflow:', tf.__version__)
#print('keras:', keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8


#To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 784)
#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

# y_new = np.zeros(y_train.shape)
# y_new[np.where(y_train==5.0)[0]] = 1
# y_train = y_new

# y_new = np.zeros(y_test.shape)
# y_new[np.where(y_test==5.0)[0]] = 1
# y_test = y_new


num_classes = 10


#Let start our work: creating a neural network
#First, we just use a single neuron. 

#####TO COMPLETE

# One hot encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
# dimention de la premier couche
dim1 = 1  #default=40
def neural_network_1():
    nn = Sequential()
    # 1ere couche : couche dense => qq chose completement connecter
    nn.add(Dense(dim1,input_dim=num_pixels,kernel_initializer='normal',activation='relu'))

    nn.add(Dense(dim1,input_dim=num_pixels,kernel_initializer='normal',activation='sigmoid'))

    # 2nd couche : num_classes => nb de neurons
    nn.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))
    # sgd : descente de gradient batch par batch
    # adam : like gradient but with gravity
    nn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return nn 

# on instancie notre reseau
nn = neural_network_1()
nn.summary()

# on entraine le modele
batch_size = 128 #default 64
epochs = 1 #default 60 #num de layers
# so the args will be update batch_size * epocks times
nn.fit(x_train,y_train,validation_data =(x_test,y_test),batch_size=batch_size, epochs=epochs)

# evaluate on tests with trained nn
score = nn.evaluate(x_test,y_test)
print('Test accuracy : %.2f%%'%(score[1]*100))






