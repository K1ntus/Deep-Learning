'''
Created on 20 nov. 2019

@author: jmasson
'''

import numpy as np

#In this first part, we just prepare our data (mnist) 
#for training and testing

import keras
from keras.datasets import mnist

##Exercice 1.1
def sigmoid (z):
    return 1./(1+np.exp(-z))
'''
y = sigma(z)
y = sigma( sum(i=1, i < 784, w_i * x_i + b) ) #28*28 = 784 (nb de pixels)
y = sigma( (W^T).x +b )


'''


#Exercice 1.3
def my_crossentropy (y, y_hat):
    m = y.shape[1]
    perte = -(1/m) * (np.sum(
        np.multiply(y, np.log(y_hat)) 
        + np.multiply(1-y, np.log(1-y_hat)) 
    ))  #perte L bizarre

    return perte

#Exercice 1.4
'''
theta(L)/theta(w_i) = theta(L)/theta(y_hat) * theta(y_hat)/theta(z) * theta(z)*theta(w_i)
Simplification theta(y_hat) et theta(z)


L =  -(y*ln(y_hat) + (1 - y) * ln(1 - y_hat))
y_hat = theta(z)
z = sum(i=1, m, w_i*x_i + b) = w_1*x_1 + ... + w_i*x_i + ... w_m*x_m
theta(L)/theta(w_i) => dérivée de tout sauf de ce qui dépend de w_i
'''

#In this first part, we just prepare our data (mnist) 
#for training and testing
#In this first part, we just prepare our data (mnist) 
#for training and testing
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 5 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==5.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==5.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


#Display one image and corresponding label 
import matplotlib
import matplotlib.pyplot as plt
#Display one image and corresponding label 
i = 3 #6 for number 5
print('y[{}]={}'.format(i, y_train[:,i]))
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()


#Let start our work: creating a neural network
#First, we just use a single neuron. 


#####TO COMPLETE

