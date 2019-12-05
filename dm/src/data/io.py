from __future__ import print_function

import tensorflow as tf
import keras
from keras.models import Sequential,model_from_json

root = "dm/output/"

# def save_model(model,filename):
#     model_json = model.to_json()
#     with open(root + filename + ".json", "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights(root + filename+".h5")
#     print("Saved model to disk in  files:", root + filename)

# def load_model(filename):
#     # load json and create model
#     json_file = open(root + filename+".json", 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     load weights into new model
#     loaded_model.load_weights(root + filename +".h5")
#     print("Loaded model from disk")

def save_model_ez(model, path):
    model.save(path + ".h5")

def load_model_ez(path):
    return keras.models.load_model(path + ".h5")

# cnn_model  =  Sequential()
# save_model(cnn_model, "a")

import datetime
class CustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    pri