from network.cnn import TrainerCNN
import tensorflow as tf

from keras import datasets, utils
from keras.callbacks import ModelCheckpoint
from data import io

__model_name__ = "cifar10-trained-model"

__root__ = "dm/output/"
__checkpoint_path__ = "dm/output/checkpoint/" + __model_name__

__model_loader__ = True


##############
# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config=config)

#32*32 in 10 classes with 6000img/classes
#50000training img, 10000 test image
##############

epochs = 1000
num_classes = 10
batch_size = 64
# num_predictions=20
# data_augmentation = True

###############

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
# y_train = y_train.astype('float32') #y: verite terrain
x_test = x_test.astype('float32')
# y_test = y_test.astype('float32')
x_train /= 255
x_test /= 255
# Convert class vectors to binary class matrices.
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)




##############

cnnTrainer = TrainerCNN()
cnnModel = None
if(__model_loader__):
#     # cnnTrainer = TrainerCNN(cnn_model=io.load_model(__model_name__))
    cnnModel = io.load_model_ez(path=__checkpoint_path__)
else:
    cnnModel = cnnTrainer.generateModelCNN(x_train)
    # cnnTrainer = TrainerCNN()
cnnModel.summary()

checkpoint_cb = ModelCheckpoint(filepath=__checkpoint_path__, verbose=1)
callbacks_list = [checkpoint_cb]

if(__model_loader__):
    # io.save_model(model=cnnModel, filename=__model_name__)
    cnnModel.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks_list)


    io.save_model_ez(cnnModel, path=__checkpoint_path__)
else:
    
    cnnModel.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True)
