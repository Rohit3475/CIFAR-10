from matplotlib import pyplot
#from scipy.misc import toimage
from keras.datasets import cifar10
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense,Activation,Flatten,Dropout , BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
"""def show_imgs(X):
    pyplot.figure(1)
    k=0
    for i in range (0,4):
        for j in range (0,4):
            pyplot.subplot2grid((4,4),(i,j))
            pyplot.imshow(toimage(X[k]))
            k=k+1

    pyplot.show()"""

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
weight_decay = 1e-4
num_classes = 10
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay),input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size(2,2)))
model.add(Dropout(0.2))
model.add(Con2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Con2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Con2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Con2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
