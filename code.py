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
import os
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'
train_files = [cifar10_dataset_folder_path + '/data_batch_' + str(batch_id) for batch_id in range(1, 6)]
test_files = [cifar10_dataset_folder_path + '/test_batch']
'''(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')'''
weight_decay = 1e-4
num_classes = 10
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay),input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax'))

model.summary()

datagen=ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
#datagen.fit(x_train)
batch_size = 64

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
training_set=datagen.flow(train_files, (64,64))
test_set=datagen.flow(test_files, (64,64))
Model = model.fit_generator(training_set,epochs=50,steps_per_epoch = 2000 // batch_size,validation_data=test_set,validation_steps=800 // batch_size)

'''model.fit_generator(datagen.flow(train_files, batch_size=batch_size),
                    steps_per_epoch=200 // batch_size,epochs=125,
                    verbose=1,callbacks=[LearningRateScheduler(lr_schedule)])'''
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
print(Model.history.keys())

#testing
'''scores = model.evaluate(test_files, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))'''
