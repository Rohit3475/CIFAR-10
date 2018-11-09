from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.datasets import cifar10
import os
import sys
import keras
import numpy as np
import h5py
from keras.models import load_model,model_from_json
from matplotlib import pyplot
from scipy.misc import toimage

def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            pyplot.subplot2grid((4,4),(i,j))
            pyplot.imshow(toimage(X[k]))
            k = k+1
    # show the plot
    pyplot.show()



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# mean-std normalization
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

show_imgs(x_test[:16])
fn=sys.argv[1]
# Load trained CNN model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')
labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#For testing your own image
test_image=image.load_img(fn,target_size=(32,32))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
indices = np.argmax(model.predict(test_image),1)
for x in indices:
    print ( labels[x])
