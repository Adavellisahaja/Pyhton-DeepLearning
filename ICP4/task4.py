import tensorflow as tf
from keras import Sequential
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
import os
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.constraints import maxnorm
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.common.image_dim_ordering()

from google.colab import drive
drive.mount('/content/drive')

from PIL import Image
import glob
import cv2
train_images=[]
for filename in glob.glob('/content/drive/My Drive/Colab Notebooks/train/cars/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (50,50))
    train_images.append([output,0])
from PIL import Image
import glob
import cv2
for filename in glob.glob('/content/drive/My Drive/Colab Notebooks/train/planes/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal,(50,50))
    train_images.append([output,1])
import random
random.shuffle(train_images)
x_train=[]
y_train=[]
for im,label in train_images:
  x_train.append(im)
  y_train.append(label)
x_train=np.array(x_train).reshape(-1,50,50,3)
train_images[0]
type(x_train)
x_train.shape
x_train[0]
import matplotlib.pyplot as plt
plt.imshow(x_train[7,:,:])
plt.title('Ground Truth : {}'.format(y_train[7]))
plt.show()
test_images=[]
for filename in glob.glob('/content/drive/My Drive/Colab Notebooks/test/car/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (50,50))
    test_images.append([output,0])
for filename in glob.glob('/content/drive/My Drive/Colab Notebooks/test/planes/*.jpg'):
    img_normal = cv2.imread(filename)
    #print(im.shape)
    #print(type(im.shape))
    output = cv2.resize(img_normal, (50,50))
    test_images.append([output,1])
random.shuffle(test_images)
x_test=[]
y_test=[]
for im,label in test_images:
  x_test.append(im)
  y_test.append(label)
x_test=np.array(x_test).reshape(-1,50,50,3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
epochs = 20
lrate = 0.001
decay = lrate/epochs
sgd = Adam(lr=lrate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=128)
import pickle
with open("/content/drive/My Drive/Colab Notebooks/sahaja.pk2",'wb') as file:
      pickle.dump(model,file)
x=model.predict_classes(x_train[[7],:])
print(x[0])