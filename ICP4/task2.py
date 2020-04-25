import keras
import numpy
from keras.datasets import cifar10

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Input

K.common.set_image_dim_ordering('th')
tbCallBack= keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=True)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

visible = Input(shape=(32,32,3))
x=Conv2D(32,(3,3),padding="same",activation='relu')(visible)
x=Dropout(0.2)(x)
x=Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
x=Dropout(0.5)(x)
x=Flatten()(x)
x=Dense(512, activation='relu', kernel_constraint=maxnorm(3))(x)
x=Dropout(0.3)(x)
x=Dense(num_classes, activation='softmax')(x)
m2 = Model(inputs=visible,output=x)


epochs = 3
lrate = 0.001
decay = lrate/epochs
sgd = Adam(lr=lrate)
m2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

m2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)