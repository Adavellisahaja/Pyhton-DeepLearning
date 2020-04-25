import numpy
from keras.datasets import cifar10


from keras.utils import np_utils
from keras import backend as K
from tensorflow_core.python.keras.models import load_model

K.common.image_dim_ordering()
model = load_model('/content/sample_data/my_model.tf')

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
yp=y_test

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def predict_img(model,input_img):
  from keras.models import load_model
  import numpy as np
  predicted_prob = model.predict_proba(input_img)
  predicted_prob = ["{:.7f}".format(i) for i in predicted_prob[0]]
  predict_class = model.predict_classes(input_img)
  return (predicted_prob,predict_class)


for i in range(1,5):
  input_img = np.expand_dims(X_test[i],axis=0)
 # predicted_prob,predict_class = predict_img(model,input_img)
#  print (predicted_prob,predict_class)
  predicted_prob = model.predict(input_img)
  a=1
  predicted_prob = ["{:.7f}".format(i) for i in predicted_prob[0]]
  print (predicted_prob,a)
  plt.subplot(2,2,i)
  print ("True Answer",tags[i],labels[tags[i][0]],end='\n\n\n')
  plt.imshow(X_test[i])