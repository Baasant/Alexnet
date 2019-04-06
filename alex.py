from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import keras
import csv
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as k
import sys
from keras.models import Model

import os

import pandas as pd
#np.random.seed(1000)
import cv2
'''''''''
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)
model.summary()
'''

#image_train=pd.read_csv('D:\\love\\alexnet')
##data_path = os.path.join(image_train,'*d')
##files = glob.glob(data_path)
##data = []
##for f1 in files:
    ##img = cv2.imread(f1)
    ##data.append(img)
#image_train=image_train/225.0

labels_train=pd.read_csv('D:\\love\\alexnet\\trainLabels.csv')

#keras.preprocessing.train_datagen.flow_from_directory

IG = ImageDataGenerator()
train_generator =IG.flow_from_directory(
    directory=r"D:\\love\\alexnet",
    target_size=(28, 28,1),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
model = Sequential()




#epochs=50 #check photo 5 times
#batch_size=32 #after 16 image update parameters
#learn_rate = 1e-4  # sgd learning rate
#momentum = .9  # sgd momentum to avoid local minimum
#transformation_ratio = .05

#alex net in keras
# (1) Importing dependency
import keras

# (2) Get Data
#import tflearn.datasets.oxflower17 as oxflower17
#x, y = oxflower17.load_data(one_hot=True)

# (3) Create a sequential model


# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(17))
model.add(Activation('softmax'))

model.summary()




# (4) Compile
model.compile(loss='categorical_crossentropy', optimizer='adam',
 metrics=['accuracy'])

model.fit_generator(generator=train_generator)


# (5) Train
# model.fit(train_generator, labels_train, batch_size=64, epochs=1, verbose=1,
# validation_split=0.2, shuffle=False)

#save the model
model.save('my_model.h5')
model.save_weights('my_model_weights.h5')

