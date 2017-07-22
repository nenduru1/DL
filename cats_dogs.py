# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 07:49:30 2017

@author: O222069
"""
from keras import backend as K
import os
import imp

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        imp.reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")
#Convolution NN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
#Initialize the CNN
classifier=Sequential()
#step 1 Convolution
#classifier.add(Convolution2D(64,3,3,input_shape=(3,32,32)))
classifier.add(Conv2D( 2, 3,3,
            border_mode='same',
            input_shape=(3, 32, 32)))
#step 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#step 3 Flatten
classifier.add(Flatten())
#step 4 Full Connection
classifier.add(Dense(output_dim=64,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#P2 fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datgen=ImageDataGenerator(rescale=1./255)

train_set=train_datagen.flow_from_directory('dataset/training_set/',target_size=(32,32),batch_size=128,class_mode='binary')

test_set=train_datagen.flow_from_directory('dataset/test_set/',target_size=(32,32),batch_size=128,class_mode='binary')

classifier.fit_generator(train_set,samples_per_epoch=8000,nb_epoch=2,validation_data=test_set,nb_val_samples=2000)


print(sys.version)
