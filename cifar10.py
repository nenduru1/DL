# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:35:15 2017

@author: O222069
"""
import numpy as np
import keras
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
num_classes = 10
data_augmentation=True
batch_size=32
epochs=100
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
print('x_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
#model=Sequential()
#model.add(Conv2D( 32, 3,3,input_shape=(32,32,3),activation='relu'))
##step 2 Pooling
#model.add(MaxPooling2D(pool_size=(2,2)))
#
##adding second convolution layer
#model.add(Conv2D( 2, 3,3,activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
##step 3 Flatten
#model.add(Flatten())
#model.add(Dense(32))
#model.add(Dense(1,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255
X_test/=255

if not data_augmentation:
    print('not using data augmentation')
    model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test),shuffle=True)
else:
    print('Using real time data augmentation')
    datagen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,vertical_flip=False)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))