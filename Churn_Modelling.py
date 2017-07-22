# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:49:09 2017

@author: o222069
"""

#using keras for churn Modelling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')
#label encoding for country and gender
#from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()
#le.fit_transform(dataset)


feature_col=['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X=dataset[feature_col].values
y=dataset.Exited.values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(12,input_dim=11,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(6,input_dim=11,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size = 1, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#Tuning the ANN
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(12,input_dim=11,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(6,input_dim=11,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[5,5],
           'epochs':[100,200],
           'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                        param_grid=parameters,
                        scoring='accuracy',
                        cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_Accuracy=grid_search.best_score_