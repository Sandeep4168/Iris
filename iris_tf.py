# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:08:18 2020

@author: user
"""


import pandas as pd
iris_df=pd.read_csv('iris.csv')

X = iris_df.iloc[:,1:5].values
y = iris_df.iloc[:,5].values
from sklearn.preprocessing import LabelEncoder
one=LabelEncoder()
y1=one.fit_transform(y)
y=pd.get_dummies(y1).values


from sklearn.model_selection import train_test_split
X_train_full,X_test,y_train_full,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train,X_val,y_train,y_val=train_test_split(X_train_full,y_train_full,test_size=0.2,random_state=42)




from keras.utils import to_categorical 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import numpy as np
model=Sequential()
model.add(Dense(10,input_shape=(4,),activation='tanh'))
model.add(Dense(300,activation='tanh'))
model.add(Dense(100,activation='tanh'))
model.add(Dense(3,activation='softmax'))

model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(X_train,y_train,epochs=30,validation_data=(X_val,y_val))
y_pred=model.predict(X_test)
y_test_class=np.argmax(y_test,axis=1)
y_pred_class=np.argmax(y_pred,axis=1)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))

import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard_cb])

def build_model(n_hidden=2, n_neurons=30, learning_rate=0.04, input_shape=[4,]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(3,activation='softmax'))
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=['accuracy'])
    return model

keras_clf=keras.wrappers.scikit_learn.KerasClassifier(build_model)

keras_clf.fit(X_train,y_train,validation_data=(X_val,y_val),callbacks=[keras.callbacks.EarlyStopping(patience=10)])
y_pred=keras_clf.predict(X_test)
# y_test_class=np.argmax(y_test,axis=1)
# y_pred_class=np.argmax(y_pred,axis=1)
# clf_rep=classification_report(y_test,y_pred)
#from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": [0.04,0.05,0.06],
}
rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=2, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=30,
                  validation_data=(X_val, y_val),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
y_pred=rnd_search_cv.predict(X_test)
# y_test_class=np.argmax(y_test,axis=1)
# y_pred_class=np.argmax(y_pred,axis=1)
# clf_rep=classification_report(y_test_class,y_pred_class)

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)

model=rnd_search_cv.best_estimator_.model
model.summary()











