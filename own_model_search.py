#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 22:36:14 2021

@author: gastoncavallo
"""

import tensorflow as tf
import keras_tuner as kt
#import json
import numpy as np
from sklearn.model_selection import train_test_split

# # Training Dataset

# dataset_as_JSON = False

# if dataset_as_JSON:
#     #Load training dataseet from JSON
#     with open("/Users/gastoncavallo/Desktop/Facultad/PI/JSON/TrainingsetHC.json", "r") as readfile: 
#         loaded = json.load(readfile)

#     signal_fv = None
#     for lab in loaded:
    
#         onehot = np.zeros((len(loaded[lab]), len(loaded)))
    
#         onehot[:, int(lab)] = 1.0
    
#         if signal_fv is None:
#             signal_fv = np.array(loaded[lab])
#             signal_fv_labels = onehot
#         else:
#             signal_fv = np.append(signal_fv, np.array(loaded[lab]), axis = 0)
#             signal_fv_labels = np.append(signal_fv_labels, onehot, axis = 0)

#     x_train, x_test, y_train, y_test = train_test_split(signal_fv, signal_fv_labels, test_size= 0.3)
    
# else:        
    #Load training dataset from npy
signal_fv = np.load("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_HC.npy")
signal_fv_labels = np.load("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_labels_HC.npy")
x_train, x_test, y_train, y_test = train_test_split(signal_fv, signal_fv_labels, test_size= 0.3)
    
def model_create(hp):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))

    for i in range(hp.Choice("n_layers", values = [1,2,3,4])):
        model.add(tf.keras.layers.Conv1D(filters = hp.Choice("units" + str(i), values=[8, 16, 32, 64]),
                                    kernel_size = hp.Int("ksize"+ str(i), min_value = 4, max_value = 32, step = 4),
                                    strides = hp.Choice("stride"+ str(i), values=[1,2,3,4]),
                                    activation = "relu",
                                    padding = "same"
                                    ))
        if hp.Choice("dropout" + str(i), values = [True, False]):
            model.add(tf.keras.layers.Dropout(rate = hp.Choice("rate"+str(i), values=[0.1, 0.2, 0.3])))
        if hp.Choice("max_pooling" + str(i), values = [True, False]):
            model.add(tf.keras.layers.MaxPooling1D(pool_size = hp.Choice("pool_size"+str(i), values=[2,4]),
                                                   strides = hp.Choice("stride"+str(i), values=[1,2])))
    
    model.add(tf.keras.layers.Flatten())
    
    if hp.Choice("Extra_dense", values = [True, False]):
        model.add(tf.keras.layers.Dense(hp.Int("dense_units", min_value=3, max_value=10, step=1),
                                        activation='softmax'))
    model.add(tf.keras.layers.Dense(2))

    optimizer = tf.keras.optimizers.SGD(learning_rate = hp.Choice("lr", values = [1e-3, 1e-4, 1e-5, 2e-3, 2e-4, 2e-5, 3e-3, 3e-4, 3e-5, 4e-3, 4e-4, 4e-5, 5e-3, 5e-4, 5e-5,]),
                                        momentum = hp.Int("sgd_mom", min_value=1, max_value = 10, step = 1) / 10)
    loss = hp.Choice("loss", values = ["categorical_crossentropy", "mse", "mae"])
    model.compile(optimizer, loss, metrics = ["accuracy"])
    model.summary()

    return model

def model_create(hp): #Funcion definida solo para buscar el summary
    model = tf.keras.models.Sequential()
    return model

tuner = kt.RandomSearch(model_create,
                        objective = "val_loss",
                        max_trials = 30,
                        executions_per_trial = 3,
                        overwrite = False,
                        directory = "/Users/gastoncavallo/Desktop/Facultad/PI",
                        project_name = "mod_propios"
                        )



tuner.search(x_train, y_train, epochs = 50, validation_data = (x_test, y_test))
tuner.results_summary()

