#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:54:57 2021

@author: gastoncavallo
"""

import numpy as np
import json
import os
import tensorflow as tf
from feature_processing import classifier as mod

# Training Dataset
with open("/Users/gastoncavallo/Desktop/Facultad/PI/JSON/TrainingSet.json", "r") as readfile:
    loaded = json.load(readfile)

signal_fv = None
for lab in loaded:
    
    onehot = np.zeros((len(loaded[lab]), len(loaded)))
    
    onehot[:, int(lab)] = 1.0
    
    if signal_fv is None:
        signal_fv = np.array(loaded[lab])
        signal_fv_labels = onehot
    else:
        signal_fv = np.append(signal_fv, np.array(loaded[lab]), axis = 0)
        signal_fv_labels = np.append(signal_fv_labels, onehot, axis = 0)
        
#Load models
#Debo agregar un for que recorra todos los modelos, los cargue y los entrene

path = "/Users/gastoncavallo/Desktop/Facultad/PI/models_arch" #Path donde se encuentran los modelos
models_dirs = os.listdir(path)

for element in models_dirs:
    if "model" in element:

        new_model = tf.keras.models.load_model(path + "/" + element)

        # - Model Training

        optimizer = tf.keras.optimizers.SGD(lr = 0.005, momentum = 0.1)
        loss = 'categorical_crossentropy'

        cnn_model, history = mod.model_train(new_model, X = signal_fv, y = signal_fv_labels, epochs = 10,
                                        optimizer = optimizer, loss = loss,
                                        batch_size = 32)
