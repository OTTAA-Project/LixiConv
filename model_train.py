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
with open("/Users/gastoncavallo/Desktop/Facultad/PI/JSON/TrainingsetHC.json", "r") as readfile:
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
        
# Training feature vector generator for Transfer Learning (Genera los feature vectors que serán usadas para entrenar en casso de transfer learninr)

def modificar_matriz(original, dimnec1, dimnec2, dimnec3):
    
    new = original

    while dimnec1 > (new.shape[1]):
        new = np.append(new, new, axis = 1)
    while dimnec2 > (new.shape[2]):
        new = np.append(new, new, axis = 2)
    
    if dimnec3 != None:
        new = np.expand_dims(new, axis = 3)   
        new_copy = new.copy()
        
        while dimnec3 > (new.shape[3]):
             new = np.append(new, new_copy, axis = 3)
    
    return new

signaltl_fv = modificar_matriz(signal_fv, 75,75,3)



# - Model Training with transfer learning

new_model = tf.keras.models.load_model("/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/modeltl_mobilenet")
optimizer = tf.keras.optimizers.SGD(lr = 0.005, momentum = 0.1)
loss = 'categorical_crossentropy'

cnn_model, history = mod.model_train(new_model, X = signaltl_fv, y = signal_fv_labels, epochs = 3,
                                        optimizer = optimizer, loss = loss,
                                        batch_size = 32)

        
# # - Load and Train Models
# # Función que carga y entrena todos los modelos expeto los de transfer learning.

# path = "/Users/gastoncavallo/Desktop/Facultad/PI/models_arch" #Path donde se encuentran los modelos
# models_dirs = os.listdir(path)

# for element in models_dirs:
#     if "model" in element:

#         new_model = tf.keras.models.load_model(path + "/" + element)

#         # - Model Training

#         optimizer = tf.keras.optimizers.SGD(lr = 0.005, momentum = 0.1)
#         loss = 'categorical_crossentropy'

#         cnn_model, history = mod.model_train(new_model, X = signal_fv, y = signal_fv_labels, epochs = 10,
#                                         optimizer = optimizer, loss = loss,
#                                         batch_size = 32)
