#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:38:49 2021

@author: gastoncavallo
"""

import tensorflow as tf

#Define a function to create models
def create_model():

    cnn_model = tf.keras.models.Sequential()

    cnn_model.add(tf.keras.layers.Conv1D(8, 4, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model.add(tf.keras.layers.Conv1D(4, 16, padding = 'valid', activation = 'relu', strides = 1))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model.summary()

    return cnn_model
#Create a basic model instance
model_appsim1 = create_model()

#Display model architecture
model_appsim1.summary()

#Save the model architecture

model_appsim1.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/model_appsim1')

