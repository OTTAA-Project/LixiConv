#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:38:49 2021

@author: gastoncavallo
"""

import tensorflow as tf

#Define a function to create models
#Modellos de aplicación similar

def create_model1():

    cnn_model1 = tf.keras.models.Sequential()

    cnn_model1.add(tf.keras.layers.Conv1D(8, 4, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model1.add(tf.keras.layers.Conv1D(4, 16, padding = 'valid', activation = 'relu', strides = 1))
    cnn_model1.add(tf.keras.layers.Flatten())
    cnn_model1.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model1.summary()

    return cnn_model1

def create_model2():

    cnn_model2 = tf.keras.models.Sequential()

    cnn_model2.add(tf.keras.layers.Conv1D(8, 4, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model2.add(tf.keras.layers.Conv1D(4, 16, padding = 'valid', activation = 'relu', strides = 1))
    cnn_model2.add(tf.keras.layers.Flatten())
    cnn_model2.add(tf.keras.layers.Dense(3, activation='softmax'))
    cnn_model2.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model2.summary()

    return cnn_model2

def create_model3():

    cnn_model3 = tf.keras.models.Sequential()

    cnn_model3.add(tf.keras.layers.Conv1D(10, 4, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model3.add(tf.keras.layers.Conv1D(50, 13, padding = 'valid', activation = 'relu', strides = 13))
    cnn_model3.add(tf.keras.layers.Flatten())
    cnn_model3.add(tf.keras.layers.Dense(100, activation='softmax'))
    cnn_model3.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model3.summary()

    return cnn_model3

#Modelos para transfer learning


#puedo hacer un for que recorra todos los modelos definidos, los ejecute y los guarde


#Create a basic model instance
model_appsim1 = create_model1()
model_appsim2 = create_model2()
model_appsim3 = create_model3()

#Display model architecture
model_appsim1.summary()
model_appsim2.summary()
model_appsim3.summary()

#Save the model architecture
model_appsim1.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/model_appsim1')
model_appsim2.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/model_appsim2')
model_appsim3.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/model_appsim3')


