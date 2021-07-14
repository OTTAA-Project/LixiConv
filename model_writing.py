#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:38:49 2021

@author: gastoncavallo
"""

import tensorflow as tf
from tensorflow.keras import applications

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

# --- Modelos para transfer learning ---

def create_model4():

    model_transfer1 = applications.vgg16.VGG16(include_top = False, 
                                               input_shape = (512, 128, 3), 
                                               pooling = "avg") #El pooling se puede variar entre avg o max
    
    cnn_model4 = tf.keras.models.Sequential()
    for layer in model_transfer1.layers:
        layer.trainable = False #congelo las capas que no deben ser entrenadas
   
    for capa in model_transfer1.layers:
        cnn_model4.add(capa) 
        
    cnn_model4.add(tf.keras.layers.Flatten()) #debe ser añadida para no tener incompatibilidades de dimensiones
    #cnn_model4.add(tf.keras.layers.Dense(10, activation = 'softmax')) #capa densa opcional - Parametros entrenables se va a 5152
    cnn_model4.add(tf.keras.layers.Dense(2, activation = 'softmax')) #para clasificar 2 clases
    cnn_model4.summary()
    return cnn_model4

def create_model5():
    
    model_transfer2 = applications.resnet50.ResNet50(include_top = False,
                                                     input_shape = (512,128,3),
                                                     pooling = "avg")
    
    for layer in model_transfer2.layers:
        layer.trainable = False
        
    cnn_model5 = tf.keras.models.Sequential()
    cnn_model5.add(model_transfer2)
    cnn_model5.add(tf.keras.layers.Flatten())
    #cnn_model5.add(tf.keras.layers.Dense(5, activation = 'softmax')) #capa densa opcional
    cnn_model5.add(tf.keras.layers.Dense(2, activation = 'softmax'))
    
    cnn_model5.summary()
    
    return cnn_model5

def create_model6():
    
    model_transfer3 = applications.inception_v3.InceptionV3(include_top = False,
                                                            input_shape = (512,128,3),
                                                            pooling = "avg")
    
    for layer in model_transfer3.layers:
        layer.trainable = False
        
    cnn_model6 = tf.keras.models.Sequential()
    cnn_model6.add(model_transfer3)
    cnn_model6.add(tf.keras.layers.Flatten())
    #cnn_model6.add(tf.keras.layers.Dense(5, activation = 'softmax')) #capa densa opcional
    cnn_model6.add(tf.keras.layers.Dense(2, activation = 'softmax'))
    
    cnn_model6.summary()
    
    return cnn_model6

def create_model7():
    
    model_transfer4 = applications.xception.Xception(include_top = False,
                                                            input_shape = (512,128,3),
                                                            pooling = "avg")
    
    for layer in model_transfer4.layers:
        layer.trainable = False
        
    cnn_model7 = tf.keras.models.Sequential()
    cnn_model7.add(model_transfer4)
    cnn_model7.add(tf.keras.layers.Flatten())
    #cnn_model7.add(tf.keras.layers.Dense(5, activation = 'softmax')) #capa densa opcional
    cnn_model7.add(tf.keras.layers.Dense(2, activation = 'softmax'))
    
    cnn_model7.summary()
    
    return cnn_model7

def create_model8():
    
    model_transfer5 = applications.mobilenet.MobileNet(include_top = False,
                                                       input_shape = (512, 128, 3),
                                                       pooling = "avg")
    
    for layer in model_transfer5.layers:
        layer.trainable = False
        
    cnn_model8 = tf.keras.models.Sequential()
    cnn_model8.add(model_transfer5)
    cnn_model8.add(tf.keras.layers.Flatten())
    cnn_model8.add(tf.keras.layers.Dense(5, activation = 'softmax')) #capa densa opcional
    cnn_model8.add(tf.keras.layers.Dense(2, activation = 'softmax'))
    
    cnn_model8.summary()
    
    return cnn_model8
        

#Create a basic model instance
model_appsim1 = create_model1()
model_appsim2 = create_model2()
model_appsim3 = create_model3()

modeltl_vgg16 = create_model4()
modeltl_resnet50 = create_model5()
modeltl_inceptionv3 = create_model6()
modeltl_xception = create_model7()
modeltl_mobilenet = create_model8()

#Save the model architecture
model_appsim1.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/model_appsim1')
model_appsim2.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/model_appsim2')
model_appsim3.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/model_appsim3')

modeltl_vgg16.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/modeltl_vgg16')
modeltl_resnet50.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/modeltl_resnet50')
modeltl_inceptionv3.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/modeltl_inceptionv3')
modeltl_xception.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/modeltl_xception')
modeltl_mobilenet.save('/Users/gastoncavallo/Desktop/Facultad/PI/models_arch/modeltl_mobilenet')

