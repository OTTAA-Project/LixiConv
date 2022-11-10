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

default_optimizer = tf.keras.optimizers.SGD(lr = 0.005, momentum = 0.1)
default_loss = 'categorical_crossentropy'

def create_model1():

    cnn_model1 = tf.keras.models.Sequential()

    cnn_model1.add(tf.keras.layers.Conv1D(8, 4, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model1.add(tf.keras.layers.Conv1D(4, 16, padding = 'valid', activation = 'relu', strides = 1))
    cnn_model1.add(tf.keras.layers.Flatten())
    cnn_model1.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model1.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
    cnn_model1.summary()

    return cnn_model1

def create_model2():

    cnn_model2 = tf.keras.models.Sequential()

    cnn_model2.add(tf.keras.layers.Conv1D(8, 4, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model2.add(tf.keras.layers.Conv1D(4, 16, padding = 'valid', activation = 'relu', strides = 1))
    cnn_model2.add(tf.keras.layers.Flatten())
    cnn_model2.add(tf.keras.layers.Dense(3, activation='softmax'))
    cnn_model2.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model2.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
    cnn_model2.summary()

    return cnn_model2

def create_model3():

    cnn_model3 = tf.keras.models.Sequential()

    cnn_model3.add(tf.keras.layers.Conv1D(10, 4, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model3.add(tf.keras.layers.Conv1D(50, 13, padding = 'valid', activation = 'relu', strides = 13))
    cnn_model3.add(tf.keras.layers.Flatten())
    cnn_model3.add(tf.keras.layers.Dense(100, activation='softmax'))
    cnn_model3.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model3.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
    cnn_model3.summary()

    return cnn_model3

def create_model12():
    
    cnn_model12 = tf.keras.models.Sequential()
    
    cnn_model12.add(tf.keras.layers.Conv1D(64, 3, padding = 'same', input_shape = (512,4)))
    cnn_model12.add(tf.keras.layers.BatchNormalization())
    cnn_model12.add(tf.keras.layers.ReLU())
    cnn_model12.add(tf.keras.layers.Conv1D(64, 3, padding = 'same'))
    cnn_model12.add(tf.keras.layers.BatchNormalization())
    cnn_model12.add(tf.keras.layers.ReLU())
    cnn_model12.add(tf.keras.layers.Conv1D(64, 3, padding = 'same'))
    cnn_model12.add(tf.keras.layers.BatchNormalization())
    cnn_model12.add(tf.keras.layers.ReLU())               
    cnn_model12.add(tf.keras.layers.GlobalAveragePooling1D())
    cnn_model12.add(tf.keras.layers.Dense(2, activation="softmax"))

    cnn_model12.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])    
    cnn_model12.summary()
    
    return cnn_model12

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
   
    cnn_model4.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])    
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
    
    cnn_model5.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])    
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
    
    cnn_model6.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
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
    
    cnn_model7.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
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
    
    cnn_model8.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
    cnn_model8.summary()
    
    return cnn_model8


# --- Modelos propios

def create_model9():
    
    regularizer = tf.keras.regularizers.l2(0.01) # kernel_regularizer = regularizer,
    cnn_model9 = tf.keras.models.Sequential()

    cnn_model9.add(tf.keras.layers.Conv1D(8, 32, padding = 'valid', activation = 'relu', strides = 1, kernel_regularizer = regularizer, input_shape = (512, 4)))
    cnn_model9.add(tf.keras.layers.Dropout(0.3))
    cnn_model9.add(tf.keras.layers.MaxPooling1D(pool_size =  2, strides = None))
    cnn_model9.add(tf.keras.layers.Conv1D(16, 16, padding = 'valid', activation = 'relu', strides = 1, kernel_regularizer = regularizer))
    cnn_model9.add(tf.keras.layers.Dropout(0.2))
    cnn_model9.add(tf.keras.layers.Flatten())
    cnn_model9.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model9.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
    cnn_model9.summary()
    
    return cnn_model9

def create_model10():
    
    regularizer = tf.keras.regularizers.l2(0.01) # kernel_regularizer = regularizer,
    cnn_model10 = tf.keras.models.Sequential()

    cnn_model10.add(tf.keras.layers.Conv1D(8, 32, padding = 'valid', activation = 'relu', strides = 1, kernel_regularizer = regularizer, input_shape = (512, 4)))
    cnn_model10.add(tf.keras.layers.Dropout(0.3))
    cnn_model10.add(tf.keras.layers.MaxPooling1D(pool_size =  4, strides = 2))
    cnn_model10.add(tf.keras.layers.Conv1D(16, 16, padding = 'valid', activation = 'relu', strides = 1, kernel_regularizer = regularizer))
    cnn_model10.add(tf.keras.layers.Dropout(0.3))
    cnn_model10.add(tf.keras.layers.Flatten())
    cnn_model10.add(tf.keras.layers.Dense(2, activation='softmax'))

    cnn_model10.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
    cnn_model10.summary()
    
    return cnn_model10     

def create_model11():

    cnn_model11 = tf.keras.models.Sequential()

    cnn_model11.add(tf.keras.layers.Conv1D(8, 32, padding = 'valid', activation = 'relu', strides = 1, input_shape = (512, 4)))
    cnn_model11.add(tf.keras.layers.MaxPooling1D(pool_size =  2, strides = None))
    cnn_model11.add(tf.keras.layers.Conv1D(16, 16, padding = 'valid', activation = 'relu', strides = 1))
    cnn_model11.add(tf.keras.layers.Dropout(0.3))
    cnn_model11.add(tf.keras.layers.MaxPooling1D(pool_size =  4, strides = 2))
    cnn_model11.add(tf.keras.layers.Flatten())
    cnn_model11.add(tf.keras.layers.Dense(5, activation='softmax'))
    cnn_model11.add(tf.keras.layers.Dense(2, activation='softmax'))


    cnn_model11.compile(optimizer = default_optimizer, loss = default_loss, metrics = ['accuracy'])
    cnn_model11.summary()

    return cnn_model11

#Modelos propios creados con keras tuner

def model_create1():

    model1 = tf.keras.models.Sequential()
    
    model1.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 28,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same", input_shape = (512, 4)))
    model1.add(tf.keras.layers.Dropout(rate = 0.1))
    model1.add(tf.keras.layers.Conv1D(filters = 32,
                                    kernel_size = 24,
                                    strides = 1,
                                    activation = "relu",
                                    padding = "same"))
    model1.add(tf.keras.layers.Conv1D(filters = 32,
                                    kernel_size = 8,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
    model1.add(tf.keras.layers.Dropout(rate = 0.2))
    model1.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    model1.add(tf.keras.layers.Conv1D(filters = 16,
                                    kernel_size = 24,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    model1.add(tf.keras.layers.Flatten())
    model1.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002 ,
                                        momentum = 3/10)
    loss = 'categorical_crossentropy'
    model1.compile(optimizer, loss, metrics = ['accuracy'])
    model1.summary()
    
    return model1

def model_create2():

    model2 = tf.keras.models.Sequential()

    
    model2.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 28,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same",
                                    input_shape = (512, 4)))
    model2.add(tf.keras.layers.MaxPooling1D(pool_size = 2,
                                            strides = 1))
    
    model2.add(tf.keras.layers.Conv1D(filters = 16,
                                    kernel_size = 4,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
    model2.add(tf.keras.layers.Dropout(rate = 0.2))
    model2.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 1))
 
    model2.add(tf.keras.layers.Flatten())
    model2.add(tf.keras.layers.Dense(units = 4,
                                    activation='softmax'))
    model2.add(tf.keras.layers.Dense(2, activation='softmax'))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0005 ,
                                        momentum = 2/10)
    loss = 'categorical_crossentropy'
    model2.compile(optimizer, loss, metrics = ['accuracy'])
    model2.summary()
    
    return model2

def model_create3():

    model3 = tf.keras.models.Sequential()
   
    model3.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 28,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same", input_shape = (512, 4)))
    model3.add(tf.keras.layers.Dropout(rate = 0.2))
    
    model3.add(tf.keras.layers.Conv1D(filters = 16,
                                    kernel_size = 24,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    model3.add(tf.keras.layers.Dropout(rate = 0.2))
    
    model3.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 12,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
    model3.add(tf.keras.layers.Dropout(rate = 0.1))
    
    model3.add(tf.keras.layers.Flatten())
    model3.add(tf.keras.layers.Dense(units = 5,
                                    activation='softmax'))
    model3.add(tf.keras.layers.Dense(2, activation='softmax'))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0003 ,
                                        momentum = 9/10)
    loss = 'categorical_crossentropy'
    model3.compile(optimizer, loss, metrics = ['accuracy'])
    model3.summary()
    
    return model3

def model_create6():

    model6 = tf.keras.models.Sequential()
    
    model6.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 20,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same", input_shape = (512, 4)))
    model6.add(tf.keras.layers.MaxPooling1D(pool_size = 2,
                                            strides = 1))
    
    model6.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 12,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
    model6.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model6.add(tf.keras.layers.Flatten())
    model6.add(tf.keras.layers.Dense(units = 6,
                                    activation='softmax'))
    model6.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0004 ,
                                        momentum = 7/10)
    loss = 'categorical_crossentropy'
    model6.compile(optimizer, loss, metrics = ['accuracy'])
    model6.summary()
    
    return model6

#Create a basic model instance
model_appsim1 = create_model1()
model_appsim2 = create_model2()
model_appsim3 = create_model3()
model_appsim4 = create_model12()

modeltl_vgg16 = create_model4()
modeltl_resnet50 = create_model5()
modeltl_inceptionv3 = create_model6()
modeltl_xception = create_model7()
modeltl_mobilenet = create_model8()

model_own1 = create_model9()
model_own2 = create_model10()
model_own3 = create_model11()

model_own4 = model_create1()
model_own5 = model_create2()
model_own6 = model_create3()
model_own7 = model_create6()



#Save the model architecture
model_appsim1.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_appsim1')
model_appsim2.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_appsim2')
model_appsim3.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_appsim3')
model_appsim4.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_appsim4')

modeltl_vgg16.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/modeltl_vgg16')
modeltl_resnet50.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/modeltl_resnet50')
modeltl_inceptionv3.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/modeltl_inceptionv3')
modeltl_xception.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/modeltl_xception')
modeltl_mobilenet.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/modeltl_mobilenet')

model_own1.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_own1')
model_own2.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_own2')
model_own3.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_own3')
model_own4.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_own4')
model_own5.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_own5')
model_own6.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_own6')
model_own7.save('/Users/gastoncavallo/Desktop/Facultad/PI/untrained_models/model_own7')
