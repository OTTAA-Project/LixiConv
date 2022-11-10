#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:39:39 2021

@author: gastoncavallo
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from feature_processing import classifier as mod

signal_fv = np.load("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_HC.npy")
signal_fv_labels = np.load("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_labels_HC.npy")
x_train, x_test, y_train, y_test = train_test_split(signal_fv, signal_fv_labels, test_size= 0.3)

def model_create1():

    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model1.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 28,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
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
    model1.add(tf.keras.layers.Dense(2))
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002 ,
                                        momentum = 3/10)
    loss = "mse"
    model1.compile(optimizer, loss, metrics = ["accuracy"])
    model1.summary()
    
    return model1

def model_create2():

    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model2.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 28,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
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
    model2.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0005 ,
                                        momentum = 2/10)
    loss = "mae"
    model2.compile(optimizer, loss, metrics = ["accuracy"])
    model2.summary()
    
    return model2

def model_create3():

    model3 = tf.keras.models.Sequential()
    model3.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model3.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 28,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
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
    model3.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0003 ,
                                        momentum = 9/10)
    loss = "mae"
    model3.compile(optimizer, loss, metrics = ["accuracy"])
    model3.summary()
    
    return model3

def model_create4():

    model4 = tf.keras.models.Sequential()
    model4.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model4.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 20,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    model4.add(tf.keras.layers.Dropout(rate = 0.1))
    
    model4.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 8,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    model4.add(tf.keras.layers.Dropout(rate = 0.3)) 
    
    model4.add(tf.keras.layers.Flatten())
    model4.add(tf.keras.layers.Dense(units = 6,
                                    activation='softmax'))
    model4.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.004 ,
                                        momentum = 3/10)
    loss = "mse"
    model4.compile(optimizer, loss, metrics = ["accuracy"])
    model4.summary()
    
    return model4

def model_create5():

    model5 = tf.keras.models.Sequential()
    model5.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model5.add(tf.keras.layers.Conv1D(filters = 32,
                                    kernel_size = 24,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    model5.add(tf.keras.layers.Dropout(rate = 0.2))
    model5.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model5.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 28,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    model5.add(tf.keras.layers.Dropout(rate = 0.3))
    
    model5.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 20,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
    model5.add(tf.keras.layers.MaxPooling1D(pool_size = 2,
                                            strides = 1))
    
    model5.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 4,
                                    strides = 1,
                                    activation = "relu",
                                    padding = "same"))
    model5.add(tf.keras.layers.Dropout(rate = 0.1))
    model5.add(tf.keras.layers.MaxPooling1D(pool_size = 2,
                                            strides = 1))
    
    model5.add(tf.keras.layers.Flatten())
    model5.add(tf.keras.layers.Dense(units = 6,
                                    activation='softmax'))
    model5.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002 ,
                                        momentum = 2/10)
    loss = "mae"
    model5.compile(optimizer, loss, metrics = ["accuracy"])
    model5.summary()
    
    return model5

def model_create6():

    model6 = tf.keras.models.Sequential()
    model6.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model6.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 20,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
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
    model6.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0004 ,
                                        momentum = 7/10)
    loss = "mse"
    model6.compile(optimizer, loss, metrics = ["accuracy"])
    model6.summary()
    
    return model6

def model_create7():

    model7 = tf.keras.models.Sequential()
    model7.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model7.add(tf.keras.layers.Conv1D(filters = 16,
                                    kernel_size = 8,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    model7.add(tf.keras.layers.Dropout(rate = 0.1))
    model7.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model7.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 16,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    model7.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model7.add(tf.keras.layers.Flatten())
    model7.add(tf.keras.layers.Dense(units = 9,
                                    activation='softmax'))
    model7.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.005,
                                        momentum = 5/10)
    loss = "mse"
    model7.compile(optimizer, loss, metrics = ["accuracy"])
    model7.summary()
    
    return model7

def model_create8():

    model8 = tf.keras.models.Sequential()
    model8.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model8.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 8,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    model8.add(tf.keras.layers.Dropout(rate = 0.2))
    model8.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model8.add(tf.keras.layers.Conv1D(filters = 32,
                                    kernel_size = 16,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    model8.add(tf.keras.layers.Dropout(rate = 0.3))
    model8.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model8.add(tf.keras.layers.Conv1D(filters = 16,
                                    kernel_size = 16,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    model8.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model8.add(tf.keras.layers.Flatten())
    model8.add(tf.keras.layers.Dense(units = 10,
                                    activation='softmax'))
    model8.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002 ,
                                        momentum = 8/10)
    loss = "mse"
    model8.compile(optimizer, loss, metrics = ["accuracy"])
    model8.summary()
    
    return model8

def model_create9():

    model9 = tf.keras.models.Sequential()
    model9.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model9.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 28,
                                    strides = 3,
                                    activation = "relu",
                                    padding = "same"))
    
    model9.add(tf.keras.layers.Conv1D(filters = 32,
                                    kernel_size = 16,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    model9.add(tf.keras.layers.Dropout(rate = 0.1))
    model9.add(tf.keras.layers.MaxPooling1D(pool_size = 2,
                                            strides = 1))
    
    model9.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 4,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    model9.add(tf.keras.layers.MaxPooling1D(pool_size = 2,
                                            strides = 1))
    
    model9.add(tf.keras.layers.Conv1D(filters = 64,
                                    kernel_size = 28,
                                    strides = 1,
                                    activation = "relu",
                                    padding = "same"))
    model9.add(tf.keras.layers.Dropout(rate = 0.3))
    model9.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model9.add(tf.keras.layers.Flatten())
    model9.add(tf.keras.layers.Dense(units = 6,
                                    activation='softmax'))
    model9.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 5e-5,
                                        momentum = 6/10)
    loss = "mse"
    model9.compile(optimizer, loss, metrics = ["accuracy"])
    model9.summary()
    
    return model9

def model_create10():
    

    model10 = tf.keras.models.Sequential()
    model10.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    
    model10.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 12,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
    
    model10.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 8,
                                    strides = 4,
                                    activation = "relu",
                                    padding = "same"))
    
    model10.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 24,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    model10.add(tf.keras.layers.MaxPooling1D(pool_size = 4,
                                            strides = 2))
    
    model10.add(tf.keras.layers.Flatten())
    model10.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0001 ,
                                        momentum = 5/10)
    loss = "mse"
    model10.compile(optimizer, loss, metrics = ["accuracy"])
    model10.summary()
    
    return model10

def model_create11():
    model11 = tf.keras.models.Sequential()
    model11.add(tf.keras.layers.Input(shape = signal_fv.shape[1:]))
    model11.add(tf.keras.layers.Conv1D(filters = 16,
                                    kernel_size = 32,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same"))
    
    model11.add(tf.keras.layers.Conv1D(filters = 8,
                                    kernel_size = 4,
                                    strides = 1,
                                    activation = "relu",
                                    padding = "same"))
    model11.add(tf.keras.layers.Dropout(rate = 0.1))
    model11.add(tf.keras.layers.MaxPooling1D(pool_size = 2,
                                            strides = 1))
    model11.add(tf.keras.layers.Flatten())
    model11.add(tf.keras.layers.Dense(2))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0001 ,
                                        momentum = 4/10)
    loss = "mae"
    model11.compile(optimizer, loss, metrics = ["accuracy"])
    model11.summary()
    
    return model11
    
model = model_create11()
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0001 ,
                                        momentum = 4/10)
loss = "mae"

cnn_model, history = mod.model_train(model, X = signal_fv, y = signal_fv_labels, epochs = 20,
                                        optimizer = optimizer, loss = loss,
                                        batch_size = 32)