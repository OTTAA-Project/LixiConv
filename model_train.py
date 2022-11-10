#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:54:57 2021

@author: gastoncavallo
"""

import numpy as np
import json
import os
import pandas as pd
import tensorflow as tf
import sklearn.metrics 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset_as_JSON = False

if dataset_as_JSON:

    #Training Dataset using JSON
    with open("/Users/gastoncavallo/Desktop/Facultad/PI/JSON/TrainingsetHC.json", "r") as readfile: #Trainingset solo es para prubar, hecho con AR2 unicamente
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

else:        
    #Training Dataset using NPY
    signal_fv = np.load("./np_dataset/signal_fv_HC.npy")
    signal_fv_labels = np.load("./np_dataset/signal_fv_labels_HC.npy")
        
#Training feature vector generator for Transfer Learning (Genera los feature vectors que serán usadas para entrenar en caso de transfer learninr)

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
#- Model Training 
# Primero defino path, despues veo si es Transfer learning o no y en funcion de eso entreno

untrained_path = "./untrained_models" #Path donde se encuentran los modelos
trained_path = "./trained_models"
untrained_models_dirs = [m for m in os.listdir(untrained_path) if "model" in m]

checkpoint_path = "./checkpoints"
callback_monitor = "val_accuracy"

metrics_path = "./metrics"

for element in untrained_models_dirs:
    new_model = tf.keras.models.load_model(os.path.join(untrained_path, element))
    
    if "modeltl" in element:
        feature = signaltl_fv
        # - Model Training parameters
        optimizer = tf.keras.optimizers.SGD(lr = 0.005, momentum = 0.1)
        loss = 'categorical_crossentropy'
        
    else:
        feature = signal_fv

    #Defino los callbacks
    es = tf.keras.callbacks.EarlyStopping(monitor=callback_monitor, min_delta=0.001, patience=7, restore_best_weights="False")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, element), monitor=callback_monitor, save_best_only=True)
    
    #Hago test_train_split al dataset
    X_train, X_test, y_train, y_test = train_test_split(feature, signal_fv_labels, test_size=0.3, random_state=42)
    
    #Entreno modelo
    history = new_model.fit(X_train, y_train,
                        epochs=40, validation_data=(X_test, y_test), shuffle=True,
                        batch_size = 32,
                        callbacks = [es, checkpoint]) 
    
    #Reaload checkpoint
    checkpoint_epoch = (list(np.array(history.history[callback_monitor])).index(max(history.history[callback_monitor])))
    final_model = tf.keras.models.load_model(os.path.join(checkpoint_path, element))
    
    #Calculo métricas que faltan
    preds = final_model.predict(feature)
    preds_for_metrics = preds.argmax(axis=1)
    labels_for_metrics = signal_fv_labels.argmax(axis=1)
    
    # - F1 score
    f1 = sklearn.metrics.f1_score(y_true = labels_for_metrics, y_pred = preds_for_metrics)
    # - Matthews correl coef
    mcc = sklearn.metrics.matthews_corrcoef(y_true = labels_for_metrics, y_pred = preds_for_metrics)
    # - Recall
    rec = sklearn.metrics.recall_score(y_true = labels_for_metrics, y_pred = preds_for_metrics)
    # - Precision
    prec = sklearn.metrics.precision_score(y_true = labels_for_metrics, y_pred = preds_for_metrics)

    #df de metricas de cada modelo
    metric_name = list(history.history.keys())
    metrics_df = pd.DataFrame([[checkpoint_epoch, history.history[metric_name[2]], history.history[metric_name[3]], 
                                history.history[metric_name[2]][checkpoint_epoch],
                                history.history[metric_name[3]][checkpoint_epoch],
                                prec, rec, f1, mcc]], index = [element], columns = ["ch_epoch","Val_Loss", "Val_Accuracy","Loss_ch","Acc_ch", "Precision", "Recall","F1 Score","MCC"])
    metrics_df.to_csv(os.path.join(metrics_path, element + ".csv"))

    #Que cada train guarde una imagen con dos graficos en una carpeta
    
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(element)
    axs[0].set_title("model accuracy")
    axs[0].plot(history.history['accuracy'], label="train")
    axs[0].plot(history.history['val_accuracy'], label="test")
    axs[0].set_xlabel("epochs")
    axs[0].set_ylim(0,1)
    axs[1].set_title("model loss")
    axs[1].plot(history.history['loss'], label="train")
    axs[1].plot(history.history['val_loss'], label="test")
    axs[1].set_xlabel("epochs")
    
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    plt.savefig(os.path.join("./plots", element+".png"))
    
    new_model.save(os.path.join(trained_path, element))
    
lista_table_metrics = []
metrics_list = os.listdir(metrics_path)

for met in metrics_list:
    if ".csv" in met:
        lista_table_metrics.append(pd.read_csv(os.path.join(metrics_path, met), index_col=0))

full_table_metrics = pd.concat(lista_table_metrics)
full_table_metrics.to_csv(os.path.join(metrics_path, "full_table_metrics.csv"))   
        
        
