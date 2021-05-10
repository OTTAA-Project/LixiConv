#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:54:13 2021

@author: gastoncavallo
"""

#-- Libraries Import
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
import tkinter as tk
from tkinter import filedialog as tkFileDialog
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
import datetime
from feature_processing import signal
from feature_processing import feature

#--- Entrenar, testear, cargar pesos y realizar comparaciones.

#--- Model Training

def model_train(model, X, y, epochs, optimizer = 'adam', loss = 'categorical_crossentropy',
                validation_set_mode = "split", val_part = 0.33, val_X = None, val_y = None, 
                batch_size = 32, random_state=42,
                plot_loss = True, plot_accuracy = True):
    
    """
    
    Inputs:
        - model (tensorflow.keras.Sequential): model to train
        - X (numpy.ndarray): array containing features or characteristics of the model to train
        - y (numpy.ndarray): array containing the labels or targets of the model to train
        - epochs (int): number of epochs to train the model
        - optimizer (tensorflow.keras.optimizers or str): optimizer for the training of the model, following tensorflow workflow
        - loss (tensorflow.keras.optimizers or str)): indicate the loss function to use
        - validation_set_mode (str): string with values "split" or "load" indicating how the validation set will be obtained 
        - val_part (float): fraction of the data that will be used as validation set to test the model, in validation_set_mode = "split" case
        - val_X (numpy.ndarray): features to use for validation in validation_set_mode = "load" case
        - val_y (numpy.ndarrat): labels to use for validation in validation_set_mode = "load" case
        - batch_size (int): size of the batch
        - random_state (int): repeatability of the validation set.
        - plot_loss (boolean): wether or not to plot loss.
        - plot_accuracy (boolean): wether or not to plot accuracy.
        
    Description:
        Train the Tensorflow keras Model with the training data.
        If plot_loss and/or plot_accuracy are set as True, plots the accuracy and loss of the model trhoughout the training, both on validation and training set.
        If "split", then the validation set will be obtained as a fraction val_part of the training set
        If "load", then the validation set will be load as val_X and val_y. In this case val_X and val_y cannot be None
        If a value different from "split" or "load" is passed, an error will be raised.
    
    Output:
        - model (tensorflow.keras.Sequential): trained model
        - history (tensorflow.keras.History): object containing info from the training process.
        
    
    """
    
    assert validation_set_mode == "split" or validation_set_mode == "load", str(validation_set_mode) + " is not a valid option for validation_set_mode"
    
    if validation_set_mode == "split":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_part, random_state=random_state)
    
    if validation_set_mode == "load":
        assert (val_X is not None) and (val_y is not None), "If the Validation Set will be loaded, features and labels should be loaded in val_X and val_y respectively"
        X_train = X
        y_train = y
        X_test = val_X
        y_test = val_y
        
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
   
    print(model.summary())
    
    history = model.fit(X_train, y_train,
                        epochs=epochs, validation_data=(X_test, y_test), shuffle=True,
                        batch_size = batch_size
                        )
    
    
    # summarize history for loss
    if plot_loss:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    #summarize history for accuracy
    if plot_accuracy:
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()        
    
    return model, history

#--- Model Test

def model_test(model, X_test, y_test, predictions = 10, buffer_size = 30):
    
    """
    
    Inputs:
        - model (tensorflow.keras.Sequential): trained model
        - X_test (): features to use to test the model
        - y_test (): labels to use to test the model
        - predictions (int): number of predictions to make
        - buffer_size (int): size of the buffer who contains the features and labels to make predictions
            
        
    Description:
    Function that takes an amount of predections buffer_size and make this predictions. 
        then it makes the mode, and compares it with the labels mode.
        
        
    Output:
        - None.
    """
    correct = 0
    
    for i in range(predictions):
        n = np.random.randint(0, X_test.shape[0] - predictions) #we substract predictions so that the random index does not overpass the FV amount.
        pred_matrix = np.zeros((buffer_size, y_test.shape[1])) #create a matrix of zeros of size (buffer, classes).
        label_matrix = np.zeros((buffer_size, y_test.shape[1])) 
        for j in range(buffer_size):
            pred_matrix[j, :] = model.predict(X_test[n+j:n+j+1, :])
            label_matrix[j, :] = y_test[n+j:n+j+1, :]
        
        print("Prediction no. ", i+1)
        #print(pred_matrix)
        #print(label_matrix)
        #print(np.argmax(pred_matrix, axis = 1))
        #print(np.argmax(label_matrix, axis = 1))
        pred = stats.mode(np.argmax(pred_matrix, axis = 1))
        label = stats.mode(np.argmax(label_matrix, axis = 1))
    
        print("Prediction: \n", pred)
        print("Label: \n", label)
        if pred[0][0] == label[0][0]:
            print("Correcto!!\n\n-\n\n")
            correct += 1
        else:
            print("Incorrecto!!\n\n-\n\n")  
    
    print(str(correct) + " out of " + str(predictions) + " predictions were correct.")
    print("That is " + str(100*correct/predictions) + "% Correct.")

#--- Weights Save

def save_weights(model, history, folder_dir):
    """
    
    Inputs:
        - model (tensorflow.keras.Sequential): trained model
        - history (): object containing info from the training process.
        - folder_dir(str): directory of the folder to save the weights
        
    Description:
    Save de weights of the model, exept those who came from Dropout layers.
    Also save a json object with the model characterist and the loss and accuracy results
        
    Output:
        .csv file with the weights
        
    """
    
    n = len(model.layers) # total number of layers
    
    li = 0 # Dense layers counter
    files = {} #creating an empty dictionary 
    for i in range(n):
        if ("Dropout" in str(type(model.layers[i])) ): #Dropout layers are only useful during training and it's weights are no longer needed in the OpenBCI model
            pass # discard, nothing to save
        else:
            wi = model.layers[i].get_weights()[0] #this should return an array with the weights of the i-eth layer
            bi = model.layers[i].get_weights()[1] #this should return an array with the biases of the i-eth layer

            files['w'+str(li)] = wi #"appending" the values of the weights of each layer to the dictionary with the name wi for
            files['b'+str(li)] = bi #the weights of the i-eth layer and bi for the biases

            li += 1
    
    #since to create the JSON object we need to get info on the model, we use it to create the folder too
    model_dict_tojson, dirName = save_model_json(history, folder_dir, model = model)
    
    # Save weights and bias in separate files        
    for f in files.keys():                                      #the .keys method return a dict_keys object with contains the keys (the names) of each element inside the dict
        np.savetxt(dirName + "/" + f + '.csv', files[f], delimiter=',') #since those keys are wi and bi for the weights and biases of the i-eth layer 
        print(files[f].shape)                                     #a csv for each array is saved

# -- Description of the Model Saved in JSON file

def save_model_json(history, folder_dir, **kwargs): #kwargs: model to get the layers from or layers_list with the layers numbers
    
    """
    
    Inputs:
        - history (): object containing info from the training process.
        - folder_dir (str): directory of the folder to save the weights
        - **kwargs: 
        
    Description:
    Take the layer info from the model and save the json 
    In **kwargs you could load the model to get the layers from, or a layers_list with the layers numbers 
    Also save the training model date.
    Output:
        - current_model_dict (dict): dictionary to save th JSON
            a."acc": (float) accuracy of the model
            b."val_acc": (float) validation accuracy of the model
            c. "loss" : (float) loss of the model
            d. "val_loss" : (float) validation loss
            e. "input_size" : input_size
            f. "hidden_layers_no":len(hidden_list)
            g. "hidden_size": hidden_list
            h. "output_size": output_size
            i. "train_date": date
        - dirName (str): directory for saving model JSON 
        
        
    """
    #First get model units numbers through the model itself or a list
    if "model" in kwargs.keys():
        model = kwargs["model"]
        input_size = model.layers[0].input_shape[-1]
        hidden_list = []
        for lay in model.layers[:-1]:
            if ("Dropout" in str(type(lay))):
                pass
            else:
                hidden_list.append(lay.output_shape[-1])
        output_size = model.layers[-1].output_shape[-1]
        
    elif "layers_list" in kwargs.keys():
        layers_list = kwargs["layers_list"]
        input_size = layers_list[0]
        hidden_list = layers_list[1:-1]
        output_size = layers_list[-1]
        
    else:
        raise ValueError("No object to get layers from please input model or layers_list")
    
    #Then get when the model was trained and the results
    train_date = str(datetime.datetime.now()).split(" ")[0]
    folder_name = str(input_size) + "-" + str(hidden_list) + "-" + str(output_size) + " " + train_date
    
    current_model_dict = {"acc": float(history.history['acc'][-1]),
                        "val_acc": float(history.history['val_acc'][-1]),
                        "loss" : float(history.history['loss'][-1]),
                        "val_loss" : float(history.history['val_loss'][-1]),
                        "input_size" : input_size,
                        "hidden_layers_no":len(hidden_list),
                        "hidden_size": hidden_list,
                        "output_size": output_size,
                        "train_date": train_date}
    
    #Reading other models present in the file to compare,
    #if the model is better than others it's name will be changed
    #adding "Best" to it and removing it from the latest best.
    folder_name = read_model_json(current_model_dict, folder_dir, folder_name, only_best = True)
    
    # Create a directory for saving model JSON and return it to save_weights to save the model weights
    dirName = folder_dir + "/" + folder_name
    print(dirName)
    
    try:
        os.mkdir(dirName) #this function creates a new directory, and since it's embedded in the try function, if the file already exists it raises an error and data is directly saved there
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        pass
    
    #Saving the JSON file
    with open(dirName + "/model_info.json", "w") as output:
        json.dump(current_model_dict, output)
        
    return current_model_dict, dirName

# Read other JSON files with model descriptions and compare to current
# if only_best = True then it will only compare the model to the ones with "Best" name

def read_model_json(current_model_dict, folder_dir, folder_name, only_best = True):
    """
    
    Inputs:
        - current_model_dict (dict): dictionary to save the JSON
            a."acc": (float) accuracy of the model
            b."val_acc": (float) validation accuracy of the model
            c. "loss" : (float) loss of the model
            d. "val_loss" : (float) validation loss
            e. "input_size" : input_size
            f. "hidden_layers_no":len(hidden_list)
            g. "hidden_size": hidden_list
            h. "output_size": output_size
            i. "train_date": date
        - forlder_dir (str): directory of the folder to save the weights
        - folder_name (str): name of the folder containing the actual JSON
        - only_best (bool): boolean to determinate if only the folders that have "Best" will be compared
        
    Description:
        Read inside the folder where the model is saved, and comparee the structure. If it's the same, 
        then reads the accuracy and loss and write "Best" in the folder name in case that the current 
        model has better numbers.
    Output:
        - folder_name (str): name of the folder, with or without the "Best" depending on the case.
        
    """
    nothing_tocompare = True #this boolean is used if there is no other model with the same parameters to compare to, then the current one will be the best in its category
    for elem in os.listdir(folder_dir): #run through folders of models in the file
        if "Best" in elem or not only_best: #taking only the ones that have "Best" on them, if only_best = True, if not we take all
            with open(folder_dir + "/" + elem + "/model_info.json",) as f:
                dict_fromjson = json.load(f) #load JSON into dict
                f.close() #close the file, if not we wont be able to change folder names because files inside them will be opened
                
                if (current_model_dict["input_size"] == dict_fromjson["input_size"]) and (current_model_dict["hidden_size"] == dict_fromjson["hidden_size"]) and (current_model_dict["output_size"] == dict_fromjson["output_size"]):
                    nothing_tocompare = False #if there is at least one similar model then we have something to compare to
                    if model_comparison(current_model_dict, dict_fromjson): #run model_comparison
                        print("The model is better than the latest best, so it will replace it as the best one.")
                        remove_from_foldername(folder_dir + "/" + elem, "Best") #if its true we remove "Best" from the folder of the latest best
                        folder_name = "Best " + folder_name #and add "Best" to the current model folder_name
                        break #if we found the one that our model is best to, then there's no reason to keep searching
                    else:
                        print("The model is not better than the latest best, it will still be saved as itself.")
                        
    if nothing_tocompare: #if there is no other model to compare to or if the ones that exist don't have the same architecture, then our current model is the best on its category
        folder_name = "Best " + folder_name
        print("The model is the first of it's category, it will be saved as best.")
    
    return folder_name

# Model Comparison Score

def model_comparison(current_model_dict, model_tocompare, keys_tocompare = ["acc", "val_acc", "loss", "val_loss"]):
    """
    
    Inputs:
        - current_model_dict (dict): actual dictionary to save the JSON and compare
            a."acc": (float) accuracy of the model
            b."val_acc": (float) validation accuracy of the model
            c. "loss" : (float) loss of the model
            d. "val_loss" : (float) validation loss
            e. "input_size" : input_size
            f. "hidden_layers_no":len(hidden_list)
            g. "hidden_size": hidden_list
            h. "output_size": output_size
            i. "train_date": date
        - model_tocompare (dict): JSON dictionary to be compared with the current model
        - keys_tocompare (list): keys from dictionary to compare both models
            a. acc: accuracy
            b. val_acc: validation accuracy
            c. loss
            d. val_loss: validation loss
    
        
    Description:
        Compare the current model with another saved model in JSON format and return True if the current model is better,
        or False in the other case.
        This function cheks keys_to_compare in each model dictionary to determinate which one is the best
        If the keys_to_compare are from validation (val_acc, val_loss), they will be weighted by 2
    
    Output:
        - is_best (bool): boolean that describes if the model is the beest or not
        
    """  
    #This is a simple model comparison method to compare previous models to
    #newer ones, to see which one is best, checking keys_to_compare in each model
    #dictionary
    
    score = 0
    for key in keys_tocompare:
        if "loss" in key: #loss is better when is lowest, so we make exceptions in the scoring
            delta = model_tocompare[key] - current_model_dict[key]
        else:
            delta = current_model_dict[key] - model_tocompare[key]
            
        if "val" in key: #Since validation accuracy and loss are more important for us because they represent the model generalization power, we give them more importance
            score += 2*delta
        else:
            score += delta
    
    if score >= 0:
        is_best = True
        
    else:
        is_best = False
        
    return is_best

# Removing a word or phrase from a folder's name

def remove_from_foldername(original_dir, to_remove):
    """
    
    Inputs:
        - original_dir(str): Directory of the model that is no longer the best
        - to_remove(str): word that will be removed
        
        
    Description:
        Function that deletes the "BEST" of the folder that was compared and is no longer the best. 
   
    Output:
        - None.
      
        
    """
    folder_name = original_dir.split("/")[-1] #we take the folder's name from the whole path
    folder_dir = original_dir.split(folder_name)[0] #and the folder's location
    if to_remove + " " in folder_name: #if the word we want to remove was spaced from the others we'll have an extra space that will look bad, so we create another exception
        new_folder_name = folder_name.replace(to_remove + " ", "")
        new_dir = folder_dir + new_folder_name
        while True: #when we change the folder name there might be another folder on the same location with the same name as what we want to replace our folders name to
            try:
                os.rename(original_dir, new_dir)
            except OSError: #this will raise an OSError, which we use to add the word "New" to the folder we are modyfing, if that exists to, we add another "New" and so on
                new_dir = new_dir + " New"
            else: #once we are able to change the name of the folder, we break from the infinite loop.
                break
            
    elif to_remove in folder_name:
        new_folder_name = folder_name.replace(to_remove, "")
        new_dir = folder_dir + new_folder_name
        while True: #when we change the folder name there might be another folder on the same location with the same name as what we want to replace our folders name to
            try:
                os.rename(original_dir, new_dir)
            except OSError: #this will raise an OSError, which we use to add the word "New" to the folder we are modyfing, if that exists to, we add another "New" and so on
                new_dir = new_dir + " New"
            else: #once we are able to change the name of the folder, we break from the infinite loop.
                break
    else: #if the word we want to remove is not in the folder's name, we just pass. this else wouldn't be necessary, but just in case we want to add something in the future
        pass