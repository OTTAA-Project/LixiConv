#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:54:09 2021

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
from feature_processing import signal as sig
from feature_processing import classifier

#--- Cargar, generar y exportar feature vector.


#--- Export Feature Vector

def export_feature_vector(name, save_dir, features_pd, labels_pd,
                          feature_discriminator_start = "/FV_X-", label_discriminator_start = "/FV_y-",
                          end_extension = "features.csv"):
    """
    Inputs:
        - name(str): name of the signal
        - save_dir(str): where you will save the feature vector
        - features_pd(pd): features of the feature vector
        - labels_pd(pd): labels of the feature vector
        - feature_discriminator_start(str): name to discriminate the feature
        - label_discriminator_start(str): name to discriminate the label
        - end_extension(str): end extension of the file
    Description:
        Features and labels will be saved as appropiate in each case, under the name "name", with the corresponding discriminator
        Feature and label will have the same extension at the end, but diferent discriminator start.
    Outputs:
        - None.
        
    """
    features_pd.to_csv(save_dir + feature_discriminator_start + name + end_extension, header = None, index = None)
    labels_pd.to_csv(save_dir + label_discriminator_start + name + end_extension, header = None, index = None)

# -- Load Feature Vector

def load_fv_fromdir(path):
    """
    Inputs:
        - path(str): path of the feature vector
    Description:
        Load and read feature vector. Only will read feature and label named "FV_X"+ name and "FV_y"+ name.
    Outputs:
        - X (numpy.array): features
        - y (numpy.array): labels
    """
    first = True
    for file in os.listdir(path):
        if ".txt" in file or ".csv" in file:
            if "FV_X" in file:
                name = file.split("FV_X-")[1] #this splits the name of the file in two parts from the "FV_X-" position, the first one would be an space, the second one the rest of the name
                pdX = pd.read_csv(os.path.join(path, file))
                pdy = pd.read_csv(os.path.join(path, ("FV_y-" + name)))
                print("Loaded " + name)
                
                if first == True:
                    X = pdX.values
                    y = pdy.values
                    first = False
                    
                else:
                    X = np.vstack((X, pdX.values))
                    y = np.vstack((y, pdy.values))
                    
    print("Feature Vectors Loaded. Sizes are: X = ", X.shape, "y = ", y.shape)
    
    return X, y

#--- Feature Vector Generator

def feature_vector_gen(fft_freq, fft_values, interest_freqs, neighbour_or_interval = "neighbour",
                       include_harmonics = True, apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0,
                       interest_bw = 1.0, max_bins = 5, #config for neighbour method
                       include_extremes = True): #config for interval method
    """
    Inputs:
        - fft_freq(np.ndarray): fft frequencies
        - fft_values(np.ndarray): fft values
        - interest_freqs(list or tuple): contain the interest frequencies on a list or a tuple with the minimun and maximun values. Depending on the case
        - neighbour_or_interval(str): Define if the feature vector is generated usin an interval or neighbour. One of this two options must be choosed
        - include_harmonics(bool): Boolean to choose if onclude harmonics or not. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - same_bw_forSNR(bool): 
        - bw_forSNR(float):
        - interest_bw(float): bandwidth to configure neighbour method
        - max_bins(int): maximum number of bins, to configure neighbour method
        - include_extremes(bool): Choose if include extremes or not if neighbour_orinterval = "interval"
            
    Description:
        General function where you chose if the feature vector is generated using an interval around the main freq or neighbours.
        If neighbour_or_interval = "neighbour", a list of freq must be given on interest_freqs. This list will be as long as frequencies of interest exist 
        If neighbour_or_interval = "interval", a tuple of minumum and maximum values must be given on interest_freqs. It will be a tuple of size 0 to 2.
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
    
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - feature_vector(np.array): feature vector values
    """
    assert neighbour_or_interval == "neighbour" or neighbour_or_interval == "interval", "The choice for Interval or Neighbour method isn't compatible"
    
    if neighbour_or_interval == "neighbour":
        assert type(interest_freqs) == list, "For Neighbour method, a list must be given on interest_freqs"
        feature_vector_bins, feature_vector = feature_vector_gen_neighbour(fft_freq, fft_values, interest_freqs,
                                                                           include_harmonics, apply_SNR, same_bw_forSNR, bw_forSNR,
                                                                           interest_bw, max_bins)
        
    elif neighbour_or_interval == "interval": #shouldn't be necessary since we have the assertion above, but..
        assert type(interest_freqs) == tuple, "For Interval method, a tuple must be given on interest_freqs with minimum and maximum values"
        feature_vector_bins, feature_vector = feature_vector_gen_interval(fft_freq, fft_values, interest_freqs,
                                                                          apply_SNR, include_extremes)
        
    return feature_vector_bins, feature_vector
        
def feature_vector_gen_interval(fft_freq, fft_values, interest_freqs,
                                apply_SNR = False, include_extremes = True):
    """
    Inputs:
        - fft_freq(np.ndarray): fft frequencies
        - fft_values(np.ndarray): fft values
        - interest_freqs(tuple): contain a tuple with the minimun and maximun values. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - include_extremes(bool): Choose if include extremes or not.
            
    Description:
        Generate a mask with the interval freqs and apply it to fft_values anda fft_freq to generate feature_vector anda feature_vector_bins respectively
        If interest_freqs = None, then that extreme of the interval goes to max or min accordingly
        & is a bitwise AND operation.
        ~ is a bitwise NOT operation.
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
        
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - feature_vector(np.array): feature vector values
        
    """
    #if a value in interest_freq is None, then that extreme of the interval goes to max or min accordingly
    if (interest_freqs[0] is None) and (interest_freqs[1] is None):
        interval_min = fft_freq[0]
        interval_max = fft_freq[-1]
    elif interest_freqs[0] is None:
        interval_min = fft_freq[0]
        interval_max = interest_freqs[1]
    elif interest_freqs[1] is None:
        interval_min = interest_freqs[0]
        interval_max = fft_freq[-1]
    else:
        interval_min = interest_freqs[0]
        interval_max = interest_freqs[1]
        
    if include_extremes:
        mask = (fft_freq >= interval_min) & (fft_freq <= interval_max) #& is a bitwise AND operation, it doesn't raise arror like and operator, see: https://www.geeksforgeeks.org/difference-between-and-and-in-python/
    else:
        mask = (fft_freq > interval_min) & (fft_freq < interval_max)
    
    feature_vector = fft_values[mask]
    feature_vector_bins = fft_freq[mask]
    
    noise_vector = fft_values[~mask] #~ is a bitwise NOT operation, it doesn't raise arror like not operator, see: https://www.geeksforgeeks.org/python-bitwise-operators/
    
    if apply_SNR:
        rate = np.mean(np.array(noise_vector))
    else:
        rate = 1.0
        
    feature_vector = np.array(feature_vector) / rate
    feature_vector_bins = np.array(feature_vector_bins)
    
    return feature_vector_bins, feature_vector

def feature_vector_gen_neighbour(fft_freq, fft_values, interest_freqs, 
                                  include_harmonics = True, apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0,
                                  interest_bw = 1, max_bins = 5):
    """
    Inputs:
        - fft_freq(np.ndarray): fft frequencies
        - fft_values(np.ndarray): fft values
        - interest_freqs(list): contain the interest frequencies on a list 
        - include_harmonics(bool): Boolean to choose if onclude harmonics or not. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - same_bw_forSNR(bool): To choose if use the same bandwidth for the SNR or not.
        - bw_forSNR(float): To choose a particular bandwidth for SNR calc.
        - interest_bw(float): bandwidth
        - max_bins(int): maximum number of bins
                
    Description:
        Calculate the feature vector bins and values
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
        If same_bw_forSNR = False, then you have to determinate the particular bw in bw_forSNR(float).
        If include_harmonics = True, then the harmonics will also be used to generate an interval around the freq and the harmonics.
        Each neighbour will be generated with an especific bandwidth and maximun number of bins. If the bw is too long, and the 
            max_bins too small, maybe not all the bw will be used. And vice versa.
            
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - feature_vector(np.array): feature vector values
        
    """
    N = fft_freq.shape[0]
    feature_vector = []
    feature_vector_bins = []
    noise_vector = []
    
    for particular_freqs in [freqs for freqs in interest_freqs if freqs is not None]: #if the list of interest_freqs has a None value, then there's a tag of no stimuli
        bins_sum = 0
        harm_bins_sum = 0
        for i in range(N):
            fft_bin = fft_freq[i] 
            fft_value = fft_values[i] 
            
            if fft_bin <= (particular_freqs + interest_bw) and fft_bin >= (particular_freqs - interest_bw) and bins_sum < max_bins:
                feature_vector.append(fft_value)
                feature_vector_bins.append(fft_bin)
                bins_sum += 1
            elif fft_bin <= (2*particular_freqs + interest_bw) and fft_bin >= (2*particular_freqs - interest_bw) and harm_bins_sum < max_bins and include_harmonics: 
                feature_vector.append(fft_value)
                feature_vector_bins.append(fft_bin)
                harm_bins_sum += 1
            
            elif apply_SNR:
                if same_bw_forSNR:
                    noise_vector.append(fft_value)
                else:
                    if fft_bin >= (particular_freqs + bw_forSNR) and fft_bin <= (particular_freqs - bw_forSNR):
                        noise_vector.append(fft_value)
              
      #add : 1) boolean to choose how to arrange FV (by bin value or by class)
    
    if apply_SNR:
        rate = np.mean(np.array(noise_vector))
    else:
        rate = 1.0
        
    feature_vector = np.array(feature_vector) / rate
    feature_vector_bins = np.array(feature_vector_bins)
    
    return feature_vector_bins, feature_vector

#--  Feature Matrix Generator

def feature_vector_matrix(signal, sample_rate, labels, startpoint_timestamps, interest_freqs, neighbour_or_interval = "neighbour",
                          stimulation_time = 10, stimulation_time_inseconds = True, stimulation_delay = 0, stimulation_delay_inseconds = True,
                          stride = 10, stride_inseconds = False, filter_window = None,
                          window_size = 512, window_size_inseconds = False, 
                          norm = "basic", max_bins = 5, interest_bw = 1.0, include_harmonics = True, 
                          apply_SNR = False, same_bw_forSNR = True, bw_forSNR = 1.0, 
                          include_extremes = True,
                          plot_average = False):
    """
    Inputs:
        - signal(txt): signal to generate the feature vector
        - sample_rate(int): sample rate to transform from waves/samples waves/sec ???
        - labels(list): labels of the feature vector
        - startpoint_timestamps(list):
        - interest_freqs(list or tuple): contain the interest frequencies on a list or a tuple with the minimun and maximun values. Depending on the case
        - neighbour_or_interval(str): Define if the feature vector is generated usin an interval or neighbour. One of this two options must be choose
        - stimulation_time(int or float): stimulation time to select an interval. It could be in second or samples
        - stimulation_time_inseconds(bool): to specify whether stimulation_time is written in seconds or in samples
        - stimulation_delay(int or float): to select a delay from timestartpoint_timestamps
        - stimulation_delay_inseconds(bool): to specify whether stimulation_delay is written in seconds or in samples
        - stride(int): stride of the window in each step
        - stride_inseconds(bool): to specify whether stride is written in seconds or in samples
        - filter_window(str): To select an especiifc window 
        - window_size(int): window size
        - window_size_inseconds(bool): to specify whether window_size is written in seconds or in samples
        - norm(str): normalization for the fft calculation
        - max_bins(int): maximum number of bins, to configure neighbour method
        - interest_bw(float): bandwidth to configure neighbour method
        - include_harmonics(bool): Boolean to choose if onclude harmonics or not to configure neighbour method. 
        - apply_SNR(bool): Boolean to choose if apply SNR or not. 
        - same_bw_forSNR(bool): To choose if use the same bandwidth for the SNR or not to configure neighbour method.
        - bw_forSNR(float): To choose a particular bandwidth for SNR calc to configure neighbour method.
        - include_extremes(bool): Choose if include extremes or not if neighbour_orinterval = "interval" 
        - plot_average(bool): to plot the feature vector average
        
    Description:
        You load a complete signal, with all the descriptions and it generate a full matrix with all the feature vector.
        If neighbour_or_interval = "neighbour", a list of freq must be given on interest_freqs. This list will be as long as frequencies of interest exist 
        If neighbour_or_interval = "interval", a tuple of minumum and maximum values must be given on interest_freqs. It will be a tuple of size 0 to 2.
        If apply_SNR = True, then feature vector values are divided by the mean of the values of the frequencies that are not included inside the fv.
        Stimulation time could be writte in seconds or samples
    Outputs:
        - feature_vector_bins(np.array): feature vector bins
        - fv_matrix(np.array): matrix full of feature vector
        - label_matrix(np.array): matrix with all the labels.
        
    """
    assert len(labels) == len(startpoint_timestamps), "label list and startpoint_timestamps list must be of equal length"

    #Window size
    if window_size_inseconds:
      N = window_size * sample_rate
    else:
      N = window_size
    
    #Stride size
    if stride_inseconds:
      stride = stride*sample_rate
  
    #Stimulation size
    if stimulation_time_inseconds:
      stim_size = stimulation_time*sample_rate
    else:
      stim_size = stimulation_time
      
    if stimulation_delay_inseconds:
      stim_delay = stimulation_delay*sample_rate
    else:
      stim_delay = stimulation_delay
  
    sample_loss = 0
    fv_matrix = None
    
    #FFT Calculation
    for i in range(len(labels)):
      l = labels[i]
      onehot = np.zeros(len(labels), dtype = float)
      onehot[i] = 1.0
      
      fft_average_matrix  = None
  
      for t in startpoint_timestamps[i]:
        j = 0 #this represents the stride on each particular window
        while True:
          start_sample = t + j + stim_delay
          end_sample = t + j + stim_delay + N
  
          if end_sample > (stim_size+t): #in case stride is not a divisor of the stimulation interval
            if start_sample < (stim_size+t):
              sample_loss += stim_size + t - start_sample #to count the amount of samples that did not get transformed. a method to overcome this loss is to take for that window only end_sample = stim_size and start_sample = stim_size-N
            break
          elif end_sample > (signal.shape[-1]): #in case signal gets to the end and a new interval doesn't fit correctly
            if start_sample < (signal.shape[-1]):
              sample_loss += signal.shape[-1] - start_sample #to count the amount of samples that did not get transformed. a method to overcome this loss is to take for that window only end_sample = stim_size and start_sample = stim_size-N
            break
          else:
            fft_freq, fft_val = sig.fft_calc(signal[start_sample:end_sample], sample_rate, norm, filter_window)
            fv_bins, fv = feature_vector_gen(fft_freq, fft_val, interest_freqs, neighbour_or_interval, include_harmonics, apply_SNR, same_bw_forSNR, bw_forSNR, interest_bw, max_bins, include_extremes)
          
          if fv_matrix is None:
            fv_matrix = np.array([fv])
            label_matrix = np.array([onehot])
                  
          else:
            fv_matrix = np.append(fv_matrix, [fv], axis = 0)
            label_matrix = np.append(label_matrix, [onehot], axis = 0)
  
          if plot_average:
              if fft_average_matrix is None:
                  fft_average_matrix = np.array([fft_val])
              else:
                  fft_average_matrix = np.append(fft_average_matrix, [fft_val], axis = 0)
          
          j += stride
                            
      if plot_average and (fft_average_matrix is not None):
          plt.plot(fft_freq, np.mean(fft_average_matrix, axis = 0))
          plt.legend(["label " + str(lab) for lab in labels if startpoint_timestamps[labels.index(lab)] != []]) 
          #this is called List Comprehension, it helps to create a new list by going through another list as an iterable in a short easy way
          
    print("The selected parameters led to the loss of ", str(sample_loss), " samples.")
  
    #add cheking of the labels vs onehots
  
    return fv_bins, fv_matrix, label_matrix