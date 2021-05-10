#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:54:08 2021

@author: gastoncavallo
"""
#-- Libraries Import
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
from scipy.optimize import minimize
import pandas as pd
import tkinter as tk
from tkinter import filedialog as tkFileDialog
import os
from feature_processing import feature
from feature_processing import classifier

#--- En este módulo se encuentra todo aquello relacionado a la carga, 
#    generacion del dataframe, procesamiento de la señal y cálculo de la FFT. Laplaciano

#--- File path and directory

def get_file(initial_dir = "/", window_title = "Select a File"):
    """
    Inputs:
        - initial_dir(str): initial directory for function
        - window_title(str): window title
       
    Description:
        Search a File or a Directory and return the file name, file directory and the path
        
    Outputs:
        - file_name(str): file name
        - file_dir(str): file directory
        - file_path(str): file path
            
    """
    root = tk.Tk()
  
    file_path = tkFileDialog.askopenfilename(initialdir = initial_dir, title = window_title, filetypes=(("Documento de texto", "*.txt"), ("Todos los archivos", "*.*")))
    file_dir , file_name = os.path.split(file_path)
    
    root.destroy()
    root.mainloop()
  
    return file_name, file_dir, file_path

def get_dir(initial_dir = "/", window_title = "Choose a Directory"):
    """
    Inputs:
        - initial_dir(str): initial directory for function
        - window_title(str): window title
       
    Description:
        Search a Directory and return it
        
    Outputs:
        - dir_path(str): Directory path
        
    """
    root = tk.Tk()
  
    dir_path = tkFileDialog.askdirectory(initialdir = initial_dir, title = window_title)
    
    root.destroy()
    root.mainloop()
  
    return dir_path

#--- Data Loading

def dataframe_creation(file_path, index_col = 0, sep = ",", header = None, skiprows= 10, 
                       names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                       has_tag_column = True, tag_column_name = 'mlp_labels', tag_column_index = 7):
    """
    Inputs:
        - file_path(str): file path
        - index_col(int): index column
        - sep(str): typo of separator you use
        - header(int or None): Defines if exist or not a row wich containg column names 
        - skiprows(int): number of rows to skip
        - names(list): columns name
        - has_tag_column(bool): if has tag column
        - tag_column_name(str): name of tag column
        - tag_column_index(int): index of tag column
       
    Description:
        Combine get_file or get_dir depending on the case and load a file with the inpust parameters
        Generate the Dataframe and returns it
    Outputs:
        - pd_df(pd): pandas Dataframe
        - tag_column_name(str): name of tag column
         
    """  
    if tag_column_name is not None:
      temp_tag_column_name = names[tag_column_index]
      if (temp_tag_column_name != tag_column_name):
        print("tag_column_name and tag_column_index do not coincide. Force index [Y/N]?")
        ans = input("If N, then tag_column will be chosen by tag_column_name: ")
        if (ans != "Y"):
          tag_column_name = temp_tag_column_name
    
    pd_df = pd.read_csv(file_path, sep = sep, header = header, index_col = index_col, names = names, skiprows = skiprows)
  
    return pd_df, tag_column_name

# Full Dataframe Searching and Loading

def build_dataframe(index_col = 0, sep = ",", header = None, skiprows= 10, 
                    names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                    has_tag_column = True, tag_column_name = 'mlp_labels', tag_column_index = 7, initial_dir = "/"):
    """
    Inputs:
        - index_col(int): index column
        - sep(str): typo of separator you use
        - header(int or None): Defines if exist or not a row wich containg column names 
        - skiprows(int): number of rows to skip
        - names(list): columns name
        - has_tag_column(bool): if has tag column
        - tag_column_name(str): name of tag column
        - tag_column_index(int): index of tag column
        - initial_dir(str): initial directory to get_file
       
    Description:
        Join get_file with dataframe_creation for full Dataframe Searching and Loading
       
    Outputs:
        - pd_df(pd): pandas Dataframe
        - pd_tag_column(pd): pandas tag column
        - pd_name(str): pandas name
        - pd_dir(str): pandas directory
        - pd_path(str): pandas path
    
    """   
    pd_name, pd_dir, pd_path = get_file(initial_dir)
    
    pd_df, pd_tag_column = dataframe_creation(pd_path, index_col, sep, header, skiprows, 
                                              names, has_tag_column, tag_column_name, tag_column_index)
    
    return pd_df, pd_tag_column, pd_name, pd_dir, pd_path

# Many DataFrame Searching and Loading from Dir

def build_dataframe_fromdir(index_col = 0, sep = ",", header = None, skiprows= 10, 
                            names= ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'x', 'y', 'z', 'mlp_labels', 'time', 'rand'],
                            has_tag_column = True, tag_column_name = 'mlp_labels', tag_column_index = 7, 
                            initial_dir = "/", return_dict = False, return_joined = True):
    """
    Inputs:
        - index_col(int): index column
        - sep(str): typo of separator you use
        - header(int or None): Defines if exist or not a row wich containg column names 
        - skiprows(int): number of rows to skip
        - names(list): columns name
        - has_tag_column(bool): if has tag column
        - tag_column_name(str): name of tag column
        - tag_column_index(int): index of tag column
        - initial_dir(str): initial directory to get_file
        - return_dict(bool): boolean to choose if return or not 
        - return_joined(bool): boolean to choose
            
    Description:
        Generate a single dataframe with many signals at the same time using get_dir and dataframe_creation
        Analyze all .txt or .csv files into the directory and loads them in a single Dataframe
        return_dict and return_joined must be opposite 
        If return_dict, then returns a dictionary that in each key has the name of the file and in each value has the dataframe 
        If return_join, then returns a single Dataframe with everything attached. And tag_column_names only returns de first tag column name
        
    
    Outputs:
        - signal_dict(dict): signal dictionary 
        - tag_column_names(list): tag column names or name depending on the case
        - full_pd(pd): full pandas Dataframe
        
        
    """    
    assert return_dict != return_joined, "return_dict and return_joined must be opposite since both or none of them can be return"
    
    files_dir = get_dir(initial_dir = initial_dir)
    
    signal_dict = {}
    tag_column_names =[]
    signal_names = []

    for file in os.listdir(files_dir):
        if ".txt" in file or ".csv" in file:
            file_path = os.path.join(files_dir, file)
            print("Loaded Signal: " + file_path)
            file_name = file.split("-")[0]
            
            signal_pd, tag_column = dataframe_creation(file_path = file_path)
            
            signal_names.append(file_name)
            tag_column_names.append(tag_column)
            
            signal_dict[file_name] = signal_pd
    
    full_pd = pd.concat([element for element in signal_dict.values()],
                        ignore_index = False
                        )
    
    if return_dict:
        return signal_dict, tag_column_names
    
    elif return_joined:
        return full_pd, tag_column_names[0]
    
#---Deleting the points that are not useful for training

def signal_cleaning (sgn_pd, label_col_name='mlp_labels', labels=[1, 2, 3, 4], #labels to keep
                    just_np = False): #wether to actually do the cleaning or just redo the index and export Numpy Array
    """
    Inputs:
        - sgn_pd(pd): pandas Dataframe
        - label_col_name(str): name of the label column
        - labels(list): list with the labels to conserve
        - just_np(bool): boolean to choose if only transform the signal into a np.array
       
    Description:
        Deletes the points that are not useful for training, and transform the signal into a numpy.array
        In the chosen pandas, only keep the chosen labels 
        If just_np, then the signal is not cleaned, just transformed into a numpy.array
    Outputs:
        - signal(numpy.array): numpy signal
        - to_signal(pd): cleaned pandas Dataframe
        
    """    
    if (not just_np):
      df_calibration = sgn_pd
      
      sgn_pd = df_calibration.loc[df_calibration[label_col_name].isin(labels)]
    
    #Rebuilding the index tag, since many values were dropped in the function above
    sgn_pd.index = range(sgn_pd.shape[0]) 
    
    #Drop the colmns and generate the signal np array, transposing it so that it's easier to work later
    to_signal = sgn_pd.copy()
    signal = np.asarray(to_signal.drop(columns = ["x", "y", "z", "mlp_labels", "time", "rand"]))
    signal = np.transpose(signal)
    
    return signal, to_signal

#--- Preprocessing

def preprocess(signal, fs=200, f_notch=50, Q_notch=30, bp_order=5, bp_type='butter', bp_flo=15, bp_fhi=80,
               ch_on_column = True):
    """
    Inputs:
        - signal(np.array): numpy array signal to preprocess
        - fs(int): sample frequency
        - f_notch(float): notch filter frequency 
        - Q_notch(float): Quality factor that characterizes the notch filteer
        - bp_order(int): bandpass filter order
        - bp_type(str): type of bandpass filter
        - bp_flo(float): bandpass low frequency
        - bp_fhi(float): bandpass high frequency
       
    Description:
        Apply filters to the signal such as, notch and bp and returns the fitered signal
       
    Outputs:
        - filt_signal(np.array): filtered signal
        
    """
    fn = fs/2
  
    #Detrend
    signal = sgn.detrend(signal, type='constant')
  
    #Notch Filter
    b, a = sgn.iirnotch(f_notch, Q_notch, fs)
    signal = sgn.filtfilt(b, a, signal)
  
    #Band-pass Filter
    b, a = sgn.iirfilter(bp_order, [bp_flo/fn, bp_fhi/fn], ftype=bp_type)
    filt_signal = sgn.filtfilt(b, a, signal)
    
    if ch_on_column:
        filt_signal = np.transpose(filt_signal)
        
  
    return filt_signal

#--- Timestamps Location

def timestamps_loc(dataframe, label_column_name = 'mlp_labels', labels = [1, 2, 3, 4], 
                   export_dict = True, export_list = True):
    """
    Inputs:
        - dataframe(pd): pandas Dataframe
        - label_column_name(str): name of the label column
        - labels(list): list with the labels 
        - export_dict(bool): boolean to choose if export as a dictionary 
        - export_list(bool): boolean to choose if export as a list of lists
       
    Description:
        
        If export_dict, then returns a dictionary where the keys will be the label and the value will be a list with the startpoint_timestamp of each label
        If export_list, then returns a list of lists in order
        
    Outputs:
        - timestamps_dict(dict): timestamps dictionary
        - timestamps_list(list): timestamps list of lists
        
    """
    #WARNIGN - ERROR SI LOS DOS EXPORT SON FALSE
    timestamps_dict = {}
    for l in labels:
        timestamps_dict["Label " + str(l)] = []

    prev_tag = None
    for i in dataframe.index:
        tag = dataframe[label_column_name][i]
        for l in labels:
            if l == tag and tag != prev_tag:
                timestamps_dict["Label " + str(l)].append(i)
        prev_tag = tag
    
    # timestamps_list = []
    # for k in timestamps_dict:
    #     timestamps_list.append(timestamps_dict[k])
    #better method:
    
    timestamps_list = [element for element in timestamps_dict.values()]
    
    if export_dict and export_list:
        return timestamps_dict, timestamps_list
    
    elif export_dict:
        return timestamps_dict
    
    elif export_list:
        return timestamps_list
    
#-- FFT Calculation

def fft_calc(windowed_signal, sample_rate, norm = "basic", filter_window = None):
    """
    Inputs:
        - windowed_signal(np.array): windowed signal
        - sample_rate(int): sample frequency
        - norm(str): type of normalization to use
        - filter_window(str or np.ndarray): filter window to apply 
       
    Description:
        Function that calculates the fft using numpy
        If filter_window = np.array, then filter_window.shape must be the same as N (windowed_signal.shape)
       
    Outputs:
        - fft_freq(numpy.ndarray): fft frequencies
        - fft_values(numpy.ndarray): fft values
        
    """
    N = int(windowed_signal.shape[0])
    
    if filter_window == None:
      windowed_signal = windowed_signal
    elif type(filter_window) == str:
      filter_window_fromscipy = sgn.get_window(filter_window, N)
      windowed_signal = windowed_signal*filter_window_fromscipy
    elif type(filter_window) == np.ndarray:
      assert filter_window.shape[0] == N
      windowed_signal = windowed_signal*filter_window 
    else:
      raise ValueError    
    
    if norm == "ortho":
      norm_factor = 1/np.sqrt(N)
    elif norm == "basic":
      norm_factor = 2/N #https://www.mathworks.com/help/matlab/ref/fft.html actually the extremes (DC and Nyquist) should only be multiplied by 2
    else:
      norm_factor = 1 #replace by RaiseError
  
    fft_values = abs(np.fft.rfft(windowed_signal)*norm_factor)
    fft_freq = np.fft.rfftfreq(n = N, d = 1/sample_rate)
  
    return fft_freq, fft_values

# --- Laplacian Optimization --

# Bins Finder

def find_freq_bins(array, f, find_alternative = True):
    """
    Inputs:
        - array(np.arrat): array of frequencieese
        - f(float): requested frequency
        - find_alternative(bool): boolean to choose if find and alternative frequency in case that the requested frequency does not exist in your frequencies array
       
    Description:
        Looks for frequency bins and returns which sub-index it is in
        If find_alternative then when the requested frequency does not exist in your frequencies array, the algorithm will returns the sub-index with the closest frequency
        Example:
            >>> array = [21.5, 21.9, 22.3, 22.7, 23.1]
            >>> f = 22
            >>> find_alternative = True
            i = 1
            
            
    Outputs:
        - i(int): sub-index where finds the requested frequency
        - i+1(int): sub-index where finds the requested frequency
            
        
    """
    for i in range(len(array)-1):
        low_d = f - array[i]
        upp_d = array[i+1] - f

        if upp_d >= 0:
            if low_d == 0 or low_d < upp_d:
                return i
    
            elif low_d > upp_d:
                return i+1


# Objective Function

def objective_f(fft, fft_bins, freq, bw_bins = 2, bw_neighbour = 2):
    """
    Inputs:
        - fft(np.array): fft
        - fft_bins(np.array): frequency bins
        - freq(float): frequency of interest
        - bw_bins(int): bandwidht that will be used to calculate the main of interest frequencies 
        - bw_neighbour(int or str): bandwidht that will be used to calculate the main of non-interest frequencies 
       
    Description:
        Takes, the fft, fft_bins and freq and calculates an objective function where result = (low_neighbour_average + upp_neighbour_average)/2 - main_freq_average
        main_freq_aveerage is an average of the interest frequencies taken from freq - bw_bins to freq + bw_bins
        low_neighbour_average is an average of non-interest frequencies taken:
            - from 0 to freq - bw_bins if bw_neighbour = "all"
            - from freq - bw_neighbour - bw_bins to freq - bw_neighbour + bw_bins if bw_neighbour is an int
        upp_neighbour_average is an average of non-interest frequencies taken:
            - from freq + bw_bins to the highest frequency if bw_neighbour = "all"
            - from freq + bw_neighbour - bw_bins to freq + bw_neighbour + bw_bins if bw_neighbour is an int
        bw_bins would be in frequency samples, not values
        bw_neighbour would be in frequency values, not samples
        
    Outputs:
        - result(float): resulting value of calculating the objective function
        
    """    
    #bw_bins would be in frequency samples, not values
    low_bw_bins = abs(int(bw_bins))
    high_bw_bins = low_bw_bins + 1 #Because of intervals being [inclusive, exclusive]
    #bw_neighbour would be in frequency values, not samples
    
    main_freq_bin = find_freq_bins(fft_bins, freq, True)
    main_freq_average = np.mean(fft[main_freq_bin-low_bw_bins:main_freq_bin+high_bw_bins])
    
    if bw_neighbour == "all":
        low_neighbour_average = np.mean(fft[0:main_freq_bin - bw_bins])
        upp_neighbour_average = np.mean(fft[main_freq_bin + bw_bins: -1])        
    else:
        low_neighbour_bin = find_freq_bins(fft_bins, freq - bw_neighbour, True)
        upp_neighbour_bin = find_freq_bins(fft_bins, freq + bw_neighbour, True)
        low_neighbour_average = np.mean(fft[low_neighbour_bin-low_bw_bins:low_neighbour_bin+high_bw_bins])
        upp_neighbour_average = np.mean(fft[upp_neighbour_bin-low_bw_bins:upp_neighbour_bin+high_bw_bins])
    
    #Objective function calculation
    result = (low_neighbour_average + upp_neighbour_average)/2 - main_freq_average

    return result

# Laplacian Application with Objective Function

def laplacian_app(x, *argf):
    """
    Inputs:
        - x(list): list of weights to apply the laplacian calculation
        - argf(list of positional arguments):
            - (np.array): signal to apply the laplacian 
            - (int or float): frequency of interest to calculate the objective function
            - (int or float): smapling frequency of signal
            - (bool): wether to plot results (progress) or not
            - (int or float): bw_bins for the objective function
            - (int float or str): bw_neighbour for the objectivee function
            
    Description:
        Apply the laplacian, spacial filtering or combination, and calculate the objective function (see objective_f)
       
    Outputs:
        - objective_result(float): objective function result
        
    """
    w = np.array(x)
    func = argf[0]
    freq = argf[1]
    sample_freq = argf[2]
    plot_bool = argf[3]
    bw_bins = argf[4]
    bw_neighbour = argf[5]
    
    func2 = func.copy()
    #fft_bins = np.fft.rfftfreq(func.shape[1], d=0.001) #verification
    #fft_vals = np.zeros((w.shape[0], fft_bins.shape[0]))

    for i in range(func.shape[0]):
        func2[i] = w[i]*func[i]
        #fft_vals[i] = abs(np.fft.rfft(func2[i])) #verification

        #plt.subplot(121)
        #plt.plot(func2[i])
        #plt.xlim(0,100)                       #verification

        #plt.subplot(122)
        #plt.plot(fft_bins, fft_vals[i])       #verification
        #plt.xlim(0, 20)
  
    sum_func = np.sum(func2, axis = 0)
    fft_bins = np.fft.rfftfreq(sum_func.shape[0], d=1/sample_freq)
    fft_vals = abs(np.fft.rfft(sum_func))
  
    if plot_bool:
        plt.figure()#(figsize = (15,5))
        plt.title("Plot for Optimization for Freq: " + str(freq) + " Hz")
        plt.plot(fft_bins, fft_vals)
        plt.xlim(0, 2*freq+1) #so as to catch the harmonics

    objective_result = objective_f(fft_vals, fft_bins, freq, bw_bins, bw_neighbour)

    return objective_result

# Optimization
def optimization_lap(freq, signal, sample_freq = 200, initial_values = [0.25, 0.25, 0.25, 0.25], 
                     bw_bins = 2, bw_neighbour = 2, max_iter = 50, method = "L-BFGS-B", bounds = 5,
                     plot_progress = True):
    """
    Inputs:
        - freq(int or float): frequency of interest
        - signal(np.array): signal of interest to optimize laplacian for
        - sample_freq(int or float): smapling frequency of signal
        - initial_values(list): initial values to start the laplacian from
        - bw_bins(int or float): bw_bins for the objective function
        - bw_neighbour(int float or str): bw_neighbour for the objectivee function
        - max_iter(int): maximun amount of iterations to allow the optimization to work on
        - method(str): mothod to work the optimization with (see scipy.optimize.minimize)
        - bounds(int or floats): bouderies for the values of the weights
        - plot_progress(bool): wether to plot progress or not
       
    Description:
       Optimize the laplacian values
       Takes a set of weights, apply the laplacian, calculates the objective function and minimize its results through changing the weights values
       The optimization is done through the chosen method
       
    Outputs:
        - results(list): final weights obtained after apply the optimization
        
    """    
    assert type(signal) == np.ndarray
    assert len(initial_values) == signal.shape[0]
    
    bound_list = []
    for i in range(len(initial_values)):
        bound_list.append((-bounds, +bounds))
        
    laplacian_app(initial_values, signal, freq, sample_freq, plot_progress, bw_bins, bw_neighbour)

    results = minimize(laplacian_app, x0 = initial_values, 
                   args=(signal, freq, sample_freq, False, bw_bins, bw_neighbour), 
                   method = "L-BFGS-B",
                   bounds = bound_list,
                   options = {'maxiter': max_iter,
                              'disp': True
                              }
                   )

    laplacian_app(results.x, signal, freq, sample_freq, plot_progress, bw_bins, bw_neighbour)

    return results