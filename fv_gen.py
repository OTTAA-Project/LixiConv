#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:32:56 2021

@author: Gaston Cavallo
"""

import numpy as np
import time
import json 
import pandas as pd
from feature_processing import signal as sig


sf = 200
labels = [1,2]

bp_lo = 1
bp_hi = 70
notch = 50

stride = 6
wz = 512

#Training Dataset - JSON Generator

# signal_pd, tag_column = sig.build_dataframe_fromdir(initial_dir="/Users/gastoncavallo/Desktop/Facultad/PI/Dataset/HC 23 Hz Dataset/Training") #JL 23 Hz Dataset hacer tambien

# signal_np, signal_pd_tonp = sig.signal_cleaning(signal_pd, labels = labels)

# signal = sig.preprocess(signal_np, bp_flo = bp_lo, bp_fhi = bp_hi, f_notch = notch, ch_on_column = True) #If ch_on_column = True, chanels will be as columns and time as rows. e.g (600,4)

# timestamps_dict, timestamps_list = sig.timestamps_loc(signal_pd_tonp, tag_column, labels = labels) #Generate dict and list with timestamps

# start = time.time()
# signal_fv = None
# for ti, la in zip(timestamps_list, labels):
#     for t in ti:
#         j = 0
#         while j+wz < 6*sf:
#             if signal_fv is None:
#                 signal_fv = signal[np.newaxis, t+j:t+j+wz, :]
#                 if la == 1:
#                     signal_fv_labels = np.array([1, 0])[np.newaxis, :]
#                 else:
#                     signal_fv_labels = np.array([0, 1])[np.newaxis, :]
#             else:
#                 signal_fv = np.append(signal_fv, signal[np.newaxis, t+j:t+j+wz, :], axis = 0)
#                 if la == 1:
#                     signal_fv_labels = np.append(signal_fv_labels, np.array([1, 0])[np.newaxis, :], axis = 0)
#                 else:
#                     signal_fv_labels = np.append(signal_fv_labels, np.array([0, 1])[np.newaxis, :], axis = 0)
#             j += stride
# #signal_fv dimension = (N recortes, window size, chanels)
# print ("El tiempo que demora en generar el fv es: " + str(time.time()-start) + " segundos.") #time it takes to generate the fv

# signal_fv_dict = {}
# for lab in np.unique(np.argmax(signal_fv_labels, axis = 1)):
#     first = np.where(np.argmax(signal_fv_labels, axis = 1) == lab)[0][0]
#     last = np.where(np.argmax(signal_fv_labels, axis = 1) == lab)[0][-1] + 1
#     signal_fv_dict[str(lab)] = signal_fv[first : last].tolist()
    
#with open("/Users/gastoncavallo/Desktop/Facultad/PI/JSON/TrainingsetJL.json", "w") as output:
#    json.dump(signal_fv_dict, output)
   
#Training Dataset - NPY Generator

signal_pd, tag_column = sig.build_dataframe_fromdir(initial_dir="/Users/gastoncavallo/Desktop/Facultad/PI/Dataset/HC 23 Hz Dataset/Training") #JL 23 Hz Dataset hacer tambien

signal_np, signal_pd_tonp = sig.signal_cleaning(signal_pd, labels = labels)

signal = sig.preprocess(signal_np, bp_flo = bp_lo, bp_fhi = bp_hi, f_notch = notch, ch_on_column = True) #If ch_on_column = True, chanels will be as columns and time as rows. e.g (600,4)

timestamps_dict, timestamps_list = sig.timestamps_loc(signal_pd_tonp, tag_column, labels = labels) #Generate dict and list with timestamps

start = time.time()
signal_fv = None
for ti, la in zip(timestamps_list, labels):
    for t in ti:
        j = 0
        while j+wz < 6*sf:
            if signal_fv is None:
                signal_fv = signal[np.newaxis, t+j:t+j+wz, :]
                if la == 1:
                    signal_fv_labels = np.array([1, 0])[np.newaxis, :]
                else:
                    signal_fv_labels = np.array([0, 1])[np.newaxis, :]
            else:
                signal_fv = np.append(signal_fv, signal[np.newaxis, t+j:t+j+wz, :], axis = 0)
                if la == 1:
                    signal_fv_labels = np.append(signal_fv_labels, np.array([1, 0])[np.newaxis, :], axis = 0)
                else:
                    signal_fv_labels = np.append(signal_fv_labels, np.array([0, 1])[np.newaxis, :], axis = 0)
            j += stride
            
# signal_fv dimension = (N recortes, window size, chanels)

np.save("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_HC", signal_fv)
np.save("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_labels_HC", signal_fv_labels)

#Time test
# start = time.time()
# with open("/Users/gastoncavallo/Desktop/Facultad/PI/JSON/TrainingsetHC.json", "w") as file:
#     json.dump(signal_fv_dict, file)
# print("JSON took", int(time.time()-start))

start = time.time()
np.save("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_HC", signal_fv)
np.save("/Users/gastoncavallo/Desktop/Facultad/PI/np_dataset/signal_fv_labels_HC", signal_fv_labels)
# print("Numpy took", int(time.time()-start))

# #Training dataset for AutoML

# signal_fv = None
# for ti, la in zip(timestamps_list, labels):
#     for t in ti:
#         j = 0
#         while j+wz < 6*sf:
#             if signal_fv is None:
#                 signal_fv = np.append(signal[t+j:t+j+wz, :].T, np.array([[1],[2],[3],[4]]), axis = 1)
#                 if la == 1:
#                     signal_fv_labels = np.array([[1, 0],[1, 0],[1, 0],[1, 0]])
#                 else:
#                     signal_fv_labels = np.array([[0, 1],[0, 1],[0, 1],[0, 1]])
#             else:
#                 signal_fv = np.append(signal_fv, np.append(signal[t+j:t+j+wz, :].T, np.array([[1],[2],[3],[4]]), axis = 1), axis = 0)
#                 if la == 1:
#                     signal_fv_labels = np.append(signal_fv_labels, np.array([[1, 0],[1, 0],[1, 0],[1, 0]]), axis = 0)
#                 else:
#                     signal_fv_labels = np.append(signal_fv_labels, np.array([[0, 1],[0, 1],[0, 1],[0, 1]]), axis = 0)
#             j += stride

# features_pd = pd.DataFrame(signal_fv)
# labels_pd = pd.DataFrame(signal_fv_labels)

# print()

# features_pd.to_csv("/Users/gastoncavallo/Desktop/Facultad/PI/CSV_AutoML/FeaturesAutoML", header = False)
# labels_pd.to_csv("/Users/gastoncavallo/Desktop/Facultad/PI/CSV_AutoML/LabelsAutoML", header = False)
