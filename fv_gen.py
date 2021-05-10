#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:32:56 2021

@author: Gaston Cavallo
"""

import numpy as np
import json 
from feature_processing import signal as sig

sf = 200
labels = [1,2]

bp_lo = 1
bp_hi = 70
notch = 50

#Training Dataset - JSON Generator

signal_pd, tag_column = sig.build_dataframe_fromdir(initial_dir="/Users/gastoncavallo/Desktop/Facultad/PI/Dataset/Otros/Prueba")

signal_np, signal_pd_tonp = sig.signal_cleaning(signal_pd, labels = labels)

signal = sig.preprocess(signal_np, bp_flo = bp_lo, bp_fhi = bp_hi, f_notch = notch, ch_on_column = True)

timestamps_dict, timestamps_list = sig.timestamps_loc(signal_pd_tonp, tag_column, labels = labels)

