#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:49:39 2018

@author: Raphael
"""

import os
os.chdir('/Users/Raphael/') #Select your working directory
cwd = os.getcwd()
import pandas as pd
import numpy as np

def transform_data(time_serie, lag_input, predict_from, predict_to):
    
    X = []
    Y = []
    
    for i in range(lag_input, time_serie.shape[0]-(predict_to - predict_from)+1):
        X.append(time_serie[i-lag_input:i])
        Y.append(time_serie[i + predict_from : i + predict_to])
        
    X = np.array(X)
    Y = np.array(Y)
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    Y = Y.reshape((Y.shape[0], predict_to - predict_from))
    return(X, Y)