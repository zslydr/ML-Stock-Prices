#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:49:39 2018

@author: Raphael
"""

#import os
#os.chdir('/Users/Raphael/') #Select your working directory
#cwd = os.getcwd()
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data, wb
import pandas_datareader.data as web
from datetime import datetime, timedelta
import numpy as np


def import_stock_price(stock_name, from_ = 365):
    stock_name = stock_name
    end = datetime.now()
    start = end - timedelta(days = from_)

    stock_prices_serie = web.DataReader(stock_name, 'iex', start, end)

    stock_prices_serie.index = pd.DatetimeIndex(stock_prices_serie.index)
    
    return(stock_prices_serie)

def transform_data(time_serie, lag_input, horizon):
    
    X = []
    Y = []
    
    for i in range(lag_input, time_serie.shape[0] - horizon):
        X.append(time_serie[i-lag_input:i])
        Y.append(time_serie[i + horizon])
        
    X = np.array(X)
    Y = np.array(Y)
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    Y = Y.reshape((Y.shape[0], 1))
    return(X, Y)
    
    
def iterate_model(model, last_n_values, horizon):
    
    Y_pred = []
    #last_n_values = 
    
    for i in range(horizon):
        new_value = regressor.predict(last_n_values)
        Y_pred.append(new_value[0, 0])
        last_n_values = np.append(last_n_values[0, 1:, :], new_value).reshape((1, X_test.shape[1], 1))
        print("année n+"+str(i), new_value[0, 0])
    return(np.array(Y_pred).reshape((horizon, 1)))