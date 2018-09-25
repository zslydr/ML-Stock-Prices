#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:37:25 2018

@author: Raphael
"""

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data, wb
import os
os.chdir('/Users/Raphael/Github/ML-Stock-Prices/') #Select your working directory
cwd = os.getcwd()
import matplotlib.pyplot as plt

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

import importlib
Functions=importlib.import_module("functions")
Functions=importlib.reload(Functions)

#%% Import the data
stock_name = "AMZN"
end = datetime.now()
start = end - timedelta(days=2000)

stock_prices_serie = web.DataReader(stock_name, 'iex', start, end)

stock_prices_serie.index = pd.DatetimeIndex(stock_prices_serie.index)

stock_prices_serie.drop(["volume"], axis = 1).plot()
#%%
# Variable of interest
var = "close"

time_serie_train = stock_prices_serie.close.values.reshape(-1, 1)[:1200]

time_serie_test = stock_prices_serie.close.values.reshape(-1, 1)[1200:]

#%% Scaler

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(time_serie_train)
time_serie_train = sc.transform(time_serie_train)
time_serie_test = sc.transform(time_serie_test)

#%%
X_train, Y_train = Functions.transform_data(time_serie_train, 60, 1, 50)

X_test, Y_test = Functions.transform_data(time_serie_test, 60, 1, 50)

#%%
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout



regressor = Sequential()
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = Y_train.shape[1]))
#regressor.add(Activation("linear"))

regressor.compile(loss="mse", optimizer="rmsprop")
#%% Train the model

regressor.fit(X_train, Y_train, epochs = 20, validation_split=0.2)

#%%
Y_train_pred = regressor.predict(X_train)

plt.plot(np.cumsum(Y_train_pred), color = 'blue', label = 'Predicted Price')
plt.plot(np.cumsum(Y_train), color = 'red', label = 'Real Price')
plt.title(stock_name + ' Price Prediction')
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
plt.show()
#%% Prediction and plot predictions

Y_pred = regressor.predict(X_test)
#%%
plt.plot(np.append(sc.inverse_transform(Y_train), sc.inverse_transform(Y_pred)), color = 'blue', label = 'Predicted Price')
plt.plot(np.append(sc.inverse_transform(Y_train), sc.inverse_transform(Y_test)), color = 'red', label = 'Real Price')
plt.title(stock_name + ' Price Prediction')
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
plt.show()

#%%

def iterate_model(model, last_n_values, horizon):
    
    Y_pred = []
    #last_n_values = 
    
    for i in range(horizon):
        new_value = regressor.predict(last_n_values)
        Y_pred.append(new_value[0, 0])
        last_n_values = np.append(last_n_values[0, 1:, :], new_value).reshape((1, X_test.shape[1], 1))
        print("année n+"+str(i), new_value[0, 0])
    return(np.array(Y_pred).reshape((horizon, 1)))

#%%


Y_pred = iterate_model(regressor, 
                       X_test[0, :, :].reshape((1, X_test.shape[1], 1)), 
                       X_test.shape[0])


#%%

Y_test.shape


def Y_to_time_series(Y):
    None

#np.zeros(Y_train.shape[0])
res = []
for i in range(Y_pred.shape[0]//4):
    res.extend(list(Y_pred[i]))


