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

end = datetime.now()
start = end - timedelta(days=1500)

stock_prices_serie = web.DataReader("GOOGL", 'iex', start, end)

stock_prices_serie.index = pd.DatetimeIndex(stock_prices_serie.index)

stock_prices_serie.drop(["volume"], axis = 1).plot()
#%%

# Variable of interest
var = "close"

time_serie_train = stock_prices_serie.close.values.reshape(-1, 1)[:700]

time_serie_test = stock_prices_serie.close.values.reshape(-1, 1)[700:]

#%% Scaler

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(time_serie_train)
time_serie_train = sc.transform(time_serie_train)
time_serie_test = sc.transform(time_serie_test)

#%%
X_train, Y_train = Functions.transform_data(time_serie_train, 60, 1)

X_test, Y_test = Functions.transform_data(time_serie_test, 60, 1)

#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#%% Train the model

regressor.fit(X_train, Y_train, epochs = 20, validation_split=0.5)


#%% Prediction and plot predictions

Y_pred = regressor.predict(X_test)

plt.plot(np.append(sc.inverse_transform(Y_train), sc.inverse_transform(Y_pred)), color = 'blue', label = 'Predicted Price')
plt.plot(np.append(sc.inverse_transform(Y_train), sc.inverse_transform(Y_test)), color = 'red', label = 'Real Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



