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
os.chdir('/Users/Raphael/') #Select your working directory
cwd = os.getcwd()
import matplotlib.pyplot as plt

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

#%% Import the data

end = datetime.now()
start = end - timedelta(days=900)

stock_prices_serie = web.DataReader("GOOGL", 'iex', start, end)

stock_prices_serie.index = pd.DatetimeIndex(stock_prices_serie.index)

stock_prices_serie.drop(["volume"], axis = 1).plot()
#%%

# Variable of interest
var = "close"

time_serie = stock_prices_serie.close.values.reshape(-1, 1)

#%% Scaler

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
time_serie = sc.fit_transform(time_serie)

#%% Construction du data set sous forme "matricielle"

# Variable de lag
lag_input = 60

# Target variable:
target_day = 1

X = []
Y = []

for i in range(60, time_serie.shape[0]):
    X.append(time_serie[i-60:i])
    Y.append(time_serie[i])
    
X = np.array(X)
Y = np.array(Y)

X = X.reshape((X.shape[0], X.shape[1], 1))

#data_df.reshape((622,61))



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
#%%
regressor.fit(X, Y, epochs = 10, validation_split=0.5)
#%%
Y_pred = regressor.predict(X)
Y_pred
#%%
plt.figure
plt.plot(sc.inverse_transform(Y_pred).reshape(Y_pred.shape[0]), color = 'blue', label = 'Predicted Price')
plt.plot(stock_prices_serie.close.values[60:], color = 'red', label = 'Real Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()