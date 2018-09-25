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

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

import importlib
Functions=importlib.import_module("functions")
Functions=importlib.reload(Functions)

#%% Import the data
stock_name = "TOT"

stock_prices_serie = Functions.import_stock_price(stock_name, from_ = 2000)


#stock_prices_serie.drop(["volume"], axis = 1).plot()
#%% Prepare the data

lag = 60

sep_train_test = 700

date_sep = stock_prices_serie.iloc[sep_train_test].name

time_serie_train = stock_prices_serie.close.values.reshape(-1, 1)[:sep_train_test]

time_serie_test = stock_prices_serie.close.values.reshape(-1, 1)[sep_train_test - lag:]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(time_serie_train)
time_serie_train = sc.transform(time_serie_train)
time_serie_test = sc.transform(time_serie_test)

horizon = 0

X_train, Y_train = Functions.transform_data(time_serie_train, lag, horizon)

X_test, Y_test = Functions.transform_data(time_serie_test, lag, horizon)

#%% Prepare the model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout



regressor = Sequential()
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 100,return_sequences = True))
#regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = Y_train.shape[1]))
#regressor.add(Activation("linear"))

regressor.compile(loss="mse", optimizer="rmsprop")
#%% Train the model

regressor.fit(X_train, Y_train, epochs = 10, validation_split=0.5)

#%% Predictions on train and test sets
Y_train_pred = sc.inverse_transform(regressor.predict(X_train))
Y_test_pred = sc.inverse_transform(regressor.predict(X_test))

vide = np.empty((lag + horizon, 1))
vide[:] = np.nan

stock_prices_serie["prediction"] = np.append(vide, np.append(Y_train_pred, Y_test_pred))
#%% Plot of the results

stock_prices_serie.close.plot(label = 'Real Price')
stock_prices_serie.prediction.plot(label = 'Predicted Price')

# stock_prices_serie.iloc[[x for x in range(lag + from_to_pred[1],stock_prices_serie.shape[0]-from_to_pred[1])]].index
plt.title(stock_name + ' Price Prediction with lag ' + str(lag) + "and horizon " + str(horizon))
plt.axvline(x=date_sep, color='k', linestyle='--')
plt.axvline(x=date_sep, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
plt.show()

#%% Prepare the data

lag = 60

sep_train_test = 700

date_sep = stock_prices_serie.iloc[sep_train_test].name

time_serie_train = stock_prices_serie.close.values.reshape(-1, 1)[:sep_train_test]

time_serie_test = stock_prices_serie.close.values.reshape(-1, 1)[sep_train_test - lag:]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(time_serie_train)
time_serie_train = sc.transform(time_serie_train)
time_serie_test = sc.transform(time_serie_test)

horizon = 30

X_train, Y_train = Functions.transform_data(time_serie_train, lag, horizon)

X_test, Y_test = Functions.transform_data(time_serie_test, lag, horizon)

#%% Prepare the model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 100,return_sequences = True))
#regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = Y_train.shape[1]))
#regressor.add(Activation("linear"))

regressor.compile(loss="mse", optimizer="rmsprop")
#%% Train the model

regressor.fit(X_train, Y_train, epochs = 10, validation_split=0.05)

#%% Predictions on train and test sets
Y_train_pred = sc.inverse_transform(regressor.predict(X_train))
Y_test_pred = sc.inverse_transform(regressor.predict(X_test))


vide = np.empty((lag + horizon, 1))
vide[:] = np.nan
#%%
stock_prices_serie["prediction"] = np.append(vide, np.append(Y_train_pred, Y_test_pred))

#%% Plot of the results

stock_prices_serie.close.plot(label = 'Real Price')
stock_prices_serie.prediction.plot(label = 'Predicted Price')

plt.title(stock_name + ' Price Prediction with lag -' + str(lag) + " days and horizon +" + str(horizon) + " days")
plt.axvline(x=date_sep, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
plt.show()

#%%

Y_test_pred = regressor.predict(X_test[10].reshape((1, X_test.shape[1], 1)))

plt.plot(np.append(sc.inverse_transform(Y_train), 
                   sc.inverse_transform(Y_test)), color = 'red', label = 'Real Price')

plt.plot([X_train.shape[0] + 10, X_train.shape[0] + 10 + 50], 
         [sc.inverse_transform(X_test[10])[-1][0], sc.inverse_transform(Y_test_pred[0][0])])
plt.title(stock_name + ' Price Prediction')
plt.axvline(x=sep_train_test, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
plt.show()