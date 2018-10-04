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
#import pandas_datareader.data as web
import numpy as np
from datetime import timedelta

import importlib
import lib.functions as Functions
#Functions=importlib.import_module("/test/functions")
Functions=importlib.reload(Functions)

#%% Import the data
stock_name = "TWTR"

stock_prices_serie = Functions.import_stock_price(stock_name, from_ = 2190)

#%%
plt.figure(num=None, figsize=(8 * 5, 6 * 5), dpi=80, facecolor='w', edgecolor='k')
stock_prices_serie.close.plot()

plt.title("Closing price of " + stock_name)
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.grid()
#plt.legend()
plt.savefig(stock_name + '_closing_price.png')


#%% Prepare the data

lag = 60

sep_train_test = 1000

from_ = 1
horizon = 1

date_sep = stock_prices_serie.iloc[sep_train_test].name

time_serie_train = stock_prices_serie.close.values.reshape(-1, 1)[:sep_train_test]

time_serie_test = stock_prices_serie.close.values.reshape(-1, 1)[sep_train_test - lag - horizon:]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(time_serie_train)
time_serie_train = sc.transform(time_serie_train)
time_serie_test = sc.transform(time_serie_test)


X_train, Y_train = Functions.transform_data(time_serie_train, lag,
                                            from_ = from_,
                                            horizon = horizon)

X_test, Y_test = Functions.transform_data(time_serie_test, lag, 
                                          from_ = from_, 
                                          horizon = horizon)

#%% Prepare the model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience = 2)



regressor = Sequential()
regressor.add(LSTM(units = 100,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], X_train.shape[2])))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences = False))
regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 100,return_sequences = True))
#regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 50))
#regressor.add(Dropout(0.2))

regressor.add(Dense(units = Y_train.shape[1]))
#regressor.add(Activation("linear"))

regressor.compile(loss="mse", optimizer="rmsprop")
#%% Train the model

regressor.fit(X_train, Y_train, epochs = 20, 
              validation_split=0.05, callbacks = [early_stopping_monitor])

#%% Predictions on train and test sets
Y_train_pred = sc.inverse_transform(regressor.predict(X_train))
Y_test_pred = sc.inverse_transform(regressor.predict(X_test))

vide = np.empty((lag + horizon, 1))
vide[:] = np.nan

stock_prices_serie["prediction"] = np.append(vide, np.append(Y_train_pred, Y_test_pred))

#%%
plt.figure(num=None, figsize=(8 * 3, 6 * 3), dpi=100, facecolor='w', edgecolor='k')

stock_prices_serie.close.plot(label = 'Real Price', color = "lightgreen")
stock_prices_serie.prediction.plot(label = 'Predicted Price', color = "orange")

plt.title(stock_name + ' Price Prediction with lag ' + str(lag) + " and horizon " + str(horizon))
plt.axvline(x=date_sep, color='k', linestyle='--')
plt.text(date_sep - timedelta(days = 90),70,'Training separator',rotation=90)
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
#plt.show()
plt.grid()
plt.savefig(stock_name + '_lag60_hor_1.png')

#%%

regressor.evaluate(np.append(X_train, X_test, axis = 0), np.append(Y_train, Y_test, axis = 0))
#%% Prepare the data

lag = 90

sep_train_test = 1000

from_ = 1

horizon = 30

date_sep = stock_prices_serie.iloc[sep_train_test].name

time_serie_train = stock_prices_serie.close.values.reshape(-1, 1)[:sep_train_test]
time_serie_test = stock_prices_serie.close.values.reshape(-1, 1)[sep_train_test - lag - horizon:]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
sc.fit(time_serie_train)
time_serie_train = sc.transform(time_serie_train)
time_serie_test = sc.transform(time_serie_test)


X_train, Y_train = Functions.transform_data(time_serie_train, lag, 
                                            from_ = from_,
                                            horizon = horizon)

X_test, Y_test = Functions.transform_data(time_serie_test, lag, 
                                          from_ = from_,
                                          horizon = horizon)

#%% Prepare the model

early_stopping_monitor = EarlyStopping(patience = 2)

regressor = Sequential()
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 50,return_sequences = False))
#regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = Y_train.shape[1]))
#regressor.add(Activation("linear"))

regressor.compile(loss="mse", optimizer="rmsprop")
#%% Train the model

regressor.fit(X_train, Y_train, epochs = 20, validation_split = 0.3, callbacks = [early_stopping_monitor])

#%% Predictions on train and test sets
Y_train_pred = sc.inverse_transform(regressor.predict(X_train))
Y_test_pred = sc.inverse_transform(regressor.predict(X_test))


vide = np.empty((lag + horizon, 1))
vide[:] = np.nan

stock_prices_serie["prediction"] = np.append(vide, np.append(Y_train_pred, Y_test_pred))

#%% Plot of the results
plt.figure(num=None, figsize=(8 * 3, 6 * 3), dpi=100, facecolor='w', edgecolor='k')

stock_prices_serie.close.plot(label = 'Real Price', color = "lightgreen")
stock_prices_serie.prediction.plot(label = 'Predicted Price', color = "orange")

plt.title(stock_name + ' Price Prediction with lag -' + str(lag) + " days and horizon +" + str(horizon) + " days")
plt.axvline(x=date_sep, color='k', linestyle='--')
plt.text(date_sep - timedelta(days = 90),70,'Training separator',rotation=90)
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
#plt.show()
plt.savefig(stock_name + '_lag60_hor_30.png')

print(regressor.evaluate(X_train, Y_train))
print(regressor.evaluate(X_test, Y_test))
print(regressor.evaluate(np.append(X_train, X_test, axis = 0), np.append(Y_train, Y_test, axis = 0)))

#%%    

for i in range(lag, stock_prices_serie.shape[0] - horizon, horizon):
    print(i)
    stock_prices_serie["prediction" + str(i)] = np.nan
    sub_serie = stock_prices_serie.close.values.reshape(-1, 1)[i-lag:i]
    sub_serie = sc.transform(sub_serie).reshape((1, lag, 1))
    pred = sc.inverse_transform(regressor.predict(sub_serie))
    stock_prices_serie["prediction" + str(i)].iloc[i] = stock_prices_serie.close.values[i]
    stock_prices_serie["prediction" + str(i)].iloc[i + from_ : i + horizon + 1] = pred[0]
    
#%%
plt.figure(num=None, figsize=(8 * 3, 6 * 3), dpi=100, facecolor='w', edgecolor='k')

stock_prices_serie.close.plot(label = 'Real Price', color = "lightgreen")
for i in range(lag, stock_prices_serie.shape[0] - horizon, horizon):
    stock_prices_serie["prediction" + str(i)].dropna().plot()

plt.title(stock_name + ' Price Prediction')
plt.axvline(x=date_sep, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
#plt.legend()
plt.show()
plt.savefig(stock_name + '_lag90_hor1_30_trend.png')

#%%

lag = 30

sep_train_test = 700

from_ = 1

horizon = 1

date_sep = stock_prices_serie.iloc[sep_train_test].name

time_serie_train = stock_prices_serie[["close", "volume"]].values.reshape(-1, 2)[:sep_train_test]
time_serie_test = stock_prices_serie[["close", "volume"]].values.reshape(-1, 2)[sep_train_test - lag - horizon:]

from sklearn.preprocessing import MinMaxScaler
sc_price = MinMaxScaler(feature_range = (0,1))
sc_price.fit(time_serie_train[:, 0].reshape(-1, 1))
time_serie_train[:, 0] = sc_price.transform(time_serie_train[:, 0].reshape(-1, 1)).reshape(sep_train_test)
time_serie_test[:, 0] = sc_price.transform(time_serie_test[:, 0].reshape(-1, 1)).reshape(time_serie_test.shape[0])

sc_volume = MinMaxScaler(feature_range = (0,1))
sc_volume.fit(time_serie_train[:, 1].reshape(-1, 1))
time_serie_train[:, 1] = sc_volume.transform(time_serie_train[:, 1].reshape(-1, 1)).reshape(sep_train_test)
time_serie_test[:, 1] = sc_volume.transform(time_serie_test[:, 1].reshape(-1, 1)).reshape(time_serie_test.shape[0])

X_train, Y_train = Functions.transform_data(time_serie_train, lag, 
                                            from_ = from_,
                                            horizon = horizon)

X_test, Y_test = Functions.transform_data(time_serie_test, lag, 
                                          from_ = from_,
                                          horizon = horizon)

#%% Prepare the model
early_stopping_monitor = EarlyStopping(patience = 2)

regressor = Sequential()
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], X_train.shape[2])))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100,return_sequences = False))
regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 10))
#regressor.add(Dropout(0.2))

regressor.add(Dense(units = Y_train.shape[1], activation = "linear"))
#regressor.add(Activation("linear"))

regressor.compile(loss="mse", optimizer="rmsprop")
#%% Train the model

regressor.fit(X_train, Y_train, epochs = 10, validation_split=0.3, callbacks = [early_stopping_monitor])


#%% Predictions on train and test sets
Y_train_pred = sc_price.inverse_transform(regressor.predict(X_train))
Y_test_pred = sc_price.inverse_transform(regressor.predict(X_test))

vide = np.empty((lag + horizon, 1))
vide[:] = np.nan

stock_prices_serie["prediction"] = np.append(vide, np.append(Y_train_pred, Y_test_pred))
#%% Plot of the results

stock_prices_serie.close.plot(label = 'Real Price')
stock_prices_serie.prediction.plot(label = 'Predicted Price')

# stock_prices_serie.iloc[[x for x in range(lag + from_to_pred[1],stock_prices_serie.shape[0]-from_to_pred[1])]].index
plt.title(stock_name + ' Price Prediction with lag ' + str(lag) + " and horizon " + str(horizon))
plt.axvline(x=date_sep, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel(stock_name + ' Stock Price')
plt.legend()
plt.show()
