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
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


#%%

# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
tickers = ['AAPL', 'MSFT', '^GSPC']

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2016-12-31'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader('MSFT', 'morningstar', start_date, end_date)

#%%


import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta

#stock of interest
stock=['MSFT','SAP','V','JPM']

# period of analysis
end = datetime.now()
start = end - timedelta(days=500)

f = web.DataReader("google", 'robinhood', start, end)
#%%

f.head()


