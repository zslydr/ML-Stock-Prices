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
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta
#%%

end = datetime.now()
start = end - timedelta(days=900)

stock_prices_serie = web.DataReader("GOOGL", 'iex', start, end)

stock_prices_serie.index = pd.DatetimeIndex(stock_prices_serie.index)

stock_prices_serie.drop(["volume"], axis = 1).plot()

