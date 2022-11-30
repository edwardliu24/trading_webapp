import math
import matplotlib.dates as mdates
import numpy as np
from numpy  import array
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go


def returns(days,df,model):
    balance=[0]
    date=[]
    data=df.iloc[:,1:6].values
    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_DT=scalar.fit_transform(data)
    prediction_hist=[]
    profit = []
    datelist_train = list(df['TRADE_DT'][-days - 1:])
    datelist_train = [dt.datetime.strptime(str(date), '%Y%m%d').date() for date in datelist_train]

  # return strategy
    for i in range(days):
        x=[]
        x.append(scaled_DT[i:i+30, 0:5])
        x=np.array(x)
        x=x.reshape(x.shape[0],x.shape[1],5)
        prediction=model.predict(x, verbose = False)
        prediction = prediction * (scalar.data_max_[2] - scalar.data_min_[2]) + scalar.data_min_[2]
        prediction_hist.append(prediction[0])
        if prediction>1.015*data[i+29][3]:
            profit.append(data[i+30][3]-data[i+29][3])
            balance.append((profit[i]+balance[i]))
        elif prediction < 0.985*data[i+29][3]:
            profit.append(data[i+29][3]-data[i+30][3])
            balance.append(profit[i] + balance[i])
        else:
            profit.append(0)
            balance.append(balance[i])
    
    # long strategy
    stock_price = df.iloc[29:,4:5].values
    long_return = [0]
    for i in range(days):
        long_return.append(long_return[i] + stock_price[i + 1] - stock_price[i])
  
  # plot
    balance = pd.DataFrame(balance)
    balance.index = datelist_train
    #plt.plot(balance,c='blue')
    long_return = pd.DataFrame(long_return)
    long_return.index = datelist_train
    #plt.plot(long_return, c='red')

    START_DATE_FOR_PLOTTING = '2012-05-01'

    trace1 = go.Scatter(
        x = balance.index,
        y = balance,
        mode = 'lines',
        name = 'Return Strategy'
    )
    trace2 = go.Scatter(
        x = long_return.index,
        y = long_return,
        mode = 'lines',
        name = 'Long Strategy'
    )
    layout = go.Layout(
        title = "Return and Long Strategy",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Profit"}
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    return balance, fig