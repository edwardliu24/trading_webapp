from flask import Flask, g, render_template, request

import tensorflow as tf

import numpy as np

import json
import plotly

import yfinance as yf
import pandas as pd
from return_strat import returns

from get_data import get_preds_data
from plot_data import plot_ticker, plot_returns
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/test/', methods=['POST', 'GET'])
def test():
    if request.method == 'GET':
        return render_template('test.html')
    else:

        # assign the user's input to target
        target = request.form['target'].upper()
        resolution = '1wk'

        #try:
        #    stock = yf.Ticker(target)
        #    df = stock.history(period='7mo')
        #    df.drop(['Dividends','Stock Splits'],axis = 1,inplace = True)
        #except:
        #    return render_template('test.html', error=True)

        # assign model to the pre-trained model
        model1 = tf.keras.models.load_model('C:\market-flow-main\model\model_week')

        ##### perform prediction on target with the model
        #period = '3y'
        #indices = ['^GSPC','^VIX','^IXIC','^DJI','^HSI','^FTSE','^FCHI','GC=F','CL=F']
        #test = get_preds_data(target,indices = indices,
        #                             period = period,
        #                             resolution = resolution,
        #                              MAs = [4,21,52])

        #high_change_cols = ['volume', 'GC=F-volume']
        #test = test.drop(high_change_cols, axis = 1)
        #today = test.iloc[-1:].drop(['target','close'], axis=1)
        #test = test.iloc[:-1]
        #test.index = test.index + timedelta(days=7)
        #returns = test['close']

        #X = test.drop(columns=['target','close'],axis = 1)
        #y = test['target']
        
        #_, accuracy = model.evaluate(X,y)
        #preds = model.predict(X)
        #accuracy = np.round(accuracy*100, 1)

        #d = {0: 'down', 1: 'up'}
        #pred = d[int(tf.math.argmax(model.predict(today), 1))]
        
        #tickerJSON = json.dumps(plot_ticker(df, target), cls=plotly.utils.PlotlyJSONEncoder)
        #returnsJSON = json.dumps(plot_returns(returns, preds, target), cls=plotly.utils.PlotlyJSONEncoder)

        df = pd.read_csv('AIndexEodPrices.csv')
        del df['S_INFO_WINDCODE']
        target = int(target)
        balance, fig = returns(target, df[-target - 30: ], model1)

        returnsJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        ###### once generated, return the prediction and figure here
        return render_template('test.html', target=target,  
                               returnsJSON=returnsJSON, 
                               accuracy=balance, pred=target)