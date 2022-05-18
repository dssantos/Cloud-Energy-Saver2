from datetime import datetime, timedelta

from statsmodels.tsa.arima.model import ARIMA
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import workload, ram_usage


def naive(hostname):
    df = workload.get(hostname)
    last_time_stamp = df.tail(1).index
    if last_time_stamp > (datetime.now() - timedelta(seconds=90)):
        predict = df.iloc[-1]['mem']
    else:
        predict = ram_usage.get(hostname)
    return predict

def arima(hostname):
    df = workload.get(hostname)
    df = df.resample('s').interpolate().resample('90s').asfreq()
    order = (1,0,1)
    X = [x for x in df.mem]
    try:
        model = ARIMA(X, order=order, enforce_stationarity=False)
        model_fit = model.fit()
        predict = model_fit.forecast()[0]
    except ValueError:
        print('Insufficient data to use arima model.\nUsing default mode.')
        predict = ram_usage.get(hostname)
    return predict

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def lstm(hostname):
    '''
    Vanila LSTM model adapted from:
    Jason Brownlee, How to Develop LSTM Models for Time Series Forecasting, Machine Learning Mastery, 
    Available from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/, 
    accessed May 17th, 2022.
    '''
    #create array
    df = workload.get(hostname)
    df1 = df['mem']
    df_arr = df1.values

    # choose a number of time steps
    n_steps = 3

    # define input sequence
    raw_seq = df_arr
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    try:
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=200, verbose=0)
        # demonstrate prediction
        x_input = array(df_arr[-n_steps:])
        x_input = x_input.reshape((1, n_steps, n_features))
        predict = model.predict(x_input, verbose=0)[0][0]
    except IndexError:
        print('Insufficient data to use lstm model.\nUsing default mode.')
        predict = ram_usage.get(hostname)

    return predict
