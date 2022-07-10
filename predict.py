import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime, timedelta

from statsmodels.tsa.arima.model import ARIMA
from numpy import array, concatenate
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

def split_dataframes(df):
    df = df.reset_index()
    dataframes = [g.set_index('time_stamp', drop=True) for k,g  in df.groupby((~(df.time_stamp.diff().dt.total_seconds().fillna(0) < 240)).cumsum())]
    return dataframes

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

def concatenate_samples(full_df, n_steps):
    Xs = []
    ys = []
    for df in split_dataframes(full_df):
        if len(df) > n_steps + 1:
            df_arr = df['mem'].values
            X, y = split_sequence(df_arr, n_steps)
            Xs.append(X)
            ys.append(y)
    Xsample = concatenate((Xs))
    ysample = concatenate((ys))
    
    return Xsample, ysample, df_arr

def lstm(hostname):
    '''
    Vanila LSTM model adapted from:
    Jason Brownlee, How to Develop LSTM Models for Time Series Forecasting, Machine Learning Mastery, 
    Available from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/, 
    accessed May 17th, 2022.
    '''
    # get data workload
    df = workload.get(hostname)

    # choose a number of time steps
    n_steps = 10

    # split into samples
    X, y, df_arr = concatenate_samples(df, n_steps)

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
