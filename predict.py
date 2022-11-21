import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime, timedelta
from random import choice
import threading
import shutil

from statsmodels.tsa.arima.model import ARIMA
from numpy import array, concatenate
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

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

def loss_average(history, epochs):
    second_half_loss = history.history['loss'][epochs//2:]
    return sum(second_half_loss) / len(second_half_loss)

def train_lstm_model(hostname):
    try:
        epochs = choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        # get data workload
        df = workload.get(hostname)

        # choose a number of time steps
        n_steps = 50

        # split into samples
        X, y, df_arr = concatenate_samples(df, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # split train and test
        split_index = len(X)*2//3
        Xtrain, Xtest = X[:-split_index], X[-split_index:]
        ytrain, ytest = y[:-split_index], y[-split_index:]
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # create a model check point
        filepath = f'models/{hostname}/weights-{epochs:02d}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]
        # fit model
        history = model.fit(Xtrain, ytrain, epochs=epochs, verbose=0, validation_data=(Xtest, ytest),callbacks=callbacks_list)
        # save model named as loss value
        loss = loss_average(history, epochs)
        loss = '.'.join([str(loss).split('.')[0].zfill(3), str(loss).split('.')[-1]])
        time_stamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        model.load_weights(filepath)
        model.save(f'models/{hostname}/{loss} {time_stamp} {epochs}')

        return model
    except:
        print('Unable to train now')

def select_best_model(hostname):
    models = [f for f in os.listdir(f'./models/{hostname}') if not f.endswith('hdf5')]
    if len(models) > 10:
        shutil.rmtree(f'./models/{hostname}/{sorted(models)[-2]}')
        shutil.rmtree(f'./models/{hostname}/{sorted(models)[-1]}')
    best_model = sorted(models)[0]

    return best_model


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
    n_steps = 50

    last_df = split_dataframes(df)[-1]
    if len(last_df) > 50 and len(df) > 100:
        try:
            # split into samples
            X, y, df_arr = concatenate_samples(df, n_steps)

            # reshape from [samples, timesteps] into [samples, timesteps, features]
            n_features = 1
            # demonstrate prediction
            x_input = array(df[-n_steps:].mem.values)
            x_input = x_input.reshape((1, n_steps, n_features))
            try:
                threading.Thread(target=train_lstm_model, args=[hostname]).start()
                
                best_model = select_best_model(hostname)
                print(best_model)
                file_path = f'models/{hostname}/{best_model}'
                model = load_model(file_path)

                predict = model.predict(x_input, verbose=0)[0][0]
                print(x_input)
                print(predict, hostname)
                default_predict = ram_usage.get(hostname)
                if abs(predict - default_predict) > 15:
                    print(f'Deleting {file_path}')
                    shutil.rmtree(file_path)
                    print(f'Trying another model...')
                    model = train_lstm_model(hostname)
                    predict = model.predict(x_input, verbose=0)[0][0]
                    print(x_input)
                    print(predict, hostname)
            except FileNotFoundError:
                model = train_lstm_model(hostname)
                predict = model.predict(x_input, verbose=0)[0][0]
        except (IndexError, ValueError, AttributeError):
            print('Insufficient data to use lstm model.\nUsing default mode.')
            predict = ram_usage.get(hostname)
    else:
        print('Insufficient data to use lstm model.\nUsing default mode.')
        predict = ram_usage.get(hostname)

    return predict
