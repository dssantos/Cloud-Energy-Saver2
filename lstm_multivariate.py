import os
import random
import csv
from datetime import datetime, timedelta
from math import sqrt

import requests
import numpy as np
from pandas import read_csv, DataFrame, concat, to_datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import workload


full_df = workload.get('compute1f')
full_df.head()

def resample_fill_values(df):
    df = df.reset_index()
    df = df.resample('30s',on='time_stamp').mean()
    df = (df.ffill()+df.bfill())/2

    return df

def split_dataframes(df):
    max_hidle_interval = 240
    df = df.reset_index()
    dataframes = [resample_fill_values(g.set_index('time_stamp', drop=True)) for k,g  in df.groupby((~(df.time_stamp.diff().dt.total_seconds().fillna(0) < max_hidle_interval)).cumsum())]
    return dataframes

splited_df = split_dataframes(full_df) # lista de dataframes com os dados

features = 1 # colunas do dataframe a serem usadas no input
lags = 5 # 1 até 5 (ou 30 se feature = 1)
TIME_STEPS = 30 # 20 a 30 dias (janelas maiores podem incluir informações irrelevantes e janelas menores podem não capturar informações importantes)
percent_train = 0.8
layer = 0
lstm1 = 64 # 64 a 128 neurônios (um número muito baixo pode levar a underfitting e um número muito alto pode levar a overfitting)
lstm2 = 128
dropout = 0.2 # 0,001 a 0,005 (taxas muito baixas podem aumentar o tempo de treinamento e taxas muito altas podem levar a instabilidades)
learning_rate = 0.001
epoch = 50 # 50 a 100 épocas (um número muito alto pode levar a overfitting)
batch = 16 # 16 a 32 amostras por lote/batch (lotes muito grandes podem aumentar o tempo de treinamento)

# features, lags, TIME_STEPS, percent_train, layer, lstm1, lstm2, dropout, learning_rate, epoch, batch  = [float(x) if '.' in x else int(x) for x in '1	5	30	0.6	1	128	256	0.1	0.001	300	64'.split()]

# input_df = splited_df[0]

def predict(input_df):
    features = 1 #random.choice([1, 5, 10, 30, 50, ])
    lags = random.choice([5, 10, 20, 30, 60, 90])
    TIME_STEPS = random.choice([5, 10, 15, 20, 30, 50, 75, 100, 150])
    percent_train = random.choice([0.5, 0.6, 0.7, 0.8])
    layer = random.choice([0, 1, 2])
    lstm1 = random.choice([64, 128, 256])
    lstm2 = random.choice([64, 128, 256, 512])
    dropout = random.choice([0.1, 0.2, 0.5])
    learning_rate = random.choice([0.001, 0.002, 0.005])
    epoch = random.choice([50, 100, 150, 200, 250, 300, 500])
    batch = random.choice([8, 16, 32, 64, 128])
    start_time = datetime.now()

    df_len = len(input_df)

    # features = 1
    input_df = input_df.iloc[: , -features:]

    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        
        return agg

    # lags = 5 # 1 até 5 (ou 30 se feature = 1)
    price_lags_df = series_to_supervised(input_df, n_in=lags, n_out=1, dropnan=True)

    # drop columns we don't want to predict
    col_remove = [x for x in range(-len(input_df.columns), -1)]
    price_lags_df.drop(price_lags_df.columns[col_remove], axis=1, inplace=True)

    # Normalização dos dados
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(price_lags_df)

    # Definição do número de passos de tempo para a previsão
    # TIME_STEPS = 30 # 20 a 30 dias (janelas maiores podem incluir informações irrelevantes e janelas menores podem não capturar informações importantes)

    # Criação de sequências de dados para treinamento
    def create_sequences(data, time_steps):
        X = []
        y = []
        for i in range(time_steps, len(data)):
            X.append(data[i - time_steps:i, :]) # Até penúltima columa
            y.append(data[i, -1]) # Pega a última coluna
        X, y = np.array(X), np.array(y)
        return X, y

    X, y = create_sequences(df_scaled, TIME_STEPS)


    # Divisão dos dados em conjunto de treino e teste
    # percent_train = 0.8
    TRAIN_SIZE = int(percent_train * len(X))
    X_train, y_train = X[:TRAIN_SIZE], y[:TRAIN_SIZE]
    X_test, y_test = X[TRAIN_SIZE:], y[TRAIN_SIZE:]


    # Criação do modelo LSTM # 1 a 2 camadas LSTM (aumentar o número de camadas pode levar a overfitting)

    # layer = 0
    # lstm1 = 64
    # lstm2 = 128
    # dropout = 0.2
    # learning_rate = 0.001

    if layer == 0:
        model = keras.Sequential()
        model.add(layers.LSTM(lstm1, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')

    if layer == 1:
        model = keras.Sequential()
        model.add(layers.LSTM(lstm1, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mse')

    if layer == 2:
        model = keras.Sequential()
        model.add(layers.LSTM(lstm1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(layers.Dropout(dropout))
        model.add(layers.LSTM(lstm2))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')


    # Treinamento do modelo
    # epoch = 50 # 50 a 100 épocas (um número muito alto pode levar a overfitting)
    # batch = 16 # 16 a 32 amostras por lote/batch (lotes muito grandes podem aumentar o tempo de treinamento)
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, validation_split=0.1, verbose=1)


    # Plot da perda durante o treinamento
    # plt.plot(history.history['loss'], label='treino')
    # plt.plot(history.history['val_loss'], label='validação')
    # plt.legend()
    # plt.show()

    # Previsão dos preços de fechamento futuros
    y_pred = model.predict(X_test)

    # Inversão da normalização dos dados
    y_pred_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]
    y_test_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]

    # Plot dos resultados da previsão
    # days_plot = 14
    # plt.plot(price_lags_df.index[TRAIN_SIZE+TIME_STEPS:][-days_plot:], y_test_inv[-days_plot:], label='valor real')
    # plt.plot(price_lags_df.index[TRAIN_SIZE+TIME_STEPS:][-days_plot:], y_pred_inv[-days_plot:], label='previsão')
    # plt.legend()
    # plt.show()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    lapsed = datetime.now() - start_time
    vars = [str(x) for x in [f'{rmse:.3f}', f'{df_len:03}', features, lags , TIME_STEPS, f'{percent_train:.2f}', layer, f'{lstm1:03}', f'{lstm2:03}', dropout, learning_rate, f'{epoch:03}', batch, lapsed]]
    log = ','.join(vars)
    print(log)

    with open('results.csv', 'a+') as f:
        f.write(log+'\n')

import os
if not os.path.exists("results.csv"):
    with open('results.csv', 'a+') as f:
        f.write('rmse,df_len,features,lags,time_steps,percent_train,layer,lstm1,lstm2,dropout,learning_rate,epoch,batch,lapsed\n')

while True:
    for df in splited_df:
        predict(df)