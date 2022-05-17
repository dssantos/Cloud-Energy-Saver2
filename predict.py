from datetime import datetime, timedelta

from pandas import date_range, DataFrame
from statsmodels.tsa.arima.model import ARIMA

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
