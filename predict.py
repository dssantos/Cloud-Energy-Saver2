from datetime import datetime, timedelta

import workload, ram_usage


def naive(hostname):
    df = workload.get(hostname)
    last_time_stamp = df.tail(1).index
    if last_time_stamp > (datetime.now() - timedelta(seconds=90)):
        predict = df.iloc[-1]['mem']
    else:
        predict = ram_usage.get(hostname)
    return predict
