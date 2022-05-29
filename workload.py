from time import sleep
from datetime import datetime
import traceback

from pandas import DataFrame, read_csv, to_datetime

import ram_usage


def save(hostname):
    while True:
        try:
            mem = ram_usage.get(hostname)
            if not mem == 0:
                time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df = DataFrame({'time_stamp': [time_stamp], 'mem': mem})
                with open(f'{hostname}.csv', 'a') as f:
                    df.to_csv(f, header=f.tell()==0, index=False)
        except Exception:
            pass
        sleep(30)

def get(hostname):
    df = read_csv(f'{hostname}.csv')
    df.time_stamp = to_datetime(df.time_stamp, format='%Y-%m-%d %H:%M:%S')
    df.set_index('time_stamp', inplace=True)
    return df
