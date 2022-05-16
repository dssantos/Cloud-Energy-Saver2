from datetime import datetime

from pandas import DataFrame

import ram_usage


def save(hostname):
    mem = ram_usage.get(hostname)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = DataFrame({'timestampt': [timestamp], 'mem': mem})
    with open(f'{hostname}.csv', 'a') as f:
        df.to_csv(f, header=f.tell()==0, index=False)
