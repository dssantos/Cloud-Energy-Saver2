#coding: utf-8
"""Multivariate workload collector.

Writes one CSV per host under workload_mv/{host}.csv with:
  time_stamp, mem, vms, secs_since_vm_created, secs_since_vm_deleted, swap, loadavg

Runs as a daemon thread per host (started by the orchestrator). Does NOT touch
the univariate computeN.csv (kept intact for the existing pipeline).

Features:
  - mem, swap, loadavg: collected in a single SSH call (ram_usage.get_extended)
  - vms: read from the shared host_metrics (updated by the cluster_workload logger)
  - secs_since_vm_created/deleted: recency of the last create/delete event (leading edge)
"""
from datetime import datetime
from time import sleep
from os import makedirs
from pandas import DataFrame

import ram_usage
import host_metrics

COLUMNS = ['time_stamp', 'mem', 'vms', 'secs_since_vm_created',
           'secs_since_vm_deleted', 'swap', 'loadavg']

MV_DIR = 'workload_mv'


def save(hostname):
    """Continuously sample multivariate metrics for a host every 30s."""
    try:
        makedirs(MV_DIR, exist_ok=True)
    except Exception:
        pass
    while True:
        try:
            metrics = ram_usage.get_extended(hostname)  # mem, swap, loadavg (one SSH)
            if metrics['mem'] > 0:  # skip unreachable samples (like workload.save skips 0)
                time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                row = {
                    'time_stamp': [time_stamp],
                    'mem': [round(metrics['mem'], 2)],
                    'vms': [host_metrics.get_vms(hostname)],
                    'secs_since_vm_created': [round(host_metrics.secs_since_create(hostname), 1)],
                    'secs_since_vm_deleted': [round(host_metrics.secs_since_delete(hostname), 1)],
                    'swap': [round(metrics['swap'], 2)],
                    'loadavg': [round(metrics['loadavg'], 2)],
                }
                df = DataFrame(row)
                with open(f'{MV_DIR}/{hostname}.csv', 'a') as f:
                    df.to_csv(f, header=f.tell() == 0, index=False)
        except Exception:
            pass
        sleep(30)
