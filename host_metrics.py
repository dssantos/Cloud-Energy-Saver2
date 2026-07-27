#coding: utf-8
"""In-memory shared state for the multivariate pipeline.

Keeps, per host:
  - current VM count (updated by the cluster_workload logger)
  - timestamp of the last VM create/delete event (updated by instances)

Read by the workload_mv collector to record vms and secs_since_vm_created/deleted
in the multivariate CSV. Thread-safe.
"""
import threading
import time

_lock = threading.Lock()

# host -> int (current VM count)
_vms = {}

# host -> {'create': ts, 'delete': ts}
_events = {}

# Sentinel when no event has been recorded yet (models see "no recent event")
NO_EVENT_SENTINEL = 9999.0


def set_vms(host, n):
    with _lock:
        _vms[host] = int(n or 0)


def get_vms(host):
    with _lock:
        return _vms.get(host, 0)


def record_create(host):
    if not host:
        return
    with _lock:
        _events.setdefault(host, {})['create'] = time.time()


def record_delete(host):
    if not host:
        return
    with _lock:
        _events.setdefault(host, {})['delete'] = time.time()


def secs_since_create(host):
    with _lock:
        ts = _events.get(host, {}).get('create')
    if ts is None:
        return NO_EVENT_SENTINEL
    return time.time() - ts


def secs_since_delete(host):
    with _lock:
        ts = _events.get(host, {}).get('delete')
    if ts is None:
        return NO_EVENT_SENTINEL
    return time.time() - ts
