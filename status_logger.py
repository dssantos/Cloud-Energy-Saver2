#coding: utf-8
"""
CES Status Logger Module

Provides continuous background logging of host status.
Thread-safe JSON logging with timestamps.
"""

import threading
import time
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from typing import List
import status


@dataclass
class HostStatus:
    timestamp: str
    hostname: str
    state: str  # 'up' or 'down'
    vms: int
    ram_percent: float
    ram_mb: float


class StatusLogger:
    def __init__(self, interval: int = 30, filename: str = 'status_log.json'):
        self.interval = interval
        self.filename = filename
        self.running = False
        self.thread = None
        self.statuses: List[HostStatus] = []
        self.lock = threading.Lock()

    def start(self):
        """Start background status logging."""
        self.running = True
        self.thread = threading.Thread(target=self._log_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop background status logging."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _log_loop(self):
        """Main logging loop running in background thread."""
        while self.running:
            hosts = status.get()
            with self.lock:
                for host in hosts:
                    host_status = HostStatus(
                        timestamp=datetime.now().isoformat(),
                        hostname=host['hostname'],
                        state=host['state'],
                        vms=host['vms'],
                        ram_percent=float(host['ram']) if host.get('ram') else 0.0,
                        ram_mb=0.0  # Can be calculated if needed
                    )
                    self.statuses.append(host_status)
                self._save()
            time.sleep(self.interval)

    def _save(self):
        """Save statuses to JSON file."""
        with open(self.filename, 'w') as f:
            json.dump([asdict(s) for s in self.statuses], f, indent=2)

    def get_statuses(self) -> List[HostStatus]:
        """Get all logged statuses (thread-safe)."""
        with self.lock:
            return self.statuses.copy()


# Global instance
logger = StatusLogger()
