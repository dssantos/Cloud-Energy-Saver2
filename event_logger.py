#coding: utf-8
"""
CES Event Logger Module

Logs events to JSON for analysis and TCC data generation.
Events include: wake, shutdown, sla_violation
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import threading
import numpy as np
from typing import Optional, List


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


@dataclass
class Event:
    """Event data structure for logging CES actions."""
    timestamp: str
    event_type: str  # 'wake', 'shutdown', 'sla_violation', 'initial_state', 'final_state'
    hostname: Optional[str]
    trigger_type: str  # 'reactive', 'lstm', 'arima', 'naive', 'system'
    ram_avg: float
    lim_max: float
    lim_med: float
    running_hosts: int
    idle_hosts: int
    offline_hosts: int
    predicted_ram: Optional[float] = None
    actual_ram: Optional[float] = None
    # Additional fields for initial/final state events
    initial_state: Optional[str] = None  # 'up' or 'down'
    initial_vms: Optional[int] = None
    final_state: Optional[str] = None  # 'up' or 'down'
    final_vms: Optional[int] = None


class EventLogger:
    """Thread-safe event logger for CES events."""

    def __init__(self, filename: str = 'events.json'):
        self.filename = filename
        self.events: List[Event] = []
        self.lock = threading.Lock()

    def log(self, event: Event):
        """Log an event to file (thread-safe)."""
        with self.lock:
            self.events.append(event)
            self._save()

    def _save(self):
        """Save events to JSON file."""
        with open(self.filename, 'w') as f:
            # Convert numpy types to native Python types
            events_data = [convert_numpy_types(asdict(e)) for e in self.events]
            json.dump(events_data, f, indent=2)

    def load(self):
        """Load events from JSON file."""
        with open(self.filename, 'r') as f:
            data = json.load(f)
            self.events = [Event(**e) for e in data]

    def get_filename(self) -> str:
        """Return the current events filename."""
        return self.filename


# Global instance
logger = EventLogger()
