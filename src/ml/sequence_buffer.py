"""
SEQUENCE BUFFER
===============
Keeps the last N raw feature vectors per service in memory, so the LSTM
detector can be given a real sequence of recent readings instead of just
one moment. Same in-process rolling-window pattern as features.py's
_history dict — same reason it needs Celery's --pool=solo flag (a
multi-process pool would give each process its own separate history,
silently corrupting the sequence).
"""

from collections import deque

import numpy as np

from src.ml.registry import registry

_buffers: dict[str, deque] = {}


def record_and_get_sequence(service_name: str, raw_features: np.ndarray):
    """
    Appends this reading to the service's rolling history. Returns a full
    (seq_len, n_features) sequence once enough history has built up,
    otherwise returns None (still warming up).
    """
    seq_len = registry.lstm_seq_len or 20

    if service_name not in _buffers:
        _buffers[service_name] = deque(maxlen=seq_len)

    buffer = _buffers[service_name]
    buffer.append(raw_features)

    if len(buffer) < seq_len:
        return None

    return np.array(buffer)