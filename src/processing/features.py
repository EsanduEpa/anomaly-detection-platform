from collections import deque
import statistics

from numpy.ma import std

# How many past readings to remember per metric per service
WINDOW_SIZE = 10

# One history buffer per (service, metric) pair
# deque with maxlen automatically drops old values when full
_history: dict[str, deque] = {}


def _get_key(service_name: str, metric_name: str) -> str:
    return f"{service_name}:{metric_name}"


def compute_features(service_name: str, metric_name: str, value: float) -> dict:
    """
    Given one metric reading, returns the raw value plus 3 extra features.
    Remembers the last WINDOW_SIZE readings to compute rolling stats.
    """
    key = _get_key(service_name, metric_name)

    # Create a new history buffer if this is the first reading
    if key not in _history:
        _history[key] = deque(maxlen=WINDOW_SIZE)

    history = _history[key]

    # --- Rate of change ---
    # How much did the value jump since the last reading?
    if len(history) > 0:
        rate_of_change = value - history[-1]
    else:
        rate_of_change = 0.0  # No previous reading to compare

    # --- Rolling average ---
    # Average of everything we've seen so far (up to WINDOW_SIZE readings)
    if len(history) > 0:
        rolling_avg = statistics.mean(history)
    else:
        rolling_avg = value  # First reading — average is just itself

    # --- Z-score ---
    # How unusual is this value compared to recent history?
    if len(history) >= 2:
        std = statistics.stdev(history)
        Z_SCORE_MIN_STD = 1e-4

    if std > Z_SCORE_MIN_STD:
        z_score = (value - rolling_avg) / std
    else:
     z_score = 0.0  # Not enough history yet to calculate

    # Save this reading into history AFTER computing features
    # (we compare against past values, not including current)
    history.append(value)

    return {
        "value": value,
        "rolling_avg": round(rolling_avg, 4),
        "z_score": round(z_score, 4),
        "rate_of_change": round(rate_of_change, 4),
    }