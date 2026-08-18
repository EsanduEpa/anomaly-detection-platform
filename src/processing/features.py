from collections import deque
import statistics



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

    if key not in _history:
        _history[key] = deque(maxlen=WINDOW_SIZE)

    history = _history[key]

    # --- Rate of change ---
    if len(history) > 0:
        rate_of_change = value - history[-1]
    else:
        rate_of_change = 0.0

    # --- Rolling average ---
    if len(history) > 0:
        rolling_avg = statistics.mean(history)
    else:
        rolling_avg = value

    # --- Z-score ---
    # Z_SCORE_MIN_STD guards against exploding z-scores when std is
    # technically nonzero but tiny (this is what broke error_rate earlier —
    # see the data_prep.py fix from Step 2).
    Z_SCORE_MIN_STD = 1e-4

    if len(history) >= 2:
        std = statistics.stdev(history)
        if std > Z_SCORE_MIN_STD:
            z_score = (value - rolling_avg) / std
        else:
            z_score = 0.0
    else:
        z_score = 0.0

    history.append(value)

    return {
        "value": value,
        "rolling_avg": round(rolling_avg, 4),
        "z_score": round(z_score, 4),
        "rate_of_change": round(rate_of_change, 4),
    }