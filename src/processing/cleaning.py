# Allowed (low, high) range for each metric.
# Same numbers used in schemas/metric.py's Field(ge=..., le=...)
METRIC_BOUNDS = {
    "cpu_usage": (0, 100),
    "memory_usage": (0, 100),
    "request_latency_ms": (0, None),   # None = no upper limit
    "requests_per_sec": (0, None),
    "error_rate": (0, 1),
    "db_connections": (0, None),
    "disk_usage": (0, 100),
}


def clean_metrics(metrics: dict) -> dict:
    """Fixes missing or out-of-range values in one batch of readings."""
    cleaned = {}

    for name, value in metrics.items():
        low, high = METRIC_BOUNDS.get(name, (None, None))

        if value is None:                      # missing → fill it
            value = low if low is not None else 0
        if low is not None and value < low:     # too low → clip up
            value = low
        if high is not None and value > high:   # too high → clip down
            value = high

        cleaned[name] = value

    return cleaned