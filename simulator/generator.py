import math
import random
from datetime import datetime, timezone


def get_time_of_day_factor():
    """
    Returns a number between 0.3 and 1.0 based on the current hour.
    Higher during business hours, lower at night.
    This mimics real server traffic patterns.
    """
    hour = datetime.now().hour  # 0-23
    # Sine wave: peaks at hour 14 (2pm), lowest at hour 2 (2am)
    factor = 0.5 + 0.5 * math.sin(math.pi * (hour - 2) / 12)
    return max(0.3, min(1.0, factor))  # Keep it between 0.3 and 1.0


def add_noise(value, noise_level=0.05):
    """
    Adds small random fluctuation to a value.
    noise_level=0.05 means ±5% random variation.
    This makes the data look realistic, not perfectly flat.
    """
    noise = random.gauss(0, noise_level * value)  # Gaussian noise
    return value + noise


def generate_normal_metrics():
    """
    Generates one set of realistic, NORMAL server metrics.
    Values vary based on time of day and random noise.
    """
    factor = get_time_of_day_factor()  # How busy is the server right now?

    metrics = {
        "cpu_usage": add_noise(35 * factor + 10),              # 10-45% normally
        "memory_usage": add_noise(50 + 10 * factor),            # 50-60% normally
        "request_latency_ms": add_noise(50 + 30 * factor),      # 50-80ms normally
        "requests_per_sec": add_noise(100 * factor + 20),       # 20-120 req/s
        "error_rate": add_noise(0.002, noise_level=0.5),        # ~0.2% error rate
        "db_connections": int(add_noise(20 * factor + 5)),      # 5-25 connections
        "disk_usage": add_noise(45 + 0.001 * factor),           # Slowly grows
    }

    # Make sure values stay within realistic ranges (no negative values!)
    metrics["cpu_usage"] = max(1.0, min(100.0, metrics["cpu_usage"]))
    metrics["memory_usage"] = max(1.0, min(100.0, metrics["memory_usage"]))
    metrics["error_rate"] = max(0.0, min(1.0, metrics["error_rate"]))
    metrics["requests_per_sec"] = max(0.0, metrics["requests_per_sec"])
    metrics["db_connections"] = max(0, metrics["db_connections"])
    metrics["disk_usage"] = max(0.0, min(100.0, metrics["disk_usage"]))
    metrics["request_latency_ms"] = max(1.0, metrics["request_latency_ms"])

    return metrics


def generate_cpu_spike_metrics():
    """
    Simulates a CPU spike anomaly scenario.
    When CPU spikes, latency and error rate also increase (cascading effect).
    """
    base = generate_normal_metrics()
    base["cpu_usage"] = add_noise(92)           # CPU jumps to ~92%
    base["request_latency_ms"] = add_noise(800) # Latency jumps to ~800ms
    base["error_rate"] = add_noise(0.15, 0.2)  # Error rate jumps to ~15%
    return base


def get_metrics_payload(service_name: str, host: str, inject_anomaly: bool = False):
    """
    Builds the full payload dictionary to send to the API.
    Randomly injects an anomaly 10% of the time if inject_anomaly is True.
    """
    if inject_anomaly and random.random() < 0.10:  # 10% chance of anomaly
        metrics = generate_cpu_spike_metrics()
    else:
        metrics = generate_normal_metrics()

    return {
        "service_name": service_name,
        "host": host,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics
    }