"""
Quick sanity check: does the ensemble correctly say "normal" for a reading
that actually looks like the simulator's healthy baseline?
"""
import numpy as np
from src.ml.registry import registry
from src.ml.ensemble import score_reading

# Realistic "normal" values, matching the simulator's healthy baseline ranges
normal_values = {
    "cpu_usage": 40.0,
    "memory_usage": 55.0,
    "request_latency_ms": 70.0,
    "requests_per_sec": 100.0,
    "error_rate": 0.002,
    "db_connections": 20.0,
    "disk_usage": 47.0,
}

raw_features = np.zeros(len(registry.feature_cols))
for i, col in enumerate(registry.feature_cols):
    if col.endswith("_z_score") or col.endswith("_rate_of_change"):
        raw_features[i] = 0.0                              # steady, no deviation
    elif col.endswith("_rolling_avg"):
        metric = col.replace("_rolling_avg", "")
        raw_features[i] = normal_values[metric]             # recent average = current value
    else:
        raw_features[i] = normal_values[col]                # the raw value itself

seq_len = registry.lstm_seq_len or 20
normal_sequence = np.tile(raw_features, (seq_len, 1))       # same calm reading, 20 times in a row

result = score_reading(raw_features, normal_sequence)
print("\nEnsemble result for a REALISTIC normal reading:")
for k, v in result.items():
    print(f"  {k:<16} {v}")