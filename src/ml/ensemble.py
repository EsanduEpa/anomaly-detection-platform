"""
ENSEMBLE SCORER
===============
Takes ONE reading's raw (unscaled) 28 features, asks all 3 detectors for
their opinion via the Model Registry, and combines their votes into a
final is_anomaly decision.

Vote requirement scales with how many detectors are actually available
(fail-open): 3 available -> need 2; 2 available -> need both; 1 available
-> it decides alone. This is what makes a missing LSTM model degrade
gracefully instead of breaking scoring entirely.
"""

import numpy as np

from src.ml.registry import registry


def score_reading(raw_features: np.ndarray,
                  recent_raw_sequence: np.ndarray = None,
                  model_version: str = "v1") -> dict:
    """
    raw_features:         1D array of 28 raw feature values, in the same
                          order as registry.feature_cols, for ONE reading.
    recent_raw_sequence:  2D array (seq_len, 28) of the last N raw readings,
                          oldest first. None if not enough history yet.
    """
    if registry.scaler is None:
        return _unavailable_result(model_version, "scaler not loaded")

    scaled_features = registry.scaler.transform(raw_features.reshape(1, -1))[0]

    zscore_result = registry.score_zscore(scaled_features)
    iforest_result = registry.score_iforest(scaled_features)

    if recent_raw_sequence is not None:
        scaled_sequence = registry.scaler.transform(recent_raw_sequence)
        lstm_result = registry.score_lstm(scaled_sequence)
    else:
        lstm_result = {"available": False}

    detectors = {
        "zscore": zscore_result,
        "iforest": iforest_result,
        "lstm": lstm_result,
    }

    available = [name for name, r in detectors.items() if r.get("available")]
    total_available = len(available)

    if total_available == 0:
        return _unavailable_result(model_version, "no detectors available")

    votes = sum(1 for name in available if detectors[name]["flag"])
    required_votes = total_available // 2 + 1
    is_anomaly = votes >= required_votes
    ensemble_score = round(votes / total_available, 4)

    return {
        "zscore_value":  zscore_result.get("value"),
        "zscore_flag":   zscore_result.get("flag", False),
        "iforest_score": iforest_result.get("value"),
        "iforest_flag":  iforest_result.get("flag", False),
        "lstm_error":    lstm_result.get("value"),
        "lstm_flag":     lstm_result.get("flag", False),
        "votes":         votes,
        "ensemble_score": ensemble_score,
        "is_anomaly":    is_anomaly,
        "model_version": model_version,
    }


def _unavailable_result(model_version: str, reason: str) -> dict:
    print(f"⚠️  Ensemble scoring skipped: {reason}")
    return {
        "zscore_value": None, "zscore_flag": False,
        "iforest_score": None, "iforest_flag": False,
        "lstm_error": None, "lstm_flag": False,
        "votes": 0, "ensemble_score": 0.0, "is_anomaly": False,
        "model_version": model_version,
    }


if __name__ == "__main__":
    # quick manual test: python -m src.ml.ensemble
    n_features = len(registry.feature_cols) or 28
    seq_len = registry.lstm_seq_len or 20

    fake_normal_reading = np.random.normal(0, 1, n_features)
    fake_sequence = np.random.normal(0, 1, (seq_len, n_features))

    result = score_reading(fake_normal_reading, fake_sequence)
    print("\nEnsemble result:")
    for k, v in result.items():
        print(f"  {k:<16} {v}")