"""
MODEL REGISTRY
==============
Single place that loads and serves all three trained anomaly detectors.
Nothing else in the codebase should import sklearn/tensorflow/pickle
directly for scoring — they just ask this module for a score.

Fail-open: if any model file is missing, that detector is marked
unavailable and skipped, instead of crashing the whole pipeline.
"""

import json
import pickle
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models"


class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        self.feature_cols = self._load_feature_cols()
        self.scaler = self._load_scaler()

        self.zscore_threshold = None
        self.zscore_col_idx = []
        self._load_zscore()

        self.iforest_model = None
        self.iforest_threshold = None
        self._load_iforest()

        self.lstm_model = None
        self.lstm_threshold = None
        self.lstm_seq_len = None
        self._load_lstm()

    # ---- loaders — each one is safe if its file is missing ----

    def _load_feature_cols(self):
        path = ML_DIR / "feature_cols.json"
        if not path.exists():
            print(f"⚠️  {path} not found — feature ordering unknown")
            return []
        with open(path) as f:
            return json.load(f)

    def _load_scaler(self):
        path = ML_DIR / "scaler.pkl"
        if not path.exists():
            print(f"⚠️  {path} not found — cannot scale features")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_zscore(self):
        path = ML_DIR / "zscore_config.json"
        if not path.exists():
            print("⚠️  Z-Score config not found — Z-Score detector DISABLED")
            return
        with open(path) as f:
            config = json.load(f)
        self.zscore_threshold = config["threshold"]
        self.zscore_col_idx = [
            i for i, c in enumerate(self.feature_cols) if c.endswith("_z_score")
        ]
        print(f"✅ Z-Score detector loaded (threshold={self.zscore_threshold})")

    def _load_iforest(self):
        model_path = ML_DIR / "iforest_model.joblib"
        config_path = ML_DIR / "iforest_config.json"
        if not model_path.exists() or not config_path.exists():
            print("⚠️  Isolation Forest files not found — Isolation Forest DISABLED")
            return
        import joblib
        self.iforest_model = joblib.load(model_path)
        with open(config_path) as f:
            self.iforest_threshold = json.load(f)["threshold"]
        print(f"✅ Isolation Forest loaded (threshold={self.iforest_threshold})")

    def _load_lstm(self):
        model_path = ML_DIR / "lstm_autoencoder.keras"
        config_path = ML_DIR / "lstm_config.json"
        if not model_path.exists() or not config_path.exists():
            print("⚠️  LSTM files not found — LSTM detector DISABLED")
            return
        from tensorflow import keras
        self.lstm_model = keras.models.load_model(model_path)
        with open(config_path) as f:
            config = json.load(f)
        self.lstm_threshold = config["threshold"]
        self.lstm_seq_len = config["seq_len"]
        print(f"✅ LSTM Autoencoder loaded (threshold={self.lstm_threshold})")

    # ---- availability flags ----

    @property
    def zscore_available(self):
        return self.zscore_threshold is not None

    @property
    def iforest_available(self):
        return self.iforest_model is not None

    @property
    def lstm_available(self):
        return self.lstm_model is not None

    # ---- scoring methods — one per detector ----

    def score_zscore(self, scaled_features: np.ndarray) -> dict:
        """scaled_features: 1D array of 28 scaled values for ONE reading."""
        if not self.zscore_available or self.scaler is None:
            return {"available": False}
        raw = self.scaler.inverse_transform(scaled_features.reshape(1, -1))[0]
        z_values = raw[self.zscore_col_idx]
        max_abs_z = float(np.max(np.abs(z_values)))
        return {
            "available": True,
            "value": max_abs_z,
            "flag": max_abs_z > self.zscore_threshold,
        }

    def score_iforest(self, scaled_features: np.ndarray) -> dict:
        if not self.iforest_available:
            return {"available": False}
        score = float(self.iforest_model.decision_function(scaled_features.reshape(1, -1))[0])
        return {
            "available": True,
            "value": score,
            "flag": score < self.iforest_threshold,
        }

    def score_lstm(self, scaled_sequence: np.ndarray) -> dict:
        """scaled_sequence: 2D array shape (seq_len, 28) — the last N scaled readings."""
        if not self.lstm_available:
            return {"available": False}
        if scaled_sequence.shape[0] != self.lstm_seq_len:
            return {"available": False}  # not enough history yet
        seq = scaled_sequence.reshape(1, self.lstm_seq_len, -1)
        reconstructed = self.lstm_model.predict(seq, verbose=0)
        error = float(np.mean(np.square(seq - reconstructed)))
        return {
            "available": True,
            "value": error,
            "flag": error > self.lstm_threshold,
        }


# module-level singleton — Celery workers load models ONCE per process
registry = ModelRegistry()


if __name__ == "__main__":
    # quick manual test: python -m src.ml.registry
    r = registry
    print("\nAvailability:")
    print("  Z-Score:         ", r.zscore_available)
    print("  Isolation Forest:", r.iforest_available)
    print("  LSTM:            ", r.lstm_available)

    fake_reading = np.random.randn(28)
    print("\nZ-Score score:        ", r.score_zscore(fake_reading))
    print("Isolation Forest score:", r.score_iforest(fake_reading))

    fake_sequence = np.random.randn(r.lstm_seq_len or 20, 28)
    print("LSTM score:            ", r.score_lstm(fake_sequence))