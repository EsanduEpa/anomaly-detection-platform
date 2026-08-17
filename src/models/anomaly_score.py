from sqlalchemy import (
    Column, BigInteger, String, Float, Boolean, Integer, DateTime
)
from sqlalchemy.sql import func
from src.db.session import Base


class AnomalyScore(Base):
    """
    One row per metric reading that was scored by the ML pipeline.

    Three detectors each cast a vote:
        Z-Score       — univariate, fast, works with zero training
        Isolation Forest — multivariate, catches complex patterns
        LSTM Autoencoder — sequence-aware, catches slow drifts

    If ≥ 2 of 3 detectors agree → is_anomaly = True.
    """
    __tablename__ = "anomaly_scores"

    # ── Identity ──────────────────────────────────────────────────────────────
    id           = Column(BigInteger, primary_key=True, autoincrement=True)

    # When and where this reading came from (matches metric_datapoints)
    timestamp    = Column(DateTime(timezone=True), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    host         = Column(String(100), nullable=False)

    # Which metric was scored.
    # NULL means this row is a MULTIVARIATE score (all 7 metrics together).
    # A value like "cpu_usage" means this is a per-metric Z-score row.
    metric_name  = Column(String(100), nullable=True)

    # ── Detector 1: Z-Score ───────────────────────────────────────────────────
    # zscore_value: how many standard deviations away from the rolling mean
    # zscore_flag:  True if |zscore_value| > threshold (default 3.0)
    zscore_value = Column(Float,   nullable=True)
    zscore_flag  = Column(Boolean, nullable=False, default=False)

    # ── Detector 2: Isolation Forest ─────────────────────────────────────────
    # iforest_score: raw anomaly score from sklearn (-1 = most anomalous)
    # iforest_flag:  True if score is below the contamination threshold
    iforest_score = Column(Float,   nullable=True)
    iforest_flag  = Column(Boolean, nullable=False, default=False)

    # ── Detector 3: LSTM Autoencoder ─────────────────────────────────────────
    # lstm_error: reconstruction error — how badly the model failed to
    #             predict this sequence (high error = unusual pattern)
    # lstm_flag:  True if lstm_error > the 95th-percentile training error
    lstm_error = Column(Float,   nullable=True)
    lstm_flag  = Column(Boolean, nullable=False, default=False)

    # ── Ensemble decision ─────────────────────────────────────────────────────
    # votes:          how many detectors flagged this reading (0, 1, 2, or 3)
    # ensemble_score: weighted average of the three detector scores (0.0–1.0)
    # is_anomaly:     True if votes >= 2  ← the final verdict
    votes          = Column(Integer, nullable=False, default=0)
    ensemble_score = Column(Float,   nullable=False, default=0.0)
    is_anomaly     = Column(Boolean, nullable=False, default=False, index=True)

    # Which version of the model file produced this score.
    # Lets you see "v1 missed these, v2 caught them".
    model_version = Column(String(50), nullable=True)

    # Set automatically by PostgreSQL when the row is inserted
    created_at = Column(DateTime(timezone=True), server_default=func.now())