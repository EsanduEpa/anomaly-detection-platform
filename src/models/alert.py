import uuid
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.db.session import Base


class Alert(Base):
    """
    One alert = one ONGOING problem on one service.

    An alert is an INTERVAL, not a moment:
        created_at    -> when the problem startedß
        last_seen_at  -> the most recent anomalous reading
        resolved_at   -> when it stopped

    Step 3 (dedup) will UPDATE an existing ACTIVE row instead of inserting
    a new one, so a 43-reading memory leak stays ONE row, not 43.
    """
    __tablename__ = "alerts"

    # ── Identity & grouping ───────────────────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # The dedup key — what makes two alerts "the same problem".
    # Step 3 looks alerts up by this instead of a growing WHERE clause.
    fingerprint = Column(String(200), nullable=False, index=True)

    # Step 4 groups related alerts into one Incident.
    # Plain UUID for now — the FOREIGN KEY is added in Step 4's migration,
    # because the `incidents` table doesn't exist yet.
    incident_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # ── Lifecycle ─────────────────────────────────────────────────────
    status           = Column(String(20), nullable=False, default="ACTIVE")
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    last_seen_at     = Column(DateTime(timezone=True), nullable=False)
    resolved_at      = Column(DateTime(timezone=True), nullable=True)
    acknowledged_at  = Column(DateTime(timezone=True), nullable=True)
    # Plain text, NOT a User foreign key — no User table until Phase 8
    acknowledged_by  = Column(String(100), nullable=True)
    occurrence_count = Column(Integer, nullable=False, default=1)

    # ── Where ─────────────────────────────────────────────────────────
    service_name = Column(String(100), nullable=False, index=True)
    host         = Column(String(100), nullable=False)

    # NULL = MULTIVARIATE (all 7 metrics together).
    # Same convention AnomalyScore already uses.
    metric_name  = Column(String(100), nullable=True)

    # ── Layer 1 — what the 3 detectors saw ────────────────────────────
    severity      = Column(String(20), nullable=False)   # INFO / WARNING / CRITICAL
    anomaly_score = Column(Float, nullable=False)        # ensemble vote fraction

    # {"votes": 2, "total_available": 3,
    #  "zscore": true, "iforest": true, "lstm": false}
    detected_by = Column(JSON, nullable=True)

    # [{"metric": "memory_usage", "z_score": 4.1, "value": 91.2}, ...]
    # What was OBSERVED — not a claim about the cause.
    triggering_metrics = Column(JSON, nullable=True)

    # ── Layer 2 — what XGBoost + SHAP predicted ───────────────────────
    escalation_probability = Column(Float, nullable=True)
    explanation_text       = Column(Text, nullable=True)
    contributing_features  = Column(JSON, nullable=True)

    # ── Notifications (Step 7) ────────────────────────────────────────
    last_notified_at = Column(DateTime(timezone=True), nullable=True)