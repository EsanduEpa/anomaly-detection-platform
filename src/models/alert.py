import uuid
from sqlalchemy import Column, String, Float, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.db.session import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    severity = Column(String(20), nullable=False)        # INFO / WARNING / CRITICAL
    status = Column(String(20), default="ACTIVE")        # ACTIVE / ACKNOWLEDGED / RESOLVED
    service_name = Column(String(100), nullable=False)
    host = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    anomaly_score = Column(Float, nullable=False)         # ML confidence 0.0 to 1.0
    explanation_text = Column(Text, nullable=True)        # Human readable explanation
    contributing_features = Column(JSON, nullable=True)   # Top features that caused the alert