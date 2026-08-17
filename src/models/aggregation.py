import uuid
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.db.session import Base


class MetricAggregation(Base):
    __tablename__ = "metric_aggregations"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    window_end   = Column(DateTime(timezone=True), nullable=False)
    window_size  = Column(String(20), nullable=False)
    service_name = Column(String(100), nullable=False, index=True)
    metric_name  = Column(String(100), nullable=False)
    avg_value    = Column(Float, nullable=False)
    min_value    = Column(Float, nullable=False)
    max_value    = Column(Float, nullable=False)
    sample_count = Column(Float, nullable=False)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())