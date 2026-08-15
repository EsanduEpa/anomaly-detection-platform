from sqlalchemy import Column, BigInteger, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from src.db.session import Base

class MetricDataPoint(Base):
    __tablename__ = "metric_datapoints"  # The actual table name in PostgreSQL

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)  # e.g. "payment-service"
    host = Column(String(100), nullable=False)                       # e.g. "prod-server-01"
    metric_name = Column(String(100), nullable=False)                # e.g. "cpu_usage"
    value = Column(Float, nullable=False)                            # e.g. 94.2
    labels = Column(JSON, nullable=True)                             # Any extra info as JSON
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())  # Auto set by DB