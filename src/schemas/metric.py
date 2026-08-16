from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# ─── INPUT SCHEMAS (data coming INTO the API) ───────────────────────────────

class MetricsPayload(BaseModel):
    """The 7 metric values sent by the simulator."""
    cpu_usage: float = Field(..., ge=0, le=100)          # Must be 0-100
    memory_usage: float = Field(..., ge=0, le=100)       # Must be 0-100
    request_latency_ms: float = Field(..., ge=0)         # Must be 0 or above
    requests_per_sec: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)           # Must be 0.0 to 1.0
    db_connections: int = Field(..., ge=0)
    disk_usage: float = Field(..., ge=0, le=100)


class MetricIngest(BaseModel):
    """The full request body the simulator sends to POST /metrics."""
    service_name: str
    host: str
    timestamp: datetime
    metrics: MetricsPayload                               # Nested metrics object


# ─── OUTPUT SCHEMAS (data going OUT of the API) ──────────────────────────────

class MetricResponse(BaseModel):
    """The shape of one metric row returned by GET /metrics."""
    id: int
    service_name: str
    host: str
    metric_name: str
    value: float
    timestamp: datetime

    model_config = {"from_attributes": True}  # Allows converting SQLAlchemy row → this schema


class IngestResponse(BaseModel):
    """The response returned after successfully ingesting metrics."""
    status: str
    message: str
    rows_saved: int