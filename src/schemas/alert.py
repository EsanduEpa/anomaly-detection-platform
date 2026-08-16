from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from uuid import UUID

class AlertResponse(BaseModel):
    """The shape of an alert returned by GET /alerts."""
    id: UUID
    created_at: datetime
    severity: str
    status: str
    service_name: str
    host: str
    metric_name: str
    anomaly_score: float
    explanation_text: Optional[str] = None

    model_config = {"from_attributes": True}