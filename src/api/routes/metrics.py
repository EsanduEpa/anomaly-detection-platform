from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional

from src.db.session import get_db
from src.models.metric import MetricDataPoint
from src.schemas.metric import MetricIngest, MetricResponse, IngestResponse
from src.workers.tasks import process_metrics

# A router is like a mini-app that groups related endpoints together
router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])


@router.post("", response_model=IngestResponse, status_code=202)
def ingest_metrics(payload: MetricIngest):
    # Convert the Pydantic object to a plain dict for Celery
    # (Celery sends data as JSON — Pydantic objects aren't JSON)
    payload_dict = payload.model_dump(mode="json")

    # Hand off to Celery — don't write to DB here
    process_metrics.delay(payload_dict)

    return IngestResponse(
        status="accepted",
        message="Metrics received and queued for processing",
        rows_saved=0
    )



@router.get("", response_model=List[MetricResponse])
def get_metrics(
    service_name: Optional[str] = None,          # Filter by service (optional)
    metric_name: Optional[str] = None,            # Filter by metric type (optional)
    from_time: Optional[datetime] = None,         # Start of time range (optional)
    to_time: Optional[datetime] = None,           # End of time range (optional)
    limit: int = 100,                             # Max rows to return (default 100)
    db: Session = Depends(get_db)
):
    """
    Returns historical metric data.
    All filters are optional — you can combine them however you want.
    """
    query = db.query(MetricDataPoint)

    # Apply filters only if they were provided
    if service_name:
        query = query.filter(MetricDataPoint.service_name == service_name)
    if metric_name:
        query = query.filter(MetricDataPoint.metric_name == metric_name)
    if from_time:
        query = query.filter(MetricDataPoint.timestamp >= from_time)
    if to_time:
        query = query.filter(MetricDataPoint.timestamp <= to_time)

    # Order by newest first, limit results
    results = query.order_by(MetricDataPoint.timestamp.desc()).limit(limit).all()

    return results