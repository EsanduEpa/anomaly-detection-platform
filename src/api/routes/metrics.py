from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional

from src.db.session import get_db
from src.models.metric import MetricDataPoint
from src.schemas.metric import MetricIngest, MetricResponse, IngestResponse

# A router is like a mini-app that groups related endpoints together
router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_metrics(payload: MetricIngest, db: Session = Depends(get_db)):
    """
    Receives a batch of metrics from a service and saves them to the database.
    Each metric (cpu, memory, etc.) is saved as a separate row.
    """
    # The payload has one timestamp, one service_name, one host
    # but 7 different metric values inside payload.metrics
    # We unpack them and save each as a separate row
    metrics_dict = payload.metrics.model_dump()
    rows_saved = 0

    for metric_name, value in metrics_dict.items():
        data_point = MetricDataPoint(
            timestamp=payload.timestamp,
            service_name=payload.service_name,
            host=payload.host,
            metric_name=metric_name,
            value=value
        )
        db.add(data_point)
        rows_saved += 1

    db.commit()  # Save all rows to the database in one go

    return IngestResponse(
        status="accepted",
        message="Metrics received and saved successfully",
        rows_saved=rows_saved
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