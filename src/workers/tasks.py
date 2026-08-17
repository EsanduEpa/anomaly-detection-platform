from datetime import datetime, timedelta, timezone


from src.workers.celery_app import celery_app
from src.processing.cleaning import clean_metrics
from src.processing.features import compute_features
from src.db.session import SessionLocal

from src.models.metric import MetricDataPoint
from src.models.aggregation import MetricAggregation

from sqlalchemy import func as sa_func


@celery_app.task(name="tasks.ping")
def ping():
    return "pong"


@celery_app.task(name="tasks.process_metrics", bind=True, max_retries=3)
def process_metrics(self, payload: dict):
    """
    Receives one raw metrics payload.
    Cleans it, computes features, saves enriched rows to the database.
    """
    service_name = payload["service_name"]
    host         = payload["host"]
    timestamp    = datetime.fromisoformat(payload["timestamp"])
    raw_metrics  = payload["metrics"]

    # Step 1 — Clean
    cleaned = clean_metrics(raw_metrics)

    # Step 2 — Compute features + save
    db = SessionLocal()
    try:
        rows_saved = 0

        for metric_name, value in cleaned.items():
            features = compute_features(service_name, metric_name, value)

            db.add(MetricDataPoint(
                timestamp    = timestamp,
                service_name = service_name,
                host         = host,
                metric_name  = metric_name,
                value        = features["value"],
                labels       = {
                    "rolling_avg"    : features["rolling_avg"],
                    "z_score"        : features["z_score"],
                    "rate_of_change" : features["rate_of_change"],
                }
            ))
            rows_saved += 1

        db.commit()
        return {"status": "success", "rows_saved": rows_saved}

    except Exception as exc:
        db.rollback()
        raise self.retry(exc=exc, countdown=5)

    finally:
        db.close()






@celery_app.task(name="tasks.aggregate_metrics")
def aggregate_metrics():
    """
    Reads last 5 minutes of raw data.
    Saves one summary row per service per metric.
    """
    db = SessionLocal()
    try:
        now          = datetime.now(timezone.utc)
        window_end   = now
        window_start = now - timedelta(minutes=5)

        # Find all services that sent data in the last 5 minutes
        services = db.query(MetricDataPoint.service_name).filter(
            MetricDataPoint.timestamp >= window_start,
            MetricDataPoint.timestamp <= window_end,
        ).distinct().all()

        rows_saved = 0

        for (service_name,) in services:

            # Find all metric types for this service
            metrics = db.query(MetricDataPoint.metric_name).filter(
                MetricDataPoint.service_name == service_name,
                MetricDataPoint.timestamp    >= window_start,
                MetricDataPoint.timestamp    <= window_end,
            ).distinct().all()

            for (metric_name,) in metrics:

                # Calculate stats for this service + metric combo
                stats = db.query(
                    sa_func.avg(MetricDataPoint.value),
                    sa_func.min(MetricDataPoint.value),
                    sa_func.max(MetricDataPoint.value),
                    sa_func.count(MetricDataPoint.value),
                ).filter(
                    MetricDataPoint.service_name == service_name,
                    MetricDataPoint.metric_name  == metric_name,
                    MetricDataPoint.timestamp    >= window_start,
                    MetricDataPoint.timestamp    <= window_end,
                ).one()

                avg_val, min_val, max_val, count = stats

                if count == 0:
                    continue

                db.add(MetricAggregation(
                    window_start = window_start,
                    window_end   = window_end,
                    window_size  = "5min",
                    service_name = service_name,
                    metric_name  = metric_name,
                    avg_value    = round(avg_val, 4),
                    min_value    = round(min_val, 4),
                    max_value    = round(max_val, 4),
                    sample_count = count,
                ))
                rows_saved += 1

        db.commit()
        return {"status": "success", "aggregations_saved": rows_saved}

    except Exception as exc:
        db.rollback()
        raise exc

    finally:
        db.close()