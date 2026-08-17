from datetime import datetime

from src.workers.celery_app import celery_app
from src.processing.cleaning import clean_metrics
from src.processing.features import compute_features
from src.db.session import SessionLocal
from src.models.metric import MetricDataPoint


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