from datetime import datetime, timedelta, timezone

import numpy as np


from src.workers.celery_app import celery_app
from src.processing.cleaning import clean_metrics
from src.processing.features import compute_features
from src.db.session import SessionLocal

from src.models.metric import MetricDataPoint
from src.models.aggregation import MetricAggregation
from src.models.anomaly_score import AnomalyScore

from src.ml.registry import registry
from src.ml.ensemble import score_reading
from src.ml.sequence_buffer import record_and_get_sequence

from src.models.alert import Alert
from src.services.alerts import build_alert_fields, merge_alert_fields

from sqlalchemy import func as sa_func

RAW_METRIC_NAMES = [
    "cpu_usage", "memory_usage", "request_latency_ms",
    "requests_per_sec", "error_rate", "db_connections", "disk_usage",
]


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
        all_features = {}   # metric_name -> {"value", "rolling_avg", "z_score", "rate_of_change"}

        for metric_name, value in cleaned.items():
            features = compute_features(service_name, metric_name, value)
            all_features[metric_name] = features

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


        # Run the ensemble ONLY if every metric this reading needs is present
        anomaly_saved = False
        if all(m in all_features for m in RAW_METRIC_NAMES) and registry.feature_cols:
            raw_features = _build_feature_vector(all_features)
            sequence = record_and_get_sequence(service_name, raw_features)
            result = score_reading(raw_features, sequence)

            db.add(AnomalyScore(
                timestamp      = timestamp,
                service_name   = service_name,
                host           = host,
                metric_name    = None,   # None = this is the MULTIVARIATE ensemble verdict
                zscore_value   = result["zscore_value"],
                zscore_flag    = result["zscore_flag"],
                iforest_score  = result["iforest_score"],
                iforest_flag   = result["iforest_flag"],
                lstm_error     = result["lstm_error"],
                lstm_flag      = result["lstm_flag"],
                votes          = result["votes"],
                ensemble_score = result["ensemble_score"],
                is_anomaly     = result["is_anomaly"],
                model_version  = result["model_version"],
            ))

            
            anomaly_saved = True


            # ── Phase 5 Step 1 Part C: raise an alert ───────────────
            # NOTE: no dedup yet — this creates ONE alert per anomalous
            #       reading. Step 3 fixes that. This is expected for now.
            # ── Phase 5 Step 3: look up before creating ─────────────
            if result["is_anomaly"]:
                fresh_fields = build_alert_fields(
                    service_name = service_name,
                    host         = host,
                    timestamp    = timestamp,
                    result       = result,
                    all_features = all_features,
                )
                fingerprint = fresh_fields["fingerprint"]

                existing = (
                    db.query(Alert)
                    .filter(Alert.fingerprint == fingerprint, Alert.status == "ACTIVE")
                    .first()
                )

                if existing:
                    merged = merge_alert_fields(
                        existing_occurrence_count = existing.occurrence_count,
                        existing_severity         = existing.severity,
                        fresh_fields              = fresh_fields,
                    )
                    for key, value in merged.items():
                        setattr(existing, key, value)
                else:
                    db.add(Alert(**fresh_fields))

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


def _build_feature_vector(all_features: dict) -> np.ndarray:
    """
    Assembles the 28-number vector in the EXACT order the models were
    trained on (registry.feature_cols), pulling the right sub-value out
    of each metric's feature dict.
    """
    vector = np.zeros(len(registry.feature_cols))
    for i, col in enumerate(registry.feature_cols):
        if col.endswith("_z_score"):
            metric = col[: -len("_z_score")]
            vector[i] = all_features[metric]["z_score"]
        elif col.endswith("_rolling_avg"):
            metric = col[: -len("_rolling_avg")]
            vector[i] = all_features[metric]["rolling_avg"]
        elif col.endswith("_rate_of_change"):
            metric = col[: -len("_rate_of_change")]
            vector[i] = all_features[metric]["rate_of_change"]
        else:
            vector[i] = all_features[col]["value"]
    return vector