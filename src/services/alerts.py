"""
ALERT BUILDER  (Phase 5, Step 1, Part B)
=========================================
Turns ONE ensemble verdict into the field values an Alert row needs.

Knows nothing about the database on purpose — it only shapes data.
Part C wires this into tasks.py to create rows. Step 3 will reuse
the SAME functions to decide insert-vs-update.
"""

from src.ml.explain import explainer

TOP_METRICS = 3


def build_fingerprint(service_name: str) -> str:
    """The dedup key — what makes two alerts 'the same problem'."""
    return f"multivariate:{service_name}"


def pick_triggering_metrics(all_features: dict, top_k: int = TOP_METRICS) -> list:
    """Which metrics looked worst right now, by z-score. No new computation —
    all_features is already built in tasks.py before the ensemble runs."""
    ranked = sorted(
        all_features.items(),
        key=lambda kv: abs(kv[1]["z_score"]),
        reverse=True,
    )
    return [
        {
            "metric":  name,
            "z_score": round(feats["z_score"], 4),
            "value":   round(feats["value"], 4),
        }
        for name, feats in ranked[:top_k]
    ]


def build_detected_by(result: dict) -> dict:
    """Which detectors fired — LAYER 1 only, no XGBoost/SHAP involved."""
    total_available = sum(
        1 for key in ("zscore_value", "iforest_score", "lstm_error")
        if result.get(key) is not None
    )
    return {
        "votes":           result.get("votes", 0),
        "total_available": total_available,
        "zscore":          bool(result.get("zscore_flag")),
        "iforest":         bool(result.get("iforest_flag")),
        "lstm":            bool(result.get("lstm_flag")),
    }


def build_escalation(result: dict) -> dict:
    """LAYER 2 — XGBoost + SHAP. Fail-open if the model isn't loaded."""
    explanation = explainer.explain(result)

    if not explanation.get("available"):
        return {
            "escalation_probability": None,
            "explanation_text":       None,
            "contributing_features":  None,
        }

    return {
        "escalation_probability": explanation["probability"],
        "explanation_text":       explanation["explanation_text"],
        "contributing_features":  explanation["contributing_features"],
    }


def _placeholder_severity(result: dict, probability) -> str:
    """TEMPORARY stub — Step 2 replaces this with the real mapping.
    Alert.severity is NOT NULL, so Part B needs *something* here to run
    at all. Deliberately crude so it's obvious it's not the final rule."""
    if probability is not None:
        if probability >= 0.70:
            return "CRITICAL"
        if probability >= 0.30:
            return "WARNING"
        return "INFO"
    return "CRITICAL" if result.get("votes", 0) >= 3 else "WARNING"


def build_alert_fields(service_name: str, host: str, timestamp,
                       result: dict, all_features: dict) -> dict:
    """Everything an Alert row needs, as a plain dict — ready for Alert(**fields)."""
    fields = {
        "fingerprint":        build_fingerprint(service_name),
        "service_name":       service_name,
        "host":               host,
        "metric_name":        None,          # NULL = multivariate
        "status":             "ACTIVE",
        "last_seen_at":       timestamp,
        "occurrence_count":   1,
        "anomaly_score":      result["ensemble_score"],
        "detected_by":        build_detected_by(result),
        "triggering_metrics": pick_triggering_metrics(all_features),
    }
    fields.update(build_escalation(result))
    fields["severity"] = _placeholder_severity(result, fields["escalation_probability"])
    return fields


if __name__ == "__main__":
    # quick manual test: python -m src.services.alerts
    # Uses fake data — no database, no live pipeline involved.
    fake_result = {
        "votes": 2, "ensemble_score": 0.667,
        "zscore_value": 3.9, "zscore_flag": True,
        "iforest_score": -0.01, "iforest_flag": True,
        "lstm_error": 0.05, "lstm_flag": False,
        "is_anomaly": True, "model_version": "v1",
    }
    fake_features = {
        "cpu_usage":           {"value": 92.1, "z_score": 3.9},
        "memory_usage":        {"value": 71.0, "z_score": 1.1},
        "request_latency_ms":  {"value": 340.0, "z_score": 3.2},
    }
    import datetime
    fields = build_alert_fields(
        service_name="payment-service", host="prod-server-01",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        result=fake_result, all_features=fake_features,
    )
    for k, v in fields.items():
        print(f"{k:<22} {v}")