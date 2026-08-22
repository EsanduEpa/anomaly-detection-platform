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


def determine_severity(detected_by: dict, escalation_probability) -> str:
    """
    How urgent is this alert, right now, using only what's knowable
    in the moment (never the simulator's ground-truth severity —
    that's leakage, see docs/PHASE_4... §3).

    Primary signal: escalation_probability, from XGBoost + SHAP.
    Thresholds borrowed from real numbers already in hand:
      - 0.50 is the same cutoff Phase 4's own evaluation used to
        decide "will be severe" vs not.
      - Readings below 0.20 are meaningfully below the ~37% base
        rate of episodes that actually turned out severe.

    Fallback (escalation model unavailable): an Alert only ever gets
    created when a MAJORITY of available detectors already agreed
    (ensemble.py's required_votes rule), so "mild" isn't really a
    possible outcome here — the only distinguishing signal left is
    whether EVERY available detector agreed (unanimous) or just
    enough of them did.
    """
    if escalation_probability is not None:
        if escalation_probability >= 0.50:
            return "CRITICAL"
        if escalation_probability >= 0.20:
            return "WARNING"
        return "INFO"

    votes = detected_by.get("votes", 0)
    total_available = detected_by.get("total_available", 0)
    if total_available > 0 and votes == total_available:
        return "CRITICAL"
    return "WARNING"

def build_alert_fields(service_name: str, host: str, timestamp,
                       result: dict, all_features: dict) -> dict:
    """Everything an Alert row needs, as a plain dict — ready for Alert(**fields)."""
    detected_by = build_detected_by(result)
    escalation  = build_escalation(result)

    fields = {
        "fingerprint":        build_fingerprint(service_name),
        "service_name":       service_name,
        "host":               host,
        "metric_name":        None,          # NULL = multivariate
        "status":             "ACTIVE",
        "last_seen_at":       timestamp,
        "occurrence_count":   1,
        "anomaly_score":      result["ensemble_score"],
        "detected_by":        detected_by,
        "triggering_metrics": pick_triggering_metrics(all_features),
    }
    fields.update(escalation)
    fields["severity"] = determine_severity(detected_by, fields["escalation_probability"])
    return fields

def merge_alert_fields(existing_occurrence_count: int, existing_severity: str,
                       fresh_fields: dict) -> dict:
    """
    Combines a NEW reading's data with what's already stored on an ACTIVE
    alert for the same problem, instead of creating a duplicate row.

    Rule: keep the LATEST snapshot for almost everything — a human
    checking the alert wants to know what it looks like RIGHT NOW, not
    what it looked like when it first triggered.

    Exception: severity only ever RISES while an alert stays ACTIVE,
    never drops. This is how the alert keeps its peak severity for
    free, without a separate peak_severity column.
    """
    SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}

    merged = dict(fresh_fields)
    merged["occurrence_count"] = existing_occurrence_count + 1

    if SEVERITY_RANK[existing_severity] > SEVERITY_RANK[merged["severity"]]:
        merged["severity"] = existing_severity

    return merged

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


        print("\n--- merge_alert_fields test ---")
    fresh = build_alert_fields(
        service_name="payment-service", host="prod-server-01",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        result=fake_result, all_features=fake_features,
    )
    print("fresh severity:", fresh["severity"])

    merged_a = merge_alert_fields(existing_occurrence_count=5,
                                  existing_severity="CRITICAL",
                                  fresh_fields=fresh)
    print("existing=CRITICAL, fresh=%s  -> merged severity=%s, count=%s"
          % (fresh["severity"], merged_a["severity"], merged_a["occurrence_count"]))

    merged_b = merge_alert_fields(existing_occurrence_count=5,
                                  existing_severity="INFO",
                                  fresh_fields=fresh)
    print("existing=INFO, fresh=%s  -> merged severity=%s, count=%s"
          % (fresh["severity"], merged_b["severity"], merged_b["occurrence_count"]))