"""
DATA PREPARATION
================
Reads metric readings from the database + the ground-truth JSONL file,
computes the same 4 features that features.py computes in the live pipeline,
then saves train/test arrays to ml_models/ ready for model training.

Run once before training any model:
    python -m src.ml.data_prep

What gets saved to ml_models/:
    X_train.npy       — 2-D feature matrix for training  (Isolation Forest)
    X_test.npy        — 2-D feature matrix for testing
    X_train_seq.npy   — 3-D sequences for training       (LSTM)
    X_test_seq.npy    — 3-D sequences for testing
    y_train.npy       — ground-truth labels (1=anomaly, 0=normal) for training
    y_test.npy        — ground-truth labels for testing
    y_train_seq.npy   — labels aligned to LSTM sequences (training)
    y_test_seq.npy    — labels aligned to LSTM sequences (testing)
    scaler.pkl        — StandardScaler fitted on training data only
    feature_cols.json — the 28 feature names in the correct order
"""

import json
import pickle
from collections import deque
from pathlib import Path

from alembic.command import history
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.db.session import SessionLocal

# ── Configuration ──────────────────────────────────────────────────────────────

GROUND_TRUTH_PATH = Path("simulator/ground_truth.jsonl")
OUTPUT_DIR        = Path("ml_models")

TRAIN_RATIO    = 0.80  # first 80% of time  → training set
                       # last  20% of time  → test set
ROLLING_WINDOW = 10    # must match WINDOW_SIZE in features.py
SEQ_LEN        = 20    # how many consecutive readings the LSTM looks at

# The 7 metrics the simulator sends every tick
RAW_METRICS = [
    "cpu_usage",
    "memory_usage",
    "request_latency_ms",
    "requests_per_sec",
    "error_rate",
    "db_connections",
    "disk_usage",
]


# ── Step A: Load raw readings from the database ────────────────────────────────

def load_from_db() -> pd.DataFrame:
    """
    metric_datapoints stores ONE ROW PER METRIC per timestamp.
    e.g. one tick produces 7 rows (cpu, mem, latency, …).

    We pivot those 7 rows into ONE wide row so we can compute
    cross-metric features later.

    Result shape: [n_ticks, 9] columns →
        timestamp | service_name | host | cpu_usage | memory_usage | …
    """
    print("Loading from database …")
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT timestamp, service_name, host, metric_name, value "
            "FROM metric_datapoints "
            "ORDER BY service_name, timestamp"
        )).fetchall()
    finally:
        db.close()

    if not rows:
        raise RuntimeError(
            "metric_datapoints is empty. "
            "Run the simulator first:\n"
            "  python -m simulator.runner --time-scale 30 --duration 90"
        )

    df = pd.DataFrame(rows, columns=["timestamp", "service_name", "host",
                                     "metric_name", "value"])

    # Pivot: rows become columns
    # Before pivot: 7 rows for one tick  (metric_name=cpu_usage, value=42.1 …)
    # After pivot:  1 row  for one tick  (cpu_usage=42.1, memory_usage=58.3 …)
    df = df.pivot_table(
        index=["timestamp", "service_name", "host"],
        columns="metric_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    df.columns.name = None  # remove the leftover "metric_name" header label

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["service_name", "timestamp"]).reset_index(drop=True)

    print(f"  {len(df):,} ticks across {df['service_name'].nunique()} services")
    return df


# ── Step B: Load ground-truth labels ──────────────────────────────────────────

def load_ground_truth() -> pd.DataFrame:
    """
    The simulator wrote one label per tick to a JSONL file.
    We use is_anomaly (True/False) as the label for supervised evaluation.
    """
    print(f"Loading ground truth from {GROUND_TRUTH_PATH} …")
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Not found: {GROUND_TRUTH_PATH}\n"
            "Run the simulator with --ground-truth simulator/ground_truth.jsonl"
        )

    records = [json.loads(line) for line in GROUND_TRUTH_PATH.open()]
    gt = pd.DataFrame(records)
    gt["timestamp"] = pd.to_datetime(gt["timestamp"], utc=True)

    total     = len(gt)
    anomalous = gt["is_anomaly"].sum()
    print(f"  {total:,} labels — {anomalous:,} anomalous ({100*anomalous/total:.1f}%)")
    return gt


# ── Step C: Compute the same 4 features as features.py ────────────────────────

def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    For every metric, compute rolling_avg, z_score, and rate_of_change
    the same way features.py does it in the live pipeline.

    We process each (service, metric) pair in chronological order so the
    rolling window fills exactly as it would in production.

    Final feature count: 7 metrics × 4 features = 28 columns.
    """
    print("Computing features …")
    feature_cols = []

    for metric in RAW_METRICS:
        if metric not in df.columns:
            print(f"  ⚠  {metric} not found in DB, skipping")
            continue

        rolling_avg_col   = f"{metric}_rolling_avg"
        z_score_col       = f"{metric}_z_score"
        rate_of_change_col = f"{metric}_rate_of_change"
        feature_cols.extend([metric, rolling_avg_col, z_score_col, rate_of_change_col])

        # Process each service separately — rolling windows must not bleed
        # across services (payment-service's history ≠ user-service's history)
        for svc, grp_idx in df.groupby("service_name").groups.items():
            history: deque = deque(maxlen=ROLLING_WINDOW)
            rolling_avgs, z_scores, rates = [], [], []

            for val in df.loc[grp_idx, metric]:
                # Rate of change
                roc = float(val - history[-1]) if history else 0.0

                # Rolling average
                ravg = float(sum(history) / len(history)) if history else float(val)

                # Z-score
                Z_SCORE_MIN_STD = 1e-4  # below this, treat the metric as "not really varying" — prevents
                        # exploding z-scores when std is technically nonzero but tiny
                        # (this is exactly what broke error_rate, which naturally hovers
                        # around 0.002 and gets rounded to 4 decimal places)

                if len(history) >= 2:
                    mean = ravg
                    std  = float(np.std(list(history)))
                    zs   = (float(val) - mean) / std if std > Z_SCORE_MIN_STD else 0.0
                else:
                    zs = 0.0

                rolling_avgs.append(round(ravg, 4))
                z_scores.append(round(zs, 4))
                rates.append(round(roc, 4))
                history.append(float(val))

            df.loc[grp_idx, rolling_avg_col]    = rolling_avgs
            df.loc[grp_idx, z_score_col]        = z_scores
            df.loc[grp_idx, rate_of_change_col] = rates

    print(f"  Feature matrix: {len(df):,} rows × {len(feature_cols)} columns")
    return df, feature_cols


# ── Step D: Merge labels, split chronologically, scale, build sequences ────────

def prepare_and_save(df: pd.DataFrame, gt: pd.DataFrame,
                     feature_cols: list[str]) -> None:

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Join metrics with labels on (timestamp, service_name) ---
    # Round to the nearest second so tiny clock drift doesn't break the join
    df["_key"] = df["timestamp"].dt.round("1s").astype(str) + "|" + df["service_name"]
    gt["_key"] = gt["timestamp"].dt.round("1s").astype(str) + "|" + gt["service_name"]

    merged = df.merge(
        gt[["_key", "is_anomaly", "anomaly_type"]],
        on="_key", how="left"
    )
    merged["is_anomaly"] = merged["is_anomaly"].fillna(False).astype(bool)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    print(f"\nAfter merge: {len(merged):,} rows  "
          f"({merged['is_anomaly'].sum():,} anomalous)")

    # --- Chronological split ---
    # WHY NOT RANDOM?
    # If we shuffle, the model trains on data from Tuesday to predict Monday.
    # In production it will only ever see the past — never the future.
    # A random split leaks future information and makes accuracy look better
    # than it really is. Always split time-series data by time.
    split_at = int(len(merged) * TRAIN_RATIO)
    train = merged.iloc[:split_at]
    test  = merged.iloc[split_at:]

    print(f"Train: {len(train):,} rows  ({train['is_anomaly'].sum():,} anomalous)")
    print(f"Test:  {len(test):,} rows  ({test['is_anomaly'].sum():,} anomalous)")

    X_train_raw = train[feature_cols].values.astype(np.float32)
    X_test_raw  = test[feature_cols].values.astype(np.float32)
    y_train     = train["is_anomaly"].values.astype(np.int8)
    y_test      = test["is_anomaly"].values.astype(np.int8)

    # --- Scale ---
    # StandardScaler makes every feature have mean=0 and std=1.
    # This stops one metric (e.g. request_latency_ms ~80) from dominating
    # another (e.g. error_rate ~0.002) just because of different units.
    #
    # CRITICAL: fit() only on training data.
    # If we fit on test data too, the scaler "peeks" at the future.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # learn mean/std from train
    X_test  = scaler.transform(X_test_raw)        # apply SAME stats to test

    # --- Build LSTM sequences ---
    # The LSTM doesn't look at one row — it looks at a WINDOW of rows.
    # SEQ_LEN=20 means "show the model the last 20 readings, then predict
    # whether reading 21 is anomalous".
    #
    # Shape: [n_samples, SEQ_LEN, n_features]
    #   e.g. [4800, 20, 28]
    def make_sequences(X: np.ndarray) -> np.ndarray:
        return np.array(
            [X[i - SEQ_LEN : i] for i in range(SEQ_LEN, len(X))],
            dtype=np.float32
        )

    X_train_seq = make_sequences(X_train)
    X_test_seq  = make_sequences(X_test)
    # Label = the label for the LAST reading in each window
    y_train_seq = y_train[SEQ_LEN:]
    y_test_seq  = y_test[SEQ_LEN:]

    # --- Save ---
    np.save(OUTPUT_DIR / "X_train.npy",     X_train)
    np.save(OUTPUT_DIR / "X_test.npy",      X_test)
    np.save(OUTPUT_DIR / "X_train_seq.npy", X_train_seq)
    np.save(OUTPUT_DIR / "X_test_seq.npy",  X_test_seq)
    np.save(OUTPUT_DIR / "y_train.npy",     y_train)
    np.save(OUTPUT_DIR / "y_test.npy",      y_test)
    np.save(OUTPUT_DIR / "y_train_seq.npy", y_train_seq)
    np.save(OUTPUT_DIR / "y_test_seq.npy",  y_test_seq)

    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(OUTPUT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\n✅  Saved to ml_models/")
    print(f"   X_train.npy      shape {X_train.shape}")
    print(f"   X_test.npy       shape {X_test.shape}")
    print(f"   X_train_seq.npy  shape {X_train_seq.shape}  ← for LSTM")
    print(f"   X_test_seq.npy   shape {X_test_seq.shape}")
    print(f"   y_train.npy      {y_train.sum()} anomalies")
    print(f"   y_test.npy       {y_test.sum()} anomalies")
    print(f"   scaler.pkl       StandardScaler ({len(feature_cols)} features)")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics_df = load_from_db()
    gt_df      = load_ground_truth()
    df, feature_cols = compute_features(metrics_df)
    prepare_and_save(df, gt_df, feature_cols)