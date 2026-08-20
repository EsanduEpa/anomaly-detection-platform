"""
ESCALATION DATA PREPARATION  (Phase 4, Step 2 — v2, with trend features)
==========================================================================
Builds the training table for the XGBoost "will this incident turn severe?"
model. This is a DIFFERENT question from data_prep.py's table, so it needs
its own script — see the Phase 4 planning discussion for why.

Reuses two proven pieces from data_prep.py instead of duplicating them:
    load_from_db()      — pulls raw ticks out of Postgres
    compute_features()  — turns them into the same 28 features the
                           detectors were trained on

What's NEW here:
    1. We "replay" the history through the already-trained detectors
       (via src.ml.ensemble.score_reading — the exact function production
       uses) to get 5 real detector-output numbers per reading:
           votes, ensemble_score, zscore_value, iforest_score, lstm_error
    2. v2 ADDS 2 more features, both observable in real time (no peeking
       at the future or at simulator-only ground truth):
           ensemble_score_trend      — change in ensemble_score over the
                                        last 5 ticks for this service
           consecutive_anomalous_ticks — how many ticks in a row, up to
                                        and including now, the live
                                        ensemble has been flagging this
                                        service as anomalous
       Reasoning: the original 5 features only describe "how weird does
       THIS ONE reading look" — a snapshot. These 2 describe movement —
       is it getting worse, and how long has it already been going on.
    3. We join those replayed scores against the FULL ground-truth record
       (not just is_anomaly — this time we also keep severity, progress,
       and episode_id).
    4. We keep only "early" readings: is_anomaly=True and progress <= 0.5
       (see EARLY_PROGRESS_CUTOFF below — raised from 0.3 after the first
       version underperformed a naive baseline; 0.3 left too little
       visible signal for slow-ramping incidents).
    5. Label = 1 if that reading's episode eventually reached severity
       "severe", else 0.
    6. Split by EPISODE, ordered by time (first 80% of episodes → train,
       last 20% → test) — never split a single episode across train/test,
       and never shuffle randomly (same "no peeking at the future"
       discipline as data_prep.py's chronological split).

Run once, after data_prep.py has already been run at least once
(needs metric_datapoints populated, and needs ml_models/ to already
contain the trained scaler + Z-Score + Isolation Forest + LSTM files,
since we score through the real registry):

    python -m src.ml.escalation_prep

Saves to ml_models/:
    X_escalation_train.npy     — [n_train, 7] feature matrix
    X_escalation_test.npy      — [n_test, 7] feature matrix
    y_escalation_train.npy     — [n_train] labels (1 = turned out severe)
    y_escalation_test.npy      — [n_test] labels
    escalation_feature_cols.json   — the 7 column names, in order
    escalation_test_meta.csv       — episode_id/category/severity per test
                                       row, for later per-category analysis
                                       (same idea as 05_evaluation.ipynb's
                                       recall-by-scenario breakdown)
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.data_prep import (
    load_from_db,
    load_ground_truth,
    compute_features,
)
from src.ml.registry import registry
from src.ml.ensemble import score_reading

# ── Configuration ──────────────────────────────────────────────────────────

OUTPUT_DIR = Path("ml_models")

# "Early" = the first 50% of an incident's lifetime (progress runs 0.0 → 1.0
# inside episode-based scenarios). Raised from 0.3 to 0.5 after the v1
# model (0.3, 5 features) scored WORSE than a naive "always guess severe"
# baseline — too little signal was visible that early for slow-ramping
# (trend/cliff) incidents. Every one of the 107 episodes still has at
# least one reading at progress <= 0.5.
EARLY_PROGRESS_CUTOFF = 0.5

# 80% of episodes (earliest-starting first) → train, rest → test.
# Same "never let the model see the future" rule as data_prep.py, applied
# at the EPISODE level so no single incident is split across both sets.
TRAIN_EPISODE_RATIO = 0.80

# How many ticks back to look when computing ensemble_score_trend.
TREND_WINDOW = 5

ESCALATION_FEATURE_COLS = [
    "votes",
    "ensemble_score",
    "zscore_value",
    "iforest_score",
    "lstm_error",
]


# ── Step A: replay history through the trained detectors ───────────────────

def score_all_readings(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    df must already have the 28 feature columns (from compute_features) and
    be sorted by service_name, then timestamp — exactly what load_from_db()
    + compute_features() hand back.

    For every row, in chronological order per service, this replays the
    SAME two-step production sequence tasks.py uses per reading:
        1. append this reading to a rolling per-service history buffer
        2. call score_reading(raw_features, sequence_or_None)

    That keeps the LSTM feature honest: readings very early in the WHOLE
    simulation (before 20 ticks of history exist for that service) get
    lstm_error = NaN, exactly like a freshly-started production system
    would. score_reading()'s existing fail-open logic handles that
    automatically — we don't need to special-case it here.

    ALSO tracks, per service, a short history of ensemble_score (to compute
    the trend) and a running streak of consecutive anomalous ticks — both
    are things a real live system could observe about itself in the
    moment, so using them as features isn't leakage.
    """
    if feature_cols != registry.feature_cols:
        raise RuntimeError(
            "Recomputed feature columns don't match registry.feature_cols — "
            "the detectors were trained on a different column order than "
            "this script just built. Re-check RAW_METRICS / compute_features."
        )

    print("Replaying history through the trained detectors …")

    results = []
    seq_buffers: dict[str, deque] = {}
    trend_buffers: dict[str, deque] = {}
    streaks: dict[str, int] = {}
    seq_len = registry.lstm_seq_len or 20

    for svc, grp_idx in df.groupby("service_name").groups.items():
        buffer = seq_buffers.setdefault(svc, deque(maxlen=seq_len))
        trend_hist = trend_buffers.setdefault(svc, deque(maxlen=TREND_WINDOW))
        streaks.setdefault(svc, 0)

        for idx in grp_idx:
            raw_features = df.loc[idx, feature_cols].values.astype(np.float64)

            buffer.append(raw_features)
            sequence = np.array(buffer) if len(buffer) == seq_len else None

            result = score_reading(raw_features, sequence)

            # --- ensemble_score_trend: change vs TREND_WINDOW ticks ago ---
            if len(trend_hist) == TREND_WINDOW:
                result["ensemble_score_trend"] = result["ensemble_score"] - trend_hist[0]
            else:
                result["ensemble_score_trend"] = np.nan  # not enough history yet
            trend_hist.append(result["ensemble_score"])

            # --- consecutive_anomalous_ticks: running streak, this service ---
            if result["is_anomaly"]:
                streaks[svc] += 1
            else:
                streaks[svc] = 0
            result["consecutive_anomalous_ticks"] = streaks[svc]

            results.append(result)

    scores_df = pd.DataFrame(results)
    # score_reading() returns a column called "is_anomaly" too (the live
    # ensemble verdict for THIS reading). Ground truth also has a column
    # called "is_anomaly" (whether this reading is truly inside an
    # episode). Rename the ensemble's version now so the merge in
    # build_labelled_table() can't silently collide the two — we only
    # ever want ground truth's is_anomaly for filtering.
    scores_df = scores_df.rename(columns={"is_anomaly": "ensemble_is_anomaly"})

    scored = pd.concat(
        [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
    )
    print(f"  Scored {len(scored):,} readings")
    return scored


# ── Step B: join with full ground truth, keep only early anomalous rows ────

def build_labelled_table(scored: pd.DataFrame) -> pd.DataFrame:
    print("Joining with ground truth (severity, progress, episode_id) …")

    gt = load_ground_truth()

    before = len(gt)
    gt["_key"] = gt["timestamp"].dt.round("1s").astype(str) + "|" + gt["service_name"]
    gt = gt.drop_duplicates(subset="_key", keep="first")
    if len(gt) < before:
        print(f"  Dropped {before - len(gt)} duplicate ground-truth keys before merging")

    scored["_key"] = (
        scored["timestamp"].dt.round("1s").astype(str) + "|" + scored["service_name"]
    )

    merged = scored.merge(
        gt[["_key", "is_anomaly", "anomaly_type", "category", "severity",
            "progress", "episode_id"]],
        on="_key", how="left",
    )
    merged["is_anomaly"] = merged["is_anomaly"].fillna(False).astype(bool)

    early = merged[
        merged["is_anomaly"]
        & (merged["progress"] <= EARLY_PROGRESS_CUTOFF)
        & merged["episode_id"].notna()
    ].copy()

    early["will_be_severe"] = (early["severity"] == "severe").astype(int)

    print(f"  {len(early):,} early-window readings across "
          f"{early['episode_id'].nunique()} episodes "
          f"({early['will_be_severe'].sum():,} labelled severe)")

    return early


# ── Step C: split by episode, chronologically, train vs test ───────────────

def split_and_save(early: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    episode_order = (
        early.groupby("episode_id")["timestamp"].min().sort_values().index.tolist()
    )
    split_at = int(len(episode_order) * TRAIN_EPISODE_RATIO)
    train_episodes = set(episode_order[:split_at])
    test_episodes = set(episode_order[split_at:])

    train = early[early["episode_id"].isin(train_episodes)]
    test = early[early["episode_id"].isin(test_episodes)]

    print(f"Train: {len(train):,} rows across {len(train_episodes)} episodes "
          f"({train['will_be_severe'].sum():,} severe)")
    print(f"Test:  {len(test):,} rows across {len(test_episodes)} episodes "
          f"({test['will_be_severe'].sum():,} severe)")

    X_train = train[ESCALATION_FEATURE_COLS].values.astype(np.float32)
    X_test = test[ESCALATION_FEATURE_COLS].values.astype(np.float32)
    y_train = train["will_be_severe"].values.astype(np.int8)
    y_test = test["will_be_severe"].values.astype(np.int8)

    np.save(OUTPUT_DIR / "X_escalation_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_escalation_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_escalation_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_escalation_test.npy", y_test)

    with open(OUTPUT_DIR / "escalation_feature_cols.json", "w") as f:
        json.dump(ESCALATION_FEATURE_COLS, f, indent=2)

    test[["episode_id", "category", "anomaly_type", "severity"]].to_csv(
        OUTPUT_DIR / "escalation_test_meta.csv", index=False
    )

    print("\n✅  Saved to ml_models/")
    print(f"   X_escalation_train.npy  shape {X_train.shape}")
    print(f"   X_escalation_test.npy   shape {X_test.shape}")
    print(f"   y_escalation_train.npy  {y_train.sum()} severe / {len(y_train)} rows")
    print(f"   y_escalation_test.npy   {y_test.sum()} severe / {len(y_test)} rows")
    print(f"   escalation_feature_cols.json  {ESCALATION_FEATURE_COLS}")
    print(f"   escalation_test_meta.csv      {len(test)} rows")


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics_df = load_from_db()
    df, feature_cols = compute_features(metrics_df)
    scored = score_all_readings(df, feature_cols)
    early = build_labelled_table(scored)
    split_and_save(early)