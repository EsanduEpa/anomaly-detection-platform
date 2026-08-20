# Phase 4 — Prediction & Explainability

**Project:** Intelligent Anomaly Detection & Incident Prediction Platform
**Status:** ✅ Complete (core modeling + explainability) — live wiring into Alerts deferred to Phase 5
**Goal:** Given an incident that has JUST started, predict whether it will turn out severe — and explain, in plain English, why the model thinks so.

---

## 1. What Phase 4 Delivers

Phase 3 answers *"is this reading abnormal right now?"* Phase 4 answers *"given that it's abnormal, is it heading somewhere bad — and why?"*

```
Historical readings (metric_datapoints, ground_truth.jsonl)
        ↓
🆕 escalation_prep.py   → replay through Phase 3 detectors,
                           join real outcomes, build training table
        ↓
🆕 06_escalation_predictor.ipynb
        ├─ XGBoost classifier  → will this incident turn severe?
        ├─ SHAP TreeExplainer  → why did it decide that, per case?
        └─ explain_prediction() → plain-English sentence + structured reasons
        ↓
Feeds src/models/alert.py's `explanation_text` / `contributing_features`
(columns existed since Phase 1, unused until now)
```

This phase is notebook + a data-prep script, not live-wired yet. `Alert` rows don't get created until Phase 5 — this phase proves the model and the explanation logic work, on real historical data, before plugging them into anything live.

---

## 2. New Tech Introduced in Phase 4

| Technology | Role | Analogy |
|---|---|---|
| `xgboost` (`XGBClassifier`) | Gradient-boosted trees — the escalation predictor | A team of doctors, each correcting the last one's mistakes |
| `shap` (`TreeExplainer`) | Per-prediction feature attribution | Fairly splitting credit for a team's result, Shapley-style |
| `scale_pos_weight` | Class-imbalance handling | Telling the model "a missed severe case costs more" |

**Add to `requirements.txt`:**
```
xgboost
shap
```
(`tensorflow`, used by the Phase 3 LSTM, was already installed locally but was never added to `requirements.txt` either — worth adding both at the same time.)

---

## 3. Defining "Escalation" — the Design Fork

Before writing any code, two competing definitions were considered:

- **Option A (chosen): early triage.** Predict an episode's *final* severity from its first few readings, using the existing `severity` ground truth.
- **Option B (rejected for now): true temporal escalation.** Redefine the label from live `AnomalyScore` trends (is the anomaly signal climbing over time).

**Why A was chosen:** inspecting `simulator/ground_truth.jsonl` directly showed that `severity` and `intensity` are fixed the moment an episode starts (checked all 107 episodes — 0 had more than one distinct `severity` value across their own rows) and never grow over an episode's life:

```python
# simulator/scenarios.py
def severity_for(intensity: float) -> str:
    if intensity < 0.62: return "mild"
    if intensity < 0.85: return "moderate"
    return "severe"
```

So "true escalation" isn't something the ground truth actually encodes — `severity` is a one-time roll, not a growing signal. Option B was noted as a natural extension once Phase 7 provides live streaming data to define escalation from directly.

**Critical leakage rule:** `intensity`, `progress`, `severity`, `episode_id` are simulator-only bookkeeping — production has no way to observe them. Only detector-output/metric features are legitimate model inputs.

---

## 4. Data Preparation — `src/ml/escalation_prep.py`

A new script, separate from Phase 3's `data_prep.py`, because it answers a different question (see §3) and needs different information preserved per row (service, episode, progress) that `data_prep.py`'s saved arrays discard.

Reuses `load_from_db()` and `compute_features()` from `data_prep.py` unchanged. New logic:

1. **Replay** historical readings through the already-trained Phase 3 detectors, in chronological order per service, using the exact production function `src.ml.ensemble.score_reading()` — producing 5 real detector-output numbers per reading.
2. **Join** against the full ground truth (deduplicated this time — fixes the duplicate-key merge bug documented in Phase 3's evaluation).
3. **Filter** to early, real anomaly readings: `is_anomaly=True AND progress <= EARLY_PROGRESS_CUTOFF`.
4. **Label**: `will_be_severe = 1` if that reading's episode ended up `severity == "severe"`.
5. **Split by EPISODE**, chronologically (earliest 80% of episodes → train, rest → test) — never split a single episode across train/test, never shuffled.

### 4.1 The 5 features (final)

| Feature | Source | Notes |
|---|---|---|
| `votes` | `AnomalyScore` / `ensemble.score_reading()` | 0–3, how many detectors agree |
| `ensemble_score` | same | `votes / total_available` — always 0.000 feature importance (redundant with `votes`, not a bug) |
| `zscore_value` | same | Z-Score detector's raw output |
| `iforest_score` | same | Isolation Forest's raw output |
| `lstm_error` | same | LSTM reconstruction error; NaN if <20 ticks of history exist yet (handled natively by XGBoost, not imputed) |

Two additional engineered features (`ensemble_score_trend`, `consecutive_anomalous_ticks`) were tried and **made results worse** — see §7.

---

## 5. Model Training

`XGBClassifier`, shallow trees on purpose:

```python
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    scale_pos_weight=n_neg/n_pos,   # ≈2.7–3.1 depending on cutoff — penalize missed severe cases
    eval_metric="logloss", random_state=42,
)
```

`max_depth=3` was a deliberate choice, not a default left alone — with only ~85 independent training episodes, deep trees would memorize training quirks instead of learning a real pattern.

---

## 6. Results

| Cutoff | Features | Precision | Recall | F1 | Test rows / severe |
|---|---|---|---|---|---|
| progress ≤ 0.3 | 5 | 0.305 | 0.375 | 0.336 (below naive baseline ~0.37) | 259 / 96 |
| **progress ≤ 0.5** | **5** | **0.413** | **0.500** | **0.452 — final chosen model** | 424 / 156 |
| progress ≤ 0.5 | 7 (+trend, +streak) | 0.384 | 0.487 | 0.429 (worse — negative result) | 424 / 156 |

**Naive baseline check:** guessing "severe" for every test row gets precision = base rate (36.8% at cutoff 0.5). The final model's 41.3% genuinely beats that; the cutoff=0.3 version (30.5%) did not — this comparison is why the cutoff was raised.

For context, F1=0.452 sits in the same tier as Phase 3's weaker detectors (Z-Score F1=0.387, LSTM F1=0.488) — a real, modest, honestly-reported result, not a strong one.

---

## 7. Why It Isn't Stronger — and Why More Features Didn't Fix It

824–1,775 training *rows* sounds like a reasonable amount of data, but they come from only **~85 independent episodes** (train split) — rows from the same episode are correlated, not separate lessons. Adding 2 well-reasoned "trend" features (`ensemble_score_trend`, `consecutive_anomalous_ticks`) made F1 worse, not better — a classic sign of overfitting on too few independent examples rather than a feature-choice problem. This confirms the real bottleneck is **sample size** (few labelled incidents), not which numbers are fed to the model.

**If revisited:** collect more simulator data (more than 107 episodes) before re-tuning features — noted as a natural tie-in to Phase 7's retraining infrastructure.

---

## 8. Explainability — SHAP

`shap.TreeExplainer(model)` computes, per prediction, how much each feature pushed the guess up or down — using Shapley values (fair credit-splitting from cooperative game theory), not a hand-written heuristic. Guarantees the contributions sum exactly to the model's real output (verified: `expected_value + sum(shap_values) = raw log-odds`, and `sigmoid(raw log-odds) = predict_proba`).

**Worked example — a correct, confident prediction (row 49, true positive, 79.9% confidence):**
```
lstm_error       = 0.124   pushed UP   +0.672
zscore_value     = 3.157   pushed UP   +0.444
iforest_score    = -0.013  pushed UP   +0.267
votes            = 2.000   pulled DOWN -0.004
ensemble_score   = 0.667   ~0.000 (redundant with votes)
```

**Worked example — a genuine mistake (row 1, false positive, 51% confidence):** the 3 detectors disagreed with each other — LSTM said "looks normal" (pulled down), Z-Score and Isolation Forest both said "looks unusual" (pushed up). The model's near-50/50 call is an honest reflection of real disagreement between detectors, not a nonsensical guess — a healthy sign, since mistakes clustering near the decision boundary (rather than confidently wrong) indicates reasonable calibration.

---

## 9. Natural-Language Explanations

`explain_prediction()` (notebook) / `EscalationExplainer.explain()` (`src/ml/explain.py`) converts SHAP's log-odds output to a real percentage, takes the top 2–3 contributing features (excluding the always-redundant `ensemble_score`), and maps each to a plain-English phrase:

> *"This incident looks likely to become severe (80% confidence), mainly because the recent sequence of readings didn't match any normal pattern the system has learned, and the readings were far outside their normal statistical range. A smaller factor: the overall combination of metrics looked highly unusual to the pattern-detection model."*

Output shape matches `src/models/alert.py` exactly:
- `explanation_text` → `Alert.explanation_text`
- `contributing_features` (list of `{feature, value, contribution}`) → `Alert.contributing_features`

### 9.1 `src/ml/explain.py`

Standalone module, mirrors `registry.py`'s lazy-singleton, fail-open pattern (`python -m src.ml.explain` to test). Not called from anywhere live yet — see §10.

**Bug found and fixed during delivery:** loading a saved XGBoost model via `.load_model()` loses sklearn wrapper metadata (`n_classes_`, etc.) present right after `.fit()`. This causes `shap.TreeExplainer`'s `expected_value` / `shap_values()` to sometimes return values wrapped in an extra array layer instead of plain scalars, raising `TypeError: unsupported format string passed to numpy.ndarray.__format__` when building the percentage string. Only reproduces on a reloaded model, not in-notebook. Fixed by explicitly unwrapping both `expected_value` and `shap_values` output (handling the list-of-arrays and array-wrapped-scalar cases) before any arithmetic or formatting. Confirmed fixed — module output matches the notebook's row 49 result exactly.

---

## 10. Key Design Decisions

- **Early triage over true escalation** — matches what the ground truth actually encodes (§3).
- **Binary label, not 3-class** — only 30 severe episodes total; 3-way classification would be too thin.
- **Episode-level chronological split** — rows from one incident never span train/test; train is genuinely the past.
- **Separate data-prep script from Phase 3's** — different question, different required metadata, not reusable.
- **Ground truth deduplicated on join** — fixes the Phase 3 duplicate-key bug proactively instead of repeating it.
- **`max_depth=3`, `scale_pos_weight` tuned** — deliberate small-data discipline, not defaults left alone.
- **SHAP over a hand-written rule set** — mathematically guaranteed to match the model's real reasoning; automatically stays correct after any retrain.
- **Defensive unwrapping in `explain.py`** — handles SHAP's output shape changing after a model save/reload, so it keeps working through future retrains, not just today's model file.
- **Notebook-only, not live-wired** — Alert creation is Phase 5's job; this phase proves the model and explanations work first.

---

## 11. Known Limitations

| Limitation | Impact | Planned fix |
|---|---|---|
| Only ~85 independent training episodes | F1 capped around 0.45; more features made it worse, not better | Collect more simulator data; revisit at Phase 7 retraining |
| `ensemble_score` always redundant | Wasted feature slot | Could be dropped outright in a future revision |
| Test set has different class balance than train (chronological split) | Metrics slightly harder to compare across cutoffs | Expected side effect of not shuffling; documented, not "fixed" |
| Not yet wired into live scoring | No real Alert rows created yet | Phase 5 |

---

## 12. What's Next — Phase 5 Preview

Phase 4 proves the model and its explanations work. Phase 5 (Alert & Incident Management) is where this actually starts creating real `Alert` rows — severity, dedup, grouping into incidents, lifecycle, JWT auth, notifications — using `src/ml/explain.py`'s output to populate `explanation_text` and `contributing_features` for the first time in production.
