# Phase 5 — Alert & Incident Management

**Project:** Intelligent Anomaly Detection & Incident Prediction Platform
**Status:** 🚧 In progress — Step 1 design locked, implementation not started
**Goal:** Turn the ML pipeline's per-reading verdicts into a small number of long-lived, actionable alerts — grouped into incidents, with a real lifecycle, ready for the Phase 6 dashboard and Phase 7 monitoring to consume.

---

## 1. What Phase 5 Delivers

Phase 3 answers *"is this reading abnormal right now?"* Phase 4 answers *"is it heading somewhere bad, and why?"* Neither of them tells anyone. Phase 5 is the part that actually **raises the alarm and keeps track of it**.

```
metric reading arrives
        ↓
src/workers/tasks.py → process_metrics()
        ↓
score_reading()              ← Phase 3: 3 detectors vote
        ↓
   is_anomaly ?
        ↓ yes
🆕 explainer.explain()       ← Phase 4, called live for the FIRST time
        ↓
🆕 alert creation / dedup    ← Phase 5 Step 1 + Step 3
        ↓
🆕 Alert row (long-lived)  →  🆕 Incident grouping (Step 4)
        ↓
🆕 /api/v1/alerts, /api/v1/incidents   → Phase 6 React dashboard
🆕 Prometheus counters                 → Phase 7 Grafana
```

Before this phase, `src/models/alert.py` had existed since Phase 1 with **zero rows ever written to it**, and `src/ml/explain.py` had existed since Phase 4 **without a single caller**. Phase 5 connects both.

---

## 2. The Core Problem Found During Design

The Phase 1 `alerts` table was designed for a system that never got built.

Check the very first migration, `88789e7a81c0_create_metrics_and_alerts_tables.py` (`down_revision = None`, created 2026-08-16 — before any ML existed):

```python
sa.Column('service_name',  sa.String(100), nullable=False),
sa.Column('host',          sa.String(100), nullable=False),
sa.Column('metric_name',   sa.String(100), nullable=False),
sa.Column('anomaly_score', sa.Float(),     nullable=False),
```

One service, one host, **one metric**, **one score**, **one timestamp**. That is the shape of a classic threshold alert — *"cpu_usage on prod-server-01 crossed 90%."* Entirely reasonable for Phase 1 to assume.

But Phase 3 built something different. `src/ml/ensemble.py`'s `score_reading()` takes all 28 features (7 metrics × 4 features) **at once** and returns **one verdict about the whole service**. Phase 3 already hit this wall and solved it — the comment in `src/models/anomaly_score.py` records the decision:

```python
# NULL means this row is a MULTIVARIATE score (all 7 metrics together).
```

The `alerts` table never got that update, because nothing wrote to it until now.

### 2.1 It is three mismatches, not one

| Column | Reality in the built system |
|---|---|
| `service_name`, `host` | ✅ Fine — available |
| `status` | ✅ Fine — Step 5 drives it |
| `severity` | ⬜ Nothing produces INFO/WARNING/CRITICAL yet → Step 2 |
| `metric_name` | ❌ **Mismatch** — the verdict is multivariate, not per-metric |
| `anomaly_score` | ❌ **Ambiguous** — `ensemble_score` (a vote fraction: only 0.33 / 0.67 / 1.0) or XGBoost's escalation probability? Two different numbers, one column |
| `explanation_text` | ❌ **Overloaded** — the detection summary and the SHAP sentence both want it |
| `contributing_features` | ✅ Fine — JSON, SHAP output fits as-is |

### 2.2 The bigger miss: an alert is an interval, not a moment

The `Alert` model has exactly **one** timestamp, `created_at`. But Step 3's whole purpose is dedup — *"if there's already an ACTIVE alert for this service, update it instead of creating a new one."*

**Update what?** With the Phase 1 schema there is nothing meaningful to update.

Take a `memory_leak` episode — 43 anomalous readings over 20 minutes:

```
readings   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
           14:03 ──────────────────────────────── 14:23

NO DEDUP           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   43 rows
                   one alert per reading — pure spam

DEDUP, Phase 1     ▓                                  1 row
schema             "alert created 14:03"
                   Still happening? Unknown.
                   How long? Unknown. How bad? Unknown.

DEDUP + lifecycle  ▓━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▓   1 row
  ← CHOSEN         started 14:03 · last seen 14:23
                   43 readings · CRITICAL · still active
```

An alert is a **fire alarm that is still ringing**, not a note saying *"an alarm started at 2pm."* The Phase 1 schema can only record the second one.

---

## 3. The Decision

> **Model an `Alert` as an interval with a lifecycle, not a row stamped at one instant — and store detection facts as queryable data, not as prose.**

Guiding rule used throughout: **strings are for humans to read, columns are for code to query.** Steps 3, 4 and Phase 7 all need to *search* this data. You cannot run `COUNT` or `GROUP BY` on an English sentence.

### 3.1 Why the fix happens now, not later

- **Phase 5 is the last phase that designs data.** Phase 6 is React, Phase 7 is Prometheus/Grafana, Phase 8 is tests + auth. All three *consume* this schema; none of them redesign it.
- **You cannot backfill what you never recorded.** If alerts run for three weeks without `last_seen_at`, those three weeks are permanently missing it. No migration recovers that.
- **The migration is happening anyway** — Step 4's `Incident` model needs a foreign key on `Alert`. Fixing the columns in that same migration costs ~30 minutes extra.

### 3.2 The Phase 6 dashboard card proves every column

This is the card Phase 6 will render:

```
┌────────────────────────────────────────────────────┐
│ 🔴 CRITICAL      payment-service                   │
│                                                    │
│ Started 14:03 · Still active (20 min)              │
│ 43 anomalous readings                              │
│                                                    │
│ Detected by 2 of 3:  ✓Z-Score ✓IForest ✗LSTM       │
│                                                    │
│ Worst metrics:  memory_usage        (z = 4.1)      │
│                 request_latency_ms  (z = 3.2)      │
│                                                    │
│ ⚠ 80% likely to become severe — mainly because     │
│   the readings were far outside their normal       │
│   statistical range                                │
│                                                    │
│      [ Acknowledge ]        [ Resolve ]            │
└────────────────────────────────────────────────────┘
```

Every line maps to exactly one column — none are decorative:

```
🔴 CRITICAL                → severity
payment-service            → service_name
Started 14:03              → created_at
Still active (20 min)      → status + last_seen_at   ← missing in Phase 1
43 anomalous readings      → occurrence_count        ← missing in Phase 1
Detected by 2 of 3 ✓✓✗     → detected_by             ← missing in Phase 1
Worst metrics + z-scores   → triggering_metrics      ← missing in Phase 1
80% likely to become...    → escalation_probability  ← missing in Phase 1
mainly because...          → explanation_text
[Acknowledge]              → acknowledged_at/_by     ← missing in Phase 1
[Resolve]                  → resolved_at             ← missing in Phase 1
```

The Phase 1 schema cannot render this card. That is the argument in one picture.

---

## 4. The `Alert` Schema (Phase 5)

### 4.1 Identity and grouping

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | exists |
| `incident_id` | UUID, nullable, FK | Step 4 — links alerts into one incident |
| `fingerprint` | String, indexed | the dedup key |

`fingerprint` follows the Prometheus Alertmanager pattern: rather than dedup via a growing `WHERE service = ? AND metric = ? AND ...` clause, compute one short string identifying *"this kind of problem in this place"* and look it up. Today it is effectively `service_name`, because the detector is multivariate and emits one verdict per service. If per-metric alerts are ever added, the fingerprint absorbs that with **no schema change** — and it makes the partial unique index in §6.2 possible.

### 4.2 Lifecycle — the part that was missing

| Column | Type | Notes |
|---|---|---|
| `status` | String | exists — ACTIVE / ACKNOWLEDGED / RESOLVED |
| `created_at` | DateTime | exists — = when the problem started |
| `last_seen_at` | DateTime | 🆕 most recent anomalous reading |
| `resolved_at` | DateTime, nullable | 🆕 when it stopped |
| `acknowledged_at` | DateTime, nullable | 🆕 |
| `acknowledged_by` | String, nullable | 🆕 **plain text, not a FK** — see §7 |
| `occurrence_count` | Integer | 🆕 how many readings this alert covers |

These six turn Step 3 from *"overwrite something"* into something real:

```
find ACTIVE alert by fingerprint
    ├─ last_seen_at    = now()
    ├─ occurrence_count += 1
    └─ severity        = max(current, new)   ← only ever goes UP
```

The 43-reading memory leak becomes one row reading *"started 14:03, last seen 14:23, 43 readings, still active, peaked at CRITICAL."*

### 4.3 Detection facts (queryable, not prose)

| Column | Type | Notes |
|---|---|---|
| `service_name`, `host` | String | exist |
| `metric_name` | String, **nullable** | 🔄 `None` = multivariate, matching `AnomalyScore` |
| `triggering_metrics` | JSON | 🆕 `[{"metric":"memory_usage","z_score":4.1}, ...]` |
| `detected_by` | JSON | 🆕 `{"votes":2,"total_available":3,"zscore":true,"iforest":true,"lstm":false}` |
| `anomaly_score` | Float | exists — the ensemble vote fraction |
| `severity` | String | exists — Step 2 fills it |

`triggering_metrics` needs **no new computation** — those z-scores already sit in `all_features` inside `process_metrics()` before the ensemble even runs.

`total_available` inside `detected_by` matters because of the fail-open design in `ensemble.py`: if the LSTM model file fails to load on one cloud worker, *"2 of 2 agreed"* is a much stronger signal than *"2 of 3"* — a distinction lost if only the vote count is stored.

### 4.4 Prediction (Phase 4's output, live at last)

| Column | Type | Notes |
|---|---|---|
| `escalation_probability` | Float, nullable | 🆕 XGBoost probability — **separate** from `anomaly_score` |
| `explanation_text` | Text | exists — SHAP sentence **only** |
| `contributing_features` | JSON | exists — SHAP output, already the right shape |

Nullable because the escalation model may not be loaded — same fail-open habit as `registry.py`.

### 4.5 Notifications (Step 7)

| Column | Type | Notes |
|---|---|---|
| `last_notified_at` | DateTime, nullable | 🆕 |

One column that prevents the most common alerting disaster: an ongoing problem paging someone every few seconds for four hours. Step 7 becomes *"only notify if `last_notified_at` is older than N minutes"* rather than needing its own migration.

---

## 5. Where Each Column Is Used

```
COLUMN                    P5      P6       P7        P8
                        alerts  React   Grafana  tests+auth
──────────────────────────────────────────────────────────
id, service_name, host    ✓       ✓        ✓         ✓
severity                  ✓       ✓        ✓
status                    ✓       ✓        ✓
fingerprint               ✓                          ✓
incident_id               ✓       ✓
created_at                ✓       ✓        ✓
last_seen_at              ✓       ✓        ✓
resolved_at               ✓       ✓        ✓
occurrence_count          ✓       ✓        ✓
acknowledged_at / _by     ✓       ✓                  ✓
metric_name (nullable)    ✓       ✓
triggering_metrics        ✓       ✓
detected_by               ✓       ✓        ✓
anomaly_score             ✓       ✓        ✓
escalation_probability    ✓       ✓        ✓
explanation_text          ✓       ✓
contributing_features     ✓       ✓
last_notified_at          ✓                          ✓
```

Phase 7 is the strongest justification. Prometheus/Grafana works by counting; with these columns each panel is one query:

```
active_alerts_total       →  COUNT WHERE status = 'ACTIVE'
alert_age_seconds         →  now() - created_at
mean_time_to_resolve      →  avg(resolved_at - created_at)
detector_agreement_rate   →  from detected_by
```

Without `resolved_at`, *"mean time to resolve"* is not a hard query — it is an **impossible** one. The data does not exist.

---

## 6. Two Production Behaviours That Are Not Columns

### 6.1 Alerts must close themselves

Nothing in the original plan ever sets `status = RESOLVED` automatically. Within a week the dashboard becomes a graveyard of ACTIVE alerts for problems that fixed themselves — worse than no dashboard, because people stop trusting it.

The machinery already exists: Celery Beat runs `aggregate_metrics` on a schedule (`celerybeat-schedule`). Add a second periodic task:

```
every 1 minute:
    for each ACTIVE alert:
        if last_seen_at older than ~2 minutes:
            status      = RESOLVED
            resolved_at = now()
```

This is why `last_seen_at` earns its place — it is not only for display, it is what makes auto-close possible. Tracked as **Step 3b** below.

### 6.2 Duplicate alerts must be stopped by the database, not Python

On cloud servers several Celery workers run concurrently:

```
worker A:  "any ACTIVE alert for payment-service?" → no   ┐
worker B:  "any ACTIVE alert for payment-service?" → no   ├ same instant
worker A:  INSERT alert                                   │
worker B:  INSERT alert          ← two active alerts      ┘
```

Application code cannot reliably prevent this. Postgres can, with a **partial unique index** — unique on `fingerprint`, but only over rows `WHERE status = 'ACTIVE'`. The database then physically refuses the second insert. A few lines in the migration; the difference between "works on my laptop" and "works with 4 workers."

---

## 7. Auth Overlap With Phase 8

The original Phase 5 plan listed *Step 6 — JWT auth*, but the roadmap also has *Phase 8 — Production Hardening (tests + auth)*. These overlap.

**Decision:** Phase 5 does the **minimum** — one hardcoded admin login, enough to protect the acknowledge/resolve endpoints. The real user system moves to Phase 8.

**Schema consequence:** `acknowledged_by` is a **plain String column, not a foreign key to a `User` table**, because no `User` table exists until Phase 8. Building an FK to a table that does not exist would block Phase 5 entirely.

---

## 8. Key Design Decisions

- **Alert = interval, not instant** — six lifecycle columns so dedup has something real to update, and so Phases 6/7 can answer "how long", "how bad", "still happening".
- **`metric_name` nullable, `None` = multivariate** — matches the convention `AnomalyScore` already established in Phase 3, rather than inventing a second one.
- **`triggering_metrics` instead of guessing one "cause" metric** — records what was actually observed (the worst z-scores) rather than writing a guess into a column that reads like a fact. Anyone later querying `WHERE metric_name = 'cpu_usage'` would otherwise believe the system was certain. It was not.
- **`detected_by` as structured JSON, not a sentence** — the *"2 of 3: Z-Score and Isolation Forest"* line is built in the API response at display time. Storing prose would make Step 4 and Phase 7 parse strings.
- **`anomaly_score` and `escalation_probability` split** — they answer different questions: *"how many detectors agree right now"* vs *"how likely is this to get worse."* One column cannot mean both.
- **`fingerprint` column** — makes dedup a single indexed lookup, and enables the partial unique index.
- **Severity only ever rises while ACTIVE** — keeps the peak for free, avoiding a separate `peak_severity` column.
- **Auto-resolve via Celery Beat** — an alerting system that cannot close alerts is not trustworthy.
- **Partial unique index for concurrency** — correctness enforced by the database, since this is intended for multi-worker cloud deployment.
- **One migration, not several** — Step 4's `Incident` FK forces a migration anyway; backfilling never-recorded data is impossible.

### 8.1 Deliberately excluded

| Not added | Why |
|---|---|
| `peak_severity` column | Severity-only-rises rule keeps the peak for free |
| Separate alert-history / audit table | `occurrence_count` + timestamps cover what would actually be queried |
| Free-form `labels` JSON (Prometheus style) | Prometheus needs it for arbitrary targets; here there are exactly 7 known metrics |
| Silences / maintenance windows | Real feature in mature systems, but a whole feature, not a column |
| `User` foreign key | Phase 8's job — see §7 |

---

## 9. Revised Phase 5 Step List

| Step | What | Status |
|---|---|---|
| 1 | Alert creation trigger + the schema above | 🔒 design locked |
| 2 | Severity mapping (probability / votes → INFO / WARNING / CRITICAL) | ⬜ |
| 3 | Dedup by `fingerprint` | ⬜ |
| 3b | 🆕 Auto-resolve Celery Beat task | ⬜ |
| 4 | `Incident` model + grouping | ⬜ |
| 5 | Lifecycle endpoints (acknowledge / resolve) | ⬜ |
| 6 | Minimal auth only — full auth deferred to Phase 8 | ⬜ |
| 7 | Notifications (`last_notified_at` makes this cheap) | ⬜ |
| 8 | `/api/v1/alerts`, `/api/v1/incidents`, wire into `main.py` | ⬜ |

Steps 3, 5 and 7 all became **easier** as a result of Step 1's schema work — each needs fields that now already exist rather than its own migration.

---

## 10. Step 1 in Detail — Where the Alert Is Created

The trigger point is inside `src/workers/tasks.py` → `process_metrics()`, immediately after `score_reading()` returns, guarded by `if result["is_anomaly"]:`. That is the only place in the codebase that already knows *"something was just flagged"* — nothing outside `tasks.py` needs to care.

### 10.1 Phase 4 plugs in with no adapter

`explain()` in `src/ml/explain.py` expects a dict keyed by its `feature_cols`:

```python
["votes", "ensemble_score", "zscore_value", "iforest_score", "lstm_error"]
```

`score_reading()` in `src/ml/ensemble.py` already returns exactly those keys. The `result` dict already sitting in `tasks.py` **is** the shape `explain()` wants — by design, from Phase 4 §9. The wiring is one import and one call.

### 10.2 Two ML layers — not to be confused

| | Layer 1 — Detection | Layer 2 — Escalation |
|---|---|---|
| Files | `registry.py`, `ensemble.py` | `explain.py` |
| Models | Z-Score, Isolation Forest, LSTM | XGBoost + SHAP |
| Question | *"Is this reading abnormal?"* | *"Given it is abnormal, will it get worse?"* |
| Input | 28 raw features | Layer 1's 5 output numbers |
| Output | `votes`, 3 × `_flag`, `is_anomaly` | `probability`, SHAP sentence |
| Fills | `detected_by`, `anomaly_score` | `escalation_probability`, `explanation_text`, `contributing_features` |

The *"2 of 3 detectors agreed"* summary comes from **Layer 1 only** — no XGBoost, no SHAP, just reading booleans already present in `result`. SHAP's sentence explains the **escalation probability**, not which detectors fired. Two different questions, two different models.

---

## 11. Known Limitations (carried into Phase 5)

| Limitation | Impact | Planned handling |
|---|---|---|
| Detector is multivariate — cannot separate two simultaneous problems on one service | Dedup granularity is per-service, not per-problem | Correct given what the detector knows; `triggering_metrics` preserves the detail |
| Escalation model F1 ≈ 0.452 (Phase 4 §7) | `escalation_probability` is a modest signal, not a strong one | Nullable + fail-open; revisit at Phase 7 retraining |
| Auto-resolve timeout is a fixed guess (~2 min) | May close flapping alerts early, or hold stale ones | Tune once real alert data exists |
| Minimal auth only | Not production-secure | Phase 8 |
