# Phase 3 — ML Anomaly Detection

**Project:** Intelligent Anomaly Detection & Incident Prediction Platform
**Status:** 🔨 In Progress — Step 0 (simulator) complete, Steps 1–9 pending
**Goal:** Score every incoming metric reading for how anomalous it is, using three independent detectors combined by majority vote, and persist those scores to the database.

---

## 1. What Phase 3 Delivers

Phase 2 built the pipeline. Phase 3 adds the brain.

```
Metrics arrive at POST /api/v1/metrics
        ↓
.delay() → Redis → Celery worker
        ↓
clean_metrics()      → fix bad/missing values          [Phase 2]
        ↓
compute_features()   → rolling avg, z-score, rate      [Phase 2]
        ↓
🆕 score_anomaly()   → 3 detectors + ensemble vote     [Phase 3]
        ↓
Save MetricDataPoint rows      (as before)
Save AnomalyScore row          🆕
```

The API contract does not change. The Celery wiring does not change. Phase 3 inserts **one step** into the existing task and **one new table**.

By the end of this phase the system can answer: *"Is this reading normal, and how confident are we?"* — with a measured precision and recall number to back it up.

---

## 2. New Tech Introduced in Phase 3

| Technology | Role | Analogy |
|---|---|---|
| scikit-learn `IsolationForest` | Unsupervised multi-metric anomaly detector | Spotting the person standing still in a moving crowd |
| PyTorch | Neural network framework for the LSTM | The engine under the hood |
| LSTM Autoencoder | Learns normal time-series patterns | A security camera that knows your office's routine |
| `StandardScaler` | Normalizes features to comparable ranges | Converting inches and kilograms to the same yardstick |
| `joblib` | Saves/loads scikit-learn models to disk | Freezing a trained brain for later |
| `torch.save` / `state_dict` | Saves/loads PyTorch model weights | Same, for the neural network |

**Add to `requirements.txt` before starting Step 4:**

```
torch==2.5.1
joblib==1.4.2
requests==2.32.3     # runner.py needs this; currently missing
```

`scikit-learn`, `pandas`, and `numpy` are already present.

---

## 3. The Three Detectors — And Why Three

### Analogy: the three doctors

A patient walks into a hospital. Instead of one doctor deciding, **three** examine them independently:

| Doctor | Method | Strength |
|---|---|---|
| Dr. Z-Score | How far is this from the recent average? | Instant, no training, always available |
| Dr. Isolation Forest | How isolated is this point from all others? | Sees all 7 metrics at once |
| Dr. LSTM | I've memorised your normal daily rhythm | Understands time and sequence |

A **head doctor (the Ensemble)** makes the final call: *if at least 2 of 3 agree it's an anomaly, it's flagged.*

```
    z-score:          ANOMALY  ✅
    Isolation Forest: ANOMALY  ✅        2 of 3  →  🚨 CONFIRMED
    LSTM:             normal   ❌

    z-score:          normal   ❌
    Isolation Forest: normal   ❌        1 of 3  →  ✅ false alarm, ignored
    LSTM:             ANOMALY  ✅
```

### 3.1 Z-Score Baseline — the ruler

Already computed in Phase 2's `features.py`. This detector just applies a threshold.

```
z = (value − rolling_avg) / standard_deviation

Normal:   CPU 38, 41, 40, 39  →  z ≈ 0.3   ✅
Anomaly:  CPU 38, 41, 40, 95  →  z = 4.2   🚨
```

**Threshold:** `|z| > 3.0` (about 1 in 370 readings under a normal distribution)

- ✅ Zero training, instant, easy to explain to stakeholders
- ❌ One metric at a time — cannot see multi-metric patterns
- ❌ **Blind to slow trends**: the rolling baseline creeps upward *with* a memory leak, so z stays near zero
- ❌ **Blind to flatlines**: when std → 0, z → 0, which reads as "perfectly normal"

### 3.2 Isolation Forest — the crowd detector

**Analogy:** a busy train station. Everyone moves in groups. One person stands perfectly still in the middle of the hall — they stand out because they're *easy to isolate* from the crowd.

The algorithm builds random decision trees and measures how many splits it takes to isolate each point:

```
Normal cluster                Anomaly
  ● ● ●                        ● ● ●
  ● ● ●   ← many splits        ● ● ●        ★  ← 1–2 splits
  ● ● ●      to isolate        ● ● ●
```

Fewer splits needed → more anomalous.

**Key parameters:**

| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 200 | More trees = more stable scores |
| `contamination` | 0.2 | Expected anomaly fraction — matches our ~20% simulator rate |
| `max_samples` | `'auto'` | Standard subsampling |
| `random_state` | 42 | Reproducibility |

- ✅ **Multivariate** — the only detector that sees relationships between metrics
- ✅ Fast to train, fast to score
- ❌ No concept of time ordering — shuffle the rows and it gives identical results
- ❌ Misses contextual anomalies (a value that's fine at 2pm but wrong at 2am)

### 3.3 LSTM Autoencoder — the pattern memory

**Analogy:** a security camera that has watched your office for six months. It knows people arrive at 9am, lunch is at noon, it's quiet after 6pm. When someone moves furniture at 3am it knows instantly — *"I have never seen this before."*

An **LSTM** is a neural network built for sequences in time. An **autoencoder** learns to compress data and then rebuild it. Combine them:

```
Input sequence          Compressed        Reconstruction
(last 10 readings)  →  (bottleneck)  →   (what it expected)

[38, 41, 40, 39...] →   [code]       →   [39, 40, 41, 38...]

reconstruction error = |input − output|

small error → the model recognised the pattern     ✅ normal
large error → it has never seen this before        🚨 anomaly
```

**Suggested architecture:**

```
Encoder:  LSTM(input=n_features, hidden=32) → LSTM(32 → 16)
Bottleneck: 16-dim latent vector
Decoder:  LSTM(16 → 32) → Linear(32 → n_features)

sequence_length : 10        (matches Phase 2's deque(maxlen=10))
loss            : MSE
optimizer       : Adam, lr=1e-3
epochs          : 50 with early stopping
threshold       : 99th percentile of reconstruction error on NORMAL data
```

> **Train the autoencoder on normal data only.** The whole idea is that it learns to reconstruct *normal* well and *anomalies* badly. Feed it anomalies during training and it learns to reconstruct those too — destroying the signal. Filter using the ground-truth labels: `is_anomaly == false`.

- ✅ Understands time and sequence — catches trends, flatlines, level shifts
- ✅ The **only** detector that can catch contextual anomalies
- ❌ Needs the most data and the longest training time
- ❌ Hardest to explain (Phase 4's SHAP work addresses explainability)

---

## 4. The 11 Anomaly Shapes → Which Detector Catches Each

**This table is the architectural justification for the whole phase.** When an interviewer asks *"why three models instead of one?"*, this is the answer.

```
SHAPE               WHAT IT LOOKS LIKE            z-score  IsoForest  LSTM
──────────────────────────────────────────────────────────────────────────────
spike           ────┐  ┌────                        ✅        ✅       ✅
                    └──┘

plateau         ────┐                               ✅        ✅       ✅
                    └───────────

trend                      ______/                  ❌        ⚠️       ✅
                    ______/                     (baseline creeps
                                                  with the leak)

cliff           ───────────┐                        ⚠️        ✅       ✅
                           └───

dip             ────┐                               ⚠️        ✅       ✅
                    └───────    (looks "healthy")

flatline        ~~~~━━━━━━━━━                       ❌        ⚠️       ✅
                    (noise vanishes)            (std→0, z→0)

variance        ~~~~/\/\/\/\/\                      ⚠️        ✅       ✅
                    (mean fine, swing wild)

level_shift     ────┐                               ⚠️        ✅       ✅
                    └━━━━━━━━ (new normal)

correlation     lat ↑↑ but cpu ↓↓                   ❌        ✅       ✅
  _break            (they normally agree)

contextual      normal values, wrong TIME           ❌        ❌       ✅

transient       ──┐                                 ✅        ✅       ⚠️
                  └─── (self-heals)
```

Read the ❌ column-wise. No single detector covers everything. That is the entire point of the ensemble.

---

## 5. Step 0 — Simulator Upgrade ✅ COMPLETE

Before collecting a single row, the simulator was rewritten so it produces every shape above.

### 5.1 The core change: anomalies are **episodes**, not rows

```
OLD                          NEW
tick 1: normal               tick 1:  normal
tick 2: 🔴 CPU SPIKE         tick 2:  🟠 memory_leak   0%   mem 54%
tick 3: normal               tick 3:  🟠 memory_leak   4%   mem 56%
tick 4: normal               ...
                             tick 60: 🟠 memory_leak  95%   mem 91%
                             tick 61: normal (cooldown)
```

**Why this matters:** an LSTM reads *sequences*. If every anomaly is one isolated row, there is no sequence to learn and the LSTM is pointless. Real incidents last minutes, not one reading.

### 5.2 New file layout

```
simulator/
├── shapes.py      ← pure math: clamp, lerp, add_noise, jitter,
│                     ramp, ease_in, plateau, decay, cliff, pulse
├── scenarios.py   ← the catalog: 23 Scenario dataclasses
├── generator.py   ← healthy baseline + ServiceState machine + simulated clock
└── runner.py      ← send loop, CLI, ground-truth logging, run summary
```

### 5.3 The 23 scenarios

```
▸ SPIKE               cpu_spike, error_burst
▸ PLATEAU             cpu_exhaustion, db_pool_exhaustion, disk_full
▸ TREND               memory_leak, disk_filling, latency_creep
▸ CLIFF               memory_exhaustion
▸ DIP                 traffic_drop, network_packet_loss
▸ FLATLINE            service_flatline
▸ VARIANCE            noisy_neighbor, intermittent_failure
▸ LEVEL_SHIFT         deployment_regression
▸ CORRELATION_BREAK   slow_query_storm, cache_failure, zombie_process
▸ SURGE               traffic_surge, ddos_attack, retry_storm
▸ CONTEXTUAL          off_hours_surge
▸ TRANSIENT           cold_start
```

**The four most instructive:**

| Scenario | Signature | Why it matters |
|---|---|---|
| `slow_query_storm` | latency ↑↑ **but CPU ↓↓** | Server is *waiting* on the DB, not computing. These two normally rise together. Only a multivariate model sees it. |
| `service_flatline` | Values frozen, **stdev = 0** | Hung monitoring agent. Z-score is structurally blind: std→0 means z→0 means "normal". |
| `off_hours_surge` | Normal values, **wrong hour** | 110 req/s and 40% CPU are great numbers — at 3am they're an incident. Only a time-aware model catches this. |
| `deployment_regression` | Permanent new baseline | Starts as an anomaly; if unfixed it *becomes* normal. That's **concept drift** — the problem Phase 7 solves. |

### 5.4 The state machine

```
     ┌──────────────────────────────────────────────────┐
     ▼                                                  │
  WARMUP ──> NORMAL ──(1.2%/tick)──> EPISODE ──> COOLDOWN
  12 ticks     ▲                    3–120 ticks   18–45 ticks
               └──────────────────────────────────────┘
```

- **Warmup** — no anomalies for the first 12 ticks, so Phase 2's `deque(maxlen=10)` fills with *clean* data. Without this, an early anomaly poisons the rolling average.
- **Cooldown** — forced healthy period so the rolling window recovers between incidents.
- **`progress`** — `ticks_elapsed / ticks_total`, 0.0 → 1.0. This is what lets `ramp()` and `ease_in()` know how far along a leak is.
- **Intensity** — each episode draws from `(0.45, 1.0)`, scaling its magnitude. Mild anomalies are included deliberately: a detector that only catches severe ones isn't much of a detector.

### 5.5 The simulated clock

Server behaviour follows a 24-hour rhythm, so training data must cover day *and* night. Waiting 24 real hours is wasteful.

```bash
--time-scale 30      # 1 real second = 30 simulated seconds
                     # → 1 real hour covers 30 simulated hours
```

**Critical:** the payload timestamp uses the same clock as the behaviour.

```python
now    = sim_now()
factor = get_time_of_day_factor(now)     # behaviour follows simulated time
...
"timestamp": now.isoformat(),             # ← and so does the timestamp
```

If the timestamp said `14:05` while metrics reflected 3am behaviour, the data would contradict itself and hour-of-day would be unlearnable.

### 5.6 Ground-truth labels — the highest-value addition

```python
payload, label = generate_tick(...)

# label:
{
  "is_anomaly":   True,
  "anomaly_type": "memory_leak",
  "category":     "trend",
  "severity":     "moderate",
  "intensity":    0.71,
  "progress":     0.34,
  "episode_id":   "a3f9c1b2",
  "hour_utc":     3.5,
  "tod_factor":   0.31
}
```

Written to a JSONL file, one line per reading.

```
WITHOUT labels:   "I built three models. They seem to work.
                   Here's a chart that looks plausible."

WITH labels:      "Precision 0.91, recall 0.87. We catch 98% of spikes
                   and 94% of trends but only 61% of contextual anomalies —
                   which is exactly why the LSTM is in the ensemble."
```

You cannot compute precision or recall without knowing which readings were genuinely anomalous. Now you do.

> Labels go to JSONL rather than Postgres because `MetricIngest` has no field for them (Pydantic silently drops extras). Join on `timestamp + service_name` in Phase 3. Adding a nullable `ground_truth` JSON column is an option if you'd rather have them in the DB.

### 5.7 Two bugs fixed during the rewrite

**Midnight discontinuity in the time-of-day curve:**

```python
# OLD — snapped discontinuously at midnight
factor = 0.5 + 0.5 * math.sin(math.pi * (hour - 2) / 12)

# NEW — smooth 24-hour cosine, continuous across midnight
wave = 0.5 + 0.5 * math.cos(2 * math.pi * (hour - 14.0) / 24.0)
```

The old version would have taught the LSTM a nightly cliff that doesn't exist in real servers.

**Unbounded compounding** in `deployment_regression` — repeated episodes multiplied the baseline shift without limit (reached 54× latency in testing). Capped via `MAX_BASELINE_SHIFT = 3.0`.

### 5.8 Simulator verification results

```
✅ All 23 scenarios fire
✅ Zero schema violations across 9,200 payloads (no 422s from MetricsPayload)
✅ db_connections always int
✅ flatline stdev = 0.000000                    (variance truly gone)
✅ slow_query_storm: lat 2478ms + cpu 17.7%     (correlation break)
✅ db_pool_exhaustion: lat 3455ms + cpu 12.6%   (counter-intuitive shape)
✅ noisy_neighbor stdev 24.1 vs normal 1.7      (14× variance)
✅ error_burst: err 56.9%, cpu 36.8%            (cpu stays normal)
✅ memory_leak: 61.6% → 87.8% across episode    (rises correctly)
✅ off_hours_surge fires at 3am, never at 2pm   (precondition works)
✅ deployment_regression capped at 3.0×
✅ time-scale 30: 24/24 hours of day covered, tod_factor spans 0.30–1.00
```

---

## 6. Data Collection Runbook

### 6.1 Pre-flight — four things that will corrupt the data

#### 🔴 Problem 1: the Celery worker shreds rolling windows

`features.py` keeps history in **process memory**:

```python
_history: dict[str, deque] = {}
```

Celery's default pool forks **one process per CPU core**. Each gets its own copy:

```
Task 1 (payment cpu=40) → process #3  → deque: [40]
Task 2 (payment cpu=42) → process #7  → deque: [42]     ← different process
Task 3 (payment cpu=41) → process #1  → deque: [41]
```

Every process sees ~1/8 of readings, out of order. `rolling_avg`, `z_score`, and `rate_of_change` become garbage — and those are exactly what Phase 3 consumes.

**Fix — always collect with a single-process worker:**

```bash
celery -A src.workers.celery_app.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo` runs tasks one at a time in the main process: one `_history`, correct order. It also avoids a known macOS fork-safety crash.

> **Known limitation worth documenting in the README:** in-memory state inside a task worker breaks horizontal scaling — the very thing Phase 2 was built for. The proper fix is to keep the rolling window in Redis. `--pool=solo` is the correct workaround for data collection, not a production answer.

#### Problem 2: poisoned rows already in the database

`test.py` deliberately sent `memory_usage: None`, which `clean_metrics()` filled with `0`. A 0% memory reading is physically impossible on a running server — Isolation Forest would learn from it.

```bash
docker exec -it anomaly_db psql -U admin -d anomaly_db -c "
  TRUNCATE metric_datapoints, metric_aggregations RESTART IDENTITY;"
```

#### Problem 3: two virtual environments

The project root contains both `.venv/` and `venv/`. Pick one; confirm with `which python`.

#### Problem 4: `.gitignore` gaps

```gitignore
simulator/gt_*.jsonl
simulator/ground_truth*.jsonl
celerybeat-schedule
```

### 6.2 Startup order

```
  1. Docker      (Postgres + Redis)     ← nothing works without this
  2. Migrations  (tables exist)
  3. API         (receives payloads)
  4. Worker      (drains the queue)     ← MUST precede the simulator
  5. Simulator   (produces data)        ← LAST
```

Redis holds queued tasks so nothing is *lost* if the worker starts late — but the backlog processes out of order, corrupting rolling windows again. Worker first.

**Terminal 0 — setup**
```bash
cd ~/Desktop/Projects/anomaly-detection-platform
source .venv/bin/activate
docker compose up -d
docker compose ps                 # both containers "running"
alembic upgrade head
```

**Terminal 1 — API**
```bash
uvicorn src.main:app --reload --port 8080
```
✅ `curl http://127.0.0.1:8080/health`

**Terminal 2 — Worker**
```bash
celery -A src.workers.celery_app.celery_app worker --loglevel=info --pool=solo
```
✅ Banner lists `tasks.process_metrics`

**Terminal 3 — Beat: SKIP during time-scaled runs.**
`aggregate_metrics` looks back 5 *real* minutes from `datetime.now()`. With `--time-scale 30` the DB timestamps race into the simulated future, so that window finds nothing. Run Beat later during a normal-speed demo.

**Terminal 4 — Simulator**
```bash
python -m simulator.runner \
    --time-scale 30 \
    --interval 1 \
    --duration 90 \
    --ground-truth simulator/gt_main.jsonl
```

### 6.3 How long to run — measured, not guessed

The real unit is **samples per service**, not rows. The table is narrow/long — 7 rows per reading — but Isolation Forest trains on all 7 metrics *together as one vector*:

```
1 payload  =  7 database rows  =  1 training SAMPLE
```

| real min | interval | scale | sim hours | hours seen | samples/service | db rows | anomaly % | scenarios |
|---|---|---|---|---|---|---|---|---|
| 18 | 1s | 30 | 24 | 24/24 | 1,080 | 22,680 | 16.7% | 14/23 |
| 30 | 1s | 30 | 25 | 24/24 | 1,800 | 37,800 | 20.7% | 18/23 |
| 45 | 1s | 40 | 30 | 24/24 | 2,700 | 56,700 | 23.9% | 23/23 |
| **60** | **1s** | **30** | **30** | **24/24** | **3,600** | **75,600** | **19.6%** | **23/23** ✅ |
| **90** | **1s** | **30** | **45** | **24/24** | **5,400** | **113,400** | **21.2%** | **23/23** ⭐ |

**Minimum viable: 60 minutes. Recommended: 90 minutes.**

Below 45 minutes not all 23 scenarios fire — the long trend episodes (`memory_leak` 40–95 ticks, `disk_filling` 55–120) don't get enough chances.

**Model requirements per service:**

| Model | Minimum | Comfortable |
|---|---|---|
| Z-score | 10 (rolling window) | any |
| Isolation Forest | ~1,500 | 5,000+ |
| LSTM Autoencoder | ~2,000 sequences | 8,000+ |

### 6.4 One run, not three

Three separate sessions would each start their simulated clock at today's date → **overlapping timestamps** in the same table, i.e. two incompatible histories interleaved at the same times. Unlearnable.

One continuous run gives one coherent timeline, and everything comes out of it:

```
Pure normal data for the LSTM?
    → filter gt_main.jsonl where is_anomaly == false

Train/test split?
    → split CHRONOLOGICALLY:  first 60% train │ next 20% val │ last 20% test
```

> Chronological splitting is **mandatory** for time-series, not a preference. A random split lets the model train on future readings while testing on past ones — data leakage that inflates accuracy and collapses on real data.

### 6.5 Post-run verification

```bash
# 1. Volume and time span
docker exec anomaly_db psql -U admin -d anomaly_db -c "
  SELECT service_name, count(*) AS rows, count(*)/7 AS samples,
         min(timestamp), max(timestamp)
  FROM metric_datapoints GROUP BY service_name;"

# 2. ⚠️ MOST IMPORTANT — are features present?
docker exec anomaly_db psql -U admin -d anomaly_db -c "
  SELECT count(*) AS null_labels FROM metric_datapoints WHERE labels IS NULL;"
#    MUST return 0. If null, feature engineering never ran → data is unusable.

# 3. Did the diurnal cycle get captured?
docker exec anomaly_db psql -U admin -d anomaly_db -c "
  SELECT extract(hour from timestamp) AS hr, count(*)
  FROM metric_datapoints WHERE metric_name='requests_per_sec'
  GROUP BY hr ORDER BY hr;"
#    Should show all 24 hours, roughly even counts.

# 4. Queue keeping up? (run during collection)
docker exec anomaly_redis redis-cli llen celery
#    Should hover near 0. Climbing = worker can't keep up.
```

---

## 7. New Project Structure

```
src/
├── ml/
│   ├── __init__.py
│   ├── data_prep.py          🆕 pivot narrow→wide, scale, build sequences
│   ├── zscore_detector.py    🆕 threshold logic (no training)
│   ├── isolation_forest.py   🆕 train / save / load / score
│   ├── lstm_autoencoder.py   🆕 PyTorch model + train / save / load / score
│   ├── ensemble.py           🆕 majority vote over the three
│   └── model_registry.py     🆕 disk persistence, versioning, lazy loading
├── models/
│   └── anomaly_score.py      🆕 AnomalyScore SQLAlchemy table
├── schemas/
│   └── anomaly.py            🆕 Pydantic response shapes
├── api/routes/
│   └── anomalies.py          🆕 GET /api/v1/anomalies
├── workers/
│   └── tasks.py              ✏️  add score_anomaly() into process_metrics
└── db/migrations/versions/
    └── xxxx_add_anomaly_scores_table.py   🆕

ml_models/                    ← saved artefacts (gitignored)
├── isolation_forest.joblib
├── scaler.joblib             ← MUST be saved with the model
├── lstm_autoencoder.pt
├── lstm_threshold.json
└── metadata.json             ← trained_at, sample count, feature order, metrics

scripts/
├── train_models.py           🆕 read DB → train all → save to ml_models/
└── evaluate_models.py        🆕 join ground truth → precision/recall report
```

---

## 8. Build Steps

One mini-step at a time; confirm each before moving on.

### Step 1 — `AnomalyScore` table

SQLAlchemy model + Alembic migration. Design below in §9.

### Step 2 — Data preparation (`data_prep.py`)

**This step is easy to underestimate.** The database is narrow/long, but ML needs wide:

```
metric_datapoints (narrow/long)          →   ML input (wide)
─────────────────────────────────────        ────────────────────────────────
ts=10:00 payment cpu_usage      42.1        ts=10:00 payment  cpu=42.1
ts=10:00 payment memory_usage   56.3                          mem=56.3
ts=10:00 payment latency        71.2                          lat=71.2
ts=10:00 payment requests_ps   104.0                          rps=104.0
ts=10:00 payment error_rate      0.002                        err=0.002
ts=10:00 payment db_connections 18                            db=18
ts=10:00 payment disk_usage     46.2                          disk=46.2
   (7 rows)                                    (1 row × 7 columns)
```

A pandas pivot on `(timestamp, service_name)`. Also in this step:

- Extract the 3 engineered features from the `labels` JSON column
- Encode hour-of-day **cyclically**: `sin(2πh/24)`, `cos(2πh/24)` — so 23:00 and 00:00 are adjacent, not 23 units apart
- Fit `StandardScaler` **on training data only**, then save it

> ⚠️ **Classic mistake:** re-fitting the scaler at inference time. The scaler is part of the model — persist it alongside and load it, never refit. Refitting on live data silently rescales everything and the model's thresholds become meaningless.

### Step 3 — Z-score detector

Reads `z_score` from the `labels` column, applies `|z| > 3.0`. No training. Build this first — it gives a working end-to-end detector to test the pipeline with before ML complexity arrives.

### Step 4 — Isolation Forest

Train on the wide, scaled feature matrix. Save with `joblib`. Note `score_samples()` returns *negative* scores where lower = more anomalous; normalise to 0–1 so the ensemble can compare detectors on the same scale.

### Step 5 — LSTM Autoencoder

PyTorch. Sequences of shape `(batch, 10, n_features)`. **Train on normal data only** (filter by ground-truth label). Threshold = 99th percentile of reconstruction error on held-out normal data; save it to `lstm_threshold.json`.

### Step 6 — Model registry

Lazy-load models once per worker process and cache them. Loading a PyTorch model from disk on every task would dominate runtime. Include a `metadata.json` recording `trained_at`, sample count, feature order, and evaluation scores — feature order especially, since a mismatch between training and inference silently produces garbage.

### Step 7 — Ensemble

Majority vote (≥2 of 3). Output a final score in 0.0–1.0 plus the individual detector verdicts, so Phase 4 and the dashboard can show *which* detectors fired.

### Step 8 — Wire into Celery

Add `score_anomaly()` between `compute_features()` and the DB write in `process_metrics`. Save one `AnomalyScore` row per reading.

**Wrap scoring in try/except.** If a model file is missing or corrupt, the task should fall back to z-score only and still save the metric rows — ingestion must never break because ML broke.

### Step 9 — Evaluate (`evaluate_models.py`)

Join DB rows to `gt_main.jsonl` on `timestamp + service_name`. Produce the report in §10. **Do not skip this step** — it's what turns the project from "I built three models" into "here are my numbers".

---

## 9. The `AnomalyScore` Table

```python
class AnomalyScore(Base):
    __tablename__ = "anomaly_scores"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp       = Column(DateTime(timezone=True), nullable=False, index=True)
    service_name    = Column(String(100), nullable=False, index=True)
    host            = Column(String(100), nullable=False)
    metric_name     = Column(String(100), nullable=True)   # null = multivariate

    # individual detector outputs — keep all three, not just the verdict
    zscore_value    = Column(Float, nullable=True)
    zscore_flag     = Column(Boolean, default=False)
    iforest_score   = Column(Float, nullable=True)
    iforest_flag    = Column(Boolean, default=False)
    lstm_error      = Column(Float, nullable=True)
    lstm_flag       = Column(Boolean, default=False)

    # ensemble result
    votes           = Column(Integer, nullable=False, default=0)   # 0–3
    ensemble_score  = Column(Float, nullable=False)                # 0.0–1.0
    is_anomaly      = Column(Boolean, nullable=False, default=False, index=True)

    model_version   = Column(String(50), nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
```

**Why store all three detector outputs rather than just the final verdict:**

1. Phase 4's SHAP explanations need to know *which* detector fired
2. Phase 6's dashboard can show a per-detector breakdown
3. You can retune the ensemble threshold later **without re-scoring anything**
4. It's how you produce the per-detector recall table in §10

**Index choices:** `timestamp` and `service_name` for dashboard time-range queries; `is_anomaly` because "show me the anomalies" is the most common filter and it's highly selective at ~20%.

---

## 10. Evaluation Methodology

This section is what makes the project credible.

### 10.1 Metrics

```
                   Predicted anomaly    Predicted normal
Actually anomaly       TP                    FN  ← missed incident (worst)
Actually normal         FP  ← false alarm     TN

precision = TP / (TP + FP)     "when we alert, how often are we right?"
recall    = TP / (TP + FN)     "of all real incidents, how many did we catch?"
F1        = harmonic mean       single number balancing both
```

For monitoring, **recall matters more than precision** — a missed outage costs far more than one extra alert. But precision below ~0.5 causes alert fatigue and people start ignoring the system, so it can't be ignored either.

### 10.2 The report to produce

```
OVERALL
  precision 0.91   recall 0.87   F1 0.89

PER DETECTOR
  detector            precision  recall
  z-score               0.94      0.41
  isolation_forest      0.88      0.79
  lstm                  0.85      0.83
  ensemble (2 of 3)     0.91      0.87    ← better than any single one

PER SHAPE  (this is the interesting table)
  category            recall   n
  spike                0.99   142
  plateau              0.96   318
  surge                0.97   201
  trend                0.88   903    ← z-score alone gets ~0.2 here
  cliff                0.91   149
  dip                  0.93   452
  flatline             0.85   244    ← z-score alone gets ~0.0
  variance             0.86   612
  level_shift          0.79   314
  correlation_break    0.84   825    ← IsoForest carries this
  contextual           0.61    69    ← LSTM only; hardest by design
  transient            0.94    76

PER SEVERITY
  severe    0.98      moderate  0.89      mild  0.68
```

Those numbers are illustrative, not predictions — you'll produce real ones. The *shape* of the story is what matters: the ensemble beats every individual detector, and the difficulty gradient across shapes and severities is visible and explainable.

### 10.3 The chronological split

```
│───────────── 60% ─────────────│──── 20% ────│──── 20% ────│
        TRAIN                        VAL           TEST
   (fit models, fit scaler)      (thresholds)   (final report)
   ←──────────────────────── time ────────────────────────→
```

Never shuffle. Never fit the scaler on val or test.

---

## 11. Key Design Decisions

- **Ensemble over a single model** — no single detector covers all 11 shapes (§4). Majority voting cuts false alarms while keeping recall.
- **Unsupervised, not supervised** — the models learn "normal" from unlabelled data. Ground-truth labels are used *only for evaluation*, never for training. Real production systems don't get labels, so training on them would be cheating.
- **Autoencoder trained on normal data only** — its entire signal comes from reconstructing normal well and anomalies badly.
- **All three detector scores persisted** — enables retuning, explainability, and the per-detector table without re-scoring.
- **Cyclical hour encoding** — so 23:00 and 00:00 are neighbours, not 23 units apart.
- **Scaler persisted with the model** — a scaler refit at inference silently invalidates every threshold.
- **Scoring wrapped in try/except** — ingestion must survive a missing or corrupt model file.
- **Models lazy-loaded and cached per worker** — disk loads on every task would dominate runtime.
- **Chronological train/test split** — random splitting leaks future information into training.
- **Episode-based anomalies in the simulator** — gives the LSTM sequences to learn and matches real incident duration.
- **`--time-scale` for collection** — full diurnal coverage in one hour instead of 24.

---

## 12. Known Limitations

| Limitation | Impact | Planned fix |
|---|---|---|
| `_history` lives in worker process memory | Must run `--pool=solo`; breaks horizontal scaling | Move the rolling window to Redis |
| Ground truth in JSONL, not Postgres | Requires a join on timestamp + service_name | Optional nullable `ground_truth` JSON column |
| `aggregate_metrics` is meaningless during time-scaled runs | No 5-min summaries in scaled data | Run Beat only during normal-speed demos |
| Models trained once, never refreshed | `deployment_regression` drift degrades accuracy over time | Phase 7 drift detection + auto-retraining |
| Contextual recall will be the weakest | `off_hours_surge` is genuinely hard | Expected; report it honestly rather than hiding it |

---

## 13. Verification Checklist

Phase 3 is complete when all of these pass:

```
Data
  ☐ 60+ minutes collected, ≥3,000 samples per service
  ☐ zero NULL values in the labels column
  ☐ all 24 hours of day present in the data
  ☐ all 23 scenarios appear in the ground-truth file
  ☐ worker ran with --pool=solo throughout

Models
  ☐ Isolation Forest trains and saves to ml_models/
  ☐ scaler.joblib saved alongside it
  ☐ LSTM Autoencoder trains, loss decreases, saves to disk
  ☐ lstm_threshold.json written from held-out normal data
  ☐ all models reload from disk in a fresh process and score correctly
  ☐ metadata.json records trained_at, sample count, feature order

Pipeline
  ☐ score_anomaly() runs inside process_metrics
  ☐ one anomaly_scores row per reading
  ☐ ensemble_score always in 0.0–1.0
  ☐ all three detector scores populated, not just the verdict
  ☐ a deleted model file degrades to z-score only WITHOUT breaking ingestion
  ☐ GET /api/v1/anomalies returns scored readings

Evaluation
  ☐ chronological 60/20/20 split, no shuffling
  ☐ overall precision / recall / F1 computed
  ☐ per-detector table produced
  ☐ per-shape recall table produced
  ☐ per-severity table produced
  ☐ ensemble outperforms every individual detector
```

---

## 14. What's Next — Phase 4 Preview

Phase 3 answers *"is this abnormal right now?"* Phase 4 answers *"is this heading for an outage, and why?"*

- **XGBoost classifier** — predicts incidents *before* they happen from the engineered features. Supervised this time, using episode outcomes as labels.
- **SHAP explainability** — turns a score into a sentence: *"CPU rising + error rate spiking + latency climbing = 87% chance of incident within 10 minutes."*
- **Natural-language explanations** stored on each prediction
- **Confidence scores** persisted alongside predictions
- Wired into the same Celery pipeline

The 23 labelled scenarios pay off again here: `episode_id` and `progress` let you train on *early* readings of an episode to predict its *later* severity — which is exactly what incident prediction is.
